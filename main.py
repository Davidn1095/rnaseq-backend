import io
import json
import math
import time
import uuid
import zipfile
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

try:
    import harmonypy as hm  # type: ignore
    _HARMONY_OK = True
except Exception:
    hm = None
    _HARMONY_OK = False


# =========================
# App
# =========================
app = FastAPI(title="RNA-seq preprocessing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Run store (demo)
# NOTE: In-memory store is NOT durable on Cloud Run if the instance restarts or scales.
# For a durable pipeline you would store artifacts in Cloud Storage / a DB keyed by run_id.
# =========================
@dataclass
class RunArtifacts:
    created_at: float
    filename: str
    counts: pd.DataFrame

    qc: Optional[Dict[str, Any]] = None
    normalized: Optional[pd.DataFrame] = None
    pca: Optional[pd.DataFrame] = None
    harmony: Optional[pd.DataFrame] = None
    clusters: Optional[pd.Series] = None
    train_metrics: Optional[Dict[str, Any]] = None

    meta: Dict[str, Any] = field(default_factory=dict)


_RUNS: Dict[str, RunArtifacts] = {}
_RUNS_LOCK = threading.Lock()
_TTL_SECONDS = 6 * 3600  # 6 hours


def _gc_runs() -> None:
    now = time.time()
    with _RUNS_LOCK:
        dead = [rid for rid, r in _RUNS.items() if (now - r.created_at) > _TTL_SECONDS]
        for rid in dead:
            del _RUNS[rid]


def _summary_5num(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {"min": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "max": float("nan")}
    return {
        "min": float(x.min()),
        "p25": float(x.quantile(0.25)),
        "p50": float(x.quantile(0.50)),
        "p75": float(x.quantile(0.75)),
        "max": float(x.max()),
    }


def _read_count_matrix(raw: bytes, filename: str) -> pd.DataFrame:
    # Accept CSV/TSV, infer separator
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read CSV/TSV: {e}")

    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Count matrix must have at least 2 columns (gene_id + >=1 cell).")

    # First column is gene IDs
    df = df.set_index(df.columns[0])

    # Coerce numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().all().all():
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion.")

    # Replace remaining NAs with 0 for counts
    df = df.fillna(0)

    # Basic validation: non-negative
    if (df.values < 0).any():
        raise HTTPException(status_code=400, detail="Counts contain negative values.")

    # Ensure unique gene IDs (keep first)
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]

    return df


def _get_or_create_run(run_id: Optional[str], filename: str, counts: pd.DataFrame) -> str:
    _gc_runs()
    rid = run_id.strip() if isinstance(run_id, str) and run_id.strip() else uuid.uuid4().hex
    with _RUNS_LOCK:
        _RUNS[rid] = RunArtifacts(created_at=time.time(), filename=filename, counts=counts)
    return rid


def _get_run(run_id: str) -> RunArtifacts:
    _gc_runs()
    with _RUNS_LOCK:
        r = _RUNS.get(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="run_id not found. Re-run /upload or /qc.")
    return r


def _ensure_normalized(r: RunArtifacts) -> pd.DataFrame:
    if r.normalized is not None:
        return r.normalized

    df = r.counts
    libsize = df.sum(axis=0)
    libsize_safe = libsize.replace(0, np.nan)
    cpm = df.div(libsize_safe, axis=1) * 1e6
    log1p_cpm = np.log1p(cpm.fillna(0))

    r.normalized = log1p_cpm
    r.meta["libsize"] = libsize.to_dict()
    return r.normalized


def _ensure_pca(r: RunArtifacts, n_components: int = 30) -> pd.DataFrame:
    if r.pca is not None:
        return r.pca

    norm = _ensure_normalized(r)  # genes x cells
    X = norm.T.values  # cells x genes

    n_cells, n_genes = X.shape
    k = int(max(2, min(n_components, n_cells - 1, n_genes)))
    pca = PCA(n_components=k, random_state=0)
    Z = pca.fit_transform(X)  # cells x k

    cols = [f"PC{i+1}" for i in range(Z.shape[1])]
    r.pca = pd.DataFrame(Z, index=norm.columns, columns=cols)
    r.meta["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    return r.pca


def _assign_fake_batches(cell_ids: np.ndarray, n_batches: int) -> pd.Series:
    # Deterministic batching by position (demo only).
    n_batches = int(max(1, n_batches))
    batches = np.array([f"batch_{(i % n_batches) + 1}" for i in range(len(cell_ids))], dtype=object)
    return pd.Series(batches, index=cell_ids, name="batch")


def _ensure_harmony(r: RunArtifacts, n_batches: int = 1) -> pd.DataFrame:
    if r.harmony is not None:
        return r.harmony

    pca_df = _ensure_pca(r)
    cell_ids = pca_df.index.to_numpy()

    # If Harmony not installed or only 1 batch, keep PCA as the "harmony" embedding.
    if (not _HARMONY_OK) or int(n_batches) <= 1:
        r.harmony = pca_df.copy()
        r.meta["harmony"] = {"applied": False, "reason": "harmonypy missing or n_batches<=1"}
        return r.harmony

    meta = pd.DataFrame({"batch": _assign_fake_batches(cell_ids, int(n_batches)).values}, index=cell_ids)

    # harmonypy expects features x samples (PCs x cells)
    Z = pca_df.values.T
    try:
        ho = hm.run_harmony(Z, meta, vars_use=["batch"])
        Zcorr = ho.Z_corr.T  # cells x PCs
    except Exception as e:
        # Fallback to PCA if Harmony fails
        r.harmony = pca_df.copy()
        r.meta["harmony"] = {"applied": False, "reason": f"harmony failed: {e}"}
        return r.harmony

    harm = pd.DataFrame(Zcorr, index=pca_df.index, columns=pca_df.columns)
    r.harmony = harm
    r.meta["harmony"] = {"applied": True, "n_batches": int(n_batches)}
    return r.harmony


def _ensure_clusters(r: RunArtifacts, k: int = 6, n_batches: int = 1) -> pd.Series:
    if r.clusters is not None:
        return r.clusters

    emb = _ensure_harmony(r, n_batches=n_batches)
    X = emb.values

    k = int(max(2, min(k, X.shape[0])))
    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = km.fit_predict(X)

    r.clusters = pd.Series(labels.astype(int), index=emb.index, name="cluster")
    r.meta["kmeans_inertia"] = float(km.inertia_)
    return r.clusters


def _ensure_train(r: RunArtifacts, n_batches: int = 1) -> Dict[str, Any]:
    if r.train_metrics is not None:
        return r.train_metrics

    emb = _ensure_harmony(r, n_batches=n_batches)  # cells x PCs
    y = _ensure_clusters(r, k=6, n_batches=n_batches).values
    X = emb.values

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        metrics = {
            "cv_folds": 0,
            "accuracy_mean": 1.0,
            "accuracy_sd": 0.0,
            "macro_f1_mean": 1.0,
            "macro_f1_sd": 0.0,
            "note": "Only one cluster present, training is trivial.",
        }
        r.train_metrics = metrics
        return metrics

    min_count = int(counts.min())
    n_splits = int(min(5, min_count))

    # If too few samples per class for CV, fit once on full data and report in-sample metrics.
    if n_splits < 2:
        clf = LogisticRegression(max_iter=2000, multi_class="auto")
        clf.fit(X, y)
        pred = clf.predict(X)
        metrics = {
            "cv_folds": 0,
            "accuracy_mean": float(accuracy_score(y, pred)),
            "accuracy_sd": 0.0,
            "macro_f1_mean": float(f1_score(y, pred, average="macro")),
            "macro_f1_sd": 0.0,
            "note": f"CV skipped because smallest cluster has {min_count} samples. Reported metrics are in-sample.",
        }
        r.train_metrics = metrics
        return metrics

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    accs: list[float] = []
    f1s: list[float] = []

    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, multi_class="auto")
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))

    metrics = {
        "cv_folds": int(n_splits),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "note": "Demo training predicts unsupervised clusters (no ground-truth labels provided).",
    }
    r.train_metrics = metrics
    return metrics



async def _extract_run_id_from_request(request: Request) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Accept run_id via:
      - application/json body: {"run_id": "...", ...}
      - multipart/form-data: run_id field
    Returns: (run_id, extra_fields_dict)
    """
    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                return None, {}
            rid = body.get("run_id")
            extra = {k: v for k, v in body.items() if k != "run_id"}
            return (str(rid) if rid is not None else None), extra
        except Exception:
            return None, {}

    # Form (multipart or urlencoded)
    try:
        form = await request.form()
        rid = form.get("run_id")
        extra = {k: form.get(k) for k in form.keys() if k != "run_id"}
        return (str(rid) if rid is not None else None), extra
    except Exception:
        return None, {}


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True, "harmony_available": _HARMONY_OK}


@app.post("/upload")
async def upload(file: UploadFile = File(...), request: Request = None):  # request kept for parity
    if not file.filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(status_code=400, detail="Upload a .csv or .tsv count matrix")

    raw = await file.read()
    counts = _read_count_matrix(raw, file.filename)
    rid = _get_or_create_run(None, file.filename, counts)

    return {
        "ok": True,
        "run_id": rid,
        "filename": file.filename,
        "shape": [int(counts.shape[0]), int(counts.shape[1])],
    }


@app.post("/qc")
async def qc(file: UploadFile = File(...), request: Request = None):
    """
    QC always accepts file (frontend sends it).
    If a run_id field is included in multipart, QC will attach results to that run_id.
    """
    rid_from_body, _ = await _extract_run_id_from_request(request) if request else (None, {})
    run_id = rid_from_body

    if not file.filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(status_code=400, detail="Upload a .csv or .tsv count matrix")

    raw = await file.read()
    counts = _read_count_matrix(raw, file.filename)

    # Create or overwrite run
    rid = _get_or_create_run(run_id, file.filename, counts)
    r = _get_run(rid)

    total_counts = counts.sum(axis=0)
    detected_genes = (counts > 0).sum(axis=0)
    pct_zeros = (counts == 0).sum(axis=0) / counts.shape[0]

    qc_summary = {
        "ok": True,
        "run_id": rid,
        "filename": file.filename,
        "shape": [int(counts.shape[0]), int(counts.shape[1])],
        "total_counts": _summary_5num(total_counts),
        "detected_genes": _summary_5num(detected_genes),
        "pct_zeros": _summary_5num(pct_zeros),
    }

    r.qc = qc_summary
    return qc_summary


@app.post("/normalize")
async def normalize(file: UploadFile = File(...), request: Request = None):
    """
    Accepts file. If run_id exists in multipart, normalization is attached to that run_id,
    otherwise a new run_id is created.
    """
    rid_from_body, _ = await _extract_run_id_from_request(request) if request else (None, {})
    run_id = rid_from_body

    if not file.filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(status_code=400, detail="Upload a .csv or .tsv count matrix")

    raw = await file.read()
    counts = _read_count_matrix(raw, file.filename)

    rid = _get_or_create_run(run_id, file.filename, counts)
    r = _get_run(rid)

    norm = _ensure_normalized(r)
    libsize = pd.Series(r.meta.get("libsize", {}))

    return {
        "ok": True,
        "run_id": rid,
        "filename": file.filename,
        "shape": [int(norm.shape[0]), int(norm.shape[1])],
        "libsize": {str(k): float(v) for k, v in libsize.to_dict().items()} if not libsize.empty else {},
        "log1p_cpm_summary": {
            "min": float(norm.min().min()),
            "median": float(norm.stack().median()),
            "max": float(norm.max().max()),
        },
        "note": "Normalized matrix stored for downstream steps and /export.",
    }


@app.post("/harmony")
async def harmony(request: Request):
    run_id, extra = await _extract_run_id_from_request(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])

    n_batches = int(extra.get("n_batches") or 1)

    r = _get_run(run_id)
    harm = _ensure_harmony(r, n_batches=n_batches)

    return {
        "ok": True,
        "run_id": run_id,
        "embedding": "harmony" if r.meta.get("harmony", {}).get("applied") else "pca_fallback",
        "shape": [int(harm.shape[0]), int(harm.shape[1])],
        "harmony_meta": r.meta.get("harmony", {}),
    }


@app.post("/cluster")
async def cluster(request: Request):
    run_id, extra = await _extract_run_id_from_request(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])

    k = int(extra.get("k") or 6)
    n_batches = int(extra.get("n_batches") or 1)

    r = _get_run(run_id)
    cl = _ensure_clusters(r, k=k, n_batches=n_batches)

    counts = cl.value_counts().sort_index()
    return {
        "ok": True,
        "run_id": run_id,
        "k": int(counts.shape[0]),
        "cluster_sizes": {str(int(i)): int(v) for i, v in counts.to_dict().items()},
        "note": "Clusters stored for /train and /export.",
    }


@app.post("/train")
async def train(request: Request):
    run_id, extra = await _extract_run_id_from_request(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])

    n_batches = int(extra.get("n_batches") or 1)

    r = _get_run(run_id)
    metrics = _ensure_train(r, n_batches=n_batches)

    return {"ok": True, "run_id": run_id, "metrics": metrics}


@app.post("/export")
async def export(request: Request):
    run_id, _ = await _extract_run_id_from_request(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])

    r = _get_run(run_id)

    # Ensure pipeline artifacts exist (best-effort)
    if r.qc is None:
        # No file is available here, so we cannot compute QC; export what exists.
        pass
    _ensure_normalized(r)
    _ensure_pca(r)
    _ensure_harmony(r, n_batches=int(r.meta.get("harmony", {}).get("n_batches", 1) or 1))
    if r.clusters is None:
        _ensure_clusters(r, k=6, n_batches=int(r.meta.get("harmony", {}).get("n_batches", 1) or 1))
    if r.train_metrics is None:
        _ensure_train(r, n_batches=int(r.meta.get("harmony", {}).get("n_batches", 1) or 1))

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # qc
        if r.qc is not None:
            z.writestr("qc_summary.json", json.dumps(r.qc, indent=2))

        # normalized (genes x cells)
        if r.normalized is not None:
            z.writestr("normalized_log1p_cpm.csv", r.normalized.to_csv())

        # pca/harmony (cells x PCs)
        if r.pca is not None:
            z.writestr("pca_embedding.csv", r.pca.to_csv(index=True))

        if r.harmony is not None:
            z.writestr("harmony_embedding.csv", r.harmony.to_csv(index=True))

        # clusters
        if r.clusters is not None:
            z.writestr("clusters.csv", r.clusters.to_frame().to_csv(index=True))

        # training metrics
        if r.train_metrics is not None:
            z.writestr("train_metrics.json", json.dumps(r.train_metrics, indent=2))

        # meta
        z.writestr("run_meta.json", json.dumps({"run_id": run_id, "filename": r.filename, "meta": r.meta}, indent=2))

    mem.seek(0)

    return StreamingResponse(
        mem,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="rnaseq_export.zip"'},
    )
