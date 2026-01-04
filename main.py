import io
import time
import uuid
import math
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Optional: Harmony. If not installed, /harmony will fall back to "no correction".
try:
    import harmonypy as hm  # type: ignore
    _HARMONY_OK = True
except Exception:
    hm = None
    _HARMONY_OK = False

# scikit-learn for PCA / clustering / training
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

app = FastAPI(title="RNA-seq preprocessing API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# In-memory run store (Cloud Run note: not guaranteed across instances/restarts)
# -----------------------------------------------------------------------------
RUN_TTL_SECONDS = 2 * 60 * 60  # 2 hours
RUNS: Dict[str, Dict[str, Any]] = {}

def _now() -> float:
    return time.time()

def _cleanup_runs() -> None:
    t = _now()
    dead = [rid for rid, r in RUNS.items() if (t - r.get("created_at", t)) > RUN_TTL_SECONDS]
    for rid in dead:
        RUNS.pop(rid, None)

def _new_run_id() -> str:
    return uuid.uuid4().hex

def _get_run(run_id: str) -> Dict[str, Any]:
    _cleanup_runs()
    r = RUNS.get(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"run_id not found (expired or never created): {run_id}")
    return r

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_matrix_from_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    # Accept CSV/TSV. Assume first column is gene IDs, remaining columns are cells.
    try:
        # sep=None with engine="python" tries to infer delimiter (comma, tab, etc.)
        df = pd.read_csv(io.BytesIO(raw), index_col=0, sep=None, engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read matrix: {e}")

    if df.shape[0] < 2 or df.shape[1] < 2:
        raise HTTPException(status_code=400, detail=f"Matrix too small: shape={df.shape}")

    # Coerce to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all().all():
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion")

    # Replace negative with 0 (some transforms could create negatives; counts should not be negative)
    df = df.clip(lower=0)

    # Ensure unique gene IDs
    if not df.index.is_unique:
        df = df.groupby(df.index).sum()

    return df

async def _read_upload(file: UploadFile) -> Tuple[pd.DataFrame, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    raw = await file.read()
    if raw is None or len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty upload")
    df = _read_matrix_from_bytes(raw, file.filename)
    return df, file.filename

def _quantile_summary(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "max": float("nan")}
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"min": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "max": float(q[4])}

def _mito_mask(gene_index: pd.Index) -> np.ndarray:
    # Common conventions: human MT-*, mouse mt-*
    g = gene_index.astype(str)
    return np.array([s.startswith("MT-") or s.startswith("mt-") for s in g], dtype=bool)

def _ensure_run(run_id: Optional[str]) -> str:
    _cleanup_runs()
    if run_id and run_id.strip():
        if run_id not in RUNS:
            raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")
        return run_id
    rid = _new_run_id()
    RUNS[rid] = {"created_at": _now()}
    return rid

def _store_df(rid: str, key: str, df: pd.DataFrame) -> None:
    RUNS[rid][key] = df

def _get_df_from_run(rid: str, key: str) -> pd.DataFrame:
    r = _get_run(rid)
    df = r.get(key)
    if df is None:
        raise HTTPException(status_code=400, detail=f"Missing '{key}' for run_id={rid}. Run previous step first.")
    return df

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "harmony_available": _HARMONY_OK}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Stores the matrix in-memory and returns run_id.
    Later steps can reference run_id without re-uploading the file.
    """
    df, fname = await _read_upload(file)
    rid = _new_run_id()
    RUNS[rid] = {
        "created_at": _now(),
        "filename": fname,
        "raw_shape": [int(df.shape[0]), int(df.shape[1])],
    }
    _store_df(rid, "raw_df", df)
    return {"ok": True, "run_id": rid, "filename": fname, "shape": [int(df.shape[0]), int(df.shape[1])]}

@app.post("/qc")
async def qc(
    file: Optional[UploadFile] = File(None),
    run_id: Optional[str] = Form(None),
):
    """
    QC summary (per-cell):
      - total_counts
      - detected_genes
      - pct_zeros
      - pct_mito (if MT-/mt- genes present)
    Accepts either:
      - file (multipart/form-data)
      - run_id (from /upload)
    """
    if file is None and (run_id is None or not run_id.strip()):
        raise HTTPException(status_code=422, detail="Provide either file or run_id")

    if file is not None:
        df, fname = await _read_upload(file)
        rid = _ensure_run(run_id)
        RUNS[rid]["filename"] = fname
        RUNS[rid]["raw_shape"] = [int(df.shape[0]), int(df.shape[1])]
        _store_df(rid, "raw_df", df)
    else:
        rid = _ensure_run(run_id)
        df = _get_df_from_run(rid, "raw_df")
        fname = RUNS[rid].get("filename", "uploaded_matrix")

    # Per-cell metrics
    X = df.to_numpy(dtype=float)
    total_counts = np.nansum(X, axis=0)
    detected_genes = np.sum(X > 0, axis=0)
    pct_zeros = np.mean(X == 0, axis=0)

    mito = _mito_mask(df.index)
    if mito.any():
        mito_counts = np.nansum(X[mito, :], axis=0)
        pct_mito = np.divide(mito_counts, np.where(total_counts == 0, np.nan, total_counts))
        pct_mito = np.nan_to_num(pct_mito, nan=0.0)
    else:
        pct_mito = None

    out = {
        "ok": True,
        "run_id": rid,
        "filename": fname,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "total_counts": _quantile_summary(total_counts.astype(float)),
        "detected_genes": _quantile_summary(detected_genes.astype(float)),
        "pct_zeros": _quantile_summary(pct_zeros.astype(float)),
    }
    if pct_mito is not None:
        out["pct_mito"] = _quantile_summary(pct_mito.astype(float))

    RUNS[rid]["qc"] = out
    return out

@app.post("/normalize")
async def normalize(
    file: Optional[UploadFile] = File(None),
    run_id: Optional[str] = Form(None),
    scale: bool = Form(True),
):
    """
    Normalization:
      - CPM per cell
      - log1p(CPM)
      - optional gene-wise scaling (zero-mean, unit-variance across cells)
    Accepts either file or run_id.
    Stores normalized matrix for downstream steps.
    """
    if file is None and (run_id is None or not run_id.strip()):
        raise HTTPException(status_code=422, detail="Provide either file or run_id")

    if file is not None:
        df, fname = await _read_upload(file)
        rid = _ensure_run(run_id)
        RUNS[rid]["filename"] = fname
        RUNS[rid]["raw_shape"] = [int(df.shape[0]), int(df.shape[1])]
        _store_df(rid, "raw_df", df)
    else:
        rid = _ensure_run(run_id)
        df = _get_df_from_run(rid, "raw_df")
        fname = RUNS[rid].get("filename", "uploaded_matrix")

    libsize = df.sum(axis=0, skipna=True).astype(float)
    libsize_safe = libsize.replace(0, np.nan)

    cpm = df.div(libsize_safe, axis=1) * 1e6
    log1p_cpm = np.log1p(cpm.fillna(0.0))

    norm_df = log1p_cpm
    if scale:
        # scale per gene across cells
        scaler = StandardScaler(with_mean=True, with_std=True)
        Z = scaler.fit_transform(norm_df.T.to_numpy(dtype=float))  # cells x genes
        norm_df = pd.DataFrame(Z.T, index=log1p_cpm.index, columns=log1p_cpm.columns)

    _store_df(rid, "norm_df", norm_df)

    # lightweight summary
    vals = norm_df.to_numpy(dtype=float).ravel()
    out = {
        "ok": True,
        "run_id": rid,
        "filename": fname,
        "shape": [int(norm_df.shape[0]), int(norm_df.shape[1])],
        "scale": bool(scale),
        "libsize_summary": _quantile_summary(libsize.to_numpy(dtype=float)),
        "normalized_summary": _quantile_summary(vals),
    }
    RUNS[rid]["normalize"] = out
    return out

@app.post("/harmony")
async def harmony(
    run_id: str = Form(...),
    n_pcs: int = Form(20),
    batch: Optional[str] = Form(None),
):
    """
    PCA + Harmony batch correction in PC space.
    Requires normalized matrix stored in run_id.
    batch:
      - optional comma-separated batch label per cell (length must equal number of cells)
      - if omitted or single batch, returns uncorrected PCs
    Stores:
      - pcs (cells x n_pcs)
      - pcs_harmony (cells x n_pcs) if harmony applied, else same as pcs
    """
    rid = _ensure_run(run_id)
    norm_df = _get_df_from_run(rid, "norm_df")

    n_pcs = int(max(2, min(n_pcs, 100)))
    X = norm_df.T.to_numpy(dtype=float)  # cells x genes

    pca = PCA(n_components=n_pcs, random_state=0)
    pcs = pca.fit_transform(X)  # cells x n_pcs

    cells = list(norm_df.columns)

    batches: List[str]
    if batch and batch.strip():
        batches = [b.strip() for b in batch.split(",")]
        if len(batches) != len(cells):
            raise HTTPException(status_code=400, detail=f"batch length {len(batches)} != number of cells {len(cells)}")
    else:
        batches = ["batch1"] * len(cells)

    unique_batches = sorted(set(batches))
    pcs_h = pcs

    applied = False
    if _HARMONY_OK and len(unique_batches) > 1:
        meta = pd.DataFrame({"batch": batches})
        # harmonypy expects features x samples => pcs.T
        ho = hm.run_harmony(pcs.T, meta_data=meta, vars_use=["batch"])
        pcs_h = ho.Z_corr.T
        applied = True

    RUNS[rid]["pca"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }
    RUNS[rid]["pcs"] = pcs
    RUNS[rid]["pcs_harmony"] = pcs_h
    RUNS[rid]["harmony_applied"] = applied
    RUNS[rid]["batches"] = batches

    return {
        "ok": True,
        "run_id": rid,
        "n_pcs": n_pcs,
        "harmony_applied": applied,
        "n_batches": len(unique_batches),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }

@app.post("/cluster")
async def cluster(
    run_id: str = Form(...),
    n_clusters: int = Form(8),
):
    """
    KMeans clustering on corrected PCs (if harmony ran), otherwise on PCs.
    Stores:
      - clusters (int list aligned to cells)
      - embedding2d (first 2 dims of corrected PCs)
    """
    rid = _ensure_run(run_id)
    if "pcs_harmony" in RUNS[rid]:
        pcs = RUNS[rid]["pcs_harmony"]
    elif "pcs" in RUNS[rid]:
        pcs = RUNS[rid]["pcs"]
    else:
        raise HTTPException(status_code=400, detail="Missing PCs. Run /harmony first.")

    n_clusters = int(max(2, min(n_clusters, 50)))

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(pcs).astype(int)

    RUNS[rid]["clusters"] = labels
    emb2 = pcs[:, :2].astype(float)

    return {
        "ok": True,
        "run_id": rid,
        "n_clusters": n_clusters,
        "counts_per_cluster": {str(i): int((labels == i).sum()) for i in range(n_clusters)},
        "embedding2d_preview": emb2[:10, :].tolist(),
    }

@app.post("/train")
async def train(
    run_id: str = Form(...),
    n_splits: int = Form(5),
):
    """
    Trains a simple classifier to predict clusters from corrected PCs (demo end-to-end).
    Uses CV accuracy as a sanity metric.
    Stores model fitted on all data for export/inspection.
    """
    rid = _ensure_run(run_id)
    if "pcs_harmony" in RUNS[rid]:
        X = RUNS[rid]["pcs_harmony"]
    elif "pcs" in RUNS[rid]:
        X = RUNS[rid]["pcs"]
    else:
        raise HTTPException(status_code=400, detail="Missing PCs. Run /harmony first.")

    y = RUNS[rid].get("clusters")
    if y is None:
        raise HTTPException(status_code=400, detail="Missing clusters. Run /cluster first.")
    y = np.asarray(y, dtype=int)

    n_splits = int(max(2, min(n_splits, 10)))
    if len(np.unique(y)) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 clusters to train a classifier.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.zeros_like(y, dtype=int)

    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=0,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict(X[test_idx])

    acc = float(accuracy_score(y, oof))

    # Fit final model on full data
    final = RandomForestClassifier(
        n_estimators=400,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    final.fit(X, y)

    RUNS[rid]["model"] = final
    RUNS[rid]["train"] = {"cv_accuracy": acc, "n_splits": n_splits}

    return {"ok": True, "run_id": rid, "cv_accuracy": acc, "n_splits": n_splits}

@app.get("/export")
def export(
    run_id: str,
    kind: str = "normalized",  # normalized | clusters | pcs | summary
):
    """
    Download artifacts.
    kind=normalized -> CSV (genes x cells) from norm_df
    kind=clusters   -> CSV (cell,cluster)
    kind=pcs        -> CSV (cell,PC1..PCn) using pcs_harmony if present else pcs
    kind=summary    -> JSON
    """
    rid = _ensure_run(run_id)
    kind = kind.strip().lower()

    if kind == "summary":
        r = _get_run(rid)
        # Avoid dumping huge matrices
        out = {
            "ok": True,
            "run_id": rid,
            "filename": r.get("filename"),
            "raw_shape": r.get("raw_shape"),
            "qc": r.get("qc"),
            "normalize": r.get("normalize"),
            "harmony_applied": r.get("harmony_applied", False),
            "train": r.get("train"),
        }
        return out

    if kind == "normalized":
        df = _get_df_from_run(rid, "norm_df")
        buf = io.StringIO()
        df.to_csv(buf)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="normalized_{rid}.csv"'},
        )

    if kind == "clusters":
        if "norm_df" not in RUNS[rid]:
            raise HTTPException(status_code=400, detail="Missing normalized data. Run /normalize first.")
        cells = list(_get_df_from_run(rid, "norm_df").columns)
        labels = RUNS[rid].get("clusters")
        if labels is None:
            raise HTTPException(status_code=400, detail="Missing clusters. Run /cluster first.")
        labels = np.asarray(labels, dtype=int)
        if labels.shape[0] != len(cells):
            raise HTTPException(status_code=500, detail="Internal mismatch: clusters length != cells length")

        out = pd.DataFrame({"cell": cells, "cluster": labels})
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="clusters_{rid}.csv"'},
        )

    if kind == "pcs":
        if "norm_df" not in RUNS[rid]:
            raise HTTPException(status_code=400, detail="Missing normalized data. Run /normalize first.")
        cells = list(_get_df_from_run(rid, "norm_df").columns)

        if "pcs_harmony" in RUNS[rid]:
            pcs = RUNS[rid]["pcs_harmony"]
        elif "pcs" in RUNS[rid]:
            pcs = RUNS[rid]["pcs"]
        else:
            raise HTTPException(status_code=400, detail="Missing PCs. Run /harmony first.")

        pcs = np.asarray(pcs, dtype=float)
        cols = ["cell"] + [f"PC{i+1}" for i in range(pcs.shape[1])]
        out = pd.DataFrame(np.column_stack([np.array(cells, dtype=object), pcs]), columns=cols)
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="pcs_{rid}.csv"'},
        )

    raise HTTPException(status_code=400, detail=f"Unknown kind: {kind}")
