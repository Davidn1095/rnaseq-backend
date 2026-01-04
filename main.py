# main.py
import io
import json
import math
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# Optional deps
try:
    import harmonypy as hm  # type: ignore
except Exception:
    hm = None

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

app = FastAPI(title="RNA-seq preprocessing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# In-memory run store (demo)
# NOTE: Cloud Run can restart; UI should call /upload again if run_id is lost.
# ----------------------------
RUNS: Dict[str, Dict[str, Any]] = {}


def _percentiles(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"min": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "max": float("nan")}
    q = np.percentile(x, [0, 25, 50, 75, 100])
    return {"min": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "max": float(q[4])}


def _infer_sep(filename: str, sample: bytes) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".tsv") or fn.endswith(".txt"):
        return "\t"
    # fallback: sniff header
    head = sample[:4096].decode("utf-8", errors="ignore")
    first_line = head.splitlines()[0] if head.splitlines() else ""
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def _read_counts_csv(bytes_blob: bytes, filename: str) -> pd.DataFrame:
    sep = _infer_sep(filename, bytes_blob)
    df = pd.read_csv(io.BytesIO(bytes_blob), sep=sep, index_col=0)
    # coerce numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all().all():
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion")
    # replace NaNs with 0 for count-like behavior
    df = df.fillna(0)
    return df


async def _parse_run_id_and_file(request: Request) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """
    Accept either:
      - multipart/form-data: fields run_id (optional), file (optional UploadFile)
      - application/json: {"run_id": "..."}
    Returns: (run_id, upload_file, filename)
    """
    ct = (request.headers.get("content-type") or "").lower()

    if ct.startswith("application/json"):
        try:
            data = await request.json()
        except Exception:
            data = {}
        run_id = data.get("run_id")
        return (str(run_id) if run_id else None, None, None)

    # multipart or others: try form()
    try:
        form = await request.form()
    except Exception:
        return (None, None, None)

    run_id = form.get("run_id")
    f = form.get("file")
    filename = getattr(f, "filename", None) if f is not None else None
    return (str(run_id) if run_id else None, f, filename)


def _ensure_run(run_id: Optional[str]) -> str:
    rid = run_id or uuid.uuid4().hex
    if rid not in RUNS:
        RUNS[rid] = {}
    return rid


def _get_counts(rid: str) -> pd.DataFrame:
    df = RUNS.get(rid, {}).get("counts")
    if df is None:
        raise HTTPException(status_code=404, detail='run_id not found. Re-run /upload or /qc.')
    return df


def _store_counts(rid: str, df: pd.DataFrame, filename: str) -> None:
    RUNS.setdefault(rid, {})
    RUNS[rid]["counts"] = df
    RUNS[rid]["filename"] = filename
    RUNS[rid]["shape"] = [int(df.shape[0]), int(df.shape[1])]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(request: Request):
    run_id, f, filename = await _parse_run_id_and_file(request)
    if f is None:
        raise HTTPException(status_code=400, detail="Missing file in multipart/form-data as field 'file'")
    raw = await f.read()
    df = _read_counts_csv(raw, filename or "counts.csv")
    rid = _ensure_run(run_id)
    _store_counts(rid, df, filename or "counts.csv")
    return {"ok": True, "run_id": rid, "filename": filename or "counts.csv", "shape": [int(df.shape[0]), int(df.shape[1])]}


@app.post("/qc")
async def qc(request: Request):
    run_id, f, filename = await _parse_run_id_and_file(request)

    if f is not None:
        raw = await f.read()
        df = _read_counts_csv(raw, filename or "counts.csv")
        rid = _ensure_run(run_id)
        _store_counts(rid, df, filename or "counts.csv")
    else:
        if not run_id:
            raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])
        rid = run_id
        df = _get_counts(rid)

    total_counts = df.sum(axis=0).to_numpy(dtype=float)
    detected_genes = (df.to_numpy(dtype=float) > 0).sum(axis=0).astype(float)
    pct_zeros = (df.to_numpy(dtype=float) == 0).sum(axis=0).astype(float) / max(1.0, float(df.shape[0]))

    out = {
        "ok": True,
        "run_id": rid,
        "filename": RUNS.get(rid, {}).get("filename", filename or "counts.csv"),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "total_counts": _percentiles(total_counts),
        "detected_genes": _percentiles(detected_genes),
        "pct_zeros": _percentiles(pct_zeros),
    }

    RUNS[rid]["qc"] = out
    return out


@app.post("/normalize")
async def normalize(request: Request):
    run_id, f, filename = await _parse_run_id_and_file(request)

    if f is not None:
        raw = await f.read()
        df = _read_counts_csv(raw, filename or "counts.csv")
        rid = _ensure_run(run_id)
        _store_counts(rid, df, filename or "counts.csv")
    else:
        if not run_id:
            raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])
        rid = run_id
        df = _get_counts(rid)

    libsize = df.sum(axis=0, skipna=True).replace(0, pd.NA)
    cpm = df.div(libsize, axis=1) * 1e6
    log1p_cpm = np.log1p(cpm.fillna(0).to_numpy(dtype=float))

    # store normalized matrix as DataFrame with same index/cols
    norm_df = pd.DataFrame(log1p_cpm, index=df.index, columns=df.columns)
    RUNS[rid]["normalized"] = norm_df

    return {
        "ok": True,
        "run_id": rid,
        "filename": RUNS.get(rid, {}).get("filename", filename or "counts.csv"),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "libsize": {str(k): float(v) for k, v in df.sum(axis=0).to_dict().items()},
        "log1p_cpm_summary": {
            "min": float(np.min(log1p_cpm)),
            "median": float(np.median(log1p_cpm)),
            "max": float(np.max(log1p_cpm)),
        },
    }


@app.post("/harmony")
async def harmony(request: Request):
    run_id, f, filename = await _parse_run_id_and_file(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])
    rid = run_id

    # If server restarted and caller provided file, allow rebuilding
    if rid not in RUNS and f is not None:
        raw = await f.read()
        df = _read_counts_csv(raw, filename or "counts.csv")
        _store_counts(rid, df, filename or "counts.csv")

    norm_df = RUNS.get(rid, {}).get("normalized")
    if norm_df is None:
        # allow fallback: compute normalized from counts if present
        df = _get_counts(rid)
        libsize = df.sum(axis=0, skipna=True).replace(0, pd.NA)
        cpm = df.div(libsize, axis=1) * 1e6
        log1p_cpm = np.log1p(cpm.fillna(0).to_numpy(dtype=float))
        norm_df = pd.DataFrame(log1p_cpm, index=df.index, columns=df.columns)
        RUNS[rid]["normalized"] = norm_df

    X = norm_df.to_numpy(dtype=float).T  # cells x genes

    n_cells, n_genes = X.shape
    n_pcs = int(min(30, max(2, n_cells - 1), max(2, n_genes - 1)))
    pca = PCA(n_components=n_pcs, random_state=0)
    Z = pca.fit_transform(X)

    # Harmony requires batch labels; in this demo we default to one batch
    batches = RUNS.get(rid, {}).get("batches")
    if batches is None:
        batches = np.zeros(n_cells, dtype=int)
    else:
        batches = np.asarray(batches)

    applied = False
    reason = ""

    Z_corr = Z
    if hm is None or len(np.unique(batches)) <= 1:
        applied = False
        reason = "harmonypy missing or n_batches<=1"
    else:
        try:
            ho = hm.run_harmony(Z, pd.DataFrame({"batch": batches}), "batch")
            Z_corr = ho.Z_corr.T
            applied = True
            reason = "ok"
        except Exception as e:
            applied = False
            reason = f"harmony error: {e}"

    RUNS[rid]["embedding"] = Z_corr
    RUNS[rid]["embedding_name"] = "pca_harmony" if applied else "pca_fallback"

    return {
        "ok": True,
        "run_id": rid,
        "embedding": RUNS[rid]["embedding_name"],
        "shape": [int(Z_corr.shape[0]), int(Z_corr.shape[1])],
        "harmony_meta": {"applied": bool(applied), "reason": reason},
    }


@app.post("/cluster")
async def cluster(request: Request):
    run_id, f, filename = await _parse_run_id_and_file(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])
    rid = run_id

    if rid not in RUNS and f is not None:
        raw = await f.read()
        df = _read_counts_csv(raw, filename or "counts.csv")
        _store_counts(rid, df, filename or "counts.csv")

    Z = RUNS.get(rid, {}).get("embedding")
    if Z is None:
        raise HTTPException(status_code=404, detail='run_id not found. Re-run /upload or /qc.')

    n_cells = int(Z.shape[0])
    k = int(min(6, max(2, n_cells // 10))) if n_cells >= 20 else 3
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(Z)

    RUNS[rid]["clusters"] = labels

    sizes = {str(i): int(np.sum(labels == i)) for i in range(k)}
    return {
        "ok": True,
        "run_id": rid,
        "k": k,
        "cluster_sizes": sizes,
        "note": "Clusters stored for /train and /export.",
    }


@app.post("/train")
async def train(request: Request):
    run_id, f, filename = await _parse_run_id_and_file(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])
    rid = run_id

    if rid not in RUNS and f is not None:
        raw = await f.read()
        df = _read_counts_csv(raw, filename or "counts.csv")
        _store_counts(rid, df, filename or "counts.csv")

    Z = RUNS.get(rid, {}).get("embedding")
    y = RUNS.get(rid, {}).get("clusters")

    if Z is None or y is None:
        raise HTTPException(status_code=400, detail="Missing embedding or clusters. Run /harmony and /cluster first.")

    X = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=int)

    n_classes = int(len(np.unique(y)))
    clf = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        solver="lbfgs",
        n_jobs=None,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))

    # Fit final model
    clf.fit(X, y)
    RUNS[rid]["model"] = clf
    RUNS[rid]["train_metrics"] = {
        "cv_folds": 5,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
    }

    return {
        "ok": True,
        "run_id": rid,
        "task": "predict_cluster_from_embedding",
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "metrics": RUNS[rid]["train_metrics"],
    }


@app.post("/export")
async def export(request: Request):
    run_id, f, filename = await _parse_run_id_and_file(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])
    rid = run_id

    # If server restarted and caller provided file, allow rebuilding counts (best effort)
    if rid not in RUNS and f is not None:
        raw = await f.read()
        df = _read_counts_csv(raw, filename or "counts.csv")
        _store_counts(rid, df, filename or "counts.csv")

    if rid not in RUNS:
        raise HTTPException(status_code=404, detail='run_id not found. Re-run /upload or /qc.')

    qc_obj = RUNS[rid].get("qc")
    norm_df: Optional[pd.DataFrame] = RUNS[rid].get("normalized")
    Z = RUNS[rid].get("embedding")
    y = RUNS[rid].get("clusters")
    metrics = RUNS[rid].get("train_metrics")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("run_id.txt", rid)

        if qc_obj is not None:
            z.writestr("qc_summary.json", json.dumps(qc_obj, indent=2))

        if norm_df is not None:
            out = io.StringIO()
            norm_df.to_csv(out)
            z.writestr("normalized_log1p_cpm.csv", out.getvalue())

        if Z is not None:
            Z = np.asarray(Z)
            emb = pd.DataFrame(Z)
            out = io.StringIO()
            emb.to_csv(out, index=False)
            z.writestr("embedding.csv", out.getvalue())

        if y is not None:
            y = np.asarray(y)
            out = io.StringIO()
            pd.DataFrame({"cluster": y}).to_csv(out, index=False)
            z.writestr("clusters.csv", out.getvalue())

        if metrics is not None:
            z.writestr("train_metrics.json", json.dumps(metrics, indent=2))

    buf.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="rnaseq_export.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
