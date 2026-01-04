# main.py
import io
import json
import math
import uuid
import zipfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


APP_VERSION = "0.2.0"

app = FastAPI(title="RNA-seq preprocessing API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Helpers
# -----------------------
def _percentiles(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": None, "p5": None, "p50": None, "p95": None, "max": None}
    q = np.quantile(x, [0.0, 0.05, 0.50, 0.95, 1.0])
    return {
        "min": float(q[0]),
        "p5": float(q[1]),
        "p50": float(q[2]),
        "p95": float(q[3]),
        "max": float(q[4]),
    }


def _infer_sep(filename: str, raw_text_head: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".tsv") or name.endswith(".txt"):
        return "\t"
    if name.endswith(".csv"):
        return ","
    return "\t" if "\t" in raw_text_head else ","


def _read_counts_matrix(file: UploadFile) -> pd.DataFrame:
    filename = file.filename or ""
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    head = raw[:4096].decode("utf-8", errors="ignore")
    sep = _infer_sep(filename, head)

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, index_col=0)
    except Exception:
        # fallback: let pandas try to infer delimiter
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", index_col=0)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read matrix: {e}")

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Matrix has zero rows or zero columns")

    df = df.apply(pd.to_numeric, errors="coerce")
    all_nan = bool(df.isna().all().all())
    if all_nan:
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion")

    # Keep missingness for QC, but for downstream counts treat NA as 0
    return df


def _qc_metrics(df_num: pd.DataFrame) -> dict:
    # df_num is genes x cells, may contain NA
    n_genes, n_cells = df_num.shape

    missing_rate = float(df_num.isna().to_numpy().mean())

    df = df_num.fillna(0.0)
    counts = df.to_numpy(dtype=float)

    total_counts = counts.sum(axis=0)
    detected_genes = (counts > 0).sum(axis=0)
    pct_zeros = (counts == 0).mean(axis=0)

    gene_names = df.index.astype(str)
    mito_mask = gene_names.str.upper().str.startswith("MT-").to_numpy()
    if mito_mask.any():
        mito_counts = counts[mito_mask, :].sum(axis=0)
        pct_mito = np.divide(mito_counts, np.maximum(total_counts, 1e-12))
    else:
        pct_mito = np.zeros_like(total_counts)

    return {
        "shape": [int(n_genes), int(n_cells)],
        "missing_rate": missing_rate,
        "total_counts": _percentiles(total_counts),
        "detected_genes": _percentiles(detected_genes.astype(float)),
        "pct_zeros": _percentiles(pct_zeros.astype(float)),
        "pct_mito": _percentiles(pct_mito.astype(float)),
    }


def _log1p_cpm(df_num: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # df_num genes x cells, may contain NA
    df = df_num.fillna(0.0).astype(float)

    libsize = df.sum(axis=0)
    libsize_safe = libsize.replace(0.0, np.nan)

    cpm = df.div(libsize_safe, axis=1) * 1e6
    cpm = cpm.fillna(0.0)

    log1p = np.log1p(cpm.to_numpy(dtype=float))
    out = pd.DataFrame(log1p, index=df.index, columns=df.columns)
    return out, libsize


def _parse_batch_labels(batch_labels: Optional[str], cell_ids: list) -> np.ndarray:
    if not batch_labels:
        return np.zeros(len(cell_ids), dtype=int)

    parts = [p.strip() for p in batch_labels.split(",")]
    if len(parts) != len(cell_ids):
        raise HTTPException(
            status_code=400,
            detail=f"batch_labels length {len(parts)} does not match number of cells {len(cell_ids)}",
        )

    # map to integer batch ids
    uniq = {}
    ids = []
    for p in parts:
        if p not in uniq:
            uniq[p] = len(uniq)
        ids.append(uniq[p])
    return np.array(ids, dtype=int)


def _pca_cells(log_expr: pd.DataFrame, n_pcs: int) -> np.ndarray:
    # log_expr genes x cells
    X = log_expr.T.to_numpy(dtype=float)  # cells x genes
    n_cells, n_genes = X.shape
    n_components = int(min(max(n_pcs, 2), n_cells, n_genes))
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(X)


def _simple_harmony_like(pc: np.ndarray, batch: np.ndarray) -> np.ndarray:
    # simple batch mean centering in PC space
    global_mean = pc.mean(axis=0, keepdims=True)
    out = pc.copy()
    for b in np.unique(batch):
        idx = np.where(batch == b)[0]
        if idx.size == 0:
            continue
        b_mean = pc[idx].mean(axis=0, keepdims=True)
        out[idx] = pc[idx] - b_mean + global_mean
    return out


# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Optional endpoint.
    Your frontend can call this to validate the file and return basic metadata.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    head = raw[:4096].decode("utf-8", errors="ignore")
    sep = _infer_sep(file.filename or "", head)

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, index_col=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read matrix: {e}")

    return {
        "ok": True,
        "filename": file.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
    }


@app.post("/qc")
async def qc_counts(file: UploadFile = File(...)):
    await file.seek(0)
    df_num = _read_counts_matrix(file)
    m = _qc_metrics(df_num)
    return {
        "filename": file.filename,
        **m,
    }


@app.post("/normalize")
async def normalize_counts(file: UploadFile = File(...)):
    """
    Returns a CSV string of the normalized matrix.
    FastAPI will serialize this as a JSON string, which your frontend already handles.
    """
    await file.seek(0)
    df_num = _read_counts_matrix(file)
    log_expr, _libsize = _log1p_cpm(df_num)

    # Return as CSV text, includes gene ids as first column
    csv_text = log_expr.to_csv()
    return csv_text


@app.post("/harmony")
async def harmony_batch_correction(
    file: UploadFile = File(...),
    batch_labels: Optional[str] = Form(default=None),
    n_pcs: int = Form(default=30),
):
    await file.seek(0)
    df_num = _read_counts_matrix(file)
    log_expr, _ = _log1p_cpm(df_num)

    cell_ids = list(log_expr.columns.astype(str))
    batch = _parse_batch_labels(batch_labels, cell_ids)

    pc = _pca_cells(log_expr, n_pcs=n_pcs)
    pc_corr = _simple_harmony_like(pc, batch)

    # keep payload small
    return {
        "filename": file.filename,
        "n_cells": int(pc.shape[0]),
        "n_pcs": int(pc.shape[1]),
        "n_batches": int(len(np.unique(batch))),
        "pc_var": {
            "mean_abs": float(np.mean(np.abs(pc_corr))),
            "std": float(np.std(pc_corr)),
        },
        "preview": {
            "cell_ids": cell_ids[:5],
            "pc1": pc_corr[:5, 0].astype(float).tolist(),
            "pc2": pc_corr[:5, 1].astype(float).tolist(),
        },
    }


@app.post("/cluster")
async def cluster_cells(
    file: UploadFile = File(...),
    batch_labels: Optional[str] = Form(default=None),
    n_pcs: int = Form(default=30),
    k: int = Form(default=6),
):
    await file.seek(0)
    df_num = _read_counts_matrix(file)
    log_expr, _ = _log1p_cpm(df_num)

    cell_ids = list(log_expr.columns.astype(str))
    batch = _parse_batch_labels(batch_labels, cell_ids)

    pc = _pca_cells(log_expr, n_pcs=n_pcs)
    pc_corr = _simple_harmony_like(pc, batch)

    k = int(max(2, min(k, pc_corr.shape[0])))
    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = km.fit_predict(pc_corr)

    uniq, cnt = np.unique(labels, return_counts=True)
    counts = {str(int(u)): int(c) for u, c in zip(uniq, cnt)}

    return {
        "filename": file.filename,
        "n_cells": int(pc_corr.shape[0]),
        "n_pcs": int(pc_corr.shape[1]),
        "k": int(k),
        "cluster_counts": counts,
        "preview": {
            "cell_ids": cell_ids[:10],
            "clusters": labels[:10].astype(int).tolist(),
        },
    }


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    batch_labels: Optional[str] = Form(default=None),
    n_pcs: int = Form(default=30),
    k: int = Form(default=6),
):
    """
    Minimal training step for an end-to-end demo.
    If you do not supply labels, this trains a model to predict the clusters from PCs.
    """
    await file.seek(0)
    df_num = _read_counts_matrix(file)
    log_expr, _ = _log1p_cpm(df_num)

    cell_ids = list(log_expr.columns.astype(str))
    batch = _parse_batch_labels(batch_labels, cell_ids)

    pc = _pca_cells(log_expr, n_pcs=n_pcs)
    pc_corr = _simple_harmony_like(pc, batch)

    k = int(max(2, min(k, pc_corr.shape[0])))
    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    y = km.fit_predict(pc_corr)

    clf = LogisticRegression(max_iter=200, multi_class="auto")
    clf.fit(pc_corr, y)
    yhat = clf.predict(pc_corr)

    acc = float(accuracy_score(y, yhat))
    cm = confusion_matrix(y, yhat).astype(int).tolist()

    return {
        "filename": file.filename,
        "task": "predict_clusters_from_pcs",
        "n_cells": int(pc_corr.shape[0]),
        "n_pcs": int(pc_corr.shape[1]),
        "k": int(k),
        "train_accuracy": acc,
        "confusion_matrix": cm,
    }


@app.post("/export")
async def export_bundle(
    file: UploadFile = File(...),
    batch_labels: Optional[str] = Form(default=None),
    n_pcs: int = Form(default=30),
    k: int = Form(default=6),
):
    """
    Returns a ZIP file with QC JSON, normalized CSV, PCs CSV, clusters CSV, and training JSON.
    """
    await file.seek(0)
    df_num = _read_counts_matrix(file)

    qc = {"filename": file.filename, **_qc_metrics(df_num)}

    log_expr, _ = _log1p_cpm(df_num)

    cell_ids = list(log_expr.columns.astype(str))
    batch = _parse_batch_labels(batch_labels, cell_ids)

    pc = _pca_cells(log_expr, n_pcs=n_pcs)
    pc_corr = _simple_harmony_like(pc, batch)

    k = int(max(2, min(int(k), pc_corr.shape[0])))
    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    clusters = km.fit_predict(pc_corr)

    clf = LogisticRegression(max_iter=200, multi_class="auto")
    clf.fit(pc_corr, clusters)
    yhat = clf.predict(pc_corr)

    train = {
        "filename": file.filename,
        "task": "predict_clusters_from_pcs",
        "k": int(k),
        "train_accuracy": float(accuracy_score(clusters, yhat)),
        "confusion_matrix": confusion_matrix(clusters, yhat).astype(int).tolist(),
    }

    # Build files in memory
    buf = io.BytesIO()
    run_id = str(uuid.uuid4())[:8]

    pcs_df = pd.DataFrame(pc_corr, index=cell_ids, columns=[f"PC{i+1}" for i in range(pc_corr.shape[1])])
    clus_df = pd.DataFrame({"cell_id": cell_ids, "cluster": clusters.astype(int)})

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("qc.json", json.dumps(qc, indent=2))
        z.writestr("normalized_log1p_cpm.csv", log_expr.to_csv())
        z.writestr("pcs_harmony_like.csv", pcs_df.to_csv(index=True))
        z.writestr("clusters.csv", clus_df.to_csv(index=False))
        z.writestr("training.json", json.dumps(train, indent=2))

    buf.seek(0)
    filename = f"rnaseq_bundle_{run_id}.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
