import io
import json
import math
import zipfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


# =========================
# App
# =========================

app = FastAPI(title="RNA-seq preprocessing API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Utilities
# =========================

def _read_count_matrix(file: UploadFile) -> pd.DataFrame:
    name = (file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Upload a .csv, .tsv, or .txt count matrix")

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        # sep=None with engine="python" infers comma vs tab reliably for small and medium files
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", index_col=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read matrix: {e}")

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Matrix has zero rows or columns")

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all().all():
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion")

    # treat missing as 0 counts, enforce non-negative
    df = df.fillna(0.0)
    if (df.values < 0).any():
        df = df.clip(lower=0.0)

    return df


def _summary_series(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"min": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "min": float(x.min()),
        "p50": float(x.quantile(0.50)),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
    }


def _log1p_cpm(df_counts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    libsize = df_counts.sum(axis=0)
    libsize_safe = libsize.replace(0, np.nan)
    cpm = df_counts.div(libsize_safe, axis=1) * 1e6
    log1p = np.log1p(cpm.to_numpy(dtype=np.float64))
    df_log1p = pd.DataFrame(log1p, index=df_counts.index, columns=df_counts.columns)
    df_log1p = df_log1p.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df_log1p, libsize


def _scale_genes(df: pd.DataFrame) -> pd.DataFrame:
    # z-score per gene across cells
    x = df.to_numpy(dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def _pca_cells(X_cells_by_genes: np.ndarray, n_pcs: int) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Missing dependency: scikit-learn is required for PCA. Add scikit-learn to requirements.txt",
        )

    n_cells, n_genes = X_cells_by_genes.shape
    n_pcs = int(max(2, min(n_pcs, n_cells - 1, n_genes)))
    pca = PCA(n_components=n_pcs, random_state=123)
    pcs = pca.fit_transform(X_cells_by_genes)
    return pcs.astype(np.float64)


def _infer_batches(cell_names: pd.Index) -> pd.Series:
    # default heuristic: prefix before first underscore, else before first dash, else "batch0"
    s = pd.Series(cell_names.astype(str), index=cell_names)
    if s.str.contains("_").any():
        b = s.str.split("_", n=1).str[0]
    elif s.str.contains("-").any():
        b = s.str.split("-", n=1).str[0]
    else:
        b = pd.Series(["batch0"] * len(s), index=cell_names)
    # if everything becomes unique or empty, collapse to one batch
    if b.nunique(dropna=True) <= 1:
        b[:] = "batch0"
    return b.astype(str)


def _simple_batch_center(pcs: np.ndarray, batches: pd.Series) -> np.ndarray:
    # mean-center PCs within each batch
    Z = pcs.copy()
    for b in batches.unique():
        idx = np.where(batches.values == b)[0]
        if idx.size == 0:
            continue
        Z[idx, :] = Z[idx, :] - Z[idx, :].mean(axis=0, keepdims=True)
    return Z


def _harmony_correct(pcs: np.ndarray, batches: pd.Series) -> np.ndarray:
    # prefer harmonypy if present, else fallback to mean-centering
    if batches.nunique() <= 1:
        return pcs

    try:
        import harmonypy as hm
        meta = pd.DataFrame({"batch": batches.values})
        ho = hm.run_harmony(pcs, meta, "batch")
        Z = ho.Z_corr.T  # harmonypy returns PCs x cells, transpose to cells x PCs
        return np.asarray(Z, dtype=np.float64)
    except Exception:
        return _simple_batch_center(pcs, batches)


def _kmeans_cluster(Z: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Missing dependency: scikit-learn is required for clustering. Add scikit-learn to requirements.txt",
        )

    k = int(max(2, min(k, Z.shape[0] - 1)))
    km = KMeans(n_clusters=k, n_init=10, random_state=123)
    return km.fit_predict(Z).astype(int)


def _train_classifier(Z: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Missing dependency: scikit-learn is required for training. Add scikit-learn to requirements.txt",
        )

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        raise HTTPException(status_code=400, detail="Training labels have only one class")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    accs, f1s = [], []

    for tr, te in skf.split(Z, y):
        model = LogisticRegression(max_iter=2000, n_jobs=1, multi_class="auto")
        model.fit(Z[tr], y[tr])
        pred = model.predict(Z[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))

    return {
        "cv_folds": 5,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_sd": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_sd": float(np.std(f1s)),
        "n_samples": int(Z.shape[0]),
        "n_features": int(Z.shape[1]),
        "n_classes": int(len(np.unique(y))),
    }


# =========================
# Routes
# =========================

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/qc")
async def qc_counts(file: UploadFile = File(...)):
    df = _read_count_matrix(file)

    total_counts = df.sum(axis=0)
    detected_genes = (df > 0).sum(axis=0)
    pct_zeros = (df == 0).sum(axis=0) / float(df.shape[0])

    return {
        "filename": file.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "total_counts": _summary_series(total_counts),
        "detected_genes": _summary_series(detected_genes),
        "pct_zeros": _summary_series(pct_zeros),
    }


@app.post("/normalize")
async def normalize_counts(file: UploadFile = File(...), scale: bool = True):
    df = _read_count_matrix(file)

    df_log1p, libsize = _log1p_cpm(df)
    df_norm = _scale_genes(df_log1p) if scale else df_log1p

    # include CSV for downstream steps and export, keep a hard cap
    # raise the cap if you want bigger matrices in responses
    max_cells_for_inline_csv = 300
    max_genes_for_inline_csv = 2000
    include_csv = (df_norm.shape[1] <= max_cells_for_inline_csv) and (df_norm.shape[0] <= max_genes_for_inline_csv)

    normalized_csv: Optional[str] = None
    if include_csv:
        buf = io.StringIO()
        df_norm.to_csv(buf)
        normalized_csv = buf.getvalue()

    return {
        "filename": file.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "method": "log1p_cpm" + ("_zscore_genes" if scale else ""),
        "libsize_summary": _summary_series(libsize),
        "normalized_summary": {
            "min": float(df_norm.min().min()),
            "median": float(df_norm.stack().median()),
            "max": float(df_norm.max().max()),
        },
        "normalized_csv": normalized_csv,
        "note": None if include_csv else "Matrix too large to inline as CSV. Use /export to download outputs.",
    }


@app.post("/harmony")
async def harmony_batch_correction(file: UploadFile = File(...), n_pcs: int = 20):
    df = _read_count_matrix(file)

    df_log1p, _ = _log1p_cpm(df)
    X = df_log1p.T.to_numpy(dtype=np.float64)  # cells x genes

    pcs = _pca_cells(X, n_pcs=n_pcs)
    batches = _infer_batches(df.columns)

    pcs_corr = _harmony_correct(pcs, batches)

    return {
        "filename": file.filename,
        "n_cells": int(df.shape[1]),
        "n_genes": int(df.shape[0]),
        "n_pcs": int(pcs_corr.shape[1]),
        "batches": {str(k): int(v) for k, v in batches.value_counts().to_dict().items()},
        "correction": "harmonypy" if batches.nunique() > 1 else "none_single_batch",
        "pcs_preview": pcs[:5, : min(5, pcs.shape[1])].tolist(),
        "pcs_corrected_preview": pcs_corr[:5, : min(5, pcs_corr.shape[1])].tolist(),
    }


@app.post("/cluster")
async def cluster_cells(file: UploadFile = File(...), n_pcs: int = 20, k: int = 8):
    df = _read_count_matrix(file)

    df_log1p, _ = _log1p_cpm(df)
    X = df_log1p.T.to_numpy(dtype=np.float64)

    pcs = _pca_cells(X, n_pcs=n_pcs)
    batches = _infer_batches(df.columns)
    pcs_corr = _harmony_correct(pcs, batches)

    labels = _kmeans_cluster(pcs_corr, k=k)

    sizes = pd.Series(labels).value_counts().sort_index()
    return {
        "filename": file.filename,
        "n_cells": int(df.shape[1]),
        "k": int(sizes.shape[0]),
        "cluster_sizes": {str(int(k)): int(v) for k, v in sizes.to_dict().items()},
        "labels_preview": [
            {"cell": str(df.columns[i]), "cluster": int(labels[i])}
            for i in range(min(20, len(labels)))
        ],
    }


@app.post("/train")
async def train_model(file: UploadFile = File(...), n_pcs: int = 20, k_for_pseudo: int = 8):
    df = _read_count_matrix(file)

    df_log1p, _ = _log1p_cpm(df)
    X = df_log1p.T.to_numpy(dtype=np.float64)

    pcs = _pca_cells(X, n_pcs=n_pcs)
    batches = _infer_batches(df.columns)

    # Prefer predicting batch if multiple batches exist, else predict clusters as pseudo-labels
    if batches.nunique() > 1:
        y = batches.values
        task = "predict_inferred_batch"
    else:
        pcs_corr = _harmony_correct(pcs, batches)
        y = _kmeans_cluster(pcs_corr, k=k_for_pseudo)
        task = "predict_kmeans_clusters_pseudolabels"

    metrics = _train_classifier(pcs, y)

    return {
        "filename": file.filename,
        "task": task,
        "metrics": metrics,
    }


@app.post("/export")
async def export_bundle(file: UploadFile = File(...), n_pcs: int = 20, k: int = 8):
    df = _read_count_matrix(file)

    # QC
    total_counts = df.sum(axis=0)
    detected_genes = (df > 0).sum(axis=0)
    pct_zeros = (df == 0).sum(axis=0) / float(df.shape[0])
    qc = {
        "filename": file.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "total_counts": _summary_series(total_counts),
        "detected_genes": _summary_series(detected_genes),
        "pct_zeros": _summary_series(pct_zeros),
    }

    # Normalize
    df_log1p, _ = _log1p_cpm(df)
    df_norm = _scale_genes(df_log1p)

    # PCA + Harmony
    X = df_log1p.T.to_numpy(dtype=np.float64)
    pcs = _pca_cells(X, n_pcs=n_pcs)
    batches = _infer_batches(df.columns)
    pcs_corr = _harmony_correct(pcs, batches)

    # Cluster
    labels = _kmeans_cluster(pcs_corr, k=k)

    # Train
    if batches.nunique() > 1:
        y = batches.values
        task = "predict_inferred_batch"
        train = {"task": task, "metrics": _train_classifier(pcs, y)}
    else:
        task = "predict_kmeans_clusters_pseudolabels"
        train = {"task": task, "metrics": _train_classifier(pcs, labels)}

    # Build ZIP in memory
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("qc.json", json.dumps(qc, indent=2))
        z.writestr("train.json", json.dumps(train, indent=2))

        # matrices
        m = io.StringIO()
        df_norm.to_csv(m)
        z.writestr("normalized_zscore_genes.csv", m.getvalue())

        pcs_df = pd.DataFrame(pcs, index=df.columns, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
        pcs_corr_df = pd.DataFrame(pcs_corr, index=df.columns, columns=[f"HPC{i+1}" for i in range(pcs_corr.shape[1])])

        p1 = io.StringIO()
        pcs_df.to_csv(p1)
        z.writestr("pca_pcs.csv", p1.getvalue())

        p2 = io.StringIO()
        pcs_corr_df.to_csv(p2)
        z.writestr("harmony_pcs.csv", p2.getvalue())

        cl = pd.DataFrame({"cell": df.columns.astype(str), "cluster": labels.astype(int)})
        c1 = io.StringIO()
        cl.to_csv(c1, index=False)
        z.writestr("clusters.csv", c1.getvalue())

    zbuf.seek(0)

    out_name = (file.filename or "matrix").rsplit(".", 1)[0]
    zip_name = f"{out_name}_pipeline_export.zip"

    return StreamingResponse(
        zbuf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )
