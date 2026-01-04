# main.py
import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


app = FastAPI(title="RNA-seq preprocessing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class RunState:
    filename: str
    raw_counts: pd.DataFrame
    qc: Optional[Dict[str, Any]] = None
    norm_log1p_cpm: Optional[pd.DataFrame] = None
    norm_z: Optional[pd.DataFrame] = None
    pca_scores: Optional[pd.DataFrame] = None
    harmony_scores: Optional[pd.DataFrame] = None
    clusters: Optional[pd.Series] = None
    train_metrics: Optional[Dict[str, Any]] = None


RUNS: Dict[str, RunState] = {}


def _read_counts(upload: UploadFile) -> pd.DataFrame:
    name = (upload.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Upload a .csv or .tsv count matrix")

    raw = upload.file.read()
    if raw is None:
        raise HTTPException(status_code=400, detail="Empty upload")

    sep = "\t" if name.endswith(".tsv") or name.endswith(".txt") else ","
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, index_col=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read matrix: {e}")

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Matrix has zero rows or columns")

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all().all():
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion")

    df = df.fillna(0.0)

    # keep gene and cell names as strings
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    return df


def _quantiles(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {"min": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "max": float("nan")}
    q = x.quantile([0.25, 0.5, 0.75])
    return {
        "min": float(x.min()),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "max": float(x.max()),
    }


def _require_run(run_id: str) -> RunState:
    st = RUNS.get(run_id)
    if st is None:
        raise HTTPException(status_code=404, detail="run_id not found. Re-run /upload or /qc.")
    return st


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload_counts(file: UploadFile = File(...)):
    df = _read_counts(file)
    run_id = uuid4().hex
    RUNS[run_id] = RunState(filename=file.filename or "counts", raw_counts=df)
    return {
        "ok": True,
        "run_id": run_id,
        "filename": file.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
    }


@app.post("/qc")
async def qc_counts(
    file: Optional[UploadFile] = File(None),
    run_id: Optional[str] = Form(None),
):
    # Accept either:
    # - run_id that already exists, or
    # - file to create or refresh a run
    if run_id:
        st = _require_run(run_id)
    else:
        if file is None:
            raise HTTPException(status_code=422, detail="Provide run_id or file")
        df = _read_counts(file)
        run_id = uuid4().hex
        st = RunState(filename=file.filename or "counts", raw_counts=df)
        RUNS[run_id] = st

    df = st.raw_counts

    total_counts = df.sum(axis=0)
    detected_genes = (df > 0).sum(axis=0)
    pct_zeros = (df == 0).mean(axis=0) * 100.0

    qc = {
        "ok": True,
        "run_id": run_id,
        "filename": st.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "total_counts": _quantiles(total_counts),
        "detected_genes": _quantiles(detected_genes),
        "pct_zeros": _quantiles(pct_zeros),
    }

    st.qc = qc
    return qc


@app.post("/normalize")
async def normalize_counts(
    file: Optional[UploadFile] = File(None),
    run_id: Optional[str] = Form(None),
):
    # Frontend can call with file (old behavior) or run_id (preferred after /upload).
    if run_id:
        st = _require_run(run_id)
    else:
        if file is None:
            raise HTTPException(status_code=422, detail="Provide run_id or file")
        df = _read_counts(file)
        run_id = uuid4().hex
        st = RunState(filename=file.filename or "counts", raw_counts=df)
        RUNS[run_id] = st

    df = st.raw_counts

    libsize = df.sum(axis=0).replace(0, np.nan)
    cpm = df.div(libsize, axis=1) * 1e6
    log1p_cpm = np.log1p(cpm).fillna(0.0)

    # z-score per gene across cells
    gene_mean = log1p_cpm.mean(axis=1)
    gene_std = log1p_cpm.std(axis=1).replace(0, np.nan)
    z = log1p_cpm.sub(gene_mean, axis=0).div(gene_std, axis=0).fillna(0.0)

    st.norm_log1p_cpm = log1p_cpm
    st.norm_z = z

    # compact summary, full matrix goes via /export
    return {
        "ok": True,
        "run_id": run_id,
        "filename": st.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "libsize_summary": _quantiles(libsize.fillna(0.0)),
        "log1p_cpm_summary": {
            "min": float(log1p_cpm.min().min()),
            "median": float(log1p_cpm.stack().median()),
            "max": float(log1p_cpm.max().max()),
        },
        "z_summary": {
            "min": float(z.min().min()),
            "median": float(z.stack().median()),
            "max": float(z.max().max()),
        },
    }


@app.post("/harmony")
async def harmony_batch_correction(
    run_id: str = Form(...),
    n_pcs: int = Form(30),
    # optional, kept for compatibility with your frontend that currently sends file again
    file: Optional[UploadFile] = File(None),
):
    st = _require_run(run_id)

    # If normalized is missing, derive it from the file if provided, else error.
    if st.norm_z is None:
        if file is not None:
            df = _read_counts(file)
            st.raw_counts = df
            libsize = df.sum(axis=0).replace(0, np.nan)
            cpm = df.div(libsize, axis=1) * 1e6
            log1p_cpm = np.log1p(cpm).fillna(0.0)
            gene_mean = log1p_cpm.mean(axis=1)
            gene_std = log1p_cpm.std(axis=1).replace(0, np.nan)
            st.norm_z = log1p_cpm.sub(gene_mean, axis=0).div(gene_std, axis=0).fillna(0.0)
        else:
            raise HTTPException(status_code=400, detail="Run /normalize first for this run_id")

    z = st.norm_z  # genes x cells
    X = z.T.values  # cells x genes

    n_cells, n_features = X.shape
    max_pcs = max(2, min(int(n_pcs), n_cells - 1, n_features))
    pca = PCA(n_components=max_pcs, random_state=0)
    pcs = pca.fit_transform(X)

    # Minimal “batch correction” placeholder:
    # If you later add batch labels, you can replace this with real Harmony.
    harmony = pcs.copy()

    pcs_df = pd.DataFrame(pcs, index=st.raw_counts.columns, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
    harm_df = pd.DataFrame(harmony, index=st.raw_counts.columns, columns=[f"HPC{i+1}" for i in range(harmony.shape[1])])

    st.pca_scores = pcs_df
    st.harmony_scores = harm_df

    return {
        "ok": True,
        "run_id": run_id,
        "n_pcs": int(harmony.shape[1]),
        "explained_variance_ratio_first5": [float(x) for x in pca.explained_variance_ratio_[:5]],
        "embedding_preview_first5": harm_df.iloc[:5, : min(5, harm_df.shape[1])].to_dict(orient="index"),
    }


@app.post("/cluster")
async def cluster_cells(
    run_id: str = Form(...),
    k: int = Form(8),
):
    st = _require_run(run_id)

    if st.harmony_scores is None:
        raise HTTPException(status_code=400, detail="Run /harmony first for this run_id")

    X = st.harmony_scores.values
    k = int(k)
    if k < 2 or k > max(2, X.shape[0] - 1):
        raise HTTPException(status_code=400, detail="Invalid k for clustering")

    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = km.fit_predict(X)

    s = pd.Series(labels, index=st.harmony_scores.index, name="cluster")
    st.clusters = s

    counts = s.value_counts().sort_index().to_dict()
    return {
        "ok": True,
        "run_id": run_id,
        "k": k,
        "cluster_counts": {str(int(k)): int(v) for k, v in counts.items()},
    }


@app.post("/train")
async def train_model(
    run_id: str = Form(...),
    test_size: float = Form(0.2),
):
    st = _require_run(run_id)

    if st.harmony_scores is None or st.clusters is None:
        raise HTTPException(status_code=400, detail="Run /harmony and /cluster first for this run_id")

    X = st.harmony_scores.values
    y = st.clusters.values

    # stratify if possible
    strat = y if len(np.unique(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=float(test_size), random_state=0, stratify=strat)

    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = float(accuracy_score(yte, pred))
    f1m = float(f1_score(yte, pred, average="macro"))

    metrics = {
        "ok": True,
        "run_id": run_id,
        "task": "predict_cluster_from_harmony_pcs",
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "accuracy": acc,
        "macro_f1": f1m,
    }

    st.train_metrics = metrics
    return metrics


@app.post("/export")
async def export_bundle(run_id: str = Form(...)):
    st = _require_run(run_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # qc
        if st.qc is not None:
            z.writestr("qc_summary.json", json.dumps(st.qc, indent=2))

        # normalized
        if st.norm_log1p_cpm is not None:
            z.writestr("normalized_log1p_cpm.csv", st.norm_log1p_cpm.to_csv())
        if st.norm_z is not None:
            z.writestr("normalized_zscore.csv", st.norm_z.to_csv())

        # embeddings
        if st.pca_scores is not None:
            z.writestr("pca_scores.csv", st.pca_scores.to_csv(index=True))
        if st.harmony_scores is not None:
            z.writestr("harmony_scores.csv", st.harmony_scores.to_csv(index=True))

        # clusters
        if st.clusters is not None:
            z.writestr("clusters.csv", st.clusters.to_csv(index=True))

        # training
        if st.train_metrics is not None:
            z.writestr("train_metrics.json", json.dumps(st.train_metrics, indent=2))

        # raw counts
        z.writestr("raw_counts.csv", st.raw_counts.to_csv())

    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="rnaseq_export_{run_id}.zip"'},
    )
