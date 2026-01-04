import io
import json
import uuid
import zipfile
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


app = FastAPI(title="RNA-seq preprocessing API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# In-memory run store
# -------------------------
RUNS: Dict[str, Dict[str, Any]] = {}


def _new_run_id() -> str:
    return uuid.uuid4().hex


def _infer_sep(filename: str, raw_bytes: bytes) -> str:
    fn = filename.lower()
    if fn.endswith(".tsv") or fn.endswith(".txt"):
        return "\t"
    head = raw_bytes[:4096].decode("utf-8", errors="ignore")
    if "\t" in head and head.count("\t") > head.count(","):
        return "\t"
    return ","


def _read_counts_matrix(file: UploadFile, raw: bytes) -> pd.DataFrame:
    if not file.filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(status_code=400, detail="Upload a .csv, .tsv, or .txt count matrix")

    sep = _infer_sep(file.filename, raw)

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, index_col=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read matrix: {e}")

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Matrix is empty")

    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().all().all():
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion")

    df = df.fillna(0)

    if (df < 0).any().any():
        raise HTTPException(status_code=400, detail="Counts matrix contains negative values")

    return df


def _five_number(x: pd.Series) -> Dict[str, float]:
    q = x.quantile([0.25, 0.5, 0.75]).to_dict()
    return {
        "min": float(x.min()),
        "p25": float(q.get(0.25, np.nan)),
        "p50": float(q.get(0.5, np.nan)),
        "p75": float(q.get(0.75, np.nan)),
        "max": float(x.max()),
    }


def _get_run_or_400(run_id: str) -> Dict[str, Any]:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return RUNS[run_id]


def _ensure_normalized(run: Dict[str, Any]) -> None:
    if "normalized_log1p" in run and "pcs" in run:
        return

    counts: pd.DataFrame = run.get("counts")
    if counts is None:
        raise HTTPException(status_code=400, detail="Run has no counts stored")

    libsize = counts.sum(axis=0)
    libsize_safe = libsize.replace(0, np.nan)

    norm = counts.div(libsize_safe, axis=1) * 1e4
    log1p = np.log1p(norm).replace([np.inf, -np.inf], 0).fillna(0)

    # PCA expects cells as rows, genes as cols
    X = log1p.T.values
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    n_cells, n_genes = Xs.shape
    n_comp = int(min(30, max(2, n_cells - 1), n_genes))
    pca = PCA(n_components=n_comp, random_state=0)
    pcs = pca.fit_transform(Xs)

    run["libsize"] = libsize.to_dict()
    run["normalized_log1p"] = log1p
    run["pca_var_ratio"] = pca.explained_variance_ratio_.tolist()
    run["pcs"] = pd.DataFrame(pcs, index=log1p.columns, columns=[f"PC{i+1}" for i in range(n_comp)])


def _default_batches(cell_names: List[str]) -> List[str]:
    # Heuristic: batch is prefix before first underscore, else "batch1"
    out = []
    for c in cell_names:
        if "_" in c:
            out.append(c.split("_", 1)[0])
        else:
            out.append("batch1")
    return out


def _batch_center_pcs(pcs: pd.DataFrame, batches: List[str]) -> pd.DataFrame:
    # Simple batch correction in PC space:
    # corrected = pcs - mean(batch) + mean(global)
    if len(batches) != pcs.shape[0]:
        raise HTTPException(status_code=400, detail="batches length must match number of cells")

    df = pcs.copy()
    df["__batch"] = batches

    global_mean = df.drop(columns="__batch").mean(axis=0)

    corrected = []
    for b, sub in df.groupby("__batch"):
        sub_pcs = sub.drop(columns="__batch")
        sub_mean = sub_pcs.mean(axis=0)
        corrected.append(sub_pcs - sub_mean + global_mean)

    out = pd.concat(corrected, axis=0).loc[pcs.index]
    out.columns = pcs.columns
    return out


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    df = _read_counts_matrix(file, raw)

    run_id = _new_run_id()
    RUNS[run_id] = {
        "filename": file.filename,
        "counts": df,
    }

    return {
        "ok": True,
        "run_id": run_id,
        "filename": file.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
    }


@app.post("/qc")
async def qc(
    file: Optional[UploadFile] = File(None),
    run_id: Optional[str] = Form(None),
):
    if run_id and run_id in RUNS:
        run = RUNS[run_id]
        df: pd.DataFrame = run["counts"]
        filename = run.get("filename", "stored")
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="Provide file or valid run_id")
        raw = await file.read()
        df = _read_counts_matrix(file, raw)
        run_id = _new_run_id()
        RUNS[run_id] = {"filename": file.filename, "counts": df}
        filename = file.filename

    total_counts = df.sum(axis=0)
    detected_genes = (df > 0).sum(axis=0)
    pct_zeros = (df == 0).mean(axis=0)

    qc_payload = {
        "total_counts": _five_number(total_counts),
        "detected_genes": _five_number(detected_genes),
        "pct_zeros": _five_number(pct_zeros),
    }

    RUNS[run_id]["qc"] = qc_payload

    return {
        "ok": True,
        "run_id": run_id,
        "filename": filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        **qc_payload,
    }


@app.post("/normalize")
async def normalize(
    file: Optional[UploadFile] = File(None),
    run_id: Optional[str] = Form(None),
):
    if run_id and run_id in RUNS:
        run = RUNS[run_id]
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="Provide file or valid run_id")
        raw = await file.read()
        df = _read_counts_matrix(file, raw)
        run_id = _new_run_id()
        run = {"filename": file.filename, "counts": df}
        RUNS[run_id] = run

    _ensure_normalized(run)

    log1p: pd.DataFrame = run["normalized_log1p"]
    libsize: Dict[str, float] = run["libsize"]

    return {
        "ok": True,
        "run_id": run_id,
        "filename": run.get("filename"),
        "shape": [int(log1p.shape[0]), int(log1p.shape[1])],
        "libsize_min": float(np.min(list(libsize.values()))),
        "libsize_median": float(np.median(list(libsize.values()))),
        "libsize_max": float(np.max(list(libsize.values()))),
        "n_pcs": int(run["pcs"].shape[1]),
        "pca_var_ratio_first5": [float(x) for x in run["pca_var_ratio"][:5]],
    }


@app.post("/harmony")
async def harmony(
    run_id: str = Form(...),
    batches: Optional[str] = Form(None),
):
    run = _get_run_or_400(run_id)
    _ensure_normalized(run)

    pcs: pd.DataFrame = run["pcs"]

    if batches:
        # batches can be JSON list or comma-separated
        b = batches.strip()
        if b.startswith("["):
            try:
                batch_list = json.loads(b)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON in batches")
        else:
            batch_list = [x.strip() for x in b.split(",") if x.strip()]
        batch_list = [str(x) for x in batch_list]
    else:
        batch_list = _default_batches(list(pcs.index))

    corrected = _batch_center_pcs(pcs, batch_list)
    run["harmony_pcs"] = corrected
    run["batches"] = batch_list

    return {
        "ok": True,
        "run_id": run_id,
        "n_cells": int(corrected.shape[0]),
        "n_pcs": int(corrected.shape[1]),
        "n_batches": int(len(set(batch_list))),
    }


@app.post("/cluster")
async def cluster(
    run_id: str = Form(...),
    k: int = Form(10),
):
    run = _get_run_or_400(run_id)
    _ensure_normalized(run)

    Xdf: pd.DataFrame = run.get("harmony_pcs") or run["pcs"]
    X = Xdf.values

    k = int(k)
    k = max(2, min(k, max(2, X.shape[0] - 1)))

    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(X)

    clusters = pd.Series(labels, index=Xdf.index, name="cluster")
    run["clusters"] = clusters
    run["kmeans_inertia"] = float(km.inertia_)

    counts = clusters.value_counts().sort_index().to_dict()

    return {
        "ok": True,
        "run_id": run_id,
        "k": k,
        "cluster_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "inertia": float(km.inertia_),
    }


@app.post("/train")
async def train(
    run_id: str = Form(...),
    algorithm: str = Form("rf"),
):
    run = _get_run_or_400(run_id)
    _ensure_normalized(run)

    clusters: pd.Series = run.get("clusters")
    if clusters is None:
        raise HTTPException(status_code=400, detail="Run clustering first")

    Xdf: pd.DataFrame = run.get("harmony_pcs") or run["pcs"]
    X = Xdf.values
    y = clusters.loc[Xdf.index].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y if len(np.unique(y)) > 1 else None
    )

    algo = algorithm.lower().strip()
    if algo != "rf":
        raise HTTPException(status_code=400, detail="Only algorithm=rf is implemented in this demo")

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    run["model"] = clf
    run["train_metrics"] = {"accuracy": acc, "f1_macro": f1m, "n_test": int(len(y_test))}

    return {
        "ok": True,
        "run_id": run_id,
        "algorithm": "rf",
        "metrics": run["train_metrics"],
    }


@app.post("/export")
async def export(
    run_id: str = Form(...),
):
    run = _get_run_or_400(run_id)

    # Ensure we have at least normalized outputs for export
    if "counts" not in run:
        raise HTTPException(status_code=400, detail="No counts in this run")

    if "normalized_log1p" not in run or "pcs" not in run:
        _ensure_normalized(run)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # qc
        qc_obj = run.get("qc", {"note": "QC not run"})
        z.writestr("qc.json", json.dumps(qc_obj, indent=2))

        # normalized
        norm: pd.DataFrame = run["normalized_log1p"]
        z.writestr("normalized_log1p.csv", norm.to_csv())

        # pcs
        pcs: pd.DataFrame = run["pcs"]
        z.writestr("pcs.csv", pcs.to_csv(index=True))

        # harmony pcs
        if "harmony_pcs" in run:
            hpcs: pd.DataFrame = run["harmony_pcs"]
            z.writestr("harmony_pcs.csv", hpcs.to_csv(index=True))

        # clusters
        if "clusters" in run:
            cl: pd.Series = run["clusters"]
            z.writestr("clusters.csv", cl.to_csv(index=True))

        # training metrics
        if "train_metrics" in run:
            z.writestr("train_metrics.json", json.dumps(run["train_metrics"], indent=2))

        # manifest
        manifest = {
            "run_id": run_id,
            "filename": run.get("filename"),
            "has_qc": "qc" in run,
            "has_harmony": "harmony_pcs" in run,
            "has_clusters": "clusters" in run,
            "has_model": "model" in run,
        }
        z.writestr("manifest.json", json.dumps(manifest, indent=2))

    buf.seek(0)

    headers = {"Content-Disposition": f'attachment; filename="rnaseq_export_{run_id}.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
