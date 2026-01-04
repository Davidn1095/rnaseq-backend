# main.py
import io
import json
import math
import os
import uuid
import zipfile
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

APP_TITLE = "RNA-seq preprocessing API"
RUN_DIR = "/tmp/rnaseq_runs"

os.makedirs(RUN_DIR, exist_ok=True)

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helpers
# -------------------------
def _run_path(run_id: str, name: str) -> str:
    return os.path.join(RUN_DIR, f"{run_id}_{name}")


def _save_json(run_id: str, name: str, obj: Any) -> None:
    p = _run_path(run_id, f"{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _load_json(run_id: str, name: str) -> Any:
    p = _run_path(run_id, f"{name}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_pickle(run_id: str, name: str, df: Any) -> None:
    p = _run_path(run_id, f"{name}.pkl")
    pd.to_pickle(df, p)


def _load_pickle(run_id: str, name: str) -> Any:
    p = _run_path(run_id, f"{name}.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_pickle(p)


def _detect_sep_from_bytes(raw: bytes) -> str:
    head = raw[:4096].decode("utf-8", errors="ignore")
    if "\t" in head and head.count("\t") > head.count(","):
        return "\t"
    return ","


def _read_matrix_from_upload(raw: bytes) -> pd.DataFrame:
    sep = _detect_sep_from_bytes(raw)
    df = pd.read_csv(io.BytesIO(raw), sep=sep, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all().all():
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion")
    df = df.fillna(0.0)
    df = df.clip(lower=0.0)
    return df


async def _get_form(request: Request) -> Dict[str, Any]:
    try:
        form = await request.form()
        return dict(form)
    except Exception:
        return {}


async def _get_run_id(request: Request) -> Optional[str]:
    ct = (request.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try:
            data = await request.json()
            if isinstance(data, dict):
                rid = data.get("run_id")
                return str(rid) if rid else None
        except Exception:
            return None

    if "multipart/form-data" in ct or "application/x-www-form-urlencoded" in ct:
        form = await _get_form(request)
        rid = form.get("run_id")
        return str(rid) if rid else None

    return None


def _qc_summary(df: pd.DataFrame) -> Dict[str, Any]:
    # df: genes x cells
    total_counts = df.sum(axis=0).astype(float)
    detected_genes = (df > 0).sum(axis=0).astype(float)
    pct_zeros = (df == 0).sum(axis=0).astype(float) / df.shape[0]

    def qstats(x: pd.Series) -> Dict[str, float]:
        return {
            "min": float(x.min()),
            "p25": float(x.quantile(0.25)),
            "p50": float(x.quantile(0.50)),
            "p75": float(x.quantile(0.75)),
            "max": float(x.max()),
        }

    return {
        "total_counts": qstats(total_counts),
        "detected_genes": qstats(detected_genes),
        "pct_zeros": qstats(pct_zeros),
    }


def _log1p_cpm(df: pd.DataFrame) -> pd.DataFrame:
    libsize = df.sum(axis=0).replace(0, np.nan)
    cpm = df.div(libsize, axis=1) * 1e6
    log1p = np.log1p(cpm.fillna(0.0))
    return log1p


def _zscore_per_gene(df: pd.DataFrame) -> pd.DataFrame:
    # z-score each gene across cells
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
    return z


def _compute_pca(cell_by_gene: pd.DataFrame, n_components: int = 30) -> np.ndarray:
    n_components = int(min(n_components, cell_by_gene.shape[1], cell_by_gene.shape[0]))
    if n_components < 2:
        n_components = 2
    pca = PCA(n_components=n_components, random_state=0)
    emb = pca.fit_transform(cell_by_gene.values)
    return emb


def _try_harmony(emb: np.ndarray, batches: np.ndarray) -> Dict[str, Any]:
    # harmonypy expects: data is features x samples
    try:
        import harmonypy as hm  # type: ignore
    except Exception:
        return {"applied": False, "reason": "harmonypy missing"}

    try:
        meta = pd.DataFrame({"batch": batches})
        ho = hm.run_harmony(emb.T, meta, "batch")
        corrected = ho.Z_corr.T
        return {"applied": True, "reason": "ok", "corrected": corrected}
    except Exception as e:
        return {"applied": False, "reason": f"harmony error: {e}"}


def _train_cluster_classifier(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    # Simple CV with multinomial LR
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    f1s = []

    for tr, te in skf.split(X, y):
        model = LogisticRegression(
            max_iter=2000,
            multi_class="auto",
            solver="lbfgs",
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))

    return {
        "cv_folds": 5,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
    }


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(request: Request):
    form = await _get_form(request)
    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "file"], "msg": "Field required", "type": "missing"}])

    filename = getattr(file, "filename", "") or "upload.csv"
    raw = await file.read()

    run_id = str(form.get("run_id") or uuid.uuid4().hex)
    df = _read_matrix_from_upload(raw)

    _save_pickle(run_id, "counts", df)
    _save_json(run_id, "meta", {"filename": filename, "shape": [int(df.shape[0]), int(df.shape[1])]})

    return {"ok": True, "run_id": run_id, "filename": filename, "shape": [int(df.shape[0]), int(df.shape[1])]}


@app.post("/qc")
async def qc(request: Request):
    form = await _get_form(request)
    file = form.get("file")
    run_id = str(form.get("run_id") or "")

    if run_id:
        try:
            df = _load_pickle(run_id, "counts")
            meta = _load_json(run_id, "meta")
            filename = meta.get("filename", "counts.csv")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail='run_id not found. Re-run /upload or /qc.')
    else:
        if file is None:
            raise HTTPException(status_code=422, detail=[{"loc": ["body", "file"], "msg": "Field required", "type": "missing"}])
        filename = getattr(file, "filename", "") or "upload.csv"
        raw = await file.read()
        run_id = uuid.uuid4().hex
        df = _read_matrix_from_upload(raw)
        _save_pickle(run_id, "counts", df)
        _save_json(run_id, "meta", {"filename": filename, "shape": [int(df.shape[0]), int(df.shape[1])]})

    summary = _qc_summary(df)
    out = {"ok": True, "run_id": run_id, "filename": filename, "shape": [int(df.shape[0]), int(df.shape[1])], **summary}
    _save_json(run_id, "qc", out)
    return out


@app.post("/normalize")
async def normalize(request: Request):
    form = await _get_form(request)
    file = form.get("file")
    run_id = str(form.get("run_id") or "")

    if run_id:
        try:
            df = _load_pickle(run_id, "counts")
            meta = _load_json(run_id, "meta")
            filename = meta.get("filename", "counts.csv")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail='run_id not found. Re-run /upload or /qc.')
    else:
        if file is None:
            raise HTTPException(status_code=422, detail=[{"loc": ["body", "file"], "msg": "Field required", "type": "missing"}])
        filename = getattr(file, "filename", "") or "upload.csv"
        raw = await file.read()
        run_id = uuid.uuid4().hex
        df = _read_matrix_from_upload(raw)
        _save_pickle(run_id, "counts", df)
        _save_json(run_id, "meta", {"filename": filename, "shape": [int(df.shape[0]), int(df.shape[1])]})

    log1p = _log1p_cpm(df)
    z = _zscore_per_gene(log1p)

    _save_pickle(run_id, "norm_log1p_cpm", log1p)
    _save_pickle(run_id, "norm_z", z)

    libsize = df.sum(axis=0, skipna=True).astype(float)

    out = {
        "ok": True,
        "run_id": run_id,
        "filename": filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "libsize": {str(k): float(v) for k, v in libsize.to_dict().items()},
        "log1p_cpm_summary": {
            "min": float(log1p.min().min()),
            "median": float(log1p.stack().median()),
            "max": float(log1p.max().max()),
        },
        "stored": ["norm_log1p_cpm", "norm_z"],
    }
    _save_json(run_id, "normalize", out)
    return out


@app.post("/harmony")
async def harmony(request: Request):
    run_id = await _get_run_id(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])

    try:
        z = _load_pickle(run_id, "norm_z")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='run_id not found. Re-run /upload or /normalize.')

    # cells x genes
    cell_by_gene = z.T

    pca_emb = _compute_pca(cell_by_gene, n_components=30)

    # No batch metadata in this demo. Treat as single batch.
    batches = np.zeros((pca_emb.shape[0],), dtype=int)

    hm = _try_harmony(pca_emb, batches)
    if hm.get("applied"):
        emb = hm["corrected"]
        meta = {"applied": True, "reason": "ok"}
        embedding_name = "harmony"
    else:
        emb = pca_emb
        meta = {"applied": False, "reason": hm.get("reason", "not applied")}
        embedding_name = "pca_fallback"

    emb_df = pd.DataFrame(emb, index=cell_by_gene.index, columns=[f"PC{i+1}" for i in range(emb.shape[1])])
    _save_pickle(run_id, "embedding", emb_df)
    _save_json(run_id, "harmony_meta", meta)

    out = {
        "ok": True,
        "run_id": run_id,
        "embedding": embedding_name,
        "shape": [int(emb.shape[0]), int(emb.shape[1])],
        "harmony_meta": meta,
        "stored": ["embedding"],
    }
    _save_json(run_id, "harmony", out)
    return out


@app.post("/cluster")
async def cluster(request: Request):
    run_id = await _get_run_id(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])

    try:
        emb_df = _load_pickle(run_id, "embedding")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='run_id not found. Re-run /harmony.')

    X = emb_df.values
    k = int(min(6, max(2, round(math.sqrt(X.shape[0] / 2)))))

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    lab_ser = pd.Series(labels, index=emb_df.index, name="cluster")
    _save_pickle(run_id, "clusters", lab_ser)

    sizes = {str(i): int((labels == i).sum()) for i in range(k)}
    out = {
        "ok": True,
        "run_id": run_id,
        "k": k,
        "cluster_sizes": sizes,
        "note": "Clusters stored for /train and /export.",
        "stored": ["clusters"],
    }
    _save_json(run_id, "cluster", out)
    return out


@app.post("/train")
async def train(request: Request):
    run_id = await _get_run_id(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])

    try:
        emb_df = _load_pickle(run_id, "embedding")
        clusters = _load_pickle(run_id, "clusters")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='run_id not found. Re-run /cluster.')

    X = emb_df.values.astype(float)
    y = clusters.values.astype(int)

    metrics = _train_cluster_classifier(X, y)

    out = {
        "ok": True,
        "run_id": run_id,
        "task": "predict_cluster_from_embedding",
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "metrics": metrics,
        "stored": ["train"],
    }
    _save_json(run_id, "train", out)
    return out


@app.post("/export")
async def export_bundle(request: Request):
    run_id = await _get_run_id(request)
    if not run_id:
        raise HTTPException(status_code=422, detail=[{"loc": ["body", "run_id"], "msg": "Field required", "type": "missing"}])

    # Collect what exists
    parts: Dict[str, bytes] = {}

    def add_json(name: str):
        try:
            obj = _load_json(run_id, name)
            parts[f"{name}.json"] = json.dumps(obj, indent=2).encode("utf-8")
        except Exception:
            return

    def add_pickle_as_csv(name: str, fname: str):
        try:
            obj = _load_pickle(run_id, name)
            if isinstance(obj, pd.DataFrame):
                parts[fname] = obj.to_csv(index=True).encode("utf-8")
            elif isinstance(obj, pd.Series):
                parts[fname] = obj.to_frame().to_csv(index=True).encode("utf-8")
        except Exception:
            return

    add_json("meta")
    add_json("qc")
    add_json("normalize")
    add_json("harmony")
    add_json("cluster")
    add_json("train")

    add_pickle_as_csv("norm_log1p_cpm", "normalized_log1p_cpm.csv")
    add_pickle_as_csv("norm_z", "normalized_zscore.csv")
    add_pickle_as_csv("embedding", "embedding.csv")
    add_pickle_as_csv("clusters", "clusters.csv")

    if not parts:
        raise HTTPException(status_code=404, detail="No artifacts found for run_id. Re-run the pipeline.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for k, v in parts.items():
            z.writestr(k, v)
    buf.seek(0)

    # FastAPI will set application/zip based on bytes response in Cloud Run via default
    # If you want explicit StreamingResponse, keep it simple and return bytes here.
    return buf.getvalue()
