import io
import math
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RNA-seq preprocessing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _infer_sep(first_line: str) -> str:
    return "\t" if "\t" in first_line else ","


async def _read_counts_matrix(upload: UploadFile) -> pd.DataFrame:
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    first_line = raw.splitlines()[0].decode("utf-8", errors="ignore")
    sep = _infer_sep(first_line)

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, index_col=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read matrix: {e}")

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Matrix has zero rows or columns")

    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().all().all():
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion")

    # counts matrix: treat missing as 0 for downstream ops
    df = df.fillna(0)

    return df


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/qc")
async def qc_counts(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Upload a .csv or .tsv count matrix")

    df = await _read_counts_matrix(file)

    total_counts = df.sum(axis=0)
    detected_genes = (df > 0).sum(axis=0)
    pct_zeros = (df == 0).mean(axis=0)

    summary = {
        "filename": file.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "total_counts": {
            "min": float(total_counts.min()),
            "p50": float(total_counts.median()),
            "p95": float(total_counts.quantile(0.95)),
            "max": float(total_counts.max()),
        },
        "detected_genes": {
            "min": float(detected_genes.min()),
            "p50": float(detected_genes.median()),
            "p95": float(detected_genes.quantile(0.95)),
            "max": float(detected_genes.max()),
        },
        "pct_zeros": {
            "min": float(pct_zeros.min()),
            "p50": float(pct_zeros.median()),
            "p95": float(pct_zeros.quantile(0.95)),
            "max": float(pct_zeros.max()),
        },
    }

    return summary


@app.post("/normalize")
async def normalize_counts(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Upload a .csv or .tsv count matrix")

    df = await _read_counts_matrix(file)

    libsize = df.sum(axis=0)
    libsize_safe = libsize.replace(0, pd.NA)

    cpm = df.div(libsize_safe, axis=1) * 1e6
    log1p_cpm = cpm.fillna(0).applymap(math.log1p)

    return {
        "filename": file.filename,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "libsize": {str(k): float(v) for k, v in libsize.to_dict().items()},
        "log1p_cpm_summary": {
            "min": float(log1p_cpm.min().min()),
            "median": float(log1p_cpm.stack().median()),
            "max": float(log1p_cpm.max().max()),
        },
    }
