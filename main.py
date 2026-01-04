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

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/normalize")
async def normalize_counts(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv count matrix")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), index_col=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read CSV: {e}")

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all().all():
        raise HTTPException(status_code=400, detail="All values are non-numeric after coercion")

    libsize = df.sum(axis=0, skipna=True)
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
