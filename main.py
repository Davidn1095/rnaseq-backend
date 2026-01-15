from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Atlas API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TISSUE = "PBMC"

ACCESSIONS: List[Dict[str, Any]] = [
    {"id": "GSE000000", "disease": "Healthy", "platform": "10x 3' v2", "donors": 12, "cells": 50210, "tissue": TISSUE},
    {"id": "GSE000001", "disease": "Healthy", "platform": "10x 3' v3", "donors": 9,  "cells": 38840, "tissue": TISSUE},

    {"id": "GSE156989", "disease": "SLE", "platform": "10x 3' v2", "donors": 24, "cells": 80312, "tissue": TISSUE},
    {"id": "GSE135779", "disease": "SLE", "platform": "10x 3' v2", "donors": 14, "cells": 42190, "tissue": TISSUE},
    {"id": "GSE157278", "disease": "SjS", "platform": "10x 3' v3", "donors": 18, "cells": 61234, "tissue": TISSUE},
    {"id": "HRA003613", "disease": "RA",  "platform": "10x 5'",     "donors": 10, "cells": 28740, "tissue": TISSUE},
]

CELL_TYPES = [
    "CD4 T cells",
    "CD8 T cells",
    "B cells",
    "NK cells",
    "Monocytes",
    "Dendritic cells",
    "Plasma cells",
]

MARKERS_DEFAULT = [
    "IL7R","CCR7","LTB","NKG7","GNLY","MS4A1","CD79A","MZB1",
    "LYZ","S100A8","FCGR3A","LST1","FCER1A","CLEC10A",
]

def _diseases() -> List[str]:
    ds = sorted({a["disease"] for a in ACCESSIONS})
    # ensure Healthy first if present
    return ["Healthy"] + [d for d in ds if d != "Healthy"] if "Healthy" in ds else ds

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/atlas/manifest")
def manifest():
    return {
        "ok": True,
        "tissue": TISSUE,
        "diseases": _diseases(),
        "accessions": ACCESSIONS,
        "cell_types": CELL_TYPES,
        "marker_panels": {"default": MARKERS_DEFAULT},
    }

@app.get("/atlas/accessions")
def accessions(disease: Optional[str] = Query(default=None)):
    if disease is None:
        return {"ok": True, "accessions": ACCESSIONS}
    return {"ok": True, "accessions": [a for a in ACCESSIONS if a["disease"] == disease]}

@app.get("/atlas/cell_types")
def cell_types():
    return {"ok": True, "cell_types": CELL_TYPES}

@app.get("/atlas/markers")
def markers(panel: str = Query(default="default")):
    if panel != "default":
        return {"ok": False, "error": "unknown panel", "available": ["default"]}
    return {"ok": True, "panel": panel, "genes": MARKERS_DEFAULT}

@app.get("/de/volcano")
def de_volcano(
    disease: str = Query(...),
    cell_type: str = Query(...),
):
    return JSONResponse(
        status_code=501,
        content={
            "ok": False,
            "message": "Differential expression volcano endpoint is not implemented yet.",
            "requested": {"disease": disease, "cell_type": cell_type},
        },
    )

@app.get("/de/overlap")
def de_overlap(
    left: str = Query(...),
    right: str = Query(...),
    cell_type: str = Query(...),
):
    return JSONResponse(
        status_code=501,
        content={
            "ok": False,
            "message": "Differential expression overlap endpoint is not implemented yet.",
            "requested": {"left": left, "right": right, "cell_type": cell_type},
        },
    )
