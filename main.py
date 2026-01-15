import json
from pathlib import Path
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

ARTIFACTS_PATH = Path(__file__).resolve().parent / "data" / "atlas_artifacts.json"

def _diseases() -> List[str]:
    ds = sorted({a["disease"] for a in ACCESSIONS})
    # ensure Healthy first if present
    return ["Healthy"] + [d for d in ds if d != "Healthy"] if "Healthy" in ds else ds

@app.on_event("startup")
def load_artifacts() -> None:
    with ARTIFACTS_PATH.open("r", encoding="utf-8") as handle:
        app.state.artifacts = json.load(handle)

def _get_artifacts() -> Dict[str, Any]:
    return app.state.artifacts

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

@app.get("/atlas/umap")
def atlas_umap(
    disease: Optional[str] = Query(default=None),
    cell_type: Optional[str] = Query(default=None),
    color_key: str = Query(default="cell_type"),
    value_key: Optional[str] = Query(default=None),
):
    artifacts = _get_artifacts()
    rows = artifacts["umap"]
    if disease is not None:
        rows = [row for row in rows if row.get("disease") == disease]
    if cell_type is not None:
        rows = [row for row in rows if row.get("cell_type") == cell_type]
    if rows and color_key not in rows[0]:
        return {"ok": False, "error": "unknown color_key", "available": ["cell_type", "disease", "module_score"]}

    response = {
        "ok": True,
        "filters": {"disease": disease, "cell_type": cell_type},
        "color_key": color_key,
        "x": [row["x"] for row in rows],
        "y": [row["y"] for row in rows],
        "cell_id": [row["cell_id"] for row in rows],
        "color": [row[color_key] for row in rows] if rows else [],
    }
    if value_key is not None:
        response["value_key"] = value_key
        response["value"] = [row.get(value_key) for row in rows]
    return response

@app.get("/atlas/dotplot")
def atlas_dotplot(
    genes: str = Query(..., description="Comma-separated list of genes"),
    group_by: str = Query(default="cell_type"),
):
    artifacts = _get_artifacts()
    groupings = artifacts["dotplot"]["groupings"]
    if group_by not in groupings:
        return {"ok": False, "error": "unknown group_by", "available": list(groupings.keys())}
    grouping = groupings[group_by]
    gene_list = [gene.strip() for gene in genes.split(",") if gene.strip()]
    unknown = [gene for gene in gene_list if gene not in grouping["genes"]]
    if unknown:
        return {"ok": False, "error": "unknown genes", "unknown": unknown}

    groups = grouping["groups"]
    avg = [grouping["genes"][gene]["avg"] for gene in gene_list]
    pct = [grouping["genes"][gene]["pct"] for gene in gene_list]
    return {
        "ok": True,
        "group_by": group_by,
        "groups": groups,
        "genes": gene_list,
        "avg": avg,
        "pct": pct,
    }

@app.get("/atlas/violin")
def atlas_violin(
    gene: str = Query(...),
    group_by: str = Query(default="cell_type"),
    kind: str = Query(default="hist", pattern="^(hist|quantile)$"),
):
    artifacts = _get_artifacts()
    groupings = artifacts["violin"]["groupings"]
    if group_by not in groupings:
        return {"ok": False, "error": "unknown group_by", "available": list(groupings.keys())}
    genes = groupings[group_by]["genes"]
    if gene not in genes:
        return {"ok": False, "error": "unknown gene", "available": list(genes.keys())}
    payload = genes[gene]
    groups = list(payload["counts"].keys())
    if kind == "hist":
        counts = [payload["counts"][group] for group in groups]
        return {
            "ok": True,
            "gene": gene,
            "group_by": group_by,
            "kind": kind,
            "groups": groups,
            "bins": payload["bins"],
            "counts": counts,
        }
    quantiles = [payload["quantiles"][group] for group in groups]
    return {
        "ok": True,
        "gene": gene,
        "group_by": group_by,
        "kind": kind,
        "groups": groups,
        "quantiles": quantiles,
    }

@app.get("/atlas/de")
def atlas_de(
    contrast: str = Query(...),
    cell_type: str = Query(...),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    top_n: int = Query(default=5, ge=1, le=50),
):
    artifacts = _get_artifacts()
    contrasts = artifacts["de"]["contrasts"]
    if contrast not in contrasts:
        return {"ok": False, "error": "unknown contrast", "available": list(contrasts.keys())}
    cell_types = contrasts[contrast]["cell_types"]
    if cell_type not in cell_types:
        return {"ok": False, "error": "unknown cell_type", "available": list(cell_types.keys())}
    rows = cell_types[cell_type]
    total = len(rows)
    paged = rows[offset:offset + limit]
    top_up = sorted(rows, key=lambda r: r["logfc"], reverse=True)[:top_n]
    top_down = sorted(rows, key=lambda r: r["logfc"])[:top_n]
    return {
        "ok": True,
        "contrast": contrast,
        "cell_type": cell_type,
        "total": total,
        "limit": limit,
        "offset": offset,
        "rows": paged,
        "top_up": top_up,
        "top_down": top_down,
    }

@app.get("/atlas/modulescore")
def atlas_modulescore(
    module: str = Query(...),
    group_by: str = Query(default="cell_type"),
    include_values: bool = Query(default=False),
):
    artifacts = _get_artifacts()
    modules = artifacts["modulescore"]["modules"]
    if module not in modules:
        return {"ok": False, "error": "unknown module", "available": list(modules.keys())}
    module_payload = modules[module]
    groupings = module_payload["groupings"]
    if group_by not in groupings:
        return {"ok": False, "error": "unknown group_by", "available": list(groupings.keys())}
    grouping = groupings[group_by]
    groups = list(grouping["counts"].keys())
    counts = [grouping["counts"][group] for group in groups]
    response = {
        "ok": True,
        "module": module,
        "group_by": group_by,
        "groups": groups,
        "bins": grouping["bins"],
        "counts": counts,
    }
    if include_values:
        response["feature_plot"] = module_payload["feature_plot"]
    return response

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
