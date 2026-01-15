# rnaseq-backend

## Endpoints

### Atlas
- `GET /health`
- `GET /atlas/manifest`
- `GET /atlas/accessions?disease=...`
- `GET /atlas/cell_types`
- `GET /atlas/markers?panel=default`

### Differential expression (placeholders)
- `GET /de/volcano?disease=SLE&cell_type=CD4%20T%20cells`
- `GET /de/overlap?left=SjS&right=SLE&cell_type=...`
