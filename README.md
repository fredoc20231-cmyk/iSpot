# iSpot: Spatial Transcriptomics Clustering Benchmark Platform

A hosted SaaS platform for benchmarking spatial transcriptomics clustering methods. Upload ST data, benchmark multiple clustering methods, and get a ranked recommendation with interactive visualizations and a written report.

## Quick Start

### Option 1: Direct (Python)

```bash
pip install -r requirements.txt
./start.sh --host 0.0.0.0 --port 8100
```

### Option 2: Docker

```bash
docker build -t ispot .
docker run -p 8100:8100 ispot
```

Then open `http://localhost:8100` in your browser.

## Features

- **12 clustering methods**: Leiden/PCA, SpaGCN, STAGATE, GraphST, BayesSpace, HyperGCN, STMSGAL, SCOIGET, Novae, BISON, SpaRTaCo, spatialMNN
- **7 ST platforms**: Visium, Slide-seqV2, MERFISH, CosMx, Xenium, Stereo-seq, DBiT-seq
- **Ground truth + no-GT modes**: When annotations exist, scores with ARI/NMI/F1. When they don't, uses a composite proxy score (SCS 0.35, CSS 0.25, ESS 0.20, CAS 0.20)
- **Meta-learning**: Learns from all benchmark runs across all users to recommend methods for new datasets
- **Plugin system**: Community-contributed methods via a registry
- **Deliverables**: Ranking table (CSV), publication figures (PNG), interactive spatial viewer (JSON), PDF report

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/methods` | List available methods |
| GET | `/api/platforms` | List supported platforms |
| POST | `/api/upload` | Upload ST data (.h5ad, .h5, .csv) |
| POST | `/api/benchmark` | Start benchmark job |
| GET | `/api/jobs/{id}` | Get job status |
| GET | `/api/jobs/{id}/results` | Get job results |
| GET | `/api/jobs/{id}/download/{filename}` | Download deliverable |
| GET | `/api/meta-learning/stats` | Meta-learning database stats |

_Planned (not yet implemented):_ `POST /api/plugins/register`, `GET /api/meta-learning/recommend`.

## Project Structure

```
iSpot/
├── ispot/
│   ├── __init__.py
│   ├── api.py                      # FastAPI backend (9 API routes + static)
│   ├── deliverables.py             # Ranking CSV, figures, viewer JSON, PDF report
│   ├── meta_learning.py            # Meta-learning engine (thread-safe SQLite)
│   ├── multiplatform_loaders.py    # Loaders for 7 ST platforms
│   ├── plugins.py                  # Plugin system + community registry
│   ├── nogt_scoring.py             # No-GT composite scoring
│   ├── loaders.py                  # Dataset loaders (DLPFC, HER2+, MOSTA)
│   ├── cluster_estimation.py       # Auto cluster count estimation
│   ├── runner.py                   # Benchmark runner orchestration
│   ├── profiling.py                # Data profiling
│   ├── metrics.py                  # Evaluation metrics (ARI, NMI, F1)
│   ├── registry.py                 # Method registry (12 methods)
│   ├── preprocessing.py            # Standardized preprocessing
│   ├── methods/                    # 12 method runners + 3 R scripts
│   │   ├── _nogt_helper.py         # Safe metrics for no-GT path
│   │   ├── leiden_pca.py
│   │   ├── spagcn.py
│   │   ├── stagate.py
│   │   ├── graphst.py
│   │   ├── bayesspace.py
│   │   ├── hypergcn.py
│   │   ├── stmsgal.py
│   │   ├── scoiget.py
│   │   ├── novae.py
│   │   ├── bison.py
│   │   ├── spartaco.py
│   │   ├── spatialmnn.py
│   │   ├── _r_methods_runner.R
│   │   ├── _bayesspace_runner.R
│   │   └── _hungarian_f1.R
│   └── frontend/                   # SPA (no build step needed)
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── data/
│   └── unified_results.csv         # Meta-learning seed data (207 rows)
├── requirements.txt
├── start.sh
├── Dockerfile
└── README.md
```

## Usage Flow

1. **Upload**: `POST /api/upload` with your `.h5ad` file → returns `job_id`
2. **Benchmark**: `POST /api/benchmark` with `job_id`, methods, and optional parameters
3. **Poll**: `GET /api/jobs/{id}` until status is `completed`
4. **Results**: `GET /api/jobs/{id}/results` → ranking, figures, viewer, report
5. **Download**: `GET /api/jobs/{id}/download/{filename}` for individual files

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ISPOT_HOST` | `0.0.0.0` | Server bind address |
| `ISPOT_PORT` | `8100` | Server port |

## Notes

- R-based methods (BayesSpace) require R + Bioconductor packages installed
- Meta-learning DB is auto-seeded from `data/unified_results.csv` on first startup
- Job files are stored in `ispot_jobs/` (created automatically)
- No-GT scores are proxy metrics, not ground truth — this is stated in every report
- Cluster count estimation uses knee detection; typically within ±2 of true count
