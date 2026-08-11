# Deploying iSpot

iSpot ships two deployment profiles.

## Slim (recommended for beta) — fast, reliable, CPU-only

Runs the API, the interactive viewer, the FastQC-style QC report, and every
method that works on a clean CPU image (Leiden/PCA and any scanpy-only method).
It excludes the heavy, build-fragile backends (torch, TensorFlow,
R/Bioconductor, torch-geometric, squidpy), so it builds in a couple of minutes.

```bash
docker build -f Dockerfile.slim -t ispot:slim .
docker run -p 8100:8100 ispot:slim
# open http://localhost:8100
```

Or without Docker:

```bash
pip install -r requirements-core.txt
./start.sh --host 0.0.0.0 --port 8100
```

This profile is exactly what the CI **integration** job installs and exercises
(`requirements-core.txt`), so "it builds and boots" is continuously verified.

At startup `GET /api/health` reports `methods_runnable` and `default_methods`,
and `GET /api/methods/availability` lists each method with the reason it can or
cannot run in the current image. A default benchmark runs only the runnable
methods (override with `ISPOT_BETA_METHODS`).

## Full — all 12 methods (heavy)

Enables the GNN / deep-learning / R methods. Large image, slower and more
fragile to build (Bioconductor version drift, CUDA/torch wheels). Only build
this when you need those specific methods.

```bash
docker build -t ispot:full .      # uses the full Dockerfile + requirements.txt
docker run -p 8100:8100 ispot:full
```

R-based methods (BayesSpace, BISON, SpaRTaCo, spatialMNN) additionally require
R + Bioconductor in the image; GNN methods (STAGATE, GraphST, HyperGCN, …) also
need their model repositories on disk (see `ISPOT_*_DIR` env vars).

## Configuration

See the Environment Variables table in `README.md`. Common ones for a hosted
deployment: `ISPOT_JOBS_DIR` (persistent volume), `ISPOT_ALLOWED_ORIGINS`,
`ISPOT_API_KEY` (require `X-API-Key` on mutating endpoints), `ISPOT_MAX_UPLOAD_MB`,
`ISPOT_MAX_SPOTS`, `ISPOT_JOB_TTL_DAYS`.

## Scaling beyond one node

The MVP runs benchmark jobs as in-process background tasks with job status
persisted to SQLite (survives restarts). For multiple replicas / heavy
throughput, move job **execution** onto Celery/Redis workers and job artifacts
to object storage — see `docker-compose.yml` for the intended topology.
