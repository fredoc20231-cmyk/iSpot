"""
iSpot FastAPI backend.

Provides REST API for:
  - POST /api/upload: Upload spatial transcriptomics data
  - POST /api/benchmark: Start a benchmark job
  - GET /api/jobs/{job_id}: Get job status
  - GET /api/jobs/{job_id}/results: Get results (ranking, figures, report)
  - GET /api/methods: List available methods
  - GET /api/health: Health check

Section 2.1 of the platform plan.

For the MVP, jobs run as FastAPI BackgroundTasks (single-server).
In production, replace with Celery + Redis for distributed execution.
"""
from __future__ import annotations

import os
import sys
import json
import uuid
import time
import shutil
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure the repo root (which contains the ``ispot`` package) is importable,
# regardless of where the server is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ispot.multiplatform_loaders import load_data, auto_detect_platform, LOADER_REGISTRY
from ispot.profiling import profile_data, DataFeatureVector
from ispot.preprocessing import preprocess
from ispot.cluster_estimation import estimate_n_clusters
from ispot.registry import (
    ALL_METHODS, METHOD_DISPLAY, METHOD_CATEGORY,
    get_runner, is_stochastic, is_r_based,
)
from ispot.metrics import compute_metrics
from ispot.nogt_scoring import compute_nogt_score, consensus_clustering, DEFAULT_WEIGHTS
from ispot.meta_learning import (
    MetaLearningDB, MetaLearningModel, seed_from_existing_results,
    pilot_then_full, evaluate_pilot_alignment,
)
from ispot.deliverables import (
    generate_ranking_table, generate_figures,
    generate_viewer_data, generate_report,
)
from ispot.plugins import discover_plugins, get_all_method_names
from ispot.job_status import classify_job_status
from ispot.stats_compare import build_comparison_table
from ispot import validation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Job storage directory. Defaults to <repo>/ispot_jobs; override with
# ISPOT_JOBS_DIR (e.g. a mounted volume in production).
WORKSPACE_DIR = Path(os.environ.get("ISPOT_JOBS_DIR", REPO_ROOT / "ispot_jobs"))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Meta-learning database (persistent across restarts)
DB_PATH = str(WORKSPACE_DIR / "meta_learning.db")

# Seed data shipped with the repo; override with ISPOT_SEED_CSV.
SEED_CSV = os.environ.get("ISPOT_SEED_CSV", str(REPO_ROOT / "data" / "unified_results.csv"))

# Initialize meta-learning
ml_db = MetaLearningDB(DB_PATH)
# Seed from existing results if database is empty
if ml_db.count_runs() == 0:
    n = seed_from_existing_results(ml_db, SEED_CSV)
    print(f"Seeded meta-learning DB with {n} records from {SEED_CSV}")

ml_model = MetaLearningModel(min_samples_per_method=3, alpha=1.0)
ml_model.train(ml_db)

# Discover all available methods (built-in + plugins)
discover_plugins()
AVAILABLE_METHODS = list(METHOD_DISPLAY.keys())

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="iSpot",
    description="Spatial Transcriptomics Clustering Benchmark Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>iSpot API</h1><p>Frontend not found. Use the API directly.</p>")


@app.get("/styles.css")
async def serve_styles():
    """Serve CSS."""
    path = FRONTEND_DIR / "styles.css"
    if path.exists():
        return FileResponse(str(path), media_type="text/css")
    return HTMLResponse("Not found", status_code=404)


@app.get("/app.js")
async def serve_js():
    """Serve JavaScript."""
    path = FRONTEND_DIR / "app.js"
    if path.exists():
        return FileResponse(str(path), media_type="application/javascript")
    return HTMLResponse("Not found", status_code=404)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class BenchmarkRequest(BaseModel):
    job_id: str
    methods: list[str] | None = None
    n_clusters: int | None = None
    seeds: list[int] | None = None
    platform: str | None = None
    ground_truth_col: str | None = None
    use_meta_learning: bool = True
    no_gt_weights: dict | None = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "completed_partial", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    created_at: str
    completed_at: str | None = None


# ---------------------------------------------------------------------------
# Job storage (in production: use a database)
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}


def get_job_dir(job_id: str) -> Path:
    return WORKSPACE_DIR / job_id


def _sanitize_json(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "methods_available": len(AVAILABLE_METHODS),
        "meta_learning_runs": ml_db.count_runs(),
        "platforms_supported": list(LOADER_REGISTRY.keys()),
    }


@app.get("/api/methods")
async def list_methods():
    """List all available clustering methods."""
    methods = []
    for method in AVAILABLE_METHODS:
        methods.append({
            "name": method,
            "display_name": METHOD_DISPLAY.get(method, method),
            "category": METHOD_CATEGORY.get(method, "unknown"),
            "is_stochastic": is_stochastic(method),
            "is_r_based": is_r_based(method),
        })
    return {"methods": methods}


@app.get("/api/platforms")
async def list_platforms():
    """List all supported spatial transcriptomics platforms."""
    return {"platforms": list(LOADER_REGISTRY.keys())}


@app.post("/api/upload")
async def upload_data(
    file: UploadFile = File(...),
    platform: str | None = Form(None),
    sample_id: str | None = Form(None),
    ground_truth_col: str | None = Form(None),
):
    """Upload spatial transcriptomics data.

    Accepts .h5ad, .h5, or .csv files. Returns a job_id that can be used
    to start a benchmark.

    The file is stored temporarily and deleted after the benchmark completes
    (or after 7 days, whichever comes first).
    """
    # Reject unsupported file types before creating any job state.
    try:
        validation.validate_extension(file.filename)
    except validation.ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    job_id = str(uuid.uuid4())[:12]
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(exist_ok=True)

    # Save uploaded file, enforcing the configured size cap while streaming.
    file_path = job_dir / file.filename
    try:
        file_size = validation.stream_to_file(file.file, str(file_path))
    except validation.ValidationError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=e.status_code, detail=e.message)

    # Auto-detect platform if not specified
    if platform is None:
        platform = auto_detect_platform(str(file_path))

    # Store job metadata
    jobs[job_id] = {
        "job_id": job_id,
        "status": "uploaded",
        "progress": 0.0,
        "message": f"File uploaded: {file.filename}",
        "created_at": datetime.now().isoformat(),
        "file_path": str(file_path),
        "platform": platform,
        "sample_id": sample_id,
        "ground_truth_col": ground_truth_col,
    }

    return {
        "job_id": job_id,
        "platform": platform,
        "filename": file.filename,
        "file_size": file_size,
        "message": "Upload successful. Use POST /api/benchmark to start analysis.",
    }


@app.post("/api/benchmark")
async def start_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
):
    """Start a benchmark job.

    Runs the selected methods on the uploaded data and generates
    ranking table, figures, interactive viewer data, and PDF report.
    """
    job_id = request.job_id
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found. Upload data first.")

    job = jobs[job_id]
    job["status"] = "queued"
    job["progress"] = 0.0
    job["message"] = "Benchmark queued"

    # Add to background tasks
    background_tasks.add_task(
        run_benchmark_task,
        job_id=job_id,
        methods=request.methods,
        n_clusters=request.n_clusters,
        seeds=request.seeds or [42],
        platform=request.platform or job.get("platform"),
        ground_truth_col=request.ground_truth_col or job.get("ground_truth_col"),
        use_meta_learning=request.use_meta_learning,
        no_gt_weights=request.no_gt_weights,
    )

    return {"job_id": job_id, "status": "queued", "message": "Benchmark started"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a benchmark job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return _sanitize_json(jobs[job_id])


@app.get("/api/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get the results of a completed benchmark job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    job = jobs[job_id]
    if job["status"] not in ("completed", "completed_partial"):
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")

    job_dir = get_job_dir(job_id)
    results_dir = job_dir / "results"

    return _sanitize_json({
        "job_id": job_id,
        "ranking_table": f"/api/jobs/{job_id}/download/ranking_table.csv",
        "figures": [f for f in os.listdir(results_dir) if f.endswith(".png")],
        "viewer_data": f"/api/jobs/{job_id}/download/viewer_data.json",
        "report": f"/api/jobs/{job_id}/download/benchmark_report.pdf",
        "data_profile": job.get("data_profile", {}),
        "n_clusters": job.get("n_clusters", None),
        "has_ground_truth": job.get("has_ground_truth", False),
        "meta_learning": job.get("meta_learning", {}),
        "status": job["status"],
        "method_summary": job.get("method_summary", {}),
    })


@app.get("/api/jobs/{job_id}/download/{filename}")
async def download_result(job_id: str, filename: str):
    """Download a specific result file."""
    job_dir = get_job_dir(job_id)
    file_path = job_dir / "results" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
    return FileResponse(str(file_path))


@app.get("/api/meta-learning/stats")
async def meta_learning_stats():
    """Get meta-learning database statistics."""
    df = ml_db.get_all_runs()
    # Sanitize cv_r2: replace NaN/inf with None for JSON compliance
    cv_r2_sanitized = None
    if ml_model.is_trained and ml_model.cv_r2:
        cv_r2_sanitized = {
            k: (None if (v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))) else float(v))
            for k, v in ml_model.cv_r2.items()
        }
    return {
        "total_runs": len(df),
        "methods": df["method_name"].value_counts().to_dict() if len(df) > 0 else {},
        "platforms": df["platform"].value_counts().to_dict() if len(df) > 0 else {},
        "cv_r2": cv_r2_sanitized,
    }


# ---------------------------------------------------------------------------
# Benchmark execution task
# ---------------------------------------------------------------------------

def run_benchmark_task(
    job_id: str,
    methods: list[str] | None,
    n_clusters: int | None,
    seeds: list[int],
    platform: str,
    ground_truth_col: str | None,
    use_meta_learning: bool,
    no_gt_weights: dict | None,
):
    """Main benchmark execution logic. Runs as a background task."""
    job = jobs[job_id]
    job_dir = get_job_dir(job_id)
    results_dir = job_dir / "results"
    results_dir.mkdir(exist_ok=True)

    try:
        # --- Step 1: Load data ---
        job["status"] = "running"
        job["message"] = "Loading data..."
        job["progress"] = 0.05

        adata = load_data(
            job["file_path"],
            platform=platform,
            sample_id=job.get("sample_id"),
            ground_truth_col=ground_truth_col,
        )

        # --- Step 2: Profile data ---
        job["message"] = "Profiling data..."
        job["progress"] = 0.10

        features = profile_data(adata, platform=platform)
        job["data_profile"] = features.to_dict()

        # Resource guard: reject datasets larger than the configured cap before
        # dispatching any compute-heavy method.
        try:
            validation.validate_spot_count(getattr(features, "n_spots", None))
        except validation.ValidationError as e:
            job["status"] = "failed"
            job["message"] = e.message
            job["progress"] = 1.0
            return

        has_gt = adata.obs["has_ground_truth"].any()
        job["has_ground_truth"] = has_gt

        # --- Step 3: Preprocess for cluster estimation ---
        job["message"] = "Preprocessing..."
        job["progress"] = 0.15

        # Preprocess a copy for cluster estimation only.
        # Method runners do their own preprocessing internally.
        adata_for_est = preprocess(adata.copy())

        # --- Step 4: Determine cluster count ---
        job["message"] = "Estimating cluster count..."
        job["progress"] = 0.20

        if n_clusters is None:
            if features.n_expected_clusters is not None:
                n_clusters = features.n_expected_clusters
            else:
                est_result = estimate_n_clusters(adata_for_est)
                n_clusters = est_result["n_clusters"]
                job["cluster_estimation"] = est_result

        job["n_clusters"] = n_clusters

        # --- Step 5: Meta-learning recommendation ---
        job["message"] = "Meta-learning prediction..."
        job["progress"] = 0.25

        if use_meta_learning:
            decision = pilot_then_full(
                features.to_dict(), ml_model,
                all_methods=AVAILABLE_METHODS,
            )
            job["meta_learning"] = {
                "confidence": decision["confidence"],
                "predicted_ranking": decision["predicted_ranking"][:5],
                "run_all": decision["run_all"],
                "reason": decision["reason"],
            }
            if methods is None:
                methods = decision["pilot_methods"]
        else:
            if methods is None:
                methods = AVAILABLE_METHODS

        # --- Step 6: Run methods ---
        all_results = []
        method_labels = {}  # method -> labels from seed=42
        method_seed_labels = {}  # method -> list of label arrays (for stability)

        total_steps = len(methods) * len(seeds)
        step = 0

        for method in methods:
            if method not in AVAILABLE_METHODS:
                continue

            method_seeds = seeds if is_stochastic(method) else [42]
            runner = get_runner(method)
            method_seed_labels[method] = []

            for seed in method_seeds:
                step += 1
                job["message"] = f"Running {method} (seed={seed})..."
                job["progress"] = 0.25 + 0.60 * (step / total_steps)

                try:
                    t0 = time.time()
                    m = runner(adata, n_clusters, seed=seed)
                    runtime = m.get("runtime", time.time() - t0)

                    result = {
                        "method": method,
                        "method_display": METHOD_DISPLAY.get(method, method),
                        "seed": seed,
                        "ari": m.get("ari"),
                        "macro_f1": m.get("macro_f1"),
                        "weighted_f1": m.get("weighted_f1"),
                        "runtime": runtime,
                        "n_spots": m.get("n_spots", adata.shape[0]),
                        "n_clusters_pred": m.get("n_clusters_pred"),
                        "n_clusters_true": n_clusters,
                        "error": None,
                    }

                    # Store labels for no-GT scoring and viewer
                    if "labels" in m:
                        labels = m["labels"]
                    else:
                        # Extract from adata if the method stored them
                        labels = None
                        for key in [f"{method.lower()}_pred", "pred", "cluster"]:
                            if key in adata.obs.columns:
                                labels = adata.obs[key].values.astype(str)
                                break
                    if labels is not None:
                        method_seed_labels[method].append(labels)
                        if seed == 42 or seed == seeds[0]:
                            method_labels[method] = labels

                except Exception as e:
                    result = {
                        "method": method,
                        "method_display": METHOD_DISPLAY.get(method, method),
                        "seed": seed,
                        "ari": None, "macro_f1": None, "weighted_f1": None,
                        "runtime": None, "n_spots": adata.shape[0],
                        "n_clusters_pred": None, "n_clusters_true": n_clusters,
                        "error": str(e),
                    }

                all_results.append(result)

        results_df = pd.DataFrame(all_results)

        # Classify the outcome: a job is only "failed" if EVERY method failed.
        # Partial success still yields usable rankings for the methods that ran.
        summary = classify_job_status(all_results)
        job["method_summary"] = summary
        if summary["status"] == "failed":
            job["status"] = "failed"
            job["progress"] = 1.0
            job["message"] = "All methods failed; no results to report."
            return

        # --- Step 7: Compute no-GT scores if needed ---
        if not has_gt and method_labels:
            job["message"] = "Computing no-GT scores..."
            job["progress"] = 0.88

            weights = no_gt_weights or DEFAULT_WEIGHTS
            coords = np.array(adata_for_est.obsm["spatial"])
            X_pca = adata_for_est.obsm["X_pca"]

            # The consensus depends only on the full set of method labels and
            # n_clusters, so compute it once here rather than re-running the
            # spectral clustering inside compute_nogt_score for every method.
            try:
                consensus_labels = consensus_clustering(method_labels, n_clusters)
            except Exception as e:
                print(f"Consensus clustering failed, CAS will fall back per-method: {e}")
                consensus_labels = None

            nogt_results = []
            for method in method_labels:
                labels = method_labels[method]
                seed_labels = method_seed_labels.get(method, [labels])

                nogt = compute_nogt_score(
                    labels=labels,
                    label_runs=seed_labels,
                    coords=coords,
                    X_pca=X_pca,
                    all_method_labels=method_labels,
                    n_clusters=n_clusters,
                    weights=weights,
                    consensus_labels=consensus_labels,
                )

                # Update results with no-GT scores
                mask = results_df["method"] == method
                for col in ["nogt_score", "scs", "css", "ess", "cas"]:
                    results_df.loc[mask, col] = nogt[col]

                nogt_results.append({"method": method, **nogt})

        # --- Step 8: Record to meta-learning DB ---
        job["message"] = "Recording to meta-learning database..."
        job["progress"] = 0.90

        score_type = "ari" if has_gt else "nogt"
        score_col = "ari" if has_gt else "nogt_score"

        for _, row in results_df.iterrows():
            if pd.isna(row.get(score_col)):
                continue
            record = {
                "timestamp": datetime.now().isoformat(),
                "user_id": job_id,
                "method_name": row["method"],
                "method_type": METHOD_CATEGORY.get(row["method"], "unknown"),
                **{k: v for k, v in features.to_dict().items()
                   if isinstance(v, (int, float, str)) or v is None},
                "score": float(row[score_col]),
                "score_type": score_type,
                "runtime": float(row.get("runtime", 0) or 0),
                "seed": int(row.get("seed", 42)),
                "n_clusters": int(row.get("n_clusters_pred", 0) or 0),
                "git_commit": "ispot-v1.0",
            }
            ml_db.record_run(record)

        # Retrain model if enough new data
        if ml_db.count_runs() % 50 == 0:
            ml_model.train(ml_db)

        # --- Step 9: Generate deliverables ---
        job["message"] = "Generating deliverables..."
        job["progress"] = 0.93

        # Save raw results
        results_df.to_csv(results_dir / "raw_results.csv", index=False)

        # Pairwise statistical comparison (PLAN 1.5.3). Meaningful only with
        # ground truth and repeated seeds, where per-seed ARI gives paired
        # samples; skipped otherwise.
        statistical_results = None
        if has_gt:
            score_map: dict[str, dict[int, float]] = {}
            for _, row in results_df.iterrows():
                if row.get("error") is not None:
                    continue
                ari = row.get("ari")
                if ari is None or pd.isna(ari):
                    continue
                score_map.setdefault(row["method"], {})[int(row.get("seed", 42))] = float(ari)
            try:
                statistical_results = build_comparison_table(score_map, metric_name="ari")
            except Exception as e:
                print(f"Statistical comparison skipped: {e}")
                statistical_results = None

        # Ranking table
        ranking_path = generate_ranking_table(
            results_df, has_ground_truth=has_gt,
            output_dir=str(results_dir),
        )

        # Figures — use preprocessed adata for HVG info
        fig_paths = generate_figures(
            results_df, adata_for_est, method_labels,
            has_ground_truth=has_gt,
            output_dir=str(results_dir),
        )

        # Viewer data — use preprocessed adata for HVG expression
        viewer_path = generate_viewer_data(
            adata_for_est, method_labels,
            has_ground_truth=has_gt,
            output_dir=str(results_dir),
        )

        # PDF report
        report_path = generate_report(
            results_df, ranking_path, fig_paths,
            has_ground_truth=has_gt,
            data_profile=features.to_dict(),
            n_clusters=n_clusters,
            output_path=str(results_dir / "benchmark_report.pdf"),
            statistical_results=statistical_results,
        )

        # --- Step 10: Complete ---
        job["status"] = summary["status"]  # "completed" or "completed_partial"
        job["progress"] = 1.0
        if summary["status"] == "completed_partial":
            failed_names = ", ".join(m["method"] for m in summary["failed_methods"])
            job["message"] = (
                f"Completed with {summary['n_failed']} failed method(s): {failed_names}"
            )
        else:
            job["message"] = "Benchmark completed successfully"
        job["completed_at"] = datetime.now().isoformat()

        # Clean up uploaded file (keep results)
        try:
            os.remove(job["file_path"])
        except Exception:
            pass

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = 1.0
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Print startup info."""
    print(f"\niSpot API server starting...")
    print(f"  Methods available: {len(AVAILABLE_METHODS)}")
    print(f"  Meta-learning runs: {ml_db.count_runs()}")
    print(f"  Platforms: {list(LOADER_REGISTRY.keys())}")
    print(f"  Job workspace: {WORKSPACE_DIR}\n")
