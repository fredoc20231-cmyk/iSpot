"""
Traceable benchmark runner. Every result carries method, seed, dataset,
timestamp, and git commit (if available).

Usage:
  from ispot.runner import run_benchmark
  results = run_benchmark(
      methods=["Leiden_PCA", "SpaGCN"],
      datasets=["DLPFC", "HER2+"],
      output_path="results/baseline_results.jsonl",
  )
"""
import json
import os
import time
import traceback
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from ispot.registry import get_runner, is_stochastic, ALL_METHODS
from ispot.loaders import load_sample, get_n_clusters, DATASET_SLIDES, qc_check


def _get_git_commit():
    """Get current git commit hash if in a git repo."""
    try:
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(repo_root)
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def run_benchmark(
    methods=None,
    datasets=None,
    seeds=None,
    output_path=None,
    resume=True,
    verbose=True,
):
    """Run benchmark for given methods on given datasets.

    Parameters
    ----------
    methods : list of str, optional
        Method names from the registry. Default: all 12 methods.
    datasets : list of str, optional
        Dataset names. Default: all 3 (DLPFC, HER2+, MOSTA).
    seeds : list of int, optional
        Seeds for stochastic methods. Default: [42].
    output_path : str, optional
        Path to write JSONL results. If None, results are returned but not saved.
    resume : bool
        If True, skip (method, dataset, slide, seed) combos already in output_path.
    verbose : bool
        Print progress messages.

    Returns
    -------
    list of dict, each with method, seed, dataset, slide_id, ari, f1, runtime, timestamp, git_commit
    """
    if methods is None:
        methods = ALL_METHODS
    if datasets is None:
        datasets = ["DLPFC", "HER2+", "MOSTA"]
    if seeds is None:
        seeds = [42]

    git_commit = _get_git_commit()
    timestamp_start = datetime.now().isoformat()

    # Resume support
    done = set()
    existing_results = []
    if resume and output_path and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_results.append(r)
                    done.add((r.get("method"), r.get("dataset"), r.get("slide_id"), r.get("seed")))
        if verbose:
            print(f"Resuming: {len(done)} results already saved")

    all_results = list(existing_results)

    for dataset in datasets:
        slides = DATASET_SLIDES.get(dataset, [])
        if verbose:
            print(f"\n{'#'*60}\n#  DATASET: {dataset} ({len(slides)} slides)\n{'#'*60}", flush=True)

        for slide_id in slides:
            n_clusters = get_n_clusters(dataset, slide_id)
            if verbose:
                print(f"\n  Slide {slide_id} (n_clusters={n_clusters})", flush=True)

            # Load data once per slide
            try:
                adata = load_sample(dataset, slide_id)
                qc = qc_check(adata, dataset, slide_id)
                if verbose:
                    print(f"    Loaded: {adata.shape[0]} spots x {adata.shape[1]} genes, "
                          f"GT clusters: {qc['n_gt_clusters']}", flush=True)
            except Exception as e:
                if verbose:
                    print(f"    LOAD ERROR: {e}", flush=True)
                for method in methods:
                    for seed in seeds:
                        key = (method, dataset, slide_id, seed)
                        if key in done:
                            continue
                        result = {
                            "method": method, "dataset": dataset, "slide_id": slide_id,
                            "seed": seed, "ari": None, "macro_f1": None,
                            "weighted_f1": None, "runtime": None,
                            "error": f"Load: {e}",
                            "timestamp": datetime.now().isoformat(),
                            "git_commit": git_commit,
                        }
                        all_results.append(result)
                        if output_path:
                            with open(output_path, "a") as f:
                                f.write(json.dumps(result) + "\n")
                continue

            for method in methods:
                method_seeds = seeds if is_stochastic(method) else [42]
                runner = get_runner(method)

                for seed in method_seeds:
                    key = (method, dataset, slide_id, seed)
                    if key in done:
                        if verbose:
                            print(f"    {method} (seed={seed}): SKIP", flush=True)
                        continue

                    if verbose:
                        print(f"    --- {method} (seed={seed}) ---", flush=True)

                    try:
                        t0 = time.time()
                        m = runner(
                            adata, n_clusters, seed=seed,
                            dataset=dataset, slide_id=slide_id,
                        )
                        result = {
                            "method": method,
                            "dataset": dataset,
                            "slide_id": slide_id,
                            "seed": seed,
                            "ari": m["ari"],
                            "macro_f1": m["macro_f1"],
                            "weighted_f1": m["weighted_f1"],
                            "runtime": m["runtime"],
                            "n_spots": m["n_spots"],
                            "n_clusters_pred": m["n_clusters_pred"],
                            "n_clusters_true": m["n_clusters_true"],
                            "error": None,
                            "timestamp": datetime.now().isoformat(),
                            "git_commit": git_commit,
                        }
                        if verbose and m.get("ari") is not None:
                            print(f"      ARI={m['ari']:.4f}, F1={m['macro_f1']:.4f}, "
                                  f"Runtime={m['runtime']:.2f}s", flush=True)
                    except Exception as e:
                        if verbose:
                            print(f"      ERROR: {e}", flush=True)
                            traceback.print_exc()
                        result = {
                            "method": method,
                            "dataset": dataset,
                            "slide_id": slide_id,
                            "seed": seed,
                            "ari": None,
                            "macro_f1": None,
                            "weighted_f1": None,
                            "runtime": None,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                            "git_commit": git_commit,
                        }

                    all_results.append(result)
                    done.add(key)

                    if output_path:
                        with open(output_path, "a") as f:
                            f.write(json.dumps(result) + "\n")

    return all_results


def jsonl_to_csv(jsonl_path, csv_path):
    """Convert JSONL results to CSV."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return df
