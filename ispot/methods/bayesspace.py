"""
BayesSpace: Bayesian spatial domain detection via t-distributed clustering.

R-based method. Called via Rscript subprocess.
BayesSpace now installs successfully (viridis dependency was the blocker, fixed).
"""
import subprocess
import json
import os
import tempfile

R_SCRIPT = os.environ.get("RSCRIPT_PATH", "/workspace/.conda/envs/persist/bin/Rscript")


def run(adata, n_clusters, seed=42, dataset="DLPFC", slide_id="unknown", **kwargs):
    """Run BayesSpace via R subprocess.

    Saves adata to temp h5ad, calls R script, reads back results.
    """
    import anndata as ad

    # Save to temp file
    tmpdir = tempfile.mkdtemp()
    h5ad_path = os.path.join(tmpdir, f"{slide_id}.h5ad")
    adata.write_h5ad(h5ad_path)

    # R script path
    r_script = os.path.join(os.path.dirname(__file__), "_bayesspace_runner.R")

    result = subprocess.run(
        [R_SCRIPT, r_script, h5ad_path, str(n_clusters), dataset, slide_id, str(seed)],
        capture_output=True, text=True, timeout=3600
    )

    if result.returncode != 0:
        raise RuntimeError(f"BayesSpace R script failed: {result.stderr[-500:]}")

    # Parse JSON output from stdout
    output_line = [l for l in result.stdout.strip().split("\n") if l.startswith("RESULT:")]
    if not output_line:
        raise RuntimeError(f"No RESULT line in BayesSpace output: {result.stdout[-500:]}")

    m = json.loads(output_line[0].replace("RESULT:", "").strip())
    return m
