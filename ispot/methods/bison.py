"""
BISON: bi-clustering of spatial omics data via MCMC.
R-based method. Called via Rscript subprocess.

BISON C++ requires gnu++14 flag (not c++14).
"""
import subprocess
import json
import os
import tempfile

R_SCRIPT = os.environ.get("RSCRIPT_PATH", "/workspace/.conda/envs/persist/bin/Rscript")


def run(adata, n_clusters, seed=42, dataset="DLPFC", slide_id="unknown", **kwargs):
    """Run BISON via R subprocess."""
    import anndata as ad

    tmpdir = tempfile.mkdtemp()
    h5ad_path = os.path.join(tmpdir, f"{slide_id}.h5ad")
    adata.write_h5ad(h5ad_path)

    r_script = os.path.join(os.path.dirname(__file__), "_r_methods_runner.R")

    result = subprocess.run(
        [R_SCRIPT, r_script, h5ad_path, str(n_clusters), dataset, slide_id, str(seed), "BISON"],
        capture_output=True, text=True, timeout=7200
    )

    if result.returncode != 0:
        raise RuntimeError(f"BISON R script failed: {result.stderr[-500:]}")

    output_line = [l for l in result.stdout.strip().split("\n") if l.startswith("RESULT:")]
    if not output_line:
        raise RuntimeError(f"No RESULT line in BISON output: {result.stdout[-500:]}")

    m = json.loads(output_line[0].replace("RESULT:", "").strip())
    return m
