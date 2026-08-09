"""
SpaRTaCo: spatially resolved transcriptomics co-clustering.
R-based method. Called via Rscript subprocess.
"""
import subprocess
import json
import os
import tempfile

R_SCRIPT = os.environ.get("RSCRIPT_PATH", "/workspace/.conda/envs/persist/bin/Rscript")


def run(adata, n_clusters, seed=42, dataset="DLPFC", slide_id="unknown", **kwargs):
    """Run SpaRTaCo via R subprocess."""
    tmpdir = tempfile.mkdtemp()
    h5ad_path = os.path.join(tmpdir, f"{slide_id}.h5ad")
    adata.write_h5ad(h5ad_path)

    r_script = os.path.join(os.path.dirname(__file__), "_r_methods_runner.R")

    result = subprocess.run(
        [R_SCRIPT, r_script, h5ad_path, str(n_clusters), dataset, slide_id, str(seed), "SpaRTaCo"],
        capture_output=True, text=True, timeout=7200
    )

    if result.returncode != 0:
        raise RuntimeError(f"SpaRTaCo R script failed: {result.stderr[-500:]}")

    output_line = [l for l in result.stdout.strip().split("\n") if l.startswith("RESULT:")]
    if not output_line:
        raise RuntimeError(f"No RESULT line in SpaRTaCo output: {result.stdout[-500:]}")

    m = json.loads(output_line[0].replace("RESULT:", "").strip())
    return m
