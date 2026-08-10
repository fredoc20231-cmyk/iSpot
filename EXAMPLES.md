# iSpot — Worked Examples

These use realistic values from the datasets the platform was actually
validated against (DLPFC, HER2+, MOSTA — see the benchmark manuscript), plus
a Xenium example to show the platform-aware QC behavior fixed in this beta.

## Example 1 — Visium, with ground truth (DLPFC slide 151507)

DLPFC slide `151507` is one of 12 Visium slides in the manuscript's
benchmark, with 7 annotated cortical layers and roughly 4,000 in-tissue
spots — typical for a Visium capture area.

```bash
# 1. Upload — sample_id and ground-truth column are known for this slide
curl -X POST http://localhost:8100/api/upload \
  -F "file=@151507.h5ad" \
  -F "platform=Visium" \
  -F "sample_id=151507" \
  -F "ground_truth_col=layer_guess"

# → {"job_id": "a1b2c3...", "platform": "Visium", "n_spots": 4226, "n_genes": 33538}

# 2. Benchmark — ground truth is present, so ranking will be by real ARI,
#    not the no-GT proxy score. n_clusters is already known (7 layers).
curl -X POST http://localhost:8100/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{
        "job_id": "a1b2c3...",
        "n_clusters": 7,
        "seeds": [42, 43, 44],
        "use_meta_learning": true
      }'

# 3. Poll until status is "completed"
curl http://localhost:8100/api/jobs/a1b2c3...

# 4. Results — ranking_table.csv will have an ARI column, not NoGTScore
curl http://localhost:8100/api/jobs/a1b2c3.../results
```

## Example 2 — Xenium, no ground truth, showing the QC-retention fix

This is the scenario the platform-aware preprocessing fix in this beta
directly addresses. A Xenium sample with a 280-gene targeted panel and
typical single-cell detection depth (commonly well under 200 genes/cell,
which is expected for imaging-based panels, not a quality problem) would,
before this fix, silently lose most cells to a Visium-tuned `min_genes=200`
filter with no visible warning.

```bash
curl -X POST http://localhost:8100/api/upload \
  -F "file=@xenium_sample.h5ad" \
  -F "platform=Xenium"

# → {"job_id": "d4e5f6...", "platform": "Xenium", "n_spots": 48213, "n_genes": 280}

curl -X POST http://localhost:8100/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{"job_id": "d4e5f6...", "use_meta_learning": true}'

curl http://localhost:8100/api/jobs/d4e5f6.../results
```

The job's data profile now includes a `data_profile_qc` block:
```json
{
  "n_spots_before_qc": 48213,
  "n_spots_after_qc": 46890,
  "qc_retention_pct": 97.3,
  "min_genes_used": 10
}
```
Before this fix, the same upload would have used `min_genes=200` (the
Visium default) regardless of platform — on a 280-gene panel with typical
single-cell detection, that could have dropped the majority of cells with
no visible signal in the response at all. Compare `qc_retention_pct` against
what you'd have gotten from the old fixed threshold if you want to see the
difference directly:
```bash
curl -X POST http://localhost:8100/api/upload -F "file=@xenium_sample.h5ad" -F "platform=Visium"
# forcing the wrong platform on purpose, for comparison only — don't do this in real use
```

## Example 3 — Exploring a recommendation before uploading anything

Useful for a quick sanity check of what the meta-learning engine currently
believes about a platform/scale combination, without running a full job:

```bash
curl "http://localhost:8100/api/meta-learning/recommend?platform=Stereo-seq&n_spots=5000&n_genes=20000&has_ground_truth=false"
```
```json
{
  "predicted_ranking": [["GraphST", 0.71], ["STAGATE", 0.68], ...],
  "confidence": "medium",
  "note": "Approximate — based on platform-typical defaults, not a real uploaded dataset's measured profile."
}
```
Note this queries the `nogt`-trained model specifically (since
`has_ground_truth=false`) — a fix in this beta ensures ARI-trained and
NoGTScore-trained predictions are never blended together (see
`CHANGES_BETA.md`).

## Example 4 — Listing available methods, including any community plugins

```bash
curl http://localhost:8100/api/plugins
```
```json
{"plugins": []}
```
Empty until a plugin package is registered:
```bash
# The package must already be pip-installed on the server — registration
# does not accept or execute arbitrary uploaded code (see CHANGES_BETA.md, B3).
pip install my-ispot-clustering-plugin
curl -X POST http://localhost:8100/api/plugins/register \
  -H "Content-Type: application/json" \
  -d '{"package_name": "my_ispot_clustering_plugin"}'
```

## A note on n_genes/n_spots realism

If you're testing with synthetic data rather than the real datasets above,
use platform-realistic magnitudes rather than arbitrary round numbers —
several of the bugs fixed in this beta (the min_genes filter, the CAS memory
blowup) only manifest at realistic scale:
- **Visium**: ~2,000–5,000 spots/slide, ~15,000–20,000 genes detected (full transcriptome), aggregate signal per spot (multiple cells)
- **Xenium/CosMx/MERFISH**: single-cell resolution, tens of thousands to ~500,000 cells, targeted panels of ~100–5,000 genes
- **Stereo-seq**: can exceed 100,000 bins per slide at high resolution
