# Fix: 2D Spatial Domain Viewer — Dots Scattered Off-Tissue

## The bug

The interactive viewer plots exactly the spots it's given
(`ispot/frontend/app.js`'s `drawViewer()` — confirmed by reading it, it does
no filtering of its own, just renders `data.spots` as-is). So the "dots all
over the slide instead of just on the tissue" report meant the backend was
handing the viewer spots that were never actually on tissue in the first
place.

Root cause, found in `ispot/multiplatform_loaders.py`'s `BaseLoader.load()`:
```python
# Set in_tissue (default: all in tissue)
if "in_tissue" not in adata.obs.columns:
    adata.obs["in_tissue"] = 1
```
This column was read and preserved from the uploaded file, but **never
actually used to filter anything**. Real 10x Visium data comes as a
rectangular grid covering the whole capture area — only some of those spots
sit under actual tissue; the rest are background. A raw/unfiltered export
(`raw_feature_bc_matrix.h5` rather than `filtered_feature_bc_matrix.h5`, or
any `.h5ad` that kept background spots with a real `in_tissue` column) would
carry every background spot straight through clustering and into the
viewer — producing exactly the symptom described: dots scattered across the
full rectangular array shape rather than confined to the tissue's actual
footprint, and clustering algorithms wasting effort partitioning pure noise
alongside the real signal.

## The fix

`BaseLoader.load()` now actually filters to `in_tissue == 1` when the column
contains real information (i.e., isn't just the all-ones default):

```python
if "in_tissue" not in adata.obs.columns:
    adata.obs["in_tissue"] = 1
else:
    in_tissue_vals = pd.to_numeric(adata.obs["in_tissue"], errors="coerce").fillna(1).astype(int)
    n_before = adata.shape[0]
    n_off_tissue = int((in_tissue_vals == 0).sum())
    if n_off_tissue > 0:
        adata = adata[in_tissue_vals == 1].copy()
        adata.uns["n_spots_excluded_off_tissue"] = n_off_tissue
        adata.uns["n_spots_before_tissue_filter"] = int(n_before)
```

This runs once, centrally, in `BaseLoader.load()` — the shared code path
every platform loader (`VisiumLoader`, `SlideSeqLoader`, etc.) already
passes through — so it's not something each loader needs to remember to do
individually, and it applies before clustering ever sees the data, not just
before the viewer renders it.

The exclusion count is also now surfaced in the job's data profile
(`ispot/api.py`), so if a real dataset drops a large fraction of spots to
this filter, that's visible to the user instead of a silent change in dot
count between upload and viewer.

## Verification

Built a realistic synthetic scenario matching what a raw Visium export looks
like: a full 50×50 rectangular capture grid (2,500 spots), with only the
spots inside a circular "tissue" region (697 of them) marked `in_tissue=1`
and the remaining 1,803 background array spots marked `in_tissue=0`:

```
Before loader: 2500 spots total, 697 actually in tissue, 1803 background
After loader: 697 spots remain
n_spots_excluded_off_tissue: 1803
All remaining spots have in_tissue==1: True
```

The loader now correctly drops every background spot and keeps only the
real tissue footprint — confirmed the app still imports and initializes
cleanly after this change (12 methods, 7 platforms, meta-learning DB intact).

## What this does and doesn't fix

This fixes the specific failure mode where the uploaded data itself already
marks which spots are real tissue (via `in_tissue`), which is the standard
way 10x Visium and similar platforms represent this. It does **not** add
tissue-boundary *detection* for data that has no `in_tissue` information at
all (e.g., a bare coordinate + expression matrix with no tissue metadata) —
in that case the default (`in_tissue=1` for everything) still applies, since
there's no signal in the data to determine a boundary from. If you're
seeing scattered dots on uploads that genuinely lack any tissue-boundary
column, that would be a separate, harder problem (inferring tissue extent
from spatial density/expression alone) — let me know if that's the actual
situation you're hitting and I'll look at that path specifically.
