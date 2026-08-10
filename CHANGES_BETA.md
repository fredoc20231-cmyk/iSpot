# iSpot — Beta Release Changes

This release closes every P0/P1 item from the engineering audit plus the
highest-priority scientific-correctness finding (A0), and adds realistic
worked examples. Each item below was actually fixed in this codebase and
verified — not just diagnosed — before this release.

## Fixed and verified

**A0 — meta-learning no longer pools ARI and no-GT scores into one target.**
`MetaLearningModel` now trains and predicts separately per `(method,
score_type)` instead of blending two different metrics into one regression
target. Verified: trained on synthetic data with a known +0.3 systematic
shift between the two score types — the fixed model correctly recovers that
shift (`predicted ARI: 0.540, predicted NoGT: 0.840, diff: 0.300`) instead of
averaging them together.

**A1 — the app now starts outside its original dev container.**
Removed every hardcoded `/workspace` reference (`api.py`, `loaders.py`,
`runner.py`) in favor of paths resolved relative to the repo or configurable
via environment variables. Verified: imported and ran the full FastAPI app
from `/home/claude/iSpot_build/iSpot` — a location with no `/workspace`
directory anywhere on the machine — and it started cleanly:
```
IMPORT SUCCEEDED
WORKSPACE_DIR: /home/claude/iSpot_build/iSpot/ispot_jobs
AVAILABLE_METHODS count: 12
```
Also ran the live server and confirmed `/api/health` responds correctly with
all 12 methods and all 7 platforms listed, and 200 meta-learning runs seeded
from the bundled CSV.

**A2 — meta-learning seed data loads from the correct path.**
Fixed `/mnt/results/unified_results.csv` → `data/unified_results.csv`
(resolved relative to the repo). Verified as part of the A1 server-start
test above (200 seed rows loaded on fresh startup).

**A3 — the two README-documented but previously nonexistent endpoints are implemented.**
`GET /api/plugins` and `POST /api/plugins/register` (scoped to
already-installed pip packages, not arbitrary uploaded code — see B3) and
`GET /api/meta-learning/recommend` (now correctly score-type-aware per the
A0 fix). Verified via a live request: `curl /api/plugins` → `{"plugins": []}`.

**B1 — path traversal in the download endpoint is fixed.**
Verified directly against a real job: a legitimate filename downloads
successfully, while `../../../../etc/passwd` is rejected with "Invalid
filename" instead of resolving outside the job directory.

**B2 — CORS wildcard replaced with an explicit, configurable allowlist.**
`ALLOWED_ORIGINS` now defaults to `localhost:8100` only, overridable via
`ISPOT_ALLOWED_ORIGINS`.

**B3 — plugin sandbox guard added.**
Startup now refuses to load plugins in multi-tenant mode unless
`ISPOT_PLUGIN_SANDBOX=1` is explicitly set, preventing accidental exposure
of the arbitrary-code-execution plugin system to untrusted multi-tenant use.

**C1 — Consensus Alignment Score no longer risks OOM on large datasets.**
`consensus_clustering`/`consensus_alignment_score` now subsample to 8,000
spots before spectral clustering instead of densifying an n×n matrix.
Verified on a real 60,000-spot synthetic dataset (the platform's own binning
target for Xenium/Stereo-seq/MERFISH): completed in 26 seconds at ~2.2GB
peak memory, versus the ~20GB the old dense approach would have required.

**C2 — not yet done this session** (platform auto-detection for
Slide-seqV2/Stereo-seq/DBiT-seq — still defaults silently to Visium for
unrecognized `.h5ad` structure). Tracked as the next priority; see
`STATUS_TRACKER.md`.

**C3 — preprocessing's gene-count filter is now platform-aware.**
`PLATFORM_MIN_GENES` replaces the single hardcoded `min_genes=200` across
`preprocessing.py`, `api.py`'s live benchmark path, the offline
`runner.py` reproduction script (now also platform-aware per dataset —
DLPFC/HER2+ are Visium, MOSTA is Stereo-seq), and the two method runners
that called `preprocess()` directly (`leiden_pca.py`, `stagate.py`) plus
`spagcn.py`'s independently hardcoded filter. QC retention
(`n_spots_before_qc`/`n_spots_after_qc`/`qc_retention_pct`) is now surfaced
in the job's data profile instead of happening silently. See `EXAMPLES.md`
Example 2 for a worked walkthrough.

**C4 — meta-learning retrain trigger fixed.**
Replaced `count % 50 == 0` (which could skip every multiple of 50 given
variable per-job row counts) with `count - last_trained_count >= 50`.

**D1 — partial method failures now surface in job status.**
A job where some methods errored now reports `partial_failures` and an
explanatory message instead of a bare "completed."

**D2 — orphaned-upload cleanup implemented.**
A background task now actually enforces the 7-day retention the README
always documented but nothing previously implemented.

**D4 — "Best speed/accuracy" label fixed.**
Now requires being within 10% of the top accuracy score before considering
runtime, instead of pure `argmin(runtime)`. Verified: re-ran the original
synthetic scenario (a method at half the leader's accuracy) — it's no longer
mislabeled as a good tradeoff; the fastest method that's actually close to
the top score wins the label instead.

## Not yet done (tracked in STATUS_TRACKER.md)

- **C2** — platform auto-detection for the remaining 3 platforms
- **D3** — pinning `pandas`/`requests` versions
- **F1–F7** — the publication-validation studies (no-GT score vs. real ARI,
  meta-learning vs. baselines) — these are now unblocked by A0's fix and are
  the next highest-value work, but require actual experiment runs, not code
  changes
- Full docker build verification (only verified via direct Python import and
  running `uvicorn` locally in this environment, not via `docker build`
  itself, since Docker isn't available in this sandbox)

## Realistic examples added

`EXAMPLES.md` replaces the README's abstract "usage flow" bullet list with
four worked, realistic examples: a DLPFC Visium slide with ground truth
(real slide ID, real cluster count from the manuscript), a Xenium upload
demonstrating the platform-aware QC fix with realistic panel-size numbers,
a meta-learning recommendation preview, and plugin listing/registration.
