# iSpot — Master Status Tracker

**How to use this file (for AI Studio / whoever is executing):**
This is a living checklist, not a one-time report. Keep it in the repo root
(`STATUS_TRACKER.md`) and update it in the same commit as any fix it
tracks — check the box, fill in the commit hash, add a one-line note if the
fix diverged from the original instruction. When resuming work, **read this
file first** to see what's already done before re-inspecting from scratch.
When new issues are found during implementation (they will be — fixing A1
may surface something in A2's assumptions, etc.), add them under
"Newly discovered" at the bottom rather than silently fixing them
un-tracked, so nothing gets lost across sessions.

There's no automated re-inspection running against this repo — this file
only gets refreshed when someone (you, or a session with Claude) re-runs an
inspection pass and updates it by hand. Re-run one when a task group is
fully checked off, before moving to the next.

Last inspected/updated: 2026-08-10 (real-data verification pass, using an
actual GEO sample, GSM6433586). Repo state: 8 commits through `5c1caae`.
All of Groups A-D are now implemented. Most items are independently
verified against either synthetic ground-truth scenarios or real public
data; a few (marked below) are only code-reviewed/smoke-tested and still
need a dedicated verification pass. Remaining work: Group E (features) and
Group F (publication validation studies) — both unblocked by A0's fix and
ready to start.

---

## TASK GROUP A — App must be able to start (blocks everything else)

- [x] **A0** — Fix meta-learning training to split by `score_type` (`ari` vs
      `nogt`) instead of pooling both into one regression target per method
      (`meta_learning.py`, `MetaLearningModel.train()`). This silently
      trains on a target variable that switches between two different,
      not-yet-validated-as-equivalent metrics — it blocks Gap 2a/2b
      publication validation from being scientifically meaningful, so it
      must land before those studies are run, not alongside them.
      Commit: 92e85e5 | Verified via synthetic score-shift sanity check: [ ]
- [x] **A1** — Remove hardcoded `/workspace` paths (`api.py` L38/64, `loaders.py` L30-31/62/233, `runner.py` L32)
      Commit: 92e85e5 | Verified: imported and ran the full app from a location with no /workspace directory anywhere on the machine — see CHANGES_BETA.md: [x]
- [x] **A2** — Fix meta-learning seed path (`/mnt/results/...` → `data/unified_results.csv`)
      Commit: 92e85e5 | Verified: fresh DB seeded with 200 rows from data/unified_results.csv on server startup: [x]
- [x] **A3** — Implement or remove README-documented dead endpoints (`/api/plugins/register`, `/api/meta-learning/recommend`)
      Commit: 92e85e5 | Verified: `curl /api/plugins` returns 200 with plugin list: [x]

## TASK GROUP B — Security (before any public/hosted deployment)

- [x] **B1** — Fix path traversal in `/api/jobs/{id}/download/{filename}`
      Commit: 92e85e5 | Verified directly against a real job: legit file downloads, `../../../../etc/passwd` rejected with 400: [x]
- [x] **B2** — Restrict CORS `allow_origins` from wildcard
      Commit: 92e85e5 | Confirmed ALLOWED_ORIGINS defaults correctly on server start; a live disallowed-origin request wasn't separately tested: [ ]
- [x] **B3** — Add multi-tenant plugin sandbox guard
      Commit: 92e85e5 | Guard code in place; ISPOT_MULTI_TENANT=1 failure path not separately re-tested after final edits: [ ]

## TASK GROUP C — Scientific correctness

- [x] **C1** — Fix CAS memory blowup (`nogt_scoring.py`, `_consensus_batched` dense `.toarray()`)
      Commit: 92e85e5 | Verified: 60,000-spot synthetic run completed in 26s at ~2.2GB peak (not ~1GB as originally estimated, but nowhere near the ~20GB dense-matrix failure mode): [x]
- [x] **C2** — Platform auto-detection covers all 7 platforms, fallback made visible
      Commit: 5c1caae | Verified: synthetic 3000-point irregular data -> "Slide-seqV2" (was "Visium"); synthetic 90000-point grid -> "Stereo-seq" (was "Visium"): [x]
- [x] **C3** — Platform-aware `min_genes` filtering in `preprocess()`; QC retention surfaced to user
      Commit: 92e85e5 | PLATFORM_MIN_GENES logic unit-testable but not re-run against real Xenium-shaped data this session — flagged for the next verification pass: [ ]
- [x] **C4** — Fix meta-learning retrain trigger (`% 50 == 0` → threshold-crossing check)
      Commit: 92e85e5 | Logic verified by code inspection (last_trained_count now tracked and compared); not separately run through a live retrain-triggering scenario: [ ]

## TASK GROUP D — Robustness

- [x] **D1** — Surface partial method failures in job status/frontend
      Commit: 92e85e5 | Code path added; not separately tested by forcing a live method failure this session: [ ]
- [x] **D2** — Orphaned-upload cleanup (7-day expiry, as already documented in README)
      Commit: 92e85e5 | Background task added and confirmed to start via the live server test; the 7-day-old-directory removal itself wasn't separately time-machine-tested: [ ]
- [x] **D3** — Pin `pandas`/`requests` in `requirements.txt`
      Commit: 5c1caae | Pinned to pandas==2.2.3, requests==2.32.3 (conservative, era-consistent with other pins). Not re-verified via a full clean-venv install this session -- do that before deploying: [ ]
- [x] **D4** — Fix misleading "Best speed/accuracy" label in ranking table (`deliverables.py`, `generate_ranking_table`) — currently pure argmin(runtime) with no accuracy constraint, meaning the weakest-accuracy method can get labeled a good tradeoff purely for being fast.
      Commit: 92e85e5 | Verified: re-ran the Leiden_PCA scenario — no longer mislabeled, STAGATE (closest-accuracy-and-fastest) wins the label instead: [x]
- [x] **D5** — Fix spatial viewer showing dots scattered off-tissue (`multiplatform_loaders.py`, `BaseLoader.load()`) — the `in_tissue` column was read and preserved from uploads but never actually used to filter background/off-tissue spots, so raw/unfiltered Visium-style uploads carried every background spot through clustering and into the viewer.
      Commit: d96ace6 | Verified: synthetic 2500-spot rectangular grid with a 697-spot circular tissue region — loader correctly drops all 1803 background spots: [x]
- [x] **D6** — Fix per-method spot-count misalignment: 3 of 12 methods (leiden_pca, stagate, spagcn) filtered spots internally while the other 9 didn't, risking different-length label arrays per method on the same job.
      Commit: 5c1caae | QC filter now applied once centrally in api.py before any method runs; hard safety-net check added. Verified via full-app smoke test; not yet re-run through a live job with a real filtering-triggering dataset: [ ]
- [x] **D7** — Add missing `scikit-misc` dependency — every real benchmark run was crashing with `ModuleNotFoundError` inside `preprocess()`'s `highly_variable_genes(flavor="seurat_v3")` call, since this package was never in requirements.txt at all.
      Commit: 5c1caae | Verified: `preprocess()` runs successfully end-to-end on real GSM6433586 data after installing scikit-misc; requirements.txt updated to declare it: [x]
- [x] **D8** — Real image-based tissue boundary detection (`ispot/tissue_segmentation.py`, new module) — the `in_tissue` column alone can only reflect "inside the capture frame," not the real irregular tissue shape; Otsu thresholding + connected-component analysis now derives the actual boundary from histology image pixels.
      Commit: 6d7396d | Verified: IoU 1.000 against a synthetic irregular (star-shaped, non-oval) tissue mask across 10 randomized shapes; 100% precision/100% recall end-to-end through the real loader: [x]
- [x] **D9** — Fix `VisiumLoader` hard-crashing on real-world Space Ranger exports missing `tissue_lowres_image.png` (common on public datasets like GEO) — `scanpy.read_visium()` requires both hires and lowres images or refuses to load anything.
      Commit: 6e1f644 | Verified against a real GEO sample (GSM6433586_092B, hires-only): loads successfully, 1356 spots, tissue detection visually confirmed against the actual H&E image: [x]

## TASK GROUP E — Feature roadmap (only after A–D are fully checked off)

Phase 1 (harden) items A1–D3 above supersede the original roadmap's Phase 1
numbering — once they're done, resume from the roadmap document's Phase 2:

- [ ] **E1** — Job queue: BackgroundTasks → Celery/Redis
- [ ] **E2** — Isolate R-based methods into their own worker container
- [ ] **E3** — Object storage for job artifacts (S3/MinIO), not local disk
- [ ] **E4** — WebSocket/SSE job progress, replacing 2s polling
- [ ] **E5** — Parallel method execution within a job
- [ ] **E6** — GPU dispatch for GNN-based methods
- [ ] **E7** — Meta-learning: Ridge → gradient-boosted trees + calibration
- [ ] **E8** — Multi-tenancy: real auth, per-user job isolation (schema already has unused `user_id`)
- [ ] **E9** — Consensus/ensemble clustering as a user-facing deliverable
- [ ] **E10** — Per-spot uncertainty quantification in the viewer
- [ ] **E11** — Reference-atlas validation (biological plausibility check)
- [ ] **E12** — Multi-sample/multi-slice batch integration
- [ ] **E13** — Visible plugin marketplace UI (registry already exists, no UI)
- [ ] **E14** — LLM-authored narrative interpretation in PDF reports

(Full detail on each is in the roadmap doc already provided — this list is
just the tracking surface. Don't re-derive these; reference the roadmap.)

## TASK GROUP F — Publication readiness (parallel track, not blocking F required before G)

**Note:** F1 and F2 below are only meaningful once **A0** (score_type fix)
has landed — running these validations against the current, confounded
training procedure would produce results that don't reflect the corrected
model's actual behavior.

- [ ] **F1** — No-GT composite score validation against real ARI on DLPFC/HER2+ (Pearson/Spearman r with CI)
      Result: _______ | Correlation strong enough to publish as-is: [ ] yes [ ] needs reweighting
- [ ] **F2** — Meta-learning recommender validated against baselines (always-best-method, run-all) on held-out data
      Result: _______ | Beats baselines: [ ] yes [ ] no — needs E7 first
- [ ] **F3** — Real (not synthetic) Xenium/Stereo-seq/MERFISH end-to-end validation, post Group C fixes
- [ ] **F4** — Related-work section drafted against i-stLearn, Genome Biology 2024, iMeta 2025, NAR 2025 benchmarks
- [ ] **F5** — Hosted instance stood up with at least a few external (non-author) users
- [ ] **F6** — CI test suite covering `metrics.py`/`nogt_scoring.py` scoring correctness
- [ ] **F7** — README quickstart actually works end-to-end on a clean machine (depends on A1/A2)

---

## Newly discovered (add here, don't silently fix)

- **2026-08-09, Beta implementation pass** — while implementing A0's fix,
  confirmed the score_type confusion wasn't isolated to `train()`/`predict()`
  alone: `pilot_then_full()` and the `/api/meta-learning/recommend` endpoint
  both needed a `has_ground_truth`/`score_type` parameter threaded through
  too, since they call `.predict()` internally. All fixed as part of A0 —
  see `CHANGES_BETA.md` in the repo for full details and verification
  evidence for every fix landed this session.
- **2026-08-09, Round 5 (publication-critical-path pass)** —
  `meta_learning.py`'s `MetaLearningModel.train()` pools `ari`-scored and
  `nogt`-scored historical runs into one regression target per method,
  ignoring the `score_type` column that exists specifically to distinguish
  them. This corrupts the recommendation engine today and would invalidate
  any held-out validation study run against it for a publication. Logged as
  **A0** above (elevated into Group A since it's a blocking prerequisite,
  not an independent robustness item). Full reasoning and fix code in
  `PATH_TO_NATURE_CRITICAL_FIX.md`.
- **2026-08-09, Round 4** — `deliverables.py`'s ranking-table "Best
  speed/accuracy" recommendation is pure `argmin(runtime)` with no accuracy
  term at all — confirmed with a synthetic scenario where a method at half
  the top accuracy (ARI 0.31 vs. 0.62) still got the label purely for being
  fastest. Logged as **D4** above. Full writeup in
  `CODE_AUDIT_ROUND4.md` if it still exists in your files; the fix and
  verification steps are folded into D4's description above so this file
  alone is sufficient going forward.

---

## Suggested cadence

1. Work Group A fully, verify, check every box, commit.
2. Work Group B fully — required before anything is exposed publicly.
3. Work Group C — required before trusting any non-Visium benchmark result.
4. Group D can be batched into the same PRs as A/C since it touches the same files.
5. Only after A–D are fully checked: pick up Group E (features) and Group F
   (publication validation) in parallel — they don't block each other.
6. Bring this file back to a Claude session periodically (e.g., after each
   group is checked off) for a fresh inspection pass — new issues surface
   once earlier ones are fixed and previously-unreachable code paths get
   exercised for the first time.
