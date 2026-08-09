# ST-Bench: A Spatial Transcriptomics Clustering Benchmark Platform

## Summary

A hosted SaaS platform where users upload spatial transcriptomics data, the platform benchmarks multiple clustering methods, and returns a ranked recommendation with interactive visualizations and a written report. The platform handles both annotated data (ground truth available) and unannotated data (the common case), uses meta-learning to recommend methods based on accumulated experience across all runs, and supports a plugin system for community-contributed methods.

Built on the existing `stbench` package (12 method runners, standardized preprocessing, Hungarian-matched metrics, resume-capable runner with git-commit traceability).

---

## 1. Core Algorithm

### 1.1 Data Ingestion & Normalization

**Input:** User uploads one or more spatial transcriptomics samples. Supported formats (Phase 3 for non-Visium):

| Platform | Input Format | Loader |
|----------|-------------|--------|
| 10x Visium | `.h5ad`, Space Ranger `filtered_feature_bc_matrix.h5` | `VisiumLoader` |
| Slide-seqV2 | `.h5ad` with bead coordinates | `SlideSeqLoader` |
| MERFISH | `.csv`/`.h5ad` with molecule coordinates | `MERFISHLoader` |
| CosMx | NanoString output `.csv` | `CosMxLoader` |
| Xenium | 10x Xenium output `.h5` | `XeniumLoader` |
| Stereo-seq | `.h5ad` with sub-cellular coordinates | `StereoSeqLoader` |
| DBiT-seq | `.h5ad` | `DBiTLoader` |

**Normalization to internal schema:** Every loader produces an AnnData with:
- `.X` = raw counts (sparse or dense)
- `.obsm['spatial']` = 2D coordinates (array, n_spots x 2)
- `.obs['ground_truth']` = labels if user provides annotations, else `None`
- `.obs['has_ground_truth']` = bool mask
- `.obs['sample_id']` = sample identifier
- `.obs['in_tissue']` = 1 for in-tissue spots (if applicable)
- `.uns['platform']` = platform name
- `.uns['spatial_layout']` = "hexagonal" | "square" | "random" (auto-detected)

**Auto-detection of platform:** Inspect file structure, coordinate spacing regularity, and metadata to determine platform. User confirms or overrides.

### 1.2 Feature Extraction (Data Profiling)

Before any method runs, extract a feature vector characterizing the dataset. This feeds the meta-learning engine and determines compute dispatch.

```
DataFeatureVector = {
    n_spots: int,                    # total spots
    n_genes: int,                    # total genes after filtering
    sparsity: float,                 # fraction of zeros in .X
    median_genes_per_spot: float,    # median complexity
    median_counts_per_spot: float,   # median library size
    spatial_layout: str,             # hexagonal | square | random
    spot_diameter_um: float,         # spatial resolution
    has_histology: bool,             # histology image available
    platform: str,                   # Visium, MERFISH, etc.
    tissue_type: str | None,         # user-provided or None
    n_expected_clusters: int | None, # user-provided or None
    spatial_extent: float,           # max pairwise distance (tissue size)
    coordinate_density: float,       # spots per mm^2
}
```

**Spatial layout detection:** Compute pairwise distances between nearest-neighbor spots. If distances cluster tightly around two values (hexagonal grid), one value (square grid), or are broadly distributed (random), classify accordingly.

### 1.3 Standardized Preprocessing

Every dataset goes through the identical pipeline (already implemented in `stbench.preprocessing`):

1. Filter cells (min_genes=200) and genes (min_cells=3)
2. Normalize total to 1e4
3. Log1p transform
4. Highly variable gene selection (seurat_v3, n_top_genes=3000)
5. PCA (50 components on HVGs)
6. Neighbors graph (15 neighbors on PCA)
7. Original counts preserved in `.layers['counts']`

**Platform-specific adjustments:**
- High-resolution platforms (MERFISH, Xenium, Stereo-seq): spots may be single cells, not capture spots. Adjust min_genes threshold based on platform. Bin sub-cellular coordinates into pseudo-spots if spot count > 50,000.
- Slide-seq: beads are irregularly spaced; use kNN spatial graph instead of grid-based adjacency.

### 1.4 Method Execution

#### 1.4.1 Method Interface

Every method (built-in or plugin) implements:

```python
def run(adata: AnnData, n_clusters: int, seed: int = 42, **kwargs) -> dict:
    """
    Returns:
        {
            "ari": float | None,
            "macro_f1": float | None,
            "weighted_f1": float | None,
            "runtime": float,          # seconds
            "n_spots": int,
            "n_clusters_pred": int,
            "n_clusters_true": int | None,
            "labels": np.ndarray,       # predicted cluster labels (REQUIRED for no-GT scoring)
            "embedding": np.ndarray | None,  # latent embedding if available
        }
    """
```

The `labels` field is mandatory — it's needed for no-GT scoring and consensus clustering even when ground truth is absent.

#### 1.4.2 Cluster Count Resolution

If the user specifies expected cluster count K, all methods target K clusters. If unknown:

1. Run Leiden/PCA at resolutions [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
2. Compute the consensus cluster count: the K that maximizes average silhouette score across resolutions
3. Use this K for all methods
4. Report the estimated K to the user with a sensitivity analysis (how rankings change if K±1)

#### 1.4.3 Seed Control

Stochastic methods run with multiple seeds (default: 5 seeds [42, 123, 456, 789, 1024]). Deterministic methods run once. This enables stability scoring (Section 1.5.2).

#### 1.4.4 Compute Dispatch

Methods are classified by compute cost:

| Tier | Runtime per slide | Methods | Execution |
|------|------------------|---------|-----------|
| Fast | < 30s | Leiden/PCA, SpaGCN, STAGATE, spatialMNN | Local (SaaS server) |
| Medium | 30s–5min | GraphST, BayesSpace, HyperGCN, Novae (zero-shot) | Local (SaaS server) |
| Heavy | 5min–1hr | Novae (fine-tuned), STMSGAL, BISON, SpaRTaCo | Cloud dispatch |
| Very heavy | > 1hr | SCOIGET | Cloud dispatch |

**Cloud dispatch:** Heavy/very-heavy methods are dispatched to auto-scaled cloud instances (Kubernetes jobs). The platform provisions CPU-only instances by default (matching our benchmarking approach); GPU instances are available if the method supports GPU acceleration. Results are streamed back via the job queue.

**Pilot optimization (Phase 2):** Before running all methods exhaustively, the meta-learning engine predicts which methods are likely to perform well on this data type (Section 1.6). The top 5 predicted methods run first as a pilot. If pilot results align with predictions (within expected variance), only the top methods are run fully. If predictions are poor (high residual), all methods run (fallback to exhaustive).

### 1.5 Evaluation & Scoring

#### 1.5.1 Ground-Truth Available: Direct Metrics

When the user provides annotations, use the existing `stbench.metrics` pipeline:
- **ARI** (Adjusted Rand Index) — primary ranking metric
- **Macro F1** — with Hungarian label matching (critical: the F1 score calculation bug we found in the original code is fixed here)
- **Weighted F1** — accounts for class imbalance
- **Runtime** — wall-clock seconds

Ranking: methods sorted by mean ARI across seeds. Ties broken by mean macro F1, then runtime.

#### 1.5.2 No Ground Truth: Multi-Criteria Composite Score

When no annotations are available, compute a composite score from four proxy metrics:

**A. Spatial Coherence Score (SCS) — weight 0.35**

Measures whether clusters form contiguous spatial regions (the defining property of good spatial domains).

```
For each cluster c:
    binary_c = 1 if spot belongs to cluster c, else 0
    moran_c = Moran's I(binary_c, spatial_weights_matrix)
    # Moran's I ranges from -1 (dispersed) to +1 (clustered)

SCS = weighted_mean(moran_c for all c, weights = cluster_sizes)
     normalized to [0, 1] via (SCS + 1) / 2
```

Spatial weights matrix: kNN graph (k=6 for hexagonal, k=4 for square, k=8 for random) on spatial coordinates, row-normalized.

**B. Cluster Stability Score (CSS) — weight 0.25**

Measures reproducibility across random seeds.

```
Run method with K seeds (default K=5).
Compute pairwise ARI between all C(K,2) seed pairs.
CSS = mean(pairwise_ARI)
     # Already in [0, 1] range (ARI can be negative, clip to [0, 1])
CSS = max(0, CSS)
```

**C. Expression Separability Score (ESS) — weight 0.20**

Measures whether clusters are distinguishable in gene expression space.

```
silhouette = silhouette_score(X_pca, predicted_labels)
ESS = (silhouette + 1) / 2   # normalize from [-1, 1] to [0, 1]
```

**D. Consensus Alignment Score (CAS) — weight 0.20**

Measures agreement with the consensus of all methods (wisdom-of-crowd).

```
1. Run all M methods (at least once, seed=42).
2. Build co-association matrix C (n_spots x n_spots):
   C[i,j] = (number of methods assigning spots i,j to same cluster) / M
3. Apply spectral clustering to C with K clusters → consensus_labels
4. For each method m:
   CAS_m = max(0, ARI(method_m_labels, consensus_labels))
```

**Composite No-GT Score:**

```
NoGTScore = 0.35 * SCS + 0.25 * CSS + 0.20 * ESS + 0.20 * CAS
```

**Weight rationale:** Spatial coherence gets the highest weight because contiguous spatial regions are the defining output of spatial domain detection — it's what makes ST clustering different from scRNA-seq clustering. Stability is second because a method that is stably wrong is still unreliable. Expression separability and consensus are lower-weighted tiebreakers because they can be gamed (a method that creates many tiny clusters can have high silhouette; consensus can be dominated by similar methods).

**Sensitivity reporting:** The platform reports each component score alongside the composite, so users can see WHY a method was recommended and adjust weights if their priority differs (e.g., a user who cares more about stability can re-rank).

#### 1.5.3 Statistical Comparison

For both GT and no-GT cases, pairwise method comparison via:
- **Wilcoxon signed-rank test** on per-slide scores (paired by slide)
- **Cliff's delta** for effect size
- **Holm-Bonferroni correction** for multiple comparisons
- Results: "Method A significantly outperforms Method B (p < 0.05, delta = 0.47)"

### 1.6 Meta-Learning Recommendation Engine

#### 1.6.1 Data Collection

Every benchmark run records a training tuple:

```
{
    data_features: DataFeatureVector,    # from Section 1.2
    method_name: str,
    method_type: str,                     # graph_nn | statistical | traditional | foundation
    score: float,                         # ARI if GT, NoGTScore if not
    runtime: float,
    seed: int,
    timestamp: datetime,
    user_id: str (anonymized),
}
```

Stored in a central PostgreSQL database. Only data features and scores are stored — never the user's expression data or coordinates.

#### 1.6.2 Model

**Phase 2 (cold start, < 500 runs):** Per-method linear regression with data features as predictors.

```
For each method m:
    model_m = Ridge(alpha=1.0)
    model_m.fit(X_data_features, y_scores_for_method_m)

Prediction for new data with features X_new:
    predicted_score_m = model_m.predict(X_new) for each m
    ranked_methods = sort methods by predicted_score_m descending
```

Ridge regression is chosen for cold start because it regularizes well with few samples and provides interpretable coefficients (which data features drive each method's performance).

**Phase 3 (warm, > 500 runs):** Gradient-boosted trees (XGBoost/LightGBM) per method, with cross-validation. Features expanded to include interaction terms (e.g., spot_count x sparsity). Model performance tracked via out-of-fold R^2.

**Phase 4 (mature, > 5000 runs):** A single multi-output model that predicts scores for all methods simultaneously, with a method-embedding to capture method-method similarities. Enables transfer learning (a new method with few runs can borrow from similar methods).

#### 1.6.3 Pilot-Then-Full Procedure

```
INPUT: user_data (AnnData), user_constraints (max_runtime, max_cost)
OUTPUT: ranked_methods with scores

1. Extract DataFeatureVector from user_data (Section 1.2)
2. Determine K (cluster count) — user-specified or auto-estimated (Section 1.4.2)
3. IF meta-learning model has sufficient confidence (R^2 > 0.3 for this data type):
     predicted_ranking = meta_learning.predict(data_features)
     pilot_methods = top 5 from predicted_ranking
   ELSE:
     pilot_methods = all fast + medium methods (fallback to exhaustive for light methods)

4. PILOT PHASE:
     Run pilot_methods on the first slide (or a representative subsample if multi-slide)
     with seed=42 only.
     Record pilot_scores.

5. DECISION:
     IF pilot_scores align with predicted_ranking (Spearman rho > 0.6):
       # Predictions are trustworthy
       full_methods = top 5 from pilot + any user-requested methods
     ELSE:
       # Predictions failed; run everything
       full_methods = ALL_METHODS

6. FULL PHASE:
     Run full_methods on all slides with all seeds.
     Compute final scores (GT metrics or NoGT composite).
     Rank methods.

7. FEEDBACK:
     Record (data_features, method, score) tuples to meta-learning database.
     Update model if enough new data has accumulated (batch retrain every 50 new runs).
```

#### 1.6.4 Confidence Reporting

The platform tells the user how confident the recommendation is:
- **High confidence:** Meta-learning R^2 > 0.5 for this data type, pilot aligned with predictions. "We're confident these are the best methods for your data."
- **Medium confidence:** R^2 0.2–0.5, or pilot partially aligned. "These methods performed best, but we recommend running the full benchmark for definitive results."
- **Low confidence:** R^2 < 0.2 or no prior data for this platform/tissue. "We ran all methods because we don't have enough experience with this data type yet."

### 1.7 Deliverable Generation

#### 1.7.1 Ranking Table

| Rank | Method | Score | ARI/NoGTScore | Stability | Spatial Coherence | Runtime | Recommendation |
|------|--------|-------|---------------|-----------|-------------------|---------|----------------|
| 1 | GraphST | 0.82 | 0.375 | 0.91 | 0.78 | 1232s | Best overall |
| 2 | SpaGCN | 0.79 | 0.328 | 0.88 | 0.75 | 23s | Best speed/accuracy |
| ... | | | | | | | |

#### 1.7.2 Publication Figures (PNG, 300 DPI)

1. **Spatial cluster maps:** Side-by-side per method, colored by cluster assignment
2. **Metric bar chart:** ARI/NoGTScore per method with error bars (across seeds)
3. **Runtime vs. accuracy scatter:** Each method as a point, x=runtime (log scale), y=score
4. **Stability heatmap:** Pairwise ARI across seeds for each method
5. **Spatial coherence map:** Moran's I per cluster per method
6. **Consensus map:** The consensus clustering result
7. **Component score radar chart:** SCS/CSS/ESS/CAS per method (no-GT case)

#### 1.7.3 Interactive Viewer

Web-based (React + DeckGL/squidpy viewer):
- Spatial cluster map per method, toggleable
- Side-by-side comparison mode (select 2-4 methods)
- Zoom/pan, hover for spot-level gene expression
- Overlay ground truth (if available) for comparison
- Click a cluster to see marker genes (differential expression)
- Export views as PNG

#### 1.7.4 Written Report (PDF)

Auto-generated report including:
- Executive summary with recommendation
- Data profile (platform, spot count, sparsity, estimated K)
- Methods evaluated and their categories
- Ranking table with statistical comparisons
- Key figures embedded
- Reproducibility statement (seeds, git commit, preprocessing parameters)
- Limitations and caveats
- Methods section (for inclusion in the user's own paper)

---

## 2. System Architecture

### 2.1 Components

```
┌─────────────────────────────────────────────────────────┐
│                    Web Frontend (React)                   │
│  Upload · Configure · Monitor · Interactive Viewer · DL  │
└──────────────────────────┬──────────────────────────────┘
                           │ REST + WebSocket
┌──────────────────────────▼──────────────────────────────┐
│                  API Gateway (FastAPI)                    │
│  Auth · Job Management · Progress Streaming · Downloads  │
└──────┬─────────────────┬───────────────┬────────────────┘
       │                 │               │
┌──────▼──────┐  ┌───────▼───────┐  ┌────▼──────────────┐
│  Job Queue   │  │  Meta-Learning │  │  Result Storage   │
│  (Celery +   │  │  Database      │  │  (S3-compatible)  │
│   Redis)     │  │  (PostgreSQL)  │  │                   │
└──────┬──────┘  └───────────────┘  └───────────────────┘
       │
┌──────▼──────────────────────────────────────────────────┐
│              Compute Engine (stbench)                     │
│                                                           │
│  ┌─────────────┐    ┌──────────────────────────────┐     │
│  │ Local Worker │    │  Cloud Dispatch (K8s Jobs)    │     │
│  │ Fast+Medium  │    │  Heavy+Very-Heavy Methods     │     │
│  │ methods      │    │  Auto-scaled CPU/GPU          │     │
│  └─────────────┘    └──────────────────────────────┘     │
│                                                           │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Method Registry + Plugin Loader                 │     │
│  │  12 built-in + community plugins                 │     │
│  └─────────────────────────────────────────────────┘     │
│                                                           │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Data Loaders (Visium, Slide-seq, MERFISH, ...)  │     │
│  └─────────────────────────────────────────────────┘     │
│                                                           │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Scoring Engine (GT metrics + NoGT composite)    │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
User uploads data
    → API validates and stores in S3 (encrypted)
    → DataFeatureVector extracted
    → Job queued in Celery
    → Local worker: loads data, runs preprocessing
    → Meta-learning: predicts method ranking (Phase 2+)
    → Pilot phase: top methods run on first slide
    → Decision: full or subset
    → Full phase: all selected methods × all slides × all seeds
        → Fast/medium methods: local execution
        → Heavy methods: dispatched to cloud K8s jobs
    → Results collected, scored (GT or NoGT)
    → Statistical comparisons computed
    → Figures generated (matplotlib/seaborn → PNG)
    → Interactive viewer data prepared (cluster labels + coordinates)
    → Report generated (ReportLab → PDF)
    → All deliverables stored in S3, user notified
    → Meta-learning database updated with (features, method, score) tuples
    → User data deleted after 7 days (configurable)
```

### 2.3 Data Privacy

- All uploads encrypted at rest (AES-256) and in transit (TLS 1.3)
- User data automatically deleted 7 days after analysis completes (configurable per user)
- Meta-learning database stores ONLY data features (spot count, sparsity, platform) and scores — never expression data, coordinates, or cluster labels
- Users can opt out of contributing to the meta-learning database
- SOC 2 / HIPAA compliance for hosted health data (Phase 3)

---

## 3. Plugin System & Community Registry

### 3.1 Plugin Interface

A plugin is a Python package with a single entry point:

```python
# my_st_method/plugin.py
from stbench.plugins import register_method

@register_method(
    name="MyMethod",
    category="new",           # "baseline" | "new"
    is_stochastic=True,
    is_r_based=False,
    compute_tier="medium",    # "fast" | "medium" | "heavy" | "very_heavy"
    gpu_optional=True,        # can use GPU but works on CPU
    min_stbench_version="1.0",
)
def run(adata, n_clusters, seed=42, **kwargs):
    # ... clustering logic ...
    return {
        "labels": predicted_labels,
        "embedding": latent_embedding,  # optional
        "runtime": elapsed_seconds,
        "n_spots": len(adata),
        "n_clusters_pred": len(np.unique(predicted_labels)),
        "n_clusters_true": n_clusters,
        # ARI/F1 computed by platform if GT available
    }
```

### 3.2 Plugin Discovery

- **Local plugins:** Drop `.py` files in `~/.stbench/plugins/` — auto-discovered on startup
- **Pip plugins:** `pip install stbench-mymethod` — discovered via Python entry points (`stbench.methods` group)
- **Community registry:** A package index at `registry.stbench.io` where users browse, search, and install community methods: `stbench install mymethod` (wraps pip with validation)

### 3.3 Plugin Validation

Before a plugin enters the community registry, it must pass automated validation:
1. **Interface test:** Does `run()` accept the standard signature and return the required dict?
2. **Smoke test:** Does it run on a 500-spot synthetic Visium dataset without crashing?
3. **Determinism test:** Does it produce identical output with the same seed?
4. **Runtime test:** Does it complete within its declared compute tier's time limit?
5. **Output validation:** Are labels non-empty, same length as n_spots, and non-degenerate (not all one cluster)?

---

## 4. Phased Build Plan

### Phase 1: MVP (Months 1–3)

**Goal:** Usable hosted platform with Visium support, all 12 methods, GT + no-GT scoring, and all deliverables.

| Component | Status | Work |
|-----------|--------|------|
| stbench core (runner, metrics, preprocessing) | Done | Already built |
| 12 method runners | Done | Already built |
| FastAPI backend | New | Auth, job queue, progress streaming |
| React frontend (upload + config + results) | New | Basic web UI |
| Visium loader | Done | Already in stbench |
| No-GT scoring (SCS, CSS, ESS, CAS) | New | Core algorithm implementation |
| Ranking table + figures | Partial | Figure generation code exists, needs web integration |
| Written report (PDF) | New | ReportLab template |
| Interactive viewer | New | squidpy-based web viewer |
| Local compute only | Done | stbench runs locally |
| Meta-learning | Deferred | Phase 2 |
| Plugin system | Deferred | Phase 2 |
| Non-Visium platforms | Deferred | Phase 3 |

**Deliverable:** A working website where a user uploads a Visium `.h5ad`, gets all 12 methods benchmarked, and receives a ranking, figures, interactive viewer, and PDF report.

### Phase 2: Intelligence + Scale (Months 4–6)

| Component | Work |
|-----------|------|
| Meta-learning database (PostgreSQL) | Schema, data collection from all Phase 1 runs |
| Meta-learning model (Ridge regression) | Per-method models, prediction API |
| Pilot-then-full procedure | Compute optimization, confidence reporting |
| Cloud dispatch (K8s jobs) | Auto-scaling for heavy methods |
| Plugin system | Interface, local discovery, validation |
| Interactive viewer enhancements | Marker gene overlay, multi-method comparison |

**Deliverable:** Platform that recommends methods before running all of them, dispatches heavy methods to cloud, and supports user-added plugins.

### Phase 3: Multi-Platform + Community (Months 7–9)

| Component | Work |
|-----------|------|
| Slide-seq, MERFISH, CosMx, Xenium, Stereo-seq, DBiT-seq loaders | Platform-specific normalization |
| Community plugin registry | Package index, search, install, validation pipeline |
| Meta-learning upgrade (XGBoost) | More data → more sophisticated model |
| BYOC (bring your own cloud) | Users provide their own cloud credentials for heavy compute |
| HIPAA compliance | For hosted patient data |

**Deliverable:** Full platform supporting all major ST platforms, community-contributed methods, and cross-user meta-learning.

---

## 5. Compute / Resource Estimates

### 5.1 Per-User Benchmark Run (Visium, 12 methods, 1 slide, 5 seeds)

| Method | Tier | Runtime/slide | Seeds | Total | Where |
|--------|------|--------------|-------|-------|-------|
| Leiden/PCA | Fast | ~5s | 5 | 25s | Local |
| SpaGCN | Fast | ~23s | 5 | 115s | Local |
| STAGATE | Fast | ~15s | 5 | 75s | Local |
| spatialMNN | Fast | ~20s | 1 | 20s | Local |
| GraphST | Medium | ~1200s | 5 | 100min | Local |
| BayesSpace | Medium | ~60s | 5 | 5min | Local |
| HyperGCN | Medium | ~120s | 5 | 10min | Local |
| Novae (zero-shot) | Medium | ~30s | 5 | 150s | Local |
| Novae (fine-tuned) | Heavy | ~100s | 5 | 8min | Cloud |
| STMSGAL | Heavy | ~200s | 5 | 17min | Cloud |
| BISON | Heavy | ~180s | 5 | 15min | Cloud |
| SpaRTaCo | Heavy | ~150s | 5 | 12min | Cloud |
| SCOIGET | Very heavy | ~1000s | 5 | 83min | Cloud |

**Total wall time (parallelized):** ~20–30 min with local + cloud parallelism (vs. ~3.5 hr sequential).
**Local server load:** ~2.5 hr CPU time (fast + medium methods).
**Cloud compute:** ~2.3 hr CPU time (heavy methods), ~$0.50–$1.00 per run at spot pricing.

### 5.2 Infrastructure (Hosted SaaS)

| Component | Spec | Cost/month |
|-----------|------|-----------|
| API server | 4 vCPU, 8 GB RAM | ~$80 |
| Local compute worker | 16 vCPU, 64 GB RAM | ~$300 |
| PostgreSQL (meta-learning) | 2 vCPU, 4 GB RAM, 50 GB | ~$50 |
| Redis (job queue) | 1 vCPU, 2 GB RAM | ~$20 |
| S3 storage | 500 GB | ~$12 |
| Cloud compute (on-demand) | Auto-scaled K8s | ~$0.50–$1.00 per user run |
| **Total fixed** | | **~$462/month** |
| **Variable** | | **~$1/run** |

### 5.3 Meta-Learning Database Growth

| Milestone | Runs | Model | Expected R^2 |
|-----------|------|-------|-------------|
| Cold start | 0–100 | None (exhaustive) | N/A |
| Phase 2 early | 100–500 | Ridge per method | 0.2–0.4 |
| Phase 2 mature | 500–2000 | Ridge per method | 0.4–0.6 |
| Phase 3 | 2000–5000 | XGBoost per method | 0.5–0.7 |
| Phase 4 | 5000+ | Multi-output + method embedding | 0.6–0.8 |

---

## 6. Key Assumptions

1. **No-GT weights (0.35/0.25/0.20/0.20)** are initial defaults based on domain reasoning. They should be validated empirically once we have enough runs with both GT and no-GT scores to correlate the no-GT composite with true ARI. The platform will report this correlation once available and adjust weights if the empirical optimum differs.

2. **Meta-learning cold start** uses our existing benchmark data (207 rows in `unified_results.csv` across 3 datasets, 10 methods) as the initial training set. This gives ~20 training tuples per method — enough for Ridge regression with 10 features.

3. **Cloud dispatch** assumes Kubernetes availability. For Phase 1, all methods run locally (the SaaS server has 64 GB RAM, sufficient for all methods on Visium-scale data). Cloud dispatch is a Phase 2 optimization for throughput, not a Phase 1 requirement.

4. **Plugin validation** is automated but not peer-reviewed. The community registry relies on user ratings and download counts for quality signaling, similar to PyPI or conda-forge.

5. **Data deletion (7 days)** is the default. Users can extend to 30 days or request immediate deletion. Institutional users may require longer retention (configured per account).

6. **The platform does not store user data in the meta-learning database.** Only anonymized data features (spot count, sparsity, platform) and scores are stored. This is explicitly communicated to users in the privacy policy.

7. **High-resolution platforms (MERFISH, Xenium, Stereo-seq)** may require binning sub-cellular coordinates into pseudo-spots. The binning strategy (square grid, hexagonal, or adaptive) is platform-specific and configurable by the user. This is a Phase 3 feature.

8. **The no-GT composite score is a proxy, not a ground truth.** The platform explicitly states this limitation in every report. The consensus component (CAS) helps mitigate individual metric biases but does not eliminate them.
