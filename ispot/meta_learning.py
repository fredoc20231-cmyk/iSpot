"""
Meta-learning recommendation engine.

Predicts which clustering methods will perform best on a new dataset based
on accumulated experience from prior runs. Uses per-method Ridge regression
on data features.

Cold start: uses existing benchmark data as initial training set.
Warm: retrains every 50 new runs.

Section 1.6 of the platform plan.
"""
from __future__ import annotations

import json
import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import asdict
from typing import Optional

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT,
    method_name TEXT NOT NULL,
    method_type TEXT,
    -- Numeric data features
    n_spots REAL,
    n_genes REAL,
    sparsity REAL,
    median_genes_per_spot REAL,
    median_counts_per_spot REAL,
    spot_diameter_um REAL,
    has_histology REAL,
    n_expected_clusters REAL,
    spatial_extent REAL,
    coordinate_density REAL,
    -- Categorical data features
    spatial_layout TEXT,
    platform TEXT,
    tissue_type TEXT,
    -- Result
    score REAL NOT NULL,          -- ARI if GT, NoGTScore if not
    score_type TEXT NOT NULL,     -- "ari" or "nogt"
    runtime REAL,
    seed INTEGER,
    n_clusters INTEGER,
    git_commit TEXT
);

CREATE INDEX IF NOT EXISTS idx_method ON benchmark_runs(method_name);
CREATE INDEX IF NOT EXISTS idx_platform ON benchmark_runs(platform);
"""


# Numeric features used by the ML model
NUMERIC_FEATURES = [
    "n_spots", "n_genes", "sparsity", "median_genes_per_spot",
    "median_counts_per_spot", "spot_diameter_um", "has_histology",
    "n_expected_clusters", "spatial_extent", "coordinate_density",
]

# Categorical features (one-hot encoded)
CATEGORICAL_FEATURES = ["spatial_layout", "platform"]

# All possible methods (from registry)
KNOWN_METHODS = [
    "Leiden_PCA", "SpaGCN", "STAGATE", "GraphST", "BayesSpace",
    "HyperGCN", "STMSGAL", "SCOIGET", "Novae", "BISON", "SpaRTaCo",
    "spatialMNN",
]


# ---------------------------------------------------------------------------
# Database interface
# ---------------------------------------------------------------------------

class MetaLearningDB:
    """SQLite database storing benchmark run records for meta-learning.

    In production, this would be PostgreSQL. SQLite is used for development
    and single-machine deployments.

    Thread-safe: uses check_same_thread=False with a threading lock to allow
    access from FastAPI background tasks running in different threads.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        import threading
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def record_run(self, record: dict) -> int:
        """Insert a benchmark run record."""
        cols = []
        vals = []
        for col in [
            "timestamp", "user_id", "method_name", "method_type",
            "n_spots", "n_genes", "sparsity", "median_genes_per_spot",
            "median_counts_per_spot", "spot_diameter_um", "has_histology",
            "n_expected_clusters", "spatial_extent", "coordinate_density",
            "spatial_layout", "platform", "tissue_type",
            "score", "score_type", "runtime", "seed", "n_clusters", "git_commit",
        ]:
            if col in record:
                cols.append(col)
                vals.append(record[col])

        placeholders = ",".join(["?"] * len(vals))
        col_str = ",".join(cols)
        with self._lock:
            cursor = self.conn.execute(
                f"INSERT INTO benchmark_runs ({col_str}) VALUES ({placeholders})",
                vals,
            )
            self.conn.commit()
        return cursor.lastrowid

    def get_all_runs(self) -> pd.DataFrame:
        """Return all runs as a DataFrame."""
        with self._lock:
            return pd.read_sql("SELECT * FROM benchmark_runs", self.conn)

    def get_runs_for_method(self, method: str) -> pd.DataFrame:
        """Return all runs for a specific method."""
        with self._lock:
            return pd.read_sql(
                "SELECT * FROM benchmark_runs WHERE method_name = ?",
                self.conn, params=[method],
            )

    def count_runs(self) -> int:
        """Total number of runs."""
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]

    def close(self):
        with self._lock:
            self.conn.close()


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
    """Encode data features into a numeric matrix for ML models.

    Numeric features are used as-is. Categorical features are one-hot encoded.

    Parameters
    ----------
    df : pd.DataFrame
        Rows from the benchmark_runs table.
    feature_columns : list[str], optional
        If provided, select only these columns (for consistency between
        train and predict).

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
    """
    if feature_columns is None:
        # Build feature matrix: numeric + one-hot categorical
        numeric = df[NUMERIC_FEATURES].fillna(0).values.astype(float)

        # One-hot encode categoricals
        cat_parts = []
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                cat_parts.append(dummies.values.astype(float))

        if cat_parts:
            cat = np.hstack(cat_parts)
            return np.hstack([numeric, cat])
        return numeric
    else:
        # Use reference columns: build one-hot consistently
        # First get numeric features
        numeric = df[NUMERIC_FEATURES].fillna(0).values.astype(float)

        # Build one-hot for categoricals, aligned to reference columns
        cat_cols_in_ref = [c for c in feature_columns if c not in NUMERIC_FEATURES]
        if cat_cols_in_ref:
            cat_matrix = np.zeros((len(df), len(cat_cols_in_ref)))
            for i, col_name in enumerate(cat_cols_in_ref):
                # Parse: prefix_value (e.g., "platform_Visium")
                for cat_col in CATEGORICAL_FEATURES:
                    prefix = f"{cat_col}_"
                    if col_name.startswith(prefix):
                        val = col_name[len(prefix):]
                        if val == "nan":
                            cat_matrix[:, i] = df[cat_col].isna().astype(float).values
                        else:
                            cat_matrix[:, i] = (df[cat_col].astype(str) == val).astype(float).values
                        break
            return np.hstack([numeric, cat_matrix])
        return numeric


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get the full list of feature column names after one-hot encoding."""
    cols = list(NUMERIC_FEATURES)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            cols.extend(dummies.columns.tolist())
    return cols


def encode_single_feature_vector(
    features: dict,
    reference_columns: list[str] | None = None,
) -> np.ndarray:
    """Encode a single DataFeatureVector dict into a feature row.

    Parameters
    ----------
    features : dict
        DataFeatureVector as a dict (from .to_dict()).
    reference_columns : list[str], optional
        Column names from get_feature_columns() to ensure consistency.

    Returns
    -------
    np.ndarray, shape (1, n_features)
    """
    # Numeric features
    numeric_vals = [float(features.get(f, 0) or 0) for f in NUMERIC_FEATURES]
    row = list(numeric_vals)

    # Categorical one-hot
    for col in CATEGORICAL_FEATURES:
        val = features.get(col, None)
        # We need to know the possible categories from reference_columns
        if reference_columns is not None:
            # Find columns that start with this categorical prefix
            prefix = f"{col}_"
            cat_cols = [c for c in reference_columns if c.startswith(prefix)]
            for c in cat_cols:
                expected_val = c[len(prefix):]
                if expected_val == "nan" and val is None:
                    row.append(1.0)
                elif str(val) == expected_val:
                    row.append(1.0)
                else:
                    row.append(0.0)
        else:
            # No reference: just use the raw value as a code
            row.append(float(hash(str(val)) % 1000) / 1000.0)

    return np.array(row).reshape(1, -1)


# ---------------------------------------------------------------------------
# Meta-learning model
# ---------------------------------------------------------------------------

class MetaLearningModel:
    """Per-method Ridge regression model for score prediction.

    Trains one Ridge model per method, using data features as predictors
    and benchmark scores as targets. Predicts which methods will perform
    best on a new dataset.

    Cold start: if fewer than min_training_samples per method, falls back
    to uniform prediction (all methods ranked equally).
    """

    def __init__(self, min_samples_per_method: int = 5, alpha: float = 1.0):
        self.min_samples = min_samples_per_method
        self.alpha = alpha
        self.models: dict[str, Ridge] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_columns: list[str] | None = None
        self.cv_r2: dict[str, float] = {}
        self.is_trained = False

    def train(self, db: MetaLearningDB) -> dict:
        """Train per-method models from the database.

        Returns
        -------
        dict: training summary with per-method sample counts and CV R^2.
        """
        df = db.get_all_runs()
        if len(df) == 0:
            self.is_trained = False
            return {"status": "no_data", "n_total": 0}

        # Get feature columns
        self.feature_columns = get_feature_columns(df)

        summary = {"status": "trained", "n_total": len(df), "methods": {}}

        for method in df["method_name"].unique():
            method_df = df[df["method_name"] == method]
            n = len(method_df)

            if n < self.min_samples:
                summary["methods"][method] = {
                    "n_samples": n,
                    "cv_r2": None,
                    "status": "insufficient_data",
                }
                continue

            X = encode_features(method_df, self.feature_columns)
            y = method_df["score"].values.astype(float)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Cross-validated R^2
            if n >= 5:
                try:
                    cv_scores = cross_val_score(
                        Ridge(alpha=self.alpha), X_scaled, y,
                        cv=min(5, n), scoring="r2",
                    )
                    cv_r2 = float(np.mean(cv_scores))
                except Exception:
                    cv_r2 = None
            else:
                cv_r2 = None

            # Train final model
            model = Ridge(alpha=self.alpha)
            model.fit(X_scaled, y)

            self.models[method] = model
            self.scalers[method] = scaler
            self.cv_r2[method] = cv_r2

            summary["methods"][method] = {
                "n_samples": n,
                "cv_r2": cv_r2,
                "status": "trained",
            }

        self.is_trained = len(self.models) > 0
        return summary

    def predict(self, features: dict) -> dict:
        """Predict scores for all methods on a new dataset.

        Parameters
        ----------
        features : dict
            DataFeatureVector as a dict.

        Returns
        -------
        dict with keys:
            - predictions: {method: predicted_score}
            - ranking: list of (method, score) sorted descending
            - confidence: "high" | "medium" | "low"
            - mean_cv_r2: average CV R^2 across trained models
        """
        if not self.is_trained:
            # Cold start: uniform prediction
            return {
                "predictions": {m: 0.5 for m in KNOWN_METHODS},
                "ranking": [(m, 0.5) for m in KNOWN_METHODS],
                "confidence": "low",
                "mean_cv_r2": None,
                "status": "cold_start",
            }

        X = encode_single_feature_vector(features, self.feature_columns)

        predictions = {}
        r2s = []
        for method, model in self.models.items():
            scaler = self.scalers[method]
            X_scaled = scaler.transform(X)
            pred = float(model.predict(X_scaled)[0])
            predictions[method] = pred
            if self.cv_r2.get(method) is not None:
                r2s.append(self.cv_r2[method])

        # Add untrained methods with default prediction
        for method in KNOWN_METHODS:
            if method not in predictions:
                predictions[method] = 0.5

        # Rank by predicted score
        ranking = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        # Confidence based on mean CV R^2
        mean_r2 = float(np.mean(r2s)) if r2s else None
        if mean_r2 is not None:
            if mean_r2 > 0.5:
                confidence = "high"
            elif mean_r2 > 0.2:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "low"

        return {
            "predictions": predictions,
            "ranking": ranking,
            "confidence": confidence,
            "mean_cv_r2": mean_r2,
            "status": "predicted",
        }


# ---------------------------------------------------------------------------
# Pilot-then-full procedure
# ---------------------------------------------------------------------------

def pilot_then_full(
    features: dict,
    model: MetaLearningModel,
    all_methods: list[str],
    pilot_top_n: int = 5,
    alignment_threshold: float = 0.6,
) -> dict:
    """Decide which methods to run based on meta-learning predictions.

    Phase 1 (no model): run all methods.
    Phase 2+ (model trained): run top-N as pilot, then decide.

    Parameters
    ----------
    features : dict
        DataFeatureVector as a dict.
    model : MetaLearningModel
        Trained (or untrained) meta-learning model.
    all_methods : list[str]
        All available method names.
    pilot_top_n : int
        Number of top methods to run in pilot phase.
    alignment_threshold : float
        Spearman rho threshold for pilot-prediction alignment.

    Returns
    -------
    dict with keys:
        - pilot_methods: list of methods to run first
        - full_methods: list of methods to run fully (after pilot)
        - run_all: bool — whether to skip pilot and run everything
        - confidence: str
        - predicted_ranking: list of (method, score)
    """
    prediction = model.predict(features)

    if prediction["status"] == "cold_start":
        # No model: run everything
        return {
            "pilot_methods": all_methods,
            "full_methods": all_methods,
            "run_all": True,
            "confidence": "low",
            "predicted_ranking": prediction["ranking"],
            "reason": "Cold start: no training data. Running all methods.",
        }

    # Select top-N for pilot
    predicted_ranking = prediction["ranking"]
    pilot_methods = [m for m, _ in predicted_ranking[:pilot_top_n]]

    # Also include at least one fast method for quick feedback
    fast_methods = ["Leiden_PCA", "SpaGCN", "STAGATE", "spatialMNN"]
    for fm in fast_methods:
        if fm not in pilot_methods:
            pilot_methods.append(fm)
            break

    return {
        "pilot_methods": pilot_methods,
        "full_methods": None,  # determined after pilot results
        "run_all": False,
        "confidence": prediction["confidence"],
        "predicted_ranking": predicted_ranking,
        "mean_cv_r2": prediction["mean_cv_r2"],
        "reason": f"Pilot phase: running top {len(pilot_methods)} methods. "
                  f"Confidence: {prediction['confidence']} (CV R^2 = {prediction['mean_cv_r2']}).",
    }


def evaluate_pilot_alignment(
    pilot_scores: dict[str, float],
    predicted_ranking: list[tuple[str, float]],
    alignment_threshold: float = 0.6,
) -> dict:
    """Compare pilot results to meta-learning predictions.

    Called after the pilot phase completes to decide whether to run
    all methods or just the top performers.

    Parameters
    ----------
    pilot_scores : dict[str, float]
        Actual scores from pilot runs: {method: score}.
    predicted_ranking : list[tuple[str, float]]
        Predicted ranking from MetaLearningModel.predict().
    alignment_threshold : float
        Spearman rho threshold. If pilot-prediction alignment exceeds
        this, we trust the predictions and run only top methods.

    Returns
    -------
    dict with keys:
        - aligned: bool
        - spearman_rho: float
        - decision: "run_top" or "run_all"
        - full_methods: list of methods to run in full phase
    """
    # Get predicted scores for pilot methods
    predicted = {m: s for m, s in predicted_ranking}

    # Methods that have both predicted and actual scores
    common = [m for m in pilot_scores if m in predicted]
    if len(common) < 3:
        # Not enough overlap to assess alignment
        return {
            "aligned": False,
            "spearman_rho": None,
            "decision": "run_all",
            "full_methods": None,
            "reason": "Insufficient overlap between pilot and predictions.",
        }

    pred_vals = [predicted[m] for m in common]
    actual_vals = [pilot_scores[m] for m in common]

    rho, p_value = spearmanr(pred_vals, actual_vals)

    if rho is not None and rho > alignment_threshold:
        # Predictions are trustworthy: run top methods only
        top_methods = [m for m, _ in predicted_ranking[:5]]
        # Ensure pilot methods are included
        for m in pilot_scores:
            if m not in top_methods:
                top_methods.append(m)
        return {
            "aligned": True,
            "spearman_rho": float(rho),
            "decision": "run_top",
            "full_methods": top_methods,
            "reason": f"Pilot aligned with predictions (rho={rho:.3f}). "
                      f"Running top {len(top_methods)} methods.",
        }
    else:
        # Predictions failed: run everything
        return {
            "aligned": False,
            "spearman_rho": float(rho) if rho is not None else None,
            "decision": "run_all",
            "full_methods": None,
            "reason": f"Pilot did not align with predictions (rho={rho:.3f}). "
                      f"Running all methods.",
        }


# ---------------------------------------------------------------------------
# Seed existing benchmark data into the database
# ---------------------------------------------------------------------------

def seed_from_existing_results(
    db: MetaLearningDB,
    unified_results_csv: str,
    master_summary_csv: str | None = None,
) -> int:
    """Seed the meta-learning database from existing benchmark results.

    Reads unified_results.csv (our reproduction runs) and inserts records
    into the database. This provides the cold-start training data.

    Parameters
    ----------
    db : MetaLearningDB
    unified_results_csv : str
        Path to unified_results.csv.
    master_summary_csv : str, optional
        Path to master_summary.csv for additional features.

    Returns
    -------
    int: number of records inserted.
    """
    if not os.path.exists(unified_results_csv):
        return 0

    df = pd.read_csv(unified_results_csv)
    n_inserted = 0

    # Dataset-level features (approximate from dataset name)
    dataset_features = {
        "DLPFC": {
            "platform": "Visium", "spatial_layout": "square",
            "tissue_type": "brain", "spot_diameter_um": 100,
            "spatial_extent": 10000, "coordinate_density": 0.0001,
        },
        "HER2+": {
            "platform": "Visium", "spatial_layout": "square",
            "tissue_type": "breast", "spot_diameter_um": 100,
            "spatial_extent": 10000, "coordinate_density": 0.0001,
        },
        "MOSTA": {
            "platform": "Stereo-seq", "spatial_layout": "hexagonal",
            "tissue_type": "embryo", "spot_diameter_um": 50,
            "spatial_extent": 50000, "coordinate_density": 0.001,
        },
    }

    for _, row in df.iterrows():
        if pd.isna(row.get("ari")):
            continue

        dataset = row.get("dataset", "unknown")
        feats = dataset_features.get(dataset, {})

        record = {
            "timestamp": datetime.now().isoformat(),
            "user_id": "seed",
            "method_name": row.get("method", "unknown"),
            "method_type": "unknown",
            "n_spots": float(row.get("n_spots", 0) or 0),
            "n_genes": 3000,  # approximate (after HVG selection)
            "sparsity": 0.95,
            "median_genes_per_spot": 1000,
            "median_counts_per_spot": 2000,
            "spot_diameter_um": feats.get("spot_diameter_um", 100),
            "has_histology": 0,
            "n_expected_clusters": float(row.get("n_clusters_true", 0) or 0),
            "spatial_extent": feats.get("spatial_extent", 10000),
            "coordinate_density": feats.get("coordinate_density", 0.0001),
            "spatial_layout": feats.get("spatial_layout", "square"),
            "platform": feats.get("platform", "Visium"),
            "tissue_type": feats.get("tissue_type", None),
            "score": float(row["ari"]),
            "score_type": "ari",
            "runtime": float(row.get("runtime", 0) or 0),
            "seed": int(row.get("seed", 42) if pd.notna(row.get("seed", 42)) else 42),
            "n_clusters": int(row.get("n_clusters_pred", 0) if pd.notna(row.get("n_clusters_pred", 0)) else 0),
            "git_commit": "seed",
        }
        db.record_run(record)
        n_inserted += 1

    return n_inserted
