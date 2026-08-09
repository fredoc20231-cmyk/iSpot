"""Unit tests for the meta-learning model (Ridge/GBT selection + calibration)."""
import pytest

from ispot.meta_learning import MetaLearningDB, MetaLearningModel


def _seed_db(db, method, n):
    """Insert n runs for a method with score correlated to the features."""
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 0.0
        score = min(1.0, max(0.0, 0.15 + 0.7 * frac))
        db.record_run({
            "timestamp": "t", "user_id": "u", "method_name": method, "method_type": "x",
            "n_spots": 1000 + i * 50, "n_genes": 2000, "sparsity": 0.9 - 0.2 * frac,
            "median_genes_per_spot": 500, "median_counts_per_spot": 1000,
            "spot_diameter_um": 55, "has_histology": 0, "n_expected_clusters": 7,
            "spatial_extent": 100.0 + frac * 10, "coordinate_density": 10.0,
            "spatial_layout": "hexagonal", "platform": "Visium" if i % 2 == 0 else "MERFISH",
            "score": score, "score_type": "ari", "runtime": 1.0, "seed": 42,
            "n_clusters": 7, "git_commit": "t",
        })


def _feature_vector():
    return {
        "n_spots": 1500, "n_genes": 2000, "sparsity": 0.85,
        "median_genes_per_spot": 500, "median_counts_per_spot": 1000,
        "spot_diameter_um": 55, "has_histology": 0, "n_expected_clusters": 7,
        "spatial_extent": 105.0, "coordinate_density": 10.0,
        "spatial_layout": "hexagonal", "platform": "Visium",
    }


def test_model_selection_ridge_vs_gbt():
    db = MetaLearningDB(":memory:")
    _seed_db(db, "Ridgey", 20)   # below gbt threshold -> ridge
    _seed_db(db, "Boosty", 50)   # at/above threshold -> gbt
    model = MetaLearningModel(min_samples_per_method=5, gbt_min_samples=40, calibrate=True)

    summary = model.train(db)

    assert summary["status"] == "trained"
    assert summary["methods"]["Ridgey"]["model_type"] == "ridge"
    assert summary["methods"]["Boosty"]["model_type"] == "gbt"
    assert model.last_trained_count == db.count_runs() == 70


def test_calibrated_predictions_in_unit_range():
    db = MetaLearningDB(":memory:")
    _seed_db(db, "Boosty", 50)
    model = MetaLearningModel(min_samples_per_method=5, gbt_min_samples=40, calibrate=True)
    summary = model.train(db)

    assert summary["methods"]["Boosty"]["calibrated"] is True

    out = model.predict(_feature_vector())
    assert out["status"] == "predicted"
    assert "Boosty" in out["predictions"]
    assert 0.0 <= out["predictions"]["Boosty"] <= 1.0
    # ranking is sorted descending by predicted score
    scores = [s for _, s in out["ranking"]]
    assert scores == sorted(scores, reverse=True)


def test_calibration_can_be_disabled():
    db = MetaLearningDB(":memory:")
    _seed_db(db, "Boosty", 50)
    model = MetaLearningModel(min_samples_per_method=5, gbt_min_samples=40, calibrate=False)
    summary = model.train(db)

    assert summary["methods"]["Boosty"]["calibrated"] is False
    assert "Boosty" not in model.calibrators


def test_cold_start_uniform_when_untrained():
    model = MetaLearningModel()
    out = model.predict(_feature_vector())
    assert out["status"] == "cold_start"
    assert out["confidence"] == "low"
    assert all(v == 0.5 for v in out["predictions"].values())
