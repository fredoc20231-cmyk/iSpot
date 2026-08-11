"""Unit tests for the FastQC-style QC report."""
import pytest

from ispot.qc import classify

np = pytest.importorskip("numpy")
ad = pytest.importorskip("anndata")

from ispot.qc import compute_qc  # noqa: E402


# --- classify() is pure python (no numpy) --------------------------------

def test_classify_higher_is_worse():
    assert classify(0.05, warn=0.10, fail=0.30, higher_is_worse=True) == "pass"
    assert classify(0.20, warn=0.10, fail=0.30, higher_is_worse=True) == "warn"
    assert classify(0.40, warn=0.10, fail=0.30, higher_is_worse=True) == "fail"


def test_classify_lower_is_worse():
    assert classify(800, warn=500, fail=100, higher_is_worse=False) == "pass"
    assert classify(300, warn=500, fail=100, higher_is_worse=False) == "warn"
    assert classify(50, warn=500, fail=100, higher_is_worse=False) == "fail"


def test_classify_none_or_nan_is_warn():
    assert classify(None, 1, 2) == "warn"
    assert classify(float("nan"), 1, 2) == "warn"


# --- compute_qc on synthetic AnnData -------------------------------------

def _adata(counts_per_spot, n_spots=200, n_genes=60, mito_boost=0.0, seed=0):
    rng = np.random.default_rng(seed)
    # Distribute each spot's target counts across genes.
    X = rng.multinomial(int(counts_per_spot), [1.0 / n_genes] * n_genes, size=n_spots).astype("float32")
    var = [f"g{i}" for i in range(n_genes)]
    var[0], var[1] = "MT-CO1", "MT-ND1"  # two mito genes
    if mito_boost > 0:
        X[:, 0] += (counts_per_spot * mito_boost)
    a = ad.AnnData(X)
    a.var_names = var
    a.obsm["spatial"] = rng.random((n_spots, 2))
    return a


def test_compute_qc_structure_and_healthy():
    qc = compute_qc(_adata(2000), platform="Visium")
    assert set(qc.keys()) == {"basic", "modules", "summary"}
    assert qc["basic"]["platform"] == "Visium"
    assert qc["basic"]["n_mito_genes"] == 2
    ids = {m["id"] for m in qc["modules"]}
    assert {"sequencing_depth", "genes_per_spot", "high_mito_spots_frac"} <= ids
    depth = next(m for m in qc["modules"] if m["id"] == "sequencing_depth")
    assert depth["status"] == "pass"  # 2000 counts/spot is healthy


def test_compute_qc_flags_shallow_depth():
    qc = compute_qc(_adata(30))  # ~30 counts/spot -> fail depth + low-count spots
    depth = next(m for m in qc["modules"] if m["id"] == "sequencing_depth")
    low = next(m for m in qc["modules"] if m["id"] == "low_count_spots_frac")
    assert depth["status"] == "fail"
    assert low["status"] in ("warn", "fail")
    assert qc["summary"]["overall"] == "fail"


def test_compute_qc_flags_high_mito():
    qc = compute_qc(_adata(1000, mito_boost=0.5))  # ~50% mito in every spot
    mito = next(m for m in qc["modules"] if m["id"] == "high_mito_spots_frac")
    assert mito["status"] == "fail"
    assert qc["basic"]["median_mito_fraction"] > 0.2
