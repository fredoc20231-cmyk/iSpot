"""Unit tests for pairwise statistical comparison."""
import pytest

from ispot.stats_compare import cliffs_delta, holm_bonferroni, build_comparison_table


# --- Cliff's delta ---------------------------------------------------------

def test_cliffs_delta_all_greater_is_one_large():
    d, mag = cliffs_delta([10, 11, 12], [1, 2, 3])
    assert d == pytest.approx(1.0)
    assert mag == "large"


def test_cliffs_delta_all_less_is_minus_one():
    d, mag = cliffs_delta([1, 2, 3], [10, 11, 12])
    assert d == pytest.approx(-1.0)
    assert mag == "large"


def test_cliffs_delta_identical_is_negligible():
    d, mag = cliffs_delta([1, 2, 3], [1, 2, 3])
    assert d == pytest.approx(0.0)
    assert mag == "negligible"


def test_cliffs_delta_empty_is_zero():
    assert cliffs_delta([], [1, 2]) == (0.0, "negligible")


# --- Holm-Bonferroni -------------------------------------------------------

def test_holm_bonferroni_known_values():
    reject, adj = holm_bonferroni([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert reject == [True, False, False, True]


def test_holm_bonferroni_monotone_and_clipped():
    reject, adj = holm_bonferroni([0.6, 0.6, 0.6], alpha=0.05)
    # 3*0.6 = 1.8 -> clipped to 1.0, and non-decreasing
    assert adj == pytest.approx([1.0, 1.0, 1.0])
    assert reject == [False, False, False]
    assert all(adj[i] <= adj[i + 1] + 1e-12 for i in range(len(adj) - 1))


def test_holm_bonferroni_empty():
    assert holm_bonferroni([]) == ([], [])


# --- comparison table ------------------------------------------------------

def test_build_comparison_table_basic():
    seeds = [42, 123, 456, 789, 1024]
    score_map = {
        "high": {s: 0.9 for s in seeds},
        "low": {s: 0.1 for s in seeds},
    }
    df = build_comparison_table(score_map, metric_name="ari")
    assert len(df) == 1
    row = df.iloc[0]
    assert {row["method_a"], row["method_b"]} == {"high", "low"}
    assert row["n_pairs"] == 5
    assert abs(row["cliffs_delta"]) == pytest.approx(1.0)
    assert isinstance(bool(row["significant"]), bool)
    assert 0.0 <= row["p_adjusted"] <= 1.0


def test_build_comparison_table_skips_insufficient_overlap():
    # Only one shared seed -> below min_pairs -> no comparable pair.
    score_map = {"a": {42: 0.5}, "b": {99: 0.4}}
    df = build_comparison_table(score_map)
    assert len(df) == 0
    assert "significant" in df.columns


def test_build_comparison_table_identical_not_significant():
    seeds = [1, 2, 3, 4]
    score_map = {"a": {s: 0.5 for s in seeds}, "b": {s: 0.5 for s in seeds}}
    df = build_comparison_table(score_map)
    assert len(df) == 1
    assert bool(df.iloc[0]["significant"]) is False
