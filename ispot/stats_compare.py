"""
Pairwise statistical comparison of methods (PLAN section 1.5.3).

Given per-seed scores for each method, compare every method pair with the
Wilcoxon signed-rank test (paired by seed), quantify effect size with Cliff's
delta, and correct for multiple comparisons with Holm-Bonferroni.

The numeric helpers (``cliffs_delta``, ``holm_bonferroni``) are pure NumPy and
unit-tested directly; ``build_comparison_table`` adds the SciPy/pandas layer
that the PDF report consumes.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Cliff's delta magnitude thresholds (Romano et al. 2006).
_CLIFF_THRESHOLDS = ((0.147, "negligible"), (0.33, "small"), (0.474, "medium"))


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> tuple[float, str]:
    """Cliff's delta effect size between two samples and its magnitude label.

    delta = (#(x_i > y_j) - #(x_i < y_j)) / (n*m), in [-1, 1]. Positive means
    x tends to exceed y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0, "negligible"
    diff = x[:, None] - y[None, :]
    greater = int(np.sum(diff > 0))
    less = int(np.sum(diff < 0))
    delta = (greater - less) / (x.size * y.size)

    mag = "large"
    for thresh, label in _CLIFF_THRESHOLDS:
        if abs(delta) < thresh:
            mag = label
            break
    return float(delta), mag


def holm_bonferroni(
    pvals: Sequence[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni step-down correction.

    Returns ``(reject, p_adjusted)`` in the original order of ``pvals``.
    Adjusted p-values are made monotonic non-decreasing and clipped to 1.
    """
    pvals = list(pvals)
    m = len(pvals)
    if m == 0:
        return [], []

    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)

    reject = [adjusted[i] <= alpha for i in range(m)]
    return reject, adjusted


def _aligned_by_seed(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Align two {seed: score} maps on their shared seeds (sorted)."""
    shared = sorted(set(a) & set(b))
    xs = np.array([a[s] for s in shared], dtype=float)
    ys = np.array([b[s] for s in shared], dtype=float)
    return xs, ys


def build_comparison_table(
    score_map: dict[str, dict],
    metric_name: str = "ari",
    alpha: float = 0.05,
    min_pairs: int = 2,
):
    """Build a pairwise comparison table (pandas DataFrame).

    Parameters
    ----------
    score_map : dict[str, dict]
        method name -> {seed: score}. Only methods sharing at least
        ``min_pairs`` seeds with another method are compared.
    metric_name : str
        Name of the metric being compared (recorded in the table).
    alpha : float
        Significance level for Holm-Bonferroni.
    min_pairs : int
        Minimum number of shared seeds required to run a test.

    Returns
    -------
    pandas.DataFrame with columns: method_a, method_b, metric, n_pairs,
    statistic, p_value, p_adjusted, cliffs_delta, effect, significant.
    Sorted by adjusted p-value ascending. Empty if no pair is comparable.
    """
    import pandas as pd
    from scipy.stats import wilcoxon

    methods = sorted(score_map)
    rows = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            xs, ys = _aligned_by_seed(score_map[a], score_map[b])
            if len(xs) < min_pairs:
                continue
            try:
                stat, p = wilcoxon(xs, ys)
                stat, p = float(stat), float(p)
            except ValueError:
                # All paired differences are zero (identical) -> no evidence.
                stat, p = 0.0, 1.0
            delta, effect = cliffs_delta(xs, ys)
            rows.append({
                "method_a": a,
                "method_b": b,
                "metric": metric_name,
                "n_pairs": int(len(xs)),
                "statistic": round(stat, 4),
                "p_value": round(p, 6),
                "cliffs_delta": round(delta, 4),
                "effect": effect,
            })

    columns = [
        "method_a", "method_b", "metric", "n_pairs", "statistic",
        "p_value", "p_adjusted", "cliffs_delta", "effect", "significant",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    reject, adjusted = holm_bonferroni([r["p_value"] for r in rows], alpha=alpha)
    for r, rej, adj in zip(rows, reject, adjusted):
        r["p_adjusted"] = round(adj, 6)
        r["significant"] = bool(rej)

    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values("p_adjusted", kind="stable").reset_index(drop=True)
