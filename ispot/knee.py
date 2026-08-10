"""
Knee/elbow detection for a 1-D curve.

Direction-agnostic elbow finder based on the maximum perpendicular distance
from the chord joining the first and last points of the curve (the "Kneedle"
distance heuristic). Unlike a cumulative-drop threshold, it does not assume the
curve is monotonically increasing or decreasing, which matters for the
spatial-coherence-vs-K curve used in cluster-count estimation: that curve is
often non-monotonic in practice, and a monotonic assumption collapses the
estimate to the smallest candidate K almost regardless of the data.

Pure NumPy, no heavy dependencies, so the logic is unit-testable in isolation.
"""
from __future__ import annotations

import numpy as np


def find_knee(ks: np.ndarray, scores: np.ndarray) -> int:
    """Return the x-value (``ks``) at the knee/elbow of the curve.

    The knee is the point of maximum perpendicular distance from the straight
    line connecting the first and last points. This locates the corner of an
    "L"-shaped curve regardless of whether the curve rises or falls, and
    degrades gracefully to the middle candidate for flat or straight curves
    where no distinct elbow exists.

    Parameters
    ----------
    ks : array-like
        Curve x-values (e.g. candidate cluster counts), sorted ascending.
    scores : array-like
        Curve y-values (e.g. spatial coherence at each K). Same length as ``ks``.

    Returns
    -------
    int
        The ``ks`` value at the detected knee.
    """
    ks = np.asarray(ks, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(ks)

    if n != len(scores):
        raise ValueError("ks and scores must have the same length")
    if n == 0:
        raise ValueError("ks and scores must be non-empty")
    if n < 3:
        # Not enough points to define an elbow; take the middle candidate.
        return int(round(ks[n // 2]))

    # Normalize both axes to [0, 1] so the distance is scale-invariant.
    x = ks - ks.min()
    x_range = x.max()
    if x_range > 0:
        x = x / x_range

    y = scores - scores.min()
    y_range = y.max()
    if y_range < 1e-12:
        # Flat curve: no knee.
        return int(round(ks[n // 2]))
    y = y / y_range

    # Perpendicular distance of each point from the chord (x0, y0)-(x1, y1).
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = float(np.hypot(dx, dy))
    if denom < 1e-12:
        return int(round(ks[n // 2]))

    dist = np.abs(dy * (x - x0) - dx * (y - y0)) / denom
    if float(dist.max()) < 1e-9:
        # Straight line: no distinct elbow.
        return int(round(ks[n // 2]))

    return int(round(ks[int(np.argmax(dist))]))
