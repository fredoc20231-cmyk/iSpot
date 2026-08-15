"""Domain-map diagnostics.

Answers the question that keeps coming up when looking at the viewer: is a
method's spatial-domain assignment genuinely coherent, or is it (a)
salt-and-pepper scatter, or (b) trivially explained by a single coordinate axis
(horizontal/vertical bands that ignore the tissue morphology)?

These run on the SAME aligned (coords, labels) the viewer renders, so the flag
describes exactly what the user sees — turning "the domains look wrong" into an
explicit, quantified signal next to ARI. Pure numpy + scipy (no scanpy), so it
runs anywhere the core stack is installed.
"""
from __future__ import annotations

import numpy as np


def _correlation_ratio(categories, values) -> float:
    """eta = sqrt(SS_between / SS_total) for a categorical -> continuous relation.

    ~1.0 means the category is almost fully determined by ``values`` (e.g. labels
    that are just horizontal bands of the y coordinate); ~0 means the two are
    unrelated. This is the standard correlation ratio for a nominal variable
    against a numeric one.
    """
    values = np.asarray(values, dtype=float)
    total_mean = values.mean()
    ss_total = float(((values - total_mean) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    categories = np.asarray(categories)
    ss_between = 0.0
    for c in np.unique(categories):
        grp = values[categories == c]
        ss_between += len(grp) * (grp.mean() - total_mean) ** 2
    return float(np.sqrt(ss_between / ss_total))


def spatial_label_coherence(coords, labels, k: int = 6) -> float:
    """Mean fraction of each spot's k nearest spatial neighbours sharing its label.

    ~1.0 = contiguous domains; low (≈ 1/n_domains) = salt-and-pepper scatter,
    the signature of labels applied to the wrong spots.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)
    if n < 2:
        return 1.0
    k = min(k, n - 1)
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # k+1 because the first hit is self
    idx = np.atleast_2d(idx)
    neigh = idx[:, 1:]  # drop the self column
    same = labels[neigh] == labels[:, None]
    return float(same.mean())


def diagnose_domains(coords, labels, k: int = 6,
                     coherence_floor: float = 0.5,
                     axis_ceiling: float = 0.9) -> dict:
    """Classify a domain map.

    Returns spatial coherence, per-axis correlation ratios, domain count, and a
    human-readable ``flag``:

      - ``"degenerate"``      : fewer than 2 distinct domains
      - ``"salt-and-pepper"`` : spatially incoherent (labels don't form regions)
      - ``"axis-banded"``     : coherent but ~determined by one coordinate axis
      - ``"coherent"``        : contiguous regions not explained by a single axis
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray([str(x) for x in np.asarray(labels).ravel()])

    # Ignore unassigned spots so they don't skew the measures.
    keep = labels != "unassigned"
    if keep.sum() >= 2:
        coords, labels = coords[keep], labels[keep]

    n_domains = int(len(np.unique(labels)))
    coherence = spatial_label_coherence(coords, labels, k=k)
    eta_x = _correlation_ratio(labels, coords[:, 0])
    eta_y = _correlation_ratio(labels, coords[:, 1])
    max_eta = max(eta_x, eta_y)
    dominant_axis = "y" if eta_y >= eta_x else "x"

    if n_domains < 2:
        flag = "degenerate"
    elif coherence < coherence_floor:
        flag = "salt-and-pepper"
    elif max_eta >= axis_ceiling:
        flag = "axis-banded"
    else:
        flag = "coherent"

    return {
        "spatial_coherence": round(coherence, 4),
        "axis_eta_x": round(eta_x, 4),
        "axis_eta_y": round(eta_y, 4),
        "dominant_axis": dominant_axis,
        "n_domains": n_domains,
        "flag": flag,
    }
