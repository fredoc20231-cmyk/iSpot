"""Unit tests for the direction-agnostic knee/elbow finder."""
import pytest

from ispot.knee import find_knee


def test_fewer_than_three_points_returns_middle():
    assert find_knee([5], [0.9]) == 5
    # n=2 -> middle index n//2 = 1
    assert find_knee([2, 3], [1.0, 0.5]) == 3


def test_flat_curve_returns_middle():
    ks = [2, 3, 4, 5, 6]
    scores = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert find_knee(ks, scores) == 4  # ks[n//2] with n=5


def test_straight_line_returns_middle():
    # Perfectly collinear points: every point lies on the endpoint chord,
    # so there is no distinct elbow and we fall back to the middle.
    ks = [2, 3, 4, 5, 6]
    scores = [1.0, 0.8, 0.6, 0.4, 0.2]
    assert find_knee(ks, scores) == 4


def test_decreasing_elbow_is_interior():
    # Steep drop then plateau: elbow must be strictly between the endpoints,
    # NOT collapsed to the smallest candidate (the previous heuristic's bug).
    ks = [2, 3, 4, 5, 6, 7, 8]
    scores = [1.0, 0.4, 0.15, 0.12, 0.10, 0.09, 0.08]
    knee = find_knee(ks, scores)
    assert ks[0] < knee < ks[-1]


def test_increasing_elbow_is_interior():
    # Steep rise then plateau (the shape the module docstring describes).
    ks = [2, 3, 4, 5, 6, 7, 8]
    scores = [0.10, 0.60, 0.85, 0.88, 0.90, 0.91, 0.92]
    knee = find_knee(ks, scores)
    assert ks[0] < knee < ks[-1]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        find_knee([1, 2, 3], [1.0, 2.0])


def test_empty_raises():
    with pytest.raises(ValueError):
        find_knee([], [])
