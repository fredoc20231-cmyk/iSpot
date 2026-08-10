"""Unit tests for method availability (dependency-injected, no heavy backends)."""
from ispot.method_availability import (
    check_method, availability_matrix, available_methods, METHOD_REQUIREMENTS,
)

ALL = list(METHOD_REQUIREMENTS.keys())


def _has(mods):
    """Build a has_module predicate that reports only `mods` as installed."""
    installed = set(mods)
    return lambda name: name in installed


def test_leiden_available_with_only_scanpy():
    info = check_method("Leiden_PCA", has_module=_has({"scanpy"}), has_rscript=lambda: False, env={})
    assert info["available"] is True
    assert info["reason"] == ""


def test_torch_method_unavailable_without_torch():
    info = check_method("GraphST", has_module=_has({"scanpy"}), has_rscript=lambda: False, env={})
    assert info["available"] is False
    assert "torch" in info["reason"]


def test_r_method_needs_rscript():
    no_r = check_method("BISON", has_module=_has(set()), has_rscript=lambda: False, env={})
    yes_r = check_method("BISON", has_module=_has(set()), has_rscript=lambda: True, env={})
    assert no_r["available"] is False and "Rscript" in no_r["reason"]
    assert yes_r["available"] is True


def test_model_dir_requirement(tmp_path):
    # SCOIGET needs torch+pyensembl AND a model dir via ISPOT_SCOIGET_DIR.
    mods = _has({"torch", "pyensembl"})
    missing_dir = check_method("SCOIGET", has_module=mods, has_rscript=lambda: False, env={})
    assert missing_dir["available"] is False and "ISPOT_SCOIGET_DIR" in missing_dir["reason"]
    ok = check_method(
        "SCOIGET", has_module=mods, has_rscript=lambda: False,
        env={"ISPOT_SCOIGET_DIR": str(tmp_path)},
    )
    assert ok["available"] is True


def test_available_methods_filters():
    only_light = _has({"scanpy"})
    runnable = available_methods(ALL, has_module=only_light, has_rscript=lambda: False, env={})
    assert runnable == ["Leiden_PCA"]


def test_matrix_covers_all_methods():
    matrix = availability_matrix(ALL, has_module=_has(set()), has_rscript=lambda: False, env={})
    assert set(matrix.keys()) == set(ALL)
    assert all("available" in v for v in matrix.values())
