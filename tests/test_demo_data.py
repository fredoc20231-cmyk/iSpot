"""Tests for the bundled synthetic demo datasets + their SpatialQC verdicts."""
import pytest

np = pytest.importorskip("numpy")
ad = pytest.importorskip("anndata")

from ispot.demo_data import (  # noqa: E402
    list_demos, make_demo, SYNTHETIC_DEMOS, REAL_DEMOS,
)


def test_list_demos_includes_synthetic_and_real():
    demos = list_demos()
    names = {d["name"] for d in demos}
    assert set(SYNTHETIC_DEMOS).issubset(names)     # synthetic always available
    assert all(d["description"] for d in demos)
    # The bundled real breast Visium demo should be present in a full checkout.
    import os
    if os.path.isdir(REAL_DEMOS["visium_breast_092a"]["path"]):
        assert "visium_breast_092a" in names
        assert next(d for d in demos if d["name"] == "visium_breast_092a")["real"] is True


def test_healthy_demo_shape_and_metadata():
    adata = make_demo("visium_healthy_demo")
    assert adata.n_obs > 500                       # a real two-lobe footprint
    assert "spatial" in adata.obsm
    assert adata.uns["platform"] == "Visium"
    # healthy demo ships a synthetic histology image
    lib = next(iter(adata.uns["spatial"].values()))
    assert "hires" in lib["images"]
    assert lib["scalefactors"]["tissue_hires_scalef"] > 0


def test_healthy_demo_passes_spatial_signal():
    from ispot.spatial_qc_report import build_spatial_qc
    adata = make_demo("visium_healthy_demo")
    report = build_spatial_qc(adata, platform="Visium", sample_id="demo",
                              platform_confidence="explicit")
    mods = {m["id"]: m for m in report["modules"]}
    # engineered spatial domains -> real spatial signal + strong SVGs
    assert mods["spatial_signal"]["status"] in ("pass", "warn")
    assert mods["spatial_signal"]["value"] > 0.1
    assert "spatially_variable_genes" in mods
    assert mods["spatially_variable_genes"]["value"] > 0.2


def test_lowquality_demo_raises_flags():
    from ispot.spatial_qc_report import build_spatial_qc
    adata = make_demo("visium_lowquality_demo")
    report = build_spatial_qc(adata, platform="Visium", sample_id="demo",
                              platform_confidence="explicit")
    # shallow + sparse + structureless -> not an all-pass report
    assert report["summary"]["overall"] in ("warn", "fail")


def test_unknown_demo_falls_back_to_healthy():
    adata = make_demo("does-not-exist")
    assert adata.n_obs > 0 and "spatial" in adata.obsm
