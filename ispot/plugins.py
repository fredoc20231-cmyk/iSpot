"""
Plugin system for community-contributed clustering methods.

A plugin is a Python module that registers a clustering method via the
@register_method decorator. Plugins are discovered from:
  1. Local plugin directory (~/.ispot/plugins/*.py)
  2. Pip-installed packages (Python entry points: ispot.methods group)
  3. Community registry (ispot install <name>)

Section 3 of the platform plan.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import sys
import time
import shutil
import tempfile
import traceback
import numpy as np
import pandas as pd
import anndata as ad
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------

@dataclass
class PluginInfo:
    """Metadata for a registered plugin method."""
    name: str
    category: str               # "baseline" | "new"
    is_stochastic: bool
    is_r_based: bool
    compute_tier: str           # "fast" | "medium" | "heavy" | "very_heavy"
    gpu_optional: bool = False
    min_ispot_version: str = "1.0"
    description: str = ""
    author: str = ""
    version: str = "0.1.0"
    source: str = "builtin"     # "builtin" | "local" | "pip" | "registry"
    package_name: str = ""
    filepath: str = ""          # source file for local plugins (for sandboxing)
    run_func: Callable = field(default=None, repr=False)


# Global registry of all discovered plugins
_PLUGIN_REGISTRY: dict[str, PluginInfo] = {}


# ---------------------------------------------------------------------------
# Registration decorator
# ---------------------------------------------------------------------------

def register_method(
    name: str,
    category: str = "new",
    is_stochastic: bool = True,
    is_r_based: bool = False,
    compute_tier: str = "medium",
    gpu_optional: bool = False,
    min_ispot_version: str = "1.0",
    description: str = "",
    author: str = "",
    version: str = "0.1.0",
):
    """Decorator to register a clustering method as a plugin.

    Usage:
        @register_method(name="MyMethod", compute_tier="fast")
        def run(adata, n_clusters, seed=42, **kwargs):
            # ... clustering logic ...
            return {"labels": labels, "runtime": elapsed, ...}

    The decorated function must accept (adata, n_clusters, seed, **kwargs)
    and return a dict with at minimum:
        - labels: np.ndarray of cluster assignments
        - runtime: float (seconds)
        - n_spots: int
        - n_clusters_pred: int
    """
    def decorator(func: Callable) -> Callable:
        info = PluginInfo(
            name=name,
            category=category,
            is_stochastic=is_stochastic,
            is_r_based=is_r_based,
            compute_tier=compute_tier,
            gpu_optional=gpu_optional,
            min_ispot_version=min_ispot_version,
            description=description,
            author=author,
            version=version,
            run_func=func,
            source="builtin",
        )
        _PLUGIN_REGISTRY[name] = info
        return func
    return decorator


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------

def discover_plugins(plugin_dir: str | None = None) -> dict[str, PluginInfo]:
    """Discover all available plugins.

    Discovery order (later sources override earlier for same name):
    1. Built-in methods (already registered via @register_method)
    2. Local plugin directory (~/.ispot/plugins/*.py)
    3. Pip-installed packages (entry points: ispot.methods)

    Parameters
    ----------
    plugin_dir : str, optional
        Custom plugin directory. Defaults to ~/.ispot/plugins/.

    Returns
    -------
    dict[str, PluginInfo]: all discovered plugins.
    """
    # 1. Built-in methods are already in _PLUGIN_REGISTRY

    # 2. Local plugin directory
    if plugin_dir is None:
        plugin_dir = os.path.expanduser("~/.ispot/plugins")

    if os.path.isdir(plugin_dir):
        for filename in os.listdir(plugin_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                filepath = os.path.join(plugin_dir, filename)
                _load_plugin_file(filepath, source="local")

    # 3. Pip entry points
    try:
        eps = importlib.metadata.entry_points(group="ispot.methods")
        for ep in eps:
            try:
                func = ep.load()
                # The loaded function should already have been registered
                # via @register_method when the module was imported.
                # If not, check if it's a callable with the right signature.
                if ep.name not in _PLUGIN_REGISTRY:
                    # Try to register it manually
                    if callable(func) and _validate_signature(func):
                        _PLUGIN_REGISTRY[ep.name] = PluginInfo(
                            name=ep.name,
                            category="new",
                            is_stochastic=True,
                            is_r_based=False,
                            compute_tier="medium",
                            run_func=func,
                            source="pip",
                            package_name=ep.value.split(":")[0],
                        )
            except Exception as e:
                print(f"Warning: failed to load plugin '{ep.name}': {e}")
    except Exception:
        # entry_points API varies across Python versions
        pass

    return dict(_PLUGIN_REGISTRY)


def _load_plugin_file(filepath: str, source: str = "local"):
    """Load a single plugin .py file."""
    # Ensure the plugin directory is on sys.path so imports work
    plugin_dir = os.path.dirname(filepath)
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    module_name = os.path.splitext(os.path.basename(filepath))[0]
    # Use a unique module name to avoid conflicts
    full_name = f"_ispot_plugin_{module_name}"

    try:
        spec = importlib.util.spec_from_file_location(full_name, filepath)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        # The @register_method decorator in the plugin file will have
        # added entries to _PLUGIN_REGISTRY automatically.
        # Update source for plugins loaded from this file.
        for info in _PLUGIN_REGISTRY.values():
            if info.source == "builtin" and info.run_func.__module__ == full_name:
                info.source = source
                info.filepath = filepath
    except Exception as e:
        print(f"Warning: failed to load plugin '{filepath}': {e}")
        traceback.print_exc()


def _validate_signature(func: Callable) -> bool:
    """Check that a function has the required plugin signature."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    # Must accept at least (adata, n_clusters)
    return len(params) >= 2 and params[0] == "adata" and params[1] == "n_clusters"


# ---------------------------------------------------------------------------
# Plugin validation
# ---------------------------------------------------------------------------

def validate_plugin(info: PluginInfo, synthetic_adata: ad.AnnData | None = None) -> dict:
    """Validate a plugin against the platform's requirements.

    Runs 5 automated tests:
    1. Interface test: correct signature and return dict
    2. Smoke test: runs on synthetic data without crashing
    3. Determinism test: same seed → same output
    4. Runtime test: completes within compute tier time limit
    5. Output validation: labels are non-empty, correct length, non-degenerate

    Parameters
    ----------
    info : PluginInfo
        Plugin to validate.
    synthetic_adata : AnnData, optional
        Test dataset. If None, a 500-spot synthetic Visium dataset is generated.

    Returns
    -------
    dict with keys:
        - passed: bool (all tests passed)
        - tests: dict of {test_name: {passed: bool, detail: str}}
    """
    if synthetic_adata is None:
        synthetic_adata = _generate_synthetic_data()

    results = {"passed": True, "tests": {}}

    # Time limits per compute tier
    TIME_LIMITS = {
        "fast": 60, "medium": 300, "heavy": 3600, "very_heavy": 7200,
    }
    time_limit = TIME_LIMITS.get(info.compute_tier, 300)

    # Test 1: Interface
    try:
        sig = inspect.signature(info.run_func)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, f"Expected >= 2 params, got {len(params)}"
        assert params[0] == "adata", f"First param should be 'adata', got '{params[0]}'"
        assert params[1] == "n_clusters", f"Second param should be 'n_clusters', got '{params[1]}'"
        results["tests"]["interface"] = {"passed": True, "detail": f"Signature: {sig}"}
    except Exception as e:
        results["tests"]["interface"] = {"passed": False, "detail": str(e)}
        results["passed"] = False
        return results  # Can't continue if interface is wrong

    # Test 2: Smoke test
    try:
        t0 = time.time()
        result = info.run_func(synthetic_adata.copy(), n_clusters=5, seed=42)
        elapsed = time.time() - t0

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "labels" in result, "Missing 'labels' in return dict"
        assert "runtime" in result, "Missing 'runtime' in return dict"

        results["tests"]["smoke"] = {
            "passed": True,
            "detail": f"Ran in {elapsed:.1f}s, returned {len(result)} keys",
        }
    except Exception as e:
        results["tests"]["smoke"] = {"passed": False, "detail": str(e)}
        results["passed"] = False
        return results

    # Test 3: Determinism (if stochastic)
    if info.is_stochastic:
        try:
            result1 = info.run_func(synthetic_adata.copy(), n_clusters=5, seed=42)
            result2 = info.run_func(synthetic_adata.copy(), n_clusters=5, seed=42)
            labels1 = np.array(result1["labels"]).astype(str)
            labels2 = np.array(result2["labels"]).astype(str)
            assert np.array_equal(labels1, labels2), "Same seed produced different results"
            results["tests"]["determinism"] = {"passed": True, "detail": "Reproducible with same seed"}
        except AssertionError as e:
            results["tests"]["determinism"] = {"passed": False, "detail": str(e)}
            results["passed"] = False
        except Exception as e:
            results["tests"]["determinism"] = {"passed": False, "detail": f"Error: {e}"}
            results["passed"] = False
    else:
        results["tests"]["determinism"] = {"passed": True, "detail": "Skipped (deterministic method)"}

    # Test 4: Runtime
    if elapsed > time_limit:
        results["tests"]["runtime"] = {
            "passed": False,
            "detail": f"Took {elapsed:.1f}s, limit is {time_limit}s for tier '{info.compute_tier}'",
        }
        results["passed"] = False
    else:
        results["tests"]["runtime"] = {
            "passed": True,
            "detail": f"{elapsed:.1f}s within {time_limit}s limit",
        }

    # Test 5: Output validation
    try:
        labels = np.array(result["labels"]).astype(str)
        n_spots = synthetic_adata.shape[0]
        assert len(labels) == n_spots, f"Labels length {len(labels)} != n_spots {n_spots}"
        assert len(np.unique(labels)) > 1, "All spots in one cluster (degenerate)"
        assert len(np.unique(labels)) <= n_spots, "More clusters than spots"
        assert not np.any(pd.isna(labels)), "NaN in labels"
        results["tests"]["output"] = {
            "passed": True,
            "detail": f"{len(np.unique(labels))} clusters, {len(labels)} spots",
        }
    except Exception as e:
        results["tests"]["output"] = {"passed": False, "detail": str(e)}
        results["passed"] = False

    return results


def _generate_synthetic_data(n_spots: int = 500, n_genes: int = 1000) -> ad.AnnData:
    """Generate a small synthetic Visium-like dataset for plugin validation."""
    import scanpy as sc
    rng = np.random.RandomState(42)

    # Grid layout — make exactly n_spots
    side = int(np.ceil(np.sqrt(n_spots)))
    coords = []
    for i in range(side):
        for j in range(side):
            coords.append((float(i), float(j)))
            if len(coords) >= n_spots:
                break
        if len(coords) >= n_spots:
            break
    coords = np.array(coords[:n_spots])

    # Random counts
    counts = rng.negative_binomial(5, 0.3, size=(n_spots, n_genes)).astype(float)

    adata = ad.AnnData(X=counts)
    adata.obsm["spatial"] = coords
    adata.obs["sample_id"] = "synthetic"
    adata.obs["ground_truth"] = "unknown"
    adata.obs["has_ground_truth"] = False
    adata.uns["platform"] = "Visium"

    # Preprocess
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=500, flavor="seurat_v3", layer="counts")
    sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)

    return adata


# ---------------------------------------------------------------------------
# Plugin runner wrapper
# ---------------------------------------------------------------------------

def run_plugin(info: PluginInfo, adata: ad.AnnData, n_clusters: int, seed: int = 42, **kwargs) -> dict:
    """Run a plugin method and normalize its output.

    Ensures the return dict has all required fields, computing ARI/F1
    if ground truth is available.

    Parameters
    ----------
    info : PluginInfo
    adata : AnnData
    n_clusters : int
    seed : int

    Returns
    -------
    dict with ari, macro_f1, weighted_f1, runtime, n_spots, n_clusters_pred, labels
    """
    from ispot.metrics import compute_metrics

    t0 = time.time()
    result = info.run_func(adata.copy(), n_clusters, seed=seed, **kwargs)
    runtime = result.get("runtime", time.time() - t0)

    labels = np.array(result["labels"]).astype(str)

    # Compute GT metrics if available
    if "has_ground_truth" in adata.obs and adata.obs["has_ground_truth"].any():
        mask = adata.obs["has_ground_truth"].values.astype(bool)
        gt = adata.obs.loc[mask, "ground_truth"].values
        pred = labels[mask]
        m = compute_metrics(gt, pred, runtime)
    else:
        m = {
            "ari": None, "macro_f1": None, "weighted_f1": None,
            "runtime": float(runtime), "n_spots": len(labels),
            "n_clusters_pred": len(np.unique(labels)), "n_clusters_true": None,
        }

    m["labels"] = labels
    m["embedding"] = result.get("embedding", None)
    return m


def run_plugin_sandboxed(
    info: PluginInfo,
    adata: ad.AnnData,
    n_clusters: int,
    seed: int = 42,
    timeout: int | None = None,
    memory_mb: int | None = None,
) -> dict:
    """Run a plugin in an isolated subprocess (see ispot.sandbox).

    Same contract as :func:`run_plugin`, but the plugin executes in a separate
    process with memory/CPU rlimits and a wall-clock timeout, so a misbehaving
    community plugin cannot take down or read the memory of the API process.
    The AnnData is serialized to a temp ``.h5ad`` the child reads. Timeout and
    memory caps default to ISPOT_PLUGIN_TIMEOUT (600s) and ISPOT_PLUGIN_MEM_MB
    (4096).
    """
    from ispot.sandbox import run_in_subprocess
    from ispot.metrics import compute_metrics

    if timeout is None:
        timeout = int(float(os.environ.get("ISPOT_PLUGIN_TIMEOUT", "600")))
    if memory_mb is None:
        memory_mb = int(float(os.environ.get("ISPOT_PLUGIN_MEM_MB", "4096")))

    tmpdir = tempfile.mkdtemp(prefix="ispot-plugin-")
    adata_path = os.path.join(tmpdir, "input.h5ad")
    try:
        adata.write_h5ad(adata_path)
        payload = {
            "name": info.name,
            "filepath": getattr(info, "filepath", "") or "",
            "adata_path": adata_path,
            "n_clusters": n_clusters,
            "seed": seed,
        }
        res = run_in_subprocess(
            "ispot.sandbox:_plugin_entry", payload,
            timeout=timeout, memory_mb=memory_mb,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    labels = np.array(res["labels"]).astype(str)
    runtime = res.get("runtime")

    if "has_ground_truth" in adata.obs and adata.obs["has_ground_truth"].any():
        mask = adata.obs["has_ground_truth"].values.astype(bool)
        gt = adata.obs.loc[mask, "ground_truth"].values
        m = compute_metrics(gt, labels[mask], runtime)
    else:
        m = {
            "ari": None, "macro_f1": None, "weighted_f1": None,
            "runtime": float(runtime) if runtime is not None else None,
            "n_spots": len(labels),
            "n_clusters_pred": len(np.unique(labels)), "n_clusters_true": None,
        }
    m["labels"] = labels
    m["embedding"] = res.get("embedding", None)
    return m


# ---------------------------------------------------------------------------
# Registry queries
# ---------------------------------------------------------------------------

def list_plugins() -> dict[str, PluginInfo]:
    """List all discovered plugins."""
    return dict(_PLUGIN_REGISTRY)


def get_plugin(name: str) -> PluginInfo:
    """Get a plugin by name."""
    if name not in _PLUGIN_REGISTRY:
        raise ValueError(f"Unknown plugin: {name}. Available: {list(_PLUGIN_REGISTRY.keys())}")
    return _PLUGIN_REGISTRY[name]


def get_all_method_names() -> list[str]:
    """Get all method names (built-in + plugins)."""
    return list(_PLUGIN_REGISTRY.keys())
