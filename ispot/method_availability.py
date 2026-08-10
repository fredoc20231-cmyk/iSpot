"""
Method availability checking.

Only ``Leiden_PCA`` runs on a clean install; the other 11 methods import heavy
backends (torch, tensorflow, R/Bioconductor, foundation-model packages) or need
local model clones, and otherwise fail at runtime and land in the per-method
error bucket. This module reports — without running anything — which methods can
actually execute in the current environment, so the API can:

  * expose an honest availability matrix (GET /api/methods/availability), and
  * default a benchmark to the methods that will actually work.

The check is dependency-injected (``has_module`` / ``has_rscript``) so it is
deterministic and unit-testable without installing the heavy backends.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from typing import Callable, Optional

# What each method needs to actually run: importable python modules, an R
# interpreter (Rscript), and/or a local model directory pointed to by an env var.
METHOD_REQUIREMENTS: dict[str, dict] = {
    "Leiden_PCA": {"python": ["scanpy"]},
    "SpaGCN": {"python": ["SpaGCN", "torch"]},
    "STAGATE": {"python": ["STAGATE", "tensorflow"]},
    "GraphST": {"python": ["GraphST", "torch"]},
    "BayesSpace": {"rscript": True},
    "HyperGCN": {"python": ["torch"], "path_env": "ISPOT_HYPERGCN_DIR"},
    "STMSGAL": {"python": ["STMSGAL", "torch"], "path_env": "ISPOT_STMSGAL_DIR"},
    "SCOIGET": {"python": ["torch", "pyensembl"], "path_env": "ISPOT_SCOIGET_DIR"},
    "Novae": {"python": ["novae"]},
    "BISON": {"rscript": True},
    "SpaRTaCo": {"rscript": True},
    "spatialMNN": {"rscript": True},
}


def _default_has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _default_has_rscript() -> bool:
    return shutil.which("Rscript") is not None


def check_method(
    name: str,
    has_module: Optional[Callable[[str], bool]] = None,
    has_rscript: Optional[Callable[[], bool]] = None,
    env: Optional[dict] = None,
) -> dict:
    """Return availability info for one method.

    {method, available: bool, reason: str, requirements: dict}
    """
    has_module = has_module or _default_has_module
    has_rscript = has_rscript or _default_has_rscript
    env = env if env is not None else os.environ

    reqs = METHOD_REQUIREMENTS.get(name, {})
    missing: list[str] = []

    for mod in reqs.get("python", []):
        if not has_module(mod):
            missing.append(f"python:{mod}")
    if reqs.get("rscript") and not has_rscript():
        missing.append("Rscript")
    path_env = reqs.get("path_env")
    if path_env:
        path = env.get(path_env)
        if not path or not os.path.isdir(path):
            missing.append(f"model-dir:{path_env}")

    available = len(missing) == 0
    return {
        "method": name,
        "available": available,
        "reason": "" if available else "missing " + ", ".join(missing),
        "requirements": reqs,
    }


def availability_matrix(methods, **kw) -> dict:
    """Availability info keyed by method name."""
    return {m: check_method(m, **kw) for m in methods}


def available_methods(methods, **kw) -> list[str]:
    """Subset of ``methods`` that can actually run now."""
    return [m for m in methods if check_method(m, **kw)["available"]]
