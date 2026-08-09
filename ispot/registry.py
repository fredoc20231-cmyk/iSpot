"""
Method registry: maps method names to their runner modules.

12 methods total:
  Baselines (5): Leiden_PCA, SpaGCN, STAGATE, GraphST, BayesSpace
  New methods (7): HyperGCN, STMSGAL, SCOIGET, Novae, BISON, SpaRTaCo, spatialMNN

Stochastic methods (need seed control): leiden_pca, spagcn, stagate, graphst,
  hypergcn, stmsgal, scoiget, bison, spartaco, novae (from_scratch/fine_tuned),
  bayesspace
Deterministic: spatialmnn, novae (zero_shot)
"""
from ispot.methods import (
    leiden_pca, spagcn, stagate, graphst,
    hypergcn, stmsgal, scoiget, novae,
    bayesspace, bison, spartaco, spatialmnn,
)

# Registry: method_name -> (module, is_stochastic, is_r_based)
METHOD_REGISTRY = {
    # Baselines
    "Leiden_PCA": (leiden_pca, True, False),
    "SpaGCN": (spagcn, True, False),
    "STAGATE": (stagate, True, False),
    "GraphST": (graphst, True, False),
    "BayesSpace": (bayesspace, True, True),
    # New methods
    "HyperGCN": (hypergcn, True, False),
    "STMSGAL": (stmsgal, True, False),
    "SCOIGET": (scoiget, True, False),
    "Novae": (novae, True, False),
    "BISON": (bison, True, True),
    "SpaRTaCo": (spartaco, True, True),
    "spatialMNN": (spatialmnn, False, True),
}

# Method display names for figures
METHOD_DISPLAY = {
    "Leiden_PCA": "Leiden/PCA",
    "SpaGCN": "SpaGCN",
    "STAGATE": "STAGATE",
    "GraphST": "GraphST",
    "BayesSpace": "BayesSpace",
    "HyperGCN": "HyperGCN",
    "STMSGAL": "STMSGAL",
    "SCOIGET": "SCOIGET",
    "Novae": "Novae",
    "BISON": "BISON",
    "SpaRTaCo": "SpaRTaCo",
    "spatialMNN": "spatialMNN",
}

# Category for each method
METHOD_CATEGORY = {
    "Leiden_PCA": "baseline",
    "SpaGCN": "baseline",
    "STAGATE": "baseline",
    "GraphST": "baseline",
    "BayesSpace": "baseline",
    "HyperGCN": "new",
    "STMSGAL": "new",
    "SCOIGET": "new",
    "Novae": "new",
    "BISON": "new",
    "SpaRTaCo": "new",
    "spatialMNN": "new",
}

# Consistent color mapping for all figures
METHOD_COLORS = {
    "Leiden_PCA": "#1f77b4",
    "SpaGCN": "#ff7f0e",
    "STAGATE": "#2ca02c",
    "GraphST": "#d62728",
    "BayesSpace": "#9467bd",
    "HyperGCN": "#8c564b",
    "STMSGAL": "#e377c2",
    "SCOIGET": "#7f7f7f",
    "Novae": "#bcbd22",
    "BISON": "#17becf",
    "SpaRTaCo": "#aec7e8",
    "spatialMNN": "#ffbb78",
}

BASELINE_METHODS = ["Leiden_PCA", "SpaGCN", "STAGATE", "GraphST", "BayesSpace"]
NEW_METHODS = ["HyperGCN", "STMSGAL", "SCOIGET", "Novae", "BISON", "SpaRTaCo", "spatialMNN"]
ALL_METHODS = BASELINE_METHODS + NEW_METHODS


def get_runner(method_name):
    """Get the run() function for a method."""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHOD_REGISTRY.keys())}")
    module, _, _ = METHOD_REGISTRY[method_name]
    return module.run


def is_stochastic(method_name):
    """Check if a method is stochastic (needs seed control)."""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}")
    return METHOD_REGISTRY[method_name][1]


def is_r_based(method_name):
    """Check if a method is R-based."""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}")
    return METHOD_REGISTRY[method_name][2]
