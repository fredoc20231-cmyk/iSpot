"""
ispot: Standardized spatial transcriptomics clustering benchmark harness.

Provides:
  - ispot.metrics: ARI/F1 with Hungarian label alignment
  - ispot.preprocessing: Standardized preprocessing pipeline
  - ispot.loaders: One loader per dataset (DLPFC, HER2+, MOSTA, Slide-seqV2)
  - ispot.registry: Method registry mapping names to runner functions
  - ispot.runner: Traceable benchmark runner (method/seed/dataset/timestamp)
  - ispot.methods: One module per method (12 methods total)
"""
__version__ = "1.0.0"
