"""
WALDO: Wasserstein-Aligned Localisation via Differential Observations

A training-free framework for zero-shot anomaly localisation using
vision-language models with optimal transport-based reference selection.
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

from .reference_selector import WassersteinReferenceSelector
from .waldo import WALDO
from .metrics import compute_iou, compute_map

__all__ = [
    "WALDO",
    "WassersteinReferenceSelector",
    "compute_iou",
    "compute_map",
]
