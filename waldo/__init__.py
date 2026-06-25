"""
WALDO: Wasserstein-Aligned Localisation via Differential Observations

A training-free framework for zero-shot anomaly localisation using
vision-language models with optimal transport-based reference selection.

Components:
-----------
- waldo: Main WALDO class for inference
- reference_selector: Wasserstein-based reference selection
- metrics: Evaluation metrics (IoU, mAP)
- data_loader: Dataset loaders for NOVA and VinDr-CXR
- preprocessing: Image preprocessing and coordinate transformations
- visualization: Bounding box visualization and figure generation
- prompting: Prompt templates and VLM-specific optimizations
- batch_inference: Batch processing with checkpointing

Example:
--------
>>> from waldo import WALDO, get_dataloader
>>> from waldo.prompting import PromptBuilder
>>>
>>> # Load dataset
>>> loader = get_dataloader("nova")
>>> sample = loader.get_sample(0)
>>> healthy_refs = loader.get_healthy_samples(50)
>>>
>>> # Initialize WALDO
>>> waldo = WALDO(vlm_client=my_client, model="gpt-4o")
>>>
>>> # Run inference
>>> result = waldo.localize(
...     query_image=sample.image,
...     reference_pool=[ref.image for ref in healthy_refs],
...     modality="mri"
... )
"""

__version__ = "1.0.0"
__author__ = "WALDO Team"
__email__ = "bernhard.kainz@fau.de"

# Core components
from .waldo import WALDO
from .reference_selector import WassersteinReferenceSelector
from .metrics import compute_iou, compute_map, compute_confidence_interval

# Data handling
from .data_loader import (
    get_dataloader,
    NOVADataLoader,
    VinDrCXRDataLoader,
    DataSample
)

# Preprocessing
from .preprocessing import (
    ImagePreprocessor,
    CoordinateTransformer,
    DINOv3FeatureExtractor,
    DINOv2FeatureExtractor,  # backwards-compatible alias of DINOv3FeatureExtractor
    resize_with_boxes
)

# Visualization
from .visualization import (
    BoundingBoxVisualizer,
    InteractiveVisualizer,
    save_results_figure,
    create_method_overview_figure
)

# Prompting
from .prompting import (
    PromptTemplate,
    PromptStrategy,
    PromptBuilder,
    ModelSpecificPrompts
)

# Batch processing
from .batch_inference import (
    BatchProcessor,
    BatchConfig,
    ExperimentTracker,
    ProgressiveEvaluator,
    RateLimiter,
    parallel_process
)

__all__ = [
    # Core
    "WALDO",
    "WassersteinReferenceSelector",

    # Metrics
    "compute_iou",
    "compute_map",
    "compute_confidence_interval",

    # Data loading
    "get_dataloader",
    "NOVADataLoader",
    "VinDrCXRDataLoader",
    "DataSample",

    # Preprocessing
    "ImagePreprocessor",
    "CoordinateTransformer",
    "DINOv3FeatureExtractor",
    "DINOv2FeatureExtractor",
    "resize_with_boxes",

    # Visualization
    "BoundingBoxVisualizer",
    "InteractiveVisualizer",
    "save_results_figure",
    "create_method_overview_figure",

    # Prompting
    "PromptTemplate",
    "PromptStrategy",
    "PromptBuilder",
    "ModelSpecificPrompts",

    # Batch processing
    "BatchProcessor",
    "BatchConfig",
    "ExperimentTracker",
    "ProgressiveEvaluator",
    "RateLimiter",
    "parallel_process",
]
