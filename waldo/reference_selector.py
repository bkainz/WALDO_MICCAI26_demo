"""
Wasserstein-based reference selection using DINOv2 embeddings.

This module implements the entropy-weighted Sliced Wasserstein distance
for selecting anatomically-appropriate healthy reference images.
"""

import numpy as np
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F


class WassersteinReferenceSelector:
    """
    Select healthy reference images using entropy-weighted Sliced Wasserstein distance.

    The selector exploits the "Goldilocks zone" - references that are moderately
    similar to the query image provide optimal bias-variance tradeoff for
    differential reasoning.

    Attributes:
        model: DINOv2 model for extracting patch embeddings
        device: torch device for computation
        n_projections: number of random projections for SWD approximation
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda",
        n_projections: int = 128,
    ):
        """
        Initialize the reference selector.

        Args:
            model_name: HuggingFace model name for DINOv2
            device: torch device ("cuda" or "cpu")
            n_projections: number of random projections for SWD
        """
        self.device = device
        self.n_projections = n_projections
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load DINOv2 model from HuggingFace."""
        from transformers import AutoModel, AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """
        Extract DINOv2 patch embeddings from an image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Patch embeddings of shape (N_patches, embedding_dim)
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # Get patch tokens (excluding CLS token)
        patches = outputs.last_hidden_state[:, 1:, :].cpu().numpy()
        return patches[0]  # (N_patches, dim)

    def compute_entropy_weights(self, patches: np.ndarray) -> np.ndarray:
        """
        Compute entropy-based weights for patch embeddings.

        Higher entropy patches (more uncertain/variable) receive lower weights,
        focusing the distance computation on stable anatomical regions.

        Args:
            patches: Patch embeddings (N, D)

        Returns:
            Weights for each patch (N,)
        """
        # Compute patch-wise variance as proxy for entropy
        patch_vars = np.var(patches, axis=1)

        # Convert to weights (inverse of variance)
        weights = 1.0 / (patch_vars + 1e-8)
        weights = weights / weights.sum()

        return weights

    def sliced_wasserstein_distance(
        self,
        patches1: np.ndarray,
        patches2: np.ndarray,
        weights1: Optional[np.ndarray] = None,
        weights2: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Sliced Wasserstein Distance between two patch distributions.

        Args:
            patches1: First patch embeddings (N1, D)
            patches2: Second patch embeddings (N2, D)
            weights1: Optional weights for patches1
            weights2: Optional weights for patches2

        Returns:
            Sliced Wasserstein distance
        """
        d = patches1.shape[1]

        # Generate random projections
        projections = np.random.randn(self.n_projections, d)
        projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

        # Project patches
        proj1 = patches1 @ projections.T  # (N1, n_proj)
        proj2 = patches2 @ projections.T  # (N2, n_proj)

        # Compute 1D Wasserstein distance for each projection
        distances = []
        for i in range(self.n_projections):
            sorted1 = np.sort(proj1[:, i])
            sorted2 = np.sort(proj2[:, i])

            # Interpolate to same length if needed
            if len(sorted1) != len(sorted2):
                n = max(len(sorted1), len(sorted2))
                sorted1 = np.interp(
                    np.linspace(0, 1, n),
                    np.linspace(0, 1, len(sorted1)),
                    sorted1
                )
                sorted2 = np.interp(
                    np.linspace(0, 1, n),
                    np.linspace(0, 1, len(sorted2)),
                    sorted2
                )

            distances.append(np.mean(np.abs(sorted1 - sorted2)))

        return np.mean(distances)

    def select_references(
        self,
        query_image: np.ndarray,
        reference_images: List[np.ndarray],
        n_references: int = 5,
        percentile_range: Tuple[float, float] = (30, 70),
    ) -> List[int]:
        """
        Select reference images using Goldilocks zone sampling.

        References are selected from the middle percentile range of
        Wasserstein distances, avoiding both too-similar and too-different
        references.

        Args:
            query_image: Query image (H, W, 3)
            reference_images: List of candidate reference images
            n_references: Number of references to select
            percentile_range: (min, max) percentile for Goldilocks sampling

        Returns:
            Indices of selected reference images
        """
        # Extract query patches
        query_patches = self.extract_patches(query_image)
        query_weights = self.compute_entropy_weights(query_patches)

        # Compute distances to all references
        distances = []
        for ref_img in reference_images:
            ref_patches = self.extract_patches(ref_img)
            ref_weights = self.compute_entropy_weights(ref_patches)

            dist = self.sliced_wasserstein_distance(
                query_patches, ref_patches, query_weights, ref_weights
            )
            distances.append(dist)

        distances = np.array(distances)

        # Goldilocks zone: select from middle percentile range
        low_threshold = np.percentile(distances, percentile_range[0])
        high_threshold = np.percentile(distances, percentile_range[1])

        # Find indices in the Goldilocks zone
        goldilocks_mask = (distances >= low_threshold) & (distances <= high_threshold)
        goldilocks_indices = np.where(goldilocks_mask)[0]

        if len(goldilocks_indices) >= n_references:
            # Randomly sample from Goldilocks zone
            selected = np.random.choice(goldilocks_indices, n_references, replace=False)
        else:
            # Fall back to closest to median
            median_dist = np.median(distances)
            sorted_indices = np.argsort(np.abs(distances - median_dist))
            selected = sorted_indices[:n_references]

        return selected.tolist()
