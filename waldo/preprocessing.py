"""
Preprocessing utilities for WALDO.

Includes image normalization, augmentation, and coordinate transformations.
"""

import numpy as np
from typing import Tuple, List, Optional
from PIL import Image


class ImagePreprocessor:
    """
    Comprehensive image preprocessing for medical images.

    Handles normalization, resizing, and standardization for both
    MRI and CXR modalities.
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        modality: str = "mri"
    ):
        """
        Initialize preprocessor.

        Args:
            target_size: Target (height, width) or None to keep original
            normalize: Whether to normalize to [0, 1]
            modality: Image modality ("mri" or "cxr")
        """
        self.target_size = target_size
        self.normalize = normalize
        self.modality = modality.lower()

    def preprocess(
        self,
        image: np.ndarray,
        preserve_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        Preprocess image.

        Args:
            image: Input image as numpy array (H, W, 3)
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing

        Returns:
            Preprocessed image
        """
        img = image.copy()

        # Normalize to [0, 1] if needed
        if self.normalize and img.max() > 1.0:
            img = img.astype(np.float32) / 255.0

        # Resize if needed
        if self.target_size is not None:
            img = self._resize(img, self.target_size, preserve_aspect_ratio)

        # Modality-specific preprocessing
        if self.modality == "mri":
            img = self._preprocess_mri(img)
        elif self.modality == "cxr":
            img = self._preprocess_cxr(img)

        return img

    def _resize(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        preserve_aspect_ratio: bool
    ) -> np.ndarray:
        """Resize image with optional aspect ratio preservation."""
        h, w = image.shape[:2]
        target_h, target_w = target_size

        if preserve_aspect_ratio:
            # Calculate scaling factor
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
            img_resized = np.array(img_pil).astype(np.float32) / 255.0

            # Pad to target size
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            img_padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
            img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized

            return img_padded
        else:
            # Direct resize
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            img_pil = img_pil.resize((target_w, target_h), Image.LANCZOS)
            return np.array(img_pil).astype(np.float32) / 255.0

    def _preprocess_mri(self, image: np.ndarray) -> np.ndarray:
        """MRI-specific preprocessing."""
        # Histogram equalization can help with MRI contrast
        # But keep it simple for now to preserve original intensities
        return image

    def _preprocess_cxr(self, image: np.ndarray) -> np.ndarray:
        """CXR-specific preprocessing."""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # is commonly used for CXR, but we'll keep it simple
        return image


class CoordinateTransformer:
    """
    Handle coordinate transformations between different formats.

    Supports:
    - Pixel coordinates (absolute)
    - Normalized 0-1 coordinates
    - Normalized 0-1000 coordinates (VLM standard)
    """

    @staticmethod
    def normalize_boxes(
        boxes: List[List[float]],
        image_size: Tuple[int, int],
        target_range: int = 1000
    ) -> List[List[float]]:
        """
        Normalize boxes from pixel coordinates to 0-target_range.

        Args:
            boxes: List of boxes in pixel coords [[x1, y1, x2, y2], ...]
            image_size: (height, width) of image
            target_range: Target coordinate range (1000 for VLMs, 1 for normalized)

        Returns:
            Normalized boxes
        """
        h, w = image_size
        normalized = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1_norm = (x1 / w) * target_range
            y1_norm = (y1 / h) * target_range
            x2_norm = (x2 / w) * target_range
            y2_norm = (y2 / h) * target_range
            normalized.append([x1_norm, y1_norm, x2_norm, y2_norm])

        return normalized

    @staticmethod
    def denormalize_boxes(
        boxes: List[List[float]],
        image_size: Tuple[int, int],
        source_range: int = 1000
    ) -> List[List[float]]:
        """
        Denormalize boxes from 0-source_range to pixel coordinates.

        Args:
            boxes: List of normalized boxes
            image_size: (height, width) of image
            source_range: Source coordinate range (1000 for VLMs, 1 for normalized)

        Returns:
            Boxes in pixel coordinates
        """
        h, w = image_size
        pixel_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1_pix = (x1 / source_range) * w
            y1_pix = (y1 / source_range) * h
            x2_pix = (x2 / source_range) * w
            y2_pix = (y2 / source_range) * h
            pixel_boxes.append([x1_pix, y1_pix, x2_pix, y2_pix])

        return pixel_boxes

    @staticmethod
    def auto_detect_and_normalize(
        boxes: List[List[float]],
        image_size: Tuple[int, int]
    ) -> List[List[float]]:
        """
        Auto-detect coordinate format and normalize to pixel coordinates.

        This handles the inconsistent coordinate formats from different VLMs:
        - 0-1 normalized: scale by image size
        - 0-1000 normalized: scale by image size / 1000
        - Pixel coordinates: use as-is

        Args:
            boxes: Boxes in unknown format
            image_size: (height, width) of image

        Returns:
            Boxes in pixel coordinates
        """
        if not boxes:
            return []

        # Flatten all coordinates to check range
        all_coords = [c for box in boxes for c in box]
        max_coord = max(all_coords)
        min_coord = min(all_coords)

        h, w = image_size

        normalized_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box

            if max_coord <= 1.0:
                # 0-1 normalized format
                x1 = x1 * w
                y1 = y1 * h
                x2 = x2 * w
                y2 = y2 * h
            elif max_coord > 500:
                # 0-1000 normalized format
                x1 = (x1 / 1000) * w
                y1 = (y1 / 1000) * h
                x2 = (x2 / 1000) * w
                y2 = (y2 / 1000) * h
            # else: already pixel coordinates

            normalized_boxes.append([x1, y1, x2, y2])

        return normalized_boxes

    @staticmethod
    def clip_boxes(
        boxes: List[List[float]],
        image_size: Tuple[int, int]
    ) -> List[List[float]]:
        """
        Clip boxes to image boundaries.

        Args:
            boxes: Boxes in pixel coordinates
            image_size: (height, width) of image

        Returns:
            Clipped boxes
        """
        h, w = image_size
        clipped = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            clipped.append([x1, y1, x2, y2])

        return clipped


class DINOv3FeatureExtractor:
    """
    Extract DINOv3-ViT-B/16 patch features for reference selection.

    This mirrors the backbone used in the paper (DINOv3-ViT-B/16, 768-d patch
    tokens). It is a thin convenience wrapper around the same loading/extraction
    logic as :class:`waldo.reference_selector.WassersteinReferenceSelector`. DINOv3
    is gated on the HuggingFace Hub; if it cannot be loaded this RAISES with setup
    instructions, unless ``allow_dinov2_fallback=True`` is passed (non-paper).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str = "cuda",
        feature_size: int = 512,
        allow_dinov2_fallback: bool = False,
    ):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.feature_size = feature_size
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model_name = model_name
        except Exception as exc:
            fallback = "facebook/dinov2-base"
            if not allow_dinov2_fallback:
                raise RuntimeError(
                    f"Could not load the paper backbone DINOv3 ('{model_name}'): {exc}\n"
                    f"DINOv3 is gated; accept its licence and run `huggingface-cli login`, "
                    f"or set allow_dinov2_fallback=True for a non-paper smoke test."
                ) from exc
            print(f"[WALDO] WARNING: DINOv3 '{model_name}' unavailable ({exc}); using "
                  f"NON-PAPER backbone '{fallback}' (allow_dinov2_fallback=True).")
            self.processor = AutoImageProcessor.from_pretrained(fallback)
            self.model = AutoModel.from_pretrained(fallback)
            self.model_name = fallback
        self.model = self.model.to(device)
        self.model.eval()
        self.num_prefix_tokens = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)

    def extract_features(
        self,
        image: np.ndarray,
        return_cls_token: bool = False
    ) -> np.ndarray:
        """Extract CLS token or patch tokens ((N_patches, dim)) from an RGB image."""
        import torch
        img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        inputs = self.processor(
            images=img,
            return_tensors="pt",
            size={"height": self.feature_size, "width": self.feature_size},
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        tokens = outputs.last_hidden_state[0]
        if return_cls_token:
            return tokens[0].float().cpu().numpy()
        return tokens[self.num_prefix_tokens:, :].float().cpu().numpy()

    @staticmethod
    def compute_entropy_weights(features: np.ndarray) -> np.ndarray:
        """Per-patch softmax-Shannon entropy weights (paper Eq.), normalised to sum 1."""
        from .reference_selector import WassersteinReferenceSelector
        return WassersteinReferenceSelector.compute_entropy_weights(features)


# Backwards-compatible alias (the demo previously exported a DINOv2 extractor).
DINOv2FeatureExtractor = DINOv3FeatureExtractor


def resize_with_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    target_size: Tuple[int, int]
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Resize image and corresponding bounding boxes.

    Args:
        image: Input image (H, W, 3)
        boxes: Boxes in pixel coordinates
        target_size: Target (height, width)

    Returns:
        Tuple of (resized_image, resized_boxes)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Resize image
    img_pil = Image.fromarray((image * 255).astype(np.uint8))
    img_pil = img_pil.resize((target_w, target_h), Image.LANCZOS)
    img_resized = np.array(img_pil).astype(np.float32) / 255.0

    # Scale boxes
    scale_x = target_w / w
    scale_y = target_h / h

    boxes_resized = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_resized.append([
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        ])

    return img_resized, boxes_resized
