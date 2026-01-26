"""
WALDO: Main inference class for anomaly localisation.

This module implements the complete WALDO pipeline including:
1. Reference selection via Wasserstein distance
2. Differential prompting with VLM
3. Self-consistency aggregation via weighted NMS
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from .reference_selector import WassersteinReferenceSelector


class WALDO:
    """
    WALDO: Wasserstein-Aligned Localisation via Differential Observations.

    A training-free framework for zero-shot medical anomaly localisation
    using vision-language models with optimal transport-based reference selection.

    Example:
        >>> waldo = WALDO(vlm_client=my_vlm_client, model="qwen2.5-vl-72b")
        >>> boxes = waldo.localize(query_image, reference_pool)
        >>> print(f"Found {len(boxes)} anomaly regions")
    """

    def __init__(
        self,
        vlm_client: Any,
        model: str = "qwen2.5-vl-72b",
        n_views: int = 5,
        n_references: int = 3,
        device: str = "cuda",
    ):
        """
        Initialize WALDO.

        Args:
            vlm_client: Vision-language model client (OpenAI-compatible API)
            model: VLM model name
            n_views: Number of self-consistency views
            n_references: Number of references per view
            device: Device for DINOv2 embeddings
        """
        self.vlm_client = vlm_client
        self.model = model
        self.n_views = n_views
        self.n_references = n_references

        self.reference_selector = WassersteinReferenceSelector(device=device)

    def _create_differential_prompt(self, modality: str = "mri") -> str:
        """Create the differential prompting template."""
        if modality.lower() == "mri":
            return """You are a medical imaging expert. Compare the QUERY image (first) with the REFERENCE image(s) (subsequent).

Task: Identify regions in the QUERY that appear DIFFERENT from the healthy reference.

Instructions:
1. Look for intensity differences, mass effects, or abnormal structures
2. The reference shows normal anatomy - use it to identify deviations
3. Return bounding boxes in normalized 0-1000 coordinates

Return JSON: {"boxes": [[x1, y1, x2, y2], ...], "description": "brief finding"}
If no abnormalities: {"boxes": [], "description": "no significant findings"}"""
        else:  # CXR
            return """You are a radiology expert. Compare the QUERY chest X-ray (first) with the healthy REFERENCE (subsequent).

Task: Identify pathological regions in the QUERY that differ from healthy anatomy.

Instructions:
1. Look for consolidations, nodules, cardiomegaly, effusions, etc.
2. The reference shows normal chest anatomy
3. Return bounding boxes in normalized 0-1000 coordinates

Return JSON: {"boxes": [[x1, y1, x2, y2], ...], "description": "brief finding"}
If no abnormalities: {"boxes": [], "description": "no significant findings"}"""

    def _parse_vlm_response(self, response: str) -> List[List[float]]:
        """Parse bounding boxes from VLM response."""
        boxes = []

        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                box_list = data.get('boxes', data.get('confirmed_boxes', []))
                if isinstance(box_list, list):
                    for box in box_list:
                        if isinstance(box, list) and len(box) >= 4:
                            # Validate and normalize coordinates
                            coords = [float(c) for c in box[:4]]
                            # Auto-detect coordinate format
                            max_coord = max(coords)
                            if max_coord <= 1.0:
                                # 0-1 format, scale to 0-1000
                                coords = [c * 1000 for c in coords]
                            elif max_coord > 1000:
                                # Pixel coordinates > 1000, normalize
                                pass  # Keep as-is for now

                            boxes.append(coords)
            except json.JSONDecodeError:
                pass

        return boxes

    def _weighted_nms(
        self,
        boxes: List[List[float]],
        confidences: List[float],
        iou_threshold: float = 0.3,
    ) -> List[List[float]]:
        """
        Apply weighted non-maximum suppression for self-consistency aggregation.

        Args:
            boxes: List of [x1, y1, x2, y2] boxes
            confidences: Confidence scores for each box
            iou_threshold: IoU threshold for merging

        Returns:
            Aggregated boxes after NMS
        """
        if not boxes:
            return []

        boxes = np.array(boxes)
        confidences = np.array(confidences)

        # Sort by confidence
        indices = np.argsort(confidences)[::-1]
        boxes = boxes[indices]
        confidences = confidences[indices]

        keep = []
        while len(boxes) > 0:
            # Keep highest confidence box
            keep.append(boxes[0].tolist())

            if len(boxes) == 1:
                break

            # Compute IoU with remaining boxes
            ious = self._compute_ious(boxes[0], boxes[1:])

            # Remove overlapping boxes
            mask = ious < iou_threshold
            boxes = boxes[1:][mask]
            confidences = confidences[1:][mask]

        return keep

    def _compute_ious(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = area1 + area2 - inter
        return inter / (union + 1e-8)

    def localize(
        self,
        query_image: np.ndarray,
        reference_pool: List[np.ndarray],
        modality: str = "mri",
    ) -> Dict[str, Any]:
        """
        Localize anomalies in the query image.

        Args:
            query_image: Query image as RGB numpy array (H, W, 3)
            reference_pool: Pool of healthy reference images
            modality: Image modality ("mri" or "cxr")

        Returns:
            Dictionary with:
                - boxes: List of predicted bounding boxes [[x1,y1,x2,y2], ...]
                - confidences: Confidence scores for each box
                - raw_responses: Raw VLM responses for each view
        """
        all_boxes = []
        all_confidences = []
        raw_responses = []

        prompt = self._create_differential_prompt(modality)

        for view_idx in range(self.n_views):
            # Select references for this view
            ref_indices = self.reference_selector.select_references(
                query_image,
                reference_pool,
                n_references=self.n_references,
            )

            # Prepare images for VLM (query + references)
            images = [query_image] + [reference_pool[i] for i in ref_indices]

            # Call VLM
            response = self._call_vlm(images, prompt)
            raw_responses.append(response)

            # Parse boxes
            boxes = self._parse_vlm_response(response)

            # Add boxes with confidence (based on view index)
            confidence = 1.0 - (view_idx * 0.1)  # Decay with view
            for box in boxes:
                all_boxes.append(box)
                all_confidences.append(confidence)

        # Apply weighted NMS for self-consistency
        final_boxes = self._weighted_nms(all_boxes, all_confidences)

        return {
            "boxes": final_boxes,
            "confidences": all_confidences[:len(final_boxes)],
            "raw_responses": raw_responses,
        }

    def _call_vlm(self, images: List[np.ndarray], prompt: str) -> str:
        """Call the VLM with images and prompt."""
        # This is a placeholder - implement based on your VLM client
        # Example for OpenAI-compatible API:
        raise NotImplementedError(
            "Implement _call_vlm based on your VLM client. "
            "See examples/run_inference.py for reference."
        )
