"""
WALDO: Wasserstein-Aligned Localisation for VLM-based OOD detection.

Top-level inference pipeline implementing the method from the paper:

  1. Reference selection  -- entropy-weighted Sliced Wasserstein + Goldilocks + DPP
                             (see reference_selector.py).
  2. Differential prompting (Stage 1)  -- compare the query against each of K1
                             selected healthy references; aggregate boxes via
                             confidence-weighted NMS.
  3. Refinement (Stage 2)  -- re-present the Stage-1 candidates against K2 further
                             references to confirm / refine / reject; aggregate again.

Per-reference predictions are down-weighted by their distance to the query:
    c~ = c * exp(-lambda * SW2(P_q, P_h)),  lambda = 0.1
and merged with confidence-weighted NMS at IoU threshold 0.5.

DISCLAIMER: this is a clean reference implementation distilled by Claude and Codex
coding agents from a larger internal experimentation codebase. It faithfully follows
the method and hyper-parameters in the paper but is not the multi-day cluster script
that produced the published tables. See DISCLAIMER.md.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .reference_selector import WassersteinReferenceSelector
from .prompting import build_differential_prompt, build_refinement_prompt

# Paper hyper-parameters (Sec. "Implementation").
K_TOTAL = 5          # references per query
K_STAGE1 = 3         # Stage-1 differential references
K_STAGE2 = 2         # Stage-2 refinement references
LAMBDA_CONF = 0.1    # confidence down-weighting:  c~ = c * exp(-lambda * SW)
NMS_IOU = 0.5        # NMS IoU threshold
TEMPERATURE = 0.7    # VLM sampling temperature
TOP_P = 0.95         # nucleus sampling
VLM_IMAGE_SIZE = 1024  # images resized to 1024x1024 for VLM inference (paper)


class WALDO:
    """Training-free zero-shot anomaly localisation with a vision-language model.

    Example:
        >>> waldo = WALDO(vlm_client=client, model="qwen2.5-vl-72b")
        >>> out = waldo.localize(query_rgb, healthy_references, modality="mri")
        >>> out["boxes"]  # final bounding boxes in 0-1000 normalised coordinates
    """

    def __init__(
        self,
        vlm_client: Any,
        model: str = "qwen2.5-vl-72b",
        n_references: int = K_TOTAL,
        n_stage1_refs: int = K_STAGE1,
        n_stage2_refs: int = K_STAGE2,
        lambda_conf: float = LAMBDA_CONF,
        nms_iou: float = NMS_IOU,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        device: str = "cuda",
        allow_dinov2_fallback: bool = False,
        reference_selector: Optional[WassersteinReferenceSelector] = None,
    ):
        self.vlm_client = vlm_client
        self.model = model
        self.n_references = max(n_references, n_stage1_refs + n_stage2_refs)
        self.n_stage1_refs = n_stage1_refs
        self.n_stage2_refs = n_stage2_refs
        self.lambda_conf = lambda_conf
        self.nms_iou = nms_iou
        self.temperature = temperature
        self.top_p = top_p
        self.reference_selector = reference_selector or WassersteinReferenceSelector(
            device=device, allow_dinov2_fallback=allow_dinov2_fallback
        )

    # ------------------------------------------------------------------ #
    # Response parsing  (handles 0-1 / 0-1000 / pixel coordinate formats)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_boxes(response: str, key: str = "boxes") -> List[List[float]]:
        """Parse bounding boxes from a VLM JSON response, normalised to 0-1000.

        Different VLMs return 0-1, 0-1000, or pixel coordinates; we auto-detect the
        scale and express everything in the paper's 0-1000 normalised convention.
        """
        boxes: List[List[float]] = []
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if not match:
            return boxes
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return boxes

        raw = data.get(key)
        if raw is None:  # tolerate the alternate key used across stages
            raw = data.get("confirmed_boxes" if key == "boxes" else "boxes", [])
        if not isinstance(raw, list):
            return boxes

        for box in raw:
            if not (isinstance(box, (list, tuple)) and len(box) >= 4):
                continue
            coords = [float(c) for c in box[:4]]
            mx = max(abs(c) for c in coords)
            if mx <= 1.0:                 # 0-1 normalised -> 0-1000
                coords = [c * 1000.0 for c in coords]
            elif mx > 1000.0:             # pixel coords larger than 1000 -> clamp later
                coords = [min(c, 1000.0) for c in coords]
            # 1 < mx <= 1000 is already in the 0-1000 convention.
            x1, y1, x2, y2 = coords
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        return boxes

    # ------------------------------------------------------------------ #
    # Confidence-weighted NMS (weighted box fusion at IoU=0.5)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        a1 = (box[2] - box[0]) * (box[3] - box[1])
        a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (a1 + a2 - inter + 1e-8)

    def _weighted_nms(
        self,
        boxes: List[List[float]],
        confidences: List[float],
    ) -> Tuple[List[List[float]], List[float]]:
        """Confidence-weighted NMS: cluster boxes at IoU>=thr and fuse each cluster
        into a confidence-weighted average (weighted box fusion).

        Returns (fused_boxes, fused_confidences) where each fused confidence is the
        max confidence in its cluster.
        """
        if not boxes:
            return [], []
        b = np.asarray(boxes, dtype=float)
        c = np.asarray(confidences, dtype=float)
        order = np.argsort(c)[::-1]
        b, c = b[order], c[order]

        kept: List[List[float]] = []
        kept_conf: List[float] = []
        used = np.zeros(len(b), dtype=bool)
        for i in range(len(b)):
            if used[i]:
                continue
            ious = self._iou(b[i], b)
            cluster = (ious >= self.nms_iou) & (~used)
            cluster[i] = True
            w = c[cluster]
            fused = np.average(b[cluster], axis=0, weights=w if w.sum() > 0 else None)
            kept.append([float(v) for v in fused])
            kept_conf.append(float(w.max()))
            used |= cluster
        return kept, kept_conf

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def localize(
        self,
        query_image: np.ndarray,
        reference_pool: Sequence[np.ndarray],
        modality: str = "mri",
    ) -> Dict[str, Any]:
        """Localise anomalies in ``query_image`` using a healthy ``reference_pool``.

        Returns a dict with ``boxes`` (final, 0-1000 coords), ``confidences``,
        intermediate ``stage1_candidates``, and ``raw_responses``.
        """
        selected = self.reference_selector.select_references_with_scores(
            query_image, reference_pool, n_references=self.n_references
        )
        raw_responses: List[str] = []

        # Confidence model (paper):  c~ = c * exp(-lambda * SW).
        # Here c is the VLM's per-box confidence; the differential prompt does not
        # request a per-box score, so c defaults to 1.0 and the effective weight is
        # exp(-lambda * SW) (i.e. c=1). The SW distance to the query down-weights
        # predictions from less-similar references.

        # ---- Stage 1: differential detection ---------------------------------
        diff_prompt = build_differential_prompt(modality)
        s1_boxes: List[List[float]] = []
        s1_conf: List[float] = []
        for idx, sw in selected[: self.n_stage1_refs]:
            resp = self._call_vlm([query_image, reference_pool[idx]], diff_prompt)
            raw_responses.append(resp)
            weight = float(np.exp(-self.lambda_conf * sw))
            for box in self._parse_boxes(resp, key="boxes"):
                s1_boxes.append(box)
                s1_conf.append(weight)
        candidates, cand_conf = self._weighted_nms(s1_boxes, s1_conf)

        # ---- Stage 2: refinement / confirmation ------------------------------
        # Stage 2 re-presents the Stage-1 candidates to K2 further references to
        # confirm / refine / reject them. If Stage 2 returns boxes they replace the
        # Stage-1 candidates (refinement); if Stage 2 returns nothing we keep the
        # Stage-1 candidates (the demo does not treat an empty Stage 2 as a global
        # reject). See DISCLAIMER.md.
        s2_boxes: List[List[float]] = []
        s2_conf: List[float] = []
        if candidates:
            refine_prompt = build_refinement_prompt(modality, candidates)
            for idx, sw in selected[self.n_stage1_refs : self.n_stage1_refs + self.n_stage2_refs]:
                resp = self._call_vlm([query_image, reference_pool[idx]], refine_prompt)
                raw_responses.append(resp)
                weight = float(np.exp(-self.lambda_conf * sw))
                for box in self._parse_boxes(resp, key="confirmed_boxes"):
                    s2_boxes.append(box)
                    s2_conf.append(weight)

        if s2_boxes:
            final_boxes, final_conf = self._weighted_nms(s2_boxes, s2_conf)
        else:
            final_boxes, final_conf = candidates, cand_conf

        return {
            "boxes": final_boxes,
            "confidences": final_conf,
            "stage1_candidates": candidates,
            "selected_references": [idx for idx, _ in selected],
            "raw_responses": raw_responses,
        }

    def _call_vlm(self, images: List[np.ndarray], prompt: str) -> str:
        """Send images + prompt to the VLM and return the raw text response.

        Default implementation targets any OpenAI-compatible chat client (OpenAI,
        OpenRouter, vLLM's OpenAI server, ...) exposed as ``self.vlm_client`` with a
        ``chat.completions.create`` method. Images are resized to 1024x1024 (paper)
        and sampled at temperature 0.7 / top_p 0.95. Override this method for a
        different client API.
        """
        import base64
        from io import BytesIO
        from PIL import Image

        if self.vlm_client is None:
            raise ValueError(
                "WALDO.vlm_client is None. Pass an OpenAI-compatible client, e.g. "
                "WALDO(vlm_client=OpenAI(api_key=...)), or override _call_vlm."
            )

        content = [{"type": "text", "text": prompt}]
        for arr in images:
            img = Image.fromarray(np.asarray(arr).astype(np.uint8)).convert("RGB")
            img = img.resize((VLM_IMAGE_SIZE, VLM_IMAGE_SIZE))  # paper: 1024x1024 for VLM
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})

        resp = self.vlm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return resp.choices[0].message.content
