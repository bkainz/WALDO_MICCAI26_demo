"""
Prompt templates and strategies for WALDO.

Provides optimized prompts for different VLMs and modalities,
including few-shot examples and chain-of-thought reasoning.
"""

from typing import List, Dict, Optional
from enum import Enum


class PromptStrategy(Enum):
    """Prompting strategy types."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    DIFFERENTIAL = "differential"  # WALDO's default


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(self, modality: str = "mri"):
        """
        Initialize prompt template.

        Args:
            modality: Medical imaging modality ("mri" or "cxr")
        """
        self.modality = modality.lower()

    def build_prompt(
        self,
        strategy: PromptStrategy = PromptStrategy.DIFFERENTIAL,
        **kwargs
    ) -> str:
        """
        Build prompt based on strategy.

        Args:
            strategy: Prompting strategy to use
            **kwargs: Additional arguments

        Returns:
            Formatted prompt string
        """
        if strategy == PromptStrategy.ZERO_SHOT:
            return self._zero_shot_prompt(**kwargs)
        elif strategy == PromptStrategy.FEW_SHOT:
            return self._few_shot_prompt(**kwargs)
        elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            return self._chain_of_thought_prompt(**kwargs)
        elif strategy == PromptStrategy.DIFFERENTIAL:
            return self._differential_prompt(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _zero_shot_prompt(self, **kwargs) -> str:
        """Generate zero-shot prompt."""
        if self.modality == "mri":
            return """You are a medical imaging expert analyzing brain MRI scans.

Task: Identify and localize abnormal regions in the image.

Instructions:
1. Look for lesions, tumors, hemorrhages, or other abnormalities
2. Provide bounding boxes for each finding
3. Use normalized coordinates in range 0-1000

Return JSON format:
{"boxes": [[x1, y1, x2, y2], ...], "description": "brief description of findings"}

If no abnormalities found: {"boxes": [], "description": "no significant findings"}"""

        else:  # CXR
            return """You are a radiologist analyzing chest X-rays.

Task: Identify and localize pathological findings.

Instructions:
1. Look for consolidations, nodules, masses, cardiomegaly, effusions, pneumothorax, etc.
2. Provide bounding boxes for each finding
3. Use normalized coordinates in range 0-1000

Return JSON format:
{"boxes": [[x1, y1, x2, y2], ...], "description": "brief description of findings"}

If no abnormalities found: {"boxes": [], "description": "no significant findings"}"""

    def _few_shot_prompt(self, examples: Optional[List[Dict]] = None, **kwargs) -> str:
        """
        Generate few-shot prompt with examples.

        Args:
            examples: List of example dicts with 'description' and 'boxes'
        """
        base_prompt = self._zero_shot_prompt()

        if examples is None:
            # Use default examples
            if self.modality == "mri":
                examples = [
                    {
                        "description": "hyperintense lesion in right frontal lobe",
                        "boxes": [[450, 300, 650, 500]]
                    },
                    {
                        "description": "no significant abnormality",
                        "boxes": []
                    }
                ]
            else:
                examples = [
                    {
                        "description": "consolidation in right lower lobe",
                        "boxes": [[600, 500, 850, 750]]
                    },
                    {
                        "description": "cardiomegaly",
                        "boxes": [[300, 400, 700, 900]]
                    }
                ]

        examples_text = "\n\nExamples:\n"
        for i, ex in enumerate(examples, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f'Output: {{"boxes": {ex["boxes"]}, "description": "{ex["description"]}"}}\n'

        return base_prompt + examples_text

    def _chain_of_thought_prompt(self, **kwargs) -> str:
        """Generate chain-of-thought reasoning prompt."""
        if self.modality == "mri":
            return """You are a medical imaging expert analyzing brain MRI scans.

Task: Identify and localize abnormal regions using step-by-step reasoning.

Instructions:
1. First, systematically scan each region (frontal, parietal, temporal, occipital lobes, cerebellum, brainstem)
2. For each region, note if there are intensity differences, mass effects, or structural abnormalities
3. Reason about which findings are clinically significant
4. Provide bounding boxes only for significant abnormalities
5. Use normalized coordinates in range 0-1000

Response format:
{
  "reasoning": "step-by-step analysis of each region...",
  "boxes": [[x1, y1, x2, y2], ...],
  "description": "summary of significant findings"
}"""

        else:  # CXR
            return """You are a radiologist analyzing chest X-rays using systematic review.

Task: Identify and localize pathological findings using the ABCDEFG approach.

Instructions:
1. Airways: Check trachea, bronchi
2. Breathing: Assess lung fields for consolidation, nodules, masses
3. Cardiac: Evaluate heart size and borders
4. Diaphragm: Check costophrenic angles, diaphragm position
5. Everything else: Bones, soft tissues, devices
6. For each abnormality found, provide a bounding box
7. Use normalized coordinates in range 0-1000

Response format:
{
  "reasoning": "systematic ABCDEFG review...",
  "boxes": [[x1, y1, x2, y2], ...],
  "description": "summary of findings"
}"""

    def _differential_prompt(self, **kwargs) -> str:
        """WALDO's core differential prompt (Stage 1), as quoted in the paper."""
        return build_differential_prompt(self.modality)


class ModelSpecificPrompts:
    """
    VLM-specific prompt optimizations.

    Different VLMs respond better to different prompt styles.
    """

    @staticmethod
    def get_prompt_for_model(
        model_name: str,
        modality: str,
        strategy: PromptStrategy = PromptStrategy.DIFFERENTIAL
    ) -> str:
        """
        Get optimized prompt for specific VLM.

        Args:
            model_name: VLM model name (e.g., "gpt-4o", "qwen2.5-vl-72b")
            modality: Medical imaging modality
            strategy: Prompting strategy

        Returns:
            Optimized prompt string
        """
        template = PromptTemplate(modality=modality)
        base_prompt = template.build_prompt(strategy=strategy)

        # Model-specific adjustments
        if "gpt-4" in model_name.lower() or "gpt4" in model_name.lower():
            # GPT-4 variants respond well to explicit JSON formatting
            base_prompt += "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no code blocks."

        elif "qwen" in model_name.lower():
            # Qwen models benefit from explicit coordinate format clarification
            base_prompt += "\n\nCoordinate format: Normalized 0-1000 where (0,0) is top-left, (1000,1000) is bottom-right."

        elif "gemini" in model_name.lower():
            # Gemini models prefer structured output
            base_prompt += "\n\nUse this exact JSON structure with 'boxes' array and 'description' string."

        elif "claude" in model_name.lower():
            # Claude models work well with detailed instructions
            base_prompt += "\n\nProvide precise bounding boxes. Err on the side of slightly larger boxes to ensure full lesion coverage."

        return base_prompt

    @staticmethod
    def parse_model_response(
        response: str,
        model_name: str
    ) -> Dict:
        """
        Parse VLM response with model-specific handling.

        Args:
            response: Raw VLM response
            model_name: VLM model name

        Returns:
            Parsed dict with 'boxes' and 'description'
        """
        import json
        import re

        # Try direct JSON parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Model-specific parsing
        if "gemini" in model_name.lower():
            # Gemini sometimes wraps in box_2d
            match = re.search(r'"box_2d":\s*(\[.*?\])', response, re.DOTALL)
            if match:
                boxes = json.loads(match.group(1))
                return {"boxes": boxes, "description": "gemini detection"}

        # Generic fallback: extract JSON from markdown or text
        json_match = re.search(r'\{[^{}]*"boxes"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: extract box arrays
        box_pattern = r'\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]'
        boxes_str = re.findall(box_pattern, response)

        if boxes_str:
            boxes = []
            for box_str in boxes_str:
                box = json.loads(box_str)
                boxes.append(box)
            return {"boxes": boxes, "description": "extracted from text"}

        # No boxes found
        return {"boxes": [], "description": "parsing failed"}


class PromptBuilder:
    """High-level builder for creating complete prompts with images."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        modality: str = "mri",
        strategy: PromptStrategy = PromptStrategy.DIFFERENTIAL
    ):
        """
        Initialize prompt builder.

        Args:
            model_name: VLM model name
            modality: Medical imaging modality
            strategy: Prompting strategy
        """
        self.model_name = model_name
        self.modality = modality
        self.strategy = strategy

    def build_message(
        self,
        query_image_b64: str,
        reference_images_b64: List[str],
        include_examples: bool = False
    ) -> List[Dict]:
        """
        Build complete message for VLM API.

        Args:
            query_image_b64: Query image as base64 string
            reference_images_b64: Reference images as base64 strings
            include_examples: Whether to include few-shot examples

        Returns:
            Message list for OpenAI-compatible API
        """
        # Get prompt
        if include_examples:
            prompt = ModelSpecificPrompts.get_prompt_for_model(
                self.model_name, self.modality, PromptStrategy.FEW_SHOT
            )
        else:
            prompt = ModelSpecificPrompts.get_prompt_for_model(
                self.model_name, self.modality, self.strategy
            )

        # Build message
        message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }

        # Add query image
        message["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{query_image_b64}"}
        })

        # Add reference images
        for ref_b64 in reference_images_b64:
            message["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"}
            })

        return [message]


# ===========================================================================
# Canonical WALDO prompts (used by waldo.WALDO)
# ===========================================================================
# The first sentence of the differential prompt is quoted verbatim from the
# paper (Sec. "Differential Prompting and Aggregation"). In this demo the query
# and reference are sent as two separate images (query first, reference second)
# rather than a side-by-side composite, so we keep the paper wording and add an
# explicit JSON output contract for robust parsing.

_PAPER_DIFFERENTIAL = (
    "Compare the patient scan (left) to the healthy reference (right). "
    "Identify and localise any regions that appear abnormal, different, or "
    "pathological. Return bounding box coordinates [x1, y1, x2, y2] normalised "
    "to [0, 1000]."
)


def build_differential_prompt(modality: str = "mri") -> str:
    """Stage-1 differential prompt comparing the query to a healthy reference."""
    organ = "brain MRI" if modality.lower() == "mri" else "chest X-ray"
    return f"""You are a medical imaging expert analysing a {organ}.

You are shown TWO images: the FIRST is the patient scan to analyse; the SECOND is
a healthy reference. {_PAPER_DIFFERENTIAL}

Use the healthy reference to decide what is normal, then mark only the regions of
the patient scan that deviate from it.

Output JSON only:
{{"abnormality_found": true/false, "boxes": [[x1, y1, x2, y2], ...], "description": "brief finding"}}
If nothing differs from the reference: {{"abnormality_found": false, "boxes": []}}"""


def build_refinement_prompt(modality: str, candidates: list) -> str:
    """Stage-2 refinement prompt: confirm / refine / reject Stage-1 candidates."""
    organ = "brain MRI" if modality.lower() == "mri" else "chest X-ray"
    cand = [[round(float(c), 1) for c in box] for box in candidates]
    return f"""You are a medical imaging expert analysing a {organ}.

You are shown TWO images: the FIRST is the patient scan; the SECOND is a healthy
reference. An initial differential analysis proposed candidate abnormal regions
(coordinates normalised to [0, 1000]):

{cand}

For each candidate, compared against the healthy reference:
1. CONFIRM it if it is genuinely abnormal (not a normal anatomical variant).
2. REFINE its bounding box to tightly fit the abnormality.
3. REJECT false positives.
You may also add a clearly missed region.

Output JSON only:
{{"confirmed_boxes": [[x1, y1, x2, y2], ...], "description": "brief finding"}}
If none are genuine abnormalities: {{"confirmed_boxes": []}}"""
