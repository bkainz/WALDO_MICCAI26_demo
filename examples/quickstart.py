#!/usr/bin/env python3
"""
WALDO quickstart.

Demonstrates the pipeline on one NOVA sample:
  1. Load a query image + its ground-truth boxes and a pool of healthy references.
  2. Run the REAL entropy-weighted Sliced Wasserstein + Goldilocks + DPP reference
     selection (DINOv3-ViT-B/16).
  3. If an API key is supplied, run full WALDO localisation and visualise the
     predicted boxes; otherwise stop after reference selection.

This script never fabricates detections: bounding boxes are only drawn when they
come from a real VLM call. Without --api-key it visualises the ground truth and the
selected references only.

Run:
    python examples/quickstart.py                 # reference selection only (no VLM)
    python examples/quickstart.py --api-key KEY    # full WALDO localisation
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from waldo.reference_selector import WassersteinReferenceSelector


def load_sample_data(n_refs: int = 20):
    """Load one annotated NOVA query + a small healthy reference pool (aligned)."""
    if not HAS_DATASETS:
        raise ImportError("datasets library required: pip install datasets")

    print("Loading NOVA dataset (first run downloads from HuggingFace)...")
    annotations = load_dataset(
        "parquet",
        data_files="hf://datasets/c-i-ber/Nova/data/nova-v1.parquet",
        split="train",
    )
    images = load_dataset("c-i-ber/Nova", split="train")

    ann_filenames = [annotations[i]["filename"] for i in range(len(annotations))]
    sorted_filenames = sorted(set(ann_filenames))
    fn_to_idx = {fn: i for i, fn in enumerate(sorted_filenames)}

    def gold_boxes(ann):
        out = []
        for b in ann.get("bboxes", []) or []:
            if b.get("source") == "gold":
                out.append([b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]])
        return out

    def to_rgb(idx):
        img = images[idx]["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return np.array(img.convert("RGB"))

    query_image, gt_boxes, filename = None, [], None
    healthy_refs = []
    for i in range(len(annotations)):
        ann = annotations[i]
        gb = gold_boxes(ann)
        if gb and query_image is None:
            query_image, gt_boxes, filename = to_rgb(fn_to_idx[ann["filename"]]), gb, ann["filename"]
        elif not gb and len(healthy_refs) < n_refs:
            healthy_refs.append(to_rgb(fn_to_idx[ann["filename"]]))
        if query_image is not None and len(healthy_refs) >= n_refs:
            break

    if query_image is None:
        raise ValueError("No annotated NOVA sample found")
    print(f"✓ Query: {filename}  |  GT boxes: {len(gt_boxes)}  |  healthy refs: {len(healthy_refs)}")
    return query_image, gt_boxes, healthy_refs, filename


def visualize(image, gt_boxes, pred_boxes, output_path):
    """Draw GT (green) and, if present, VLM predictions (red, 0-1000 normalised)."""
    img = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    h, w = image.shape[:2]
    for box in gt_boxes:
        draw.rectangle([float(c) for c in box], outline="green", width=3)
    for box in pred_boxes:
        x1, y1, x2, y2 = (box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    img.save(output_path)
    print(f"✓ Saved visualisation to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="WALDO quickstart demo")
    parser.add_argument("--api-key", type=str, default=None,
                        help="VLM API key. If omitted, runs reference selection only.")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--openrouter", action="store_true")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' picks cuda if available, else cpu")
    parser.add_argument("--allow-dinov2-fallback", action="store_true",
                        help="Allow DINOv2-base fallback if DINOv3 is unavailable (non-paper)")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    query_image, gt_boxes, healthy_refs, filename = load_sample_data()

    print("\nRunning entropy-weighted Sliced Wasserstein reference selection (DINOv3)...")
    selector = WassersteinReferenceSelector(device=device, allow_dinov2_fallback=args.allow_dinov2_fallback)
    selected = selector.select_references_with_scores(query_image, healthy_refs, n_references=3)
    print("✓ Selected Goldilocks-zone references (index, SW distance to query):")
    for idx, sw in selected:
        print(f"    ref #{idx}:  SW={sw:.4f}")

    pred_boxes = []
    if args.api_key:
        print("\nRunning full WALDO localisation via the VLM...")
        # Reuse the production client from scripts/run_inference.py
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_inference import WALDOWithVLM
        base_url = "https://openrouter.ai/api/v1" if args.openrouter else None
        waldo = WALDOWithVLM(api_key=args.api_key, model=args.model, base_url=base_url,
                             device=device, allow_dinov2_fallback=args.allow_dinov2_fallback)
        result = waldo.localize(query_image, healthy_refs, modality="mri")
        pred_boxes = result["boxes"]
        print(f"✓ WALDO predicted {len(pred_boxes)} region(s)")
    else:
        print("\n(No --api-key given: skipping VLM inference. Only ground truth and the "
              "selected references are visualised — no detections are fabricated.)")

    vis_path = out_dir / f"demo_{Path(filename).stem}.png"
    visualize(query_image, gt_boxes, pred_boxes, str(vis_path))

    print("\nNext: run the full benchmark with")
    print("    python scripts/run_inference.py --dataset nova --model gpt-4o --api-key KEY --n-samples 10")


if __name__ == "__main__":
    main()
