#!/usr/bin/env python3
"""
Run WALDO inference on NOVA or VinDr-CXR.

End-to-end pipeline: dataset loading -> entropy-weighted Sliced Wasserstein
reference selection -> two-stage differential VLM prompting -> evaluation.

Usage:
    # NOVA with GPT-4o (OpenAI)
    python scripts/run_inference.py --dataset nova --model gpt-4o \
        --api-key YOUR_KEY --n-samples 10

    # NOVA with Qwen via OpenRouter
    python scripts/run_inference.py --dataset nova \
        --model qwen/qwen2.5-vl-72b-instruct --openrouter --api-key YOUR_KEY

    # Evaluate an existing results file
    python scripts/run_inference.py --eval-only --results results/nova/nova_waldo_qwen25_72b.json

NOTE: a faithful reference implementation distilled by Claude + Codex agents from a
larger experimentation codebase (see DISCLAIMER.md). Reproducing the exact paper
tables requires the full datasets, the DINOv3-ViT-B/16 backbone, and API access.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waldo import WALDO
from waldo.metrics import compute_best_iou, compute_map
from waldo.preprocessing import CoordinateTransformer

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def load_nova_dataset(n_samples: int = None) -> Tuple[List, List, List]:
    """Load NOVA query images + gold GT boxes with correct annotation/image alignment."""
    if not HAS_DATASETS:
        raise ImportError("datasets library required: pip install datasets")

    print("Loading NOVA annotations + images from HuggingFace...")
    annotations = load_dataset(
        "parquet",
        data_files="hf://datasets/c-i-ber/Nova/data/nova-v1.parquet",
        split="train",
    )
    images = load_dataset("c-i-ber/Nova", split="train")

    # CRITICAL: annotations are NOT sorted by filename; images ARE. Build a mapping.
    ann_filenames = [annotations[i]["filename"] for i in range(len(annotations))]
    sorted_filenames = sorted(set(ann_filenames))
    filename_to_idx = {fn: i for i, fn in enumerate(sorted_filenames)}

    valid = []
    for i in range(len(annotations)):
        ann = annotations[i]
        gt_boxes = []
        for bbox in ann.get("bboxes", []) or []:
            if bbox.get("source") == "gold":
                x1, y1 = bbox["x"], bbox["y"]
                gt_boxes.append([x1, y1, x1 + bbox["width"], y1 + bbox["height"]])
        if gt_boxes:
            valid.append({"filename": ann["filename"], "gt_boxes": gt_boxes,
                          "image_index": filename_to_idx[ann["filename"]]})

    if n_samples:
        valid = valid[:n_samples]

    query_images, gt_boxes_list, image_ids = [], [], []
    for ann in tqdm(valid, desc="Loading images"):
        img = images[ann["image_index"]]["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        query_images.append(np.array(img.convert("RGB")))
        gt_boxes_list.append(ann["gt_boxes"])
        image_ids.append(ann["filename"])

    print(f"  ✓ Loaded {len(query_images)} annotated NOVA samples")
    return query_images, gt_boxes_list, image_ids


def load_cxr_dataset(data_dir: Path, n_samples: int = None) -> Tuple[List, List, List]:
    """Load VinDr-CXR query images + GT boxes via the bundled data loader."""
    from waldo.data_loader import VinDrCXRDataLoader
    loader = VinDrCXRDataLoader(data_dir=data_dir)
    n = len(loader) if n_samples is None else min(n_samples, len(loader))
    query_images, gt_boxes_list, image_ids = [], [], []
    for i in tqdm(range(n), desc="Loading CXR"):
        s = loader.get_sample(i)
        query_images.append(s.image)
        gt_boxes_list.append(s.gt_boxes)
        image_ids.append(s.image_id)
    return query_images, gt_boxes_list, image_ids


def load_healthy_references(dataset: str, data_dir: Path, n_refs: int) -> List[np.ndarray]:
    """Load healthy reference images (NOVA: no-finding scans; CXR: 'No finding' cases)."""
    if dataset == "nova":
        from waldo.data_loader import NOVADataLoader
        loader = NOVADataLoader()
        refs = [s.image for s in loader.get_healthy_samples(n_refs)]
    else:
        from waldo.data_loader import VinDrCXRDataLoader
        loader = VinDrCXRDataLoader(data_dir=data_dir)
        refs = [s.image for s in loader.get_healthy_samples(n_refs)]
    print(f"  ✓ Loaded {len(refs)} healthy references")
    return refs


def resolve_device(device: str) -> str:
    """Resolve 'auto' to cuda if available, else cpu."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class WALDOWithVLM(WALDO):
    """WALDO with a concrete OpenAI-compatible client built from an API key.

    The base WALDO._call_vlm handles the request (1024x1024 resize, temperature
    0.7, top_p 0.95) against the OpenAI-compatible client; this subclass only
    constructs the client and adds simple retry behaviour.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, **kwargs):
        if not HAS_OPENAI:
            raise ImportError("openai library required: pip install openai")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        super().__init__(vlm_client=OpenAI(**client_kwargs), model=model, **kwargs)

    def _call_vlm(self, images: List[np.ndarray], prompt: str) -> str:
        for attempt in range(3):
            try:
                return super()._call_vlm(images, prompt)
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  VLM error: {e}")
                    return '{"boxes": []}'


def run_evaluation(query_images, gt_boxes_list, reference_pool, waldo, modality, image_ids):
    """Run WALDO over the dataset and score each image by best IoU vs GT."""
    results = []
    for query_img, gt_boxes, img_id in tqdm(
        list(zip(query_images, gt_boxes_list, image_ids)), desc="Running WALDO"
    ):
        out = waldo.localize(query_image=query_img, reference_pool=reference_pool, modality=modality)
        # WALDO returns 0-1000 normalised boxes; GT is in image pixels -> put both in pixels.
        h, w = query_img.shape[:2]
        pred_px = CoordinateTransformer.denormalize_boxes(out["boxes"], (h, w), source_range=1000)
        iou = compute_best_iou(pred_px, gt_boxes)
        results.append({
            "image_id": img_id,
            "iou": float(iou),
            "hit_30": iou >= 0.30,
            "hit_50": iou >= 0.50,
            "n_pred": len(pred_px),
            "n_gt": len(gt_boxes),
            "pred_boxes": [[float(c) for c in b] for b in pred_px],
            "gt_boxes": [[float(c) for c in b] for b in gt_boxes],
        })
    return results


def _print_metrics(metrics: Dict[str, float], n: int):
    print("\n" + "=" * 70)
    print(f"Samples: {n}")
    print(f"mAP@30:  {metrics['mAP@30']:.1%}")
    print(f"mAP@50:  {metrics['mAP@50']:.1%}")
    print(f"Avg IoU: {metrics['avg_iou']:.1%}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run WALDO inference",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["nova", "cxr"], default="nova")
    parser.add_argument("--data-dir", type=str, default="data", help="Data dir (required for CXR)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="VLM model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key (or OPENAI_API_KEY env)")
    parser.add_argument("--openrouter", action="store_true", help="Use the OpenRouter endpoint")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-references", type=int, default=5, help="K references per query (paper: 5)")
    parser.add_argument("--n-ref-pool", type=int, default=50, help="Healthy reference pool size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for DINOv3 ('auto' picks cuda if available, else cpu)")
    parser.add_argument("--allow-dinov2-fallback", action="store_true",
                        help="Allow falling back to DINOv2-base if DINOv3 is unavailable "
                             "(non-paper; for a quick smoke test only)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--results", type=str, default=None)
    args = parser.parse_args()

    if args.eval_only:
        if not args.results:
            raise ValueError("--results required with --eval-only")
        with open(args.results) as f:
            data = json.load(f)
        results = data.get("results", data.get("detailed_results", []))
        _print_metrics(compute_map(results), len(results))
        return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required: pass --api-key or set OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if args.openrouter else None

    if args.dataset == "nova":
        query_images, gt_boxes_list, image_ids = load_nova_dataset(n_samples=args.n_samples)
        modality = "mri"
    else:
        query_images, gt_boxes_list, image_ids = load_cxr_dataset(
            Path(args.data_dir) / "cxr", n_samples=args.n_samples)
        modality = "cxr"

    reference_pool = load_healthy_references(args.dataset, Path(args.data_dir) / args.dataset,
                                             n_refs=args.n_ref_pool)

    print("\nInitialising WALDO (DINOv3 reference selection + two-stage VLM prompting)...")
    waldo = WALDOWithVLM(api_key=api_key, model=args.model, base_url=base_url,
                         n_references=args.n_references, device=resolve_device(args.device),
                         allow_dinov2_fallback=args.allow_dinov2_fallback)

    results = run_evaluation(query_images, gt_boxes_list, reference_pool, waldo, modality, image_ids)
    metrics = compute_map(results)
    _print_metrics(metrics, len(results))

    output_file = args.output or f"results/{args.dataset}_waldo_{args.model.replace('/', '_')}_{len(results)}samples.json"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": {"dataset": args.dataset, "model": args.model,
                       "n_samples": len(results), "n_references": args.n_references},
            "metrics": metrics,
            "results": results,
        }, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
