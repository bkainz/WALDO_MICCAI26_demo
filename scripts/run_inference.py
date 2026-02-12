#!/usr/bin/env python3
"""
Run WALDO inference on NOVA or VinDr-CXR datasets.

This script demonstrates complete end-to-end inference including:
1. Dataset loading
2. Reference selection
3. VLM-based anomaly localization
4. Evaluation

Usage:
    # Run on NOVA with OpenAI GPT-4o
    python scripts/run_inference.py --dataset nova --model gpt-4o \\
        --api-key YOUR_KEY --n-samples 10

    # Run on CXR with Qwen via OpenRouter
    python scripts/run_inference.py --dataset cxr --model qwen \\
        --api-key YOUR_KEY --openrouter

    # Evaluate only (use pre-computed results)
    python scripts/run_inference.py --eval-only --results results/nova_waldo_gpt4o.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waldo import WALDO
from waldo.reference_selector import WassersteinReferenceSelector
from waldo.metrics import compute_map, compute_iou

# VLM Client imports
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


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def load_nova_dataset(data_dir: Path, n_samples: int = None) -> Tuple[List, List, List]:
    """
    Load NOVA dataset with proper alignment.

    Returns:
        Tuple of (query_images, gt_boxes_list, image_ids)
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required")

    print("Loading NOVA dataset...")

    # Check if annotations exist locally
    ann_file = data_dir / "annotations" / "nova_annotations.json"
    if ann_file.exists():
        print(f"  Using local annotations from {ann_file}")
        with open(ann_file) as f:
            annotations = json.load(f)
    else:
        print("  Downloading annotations from HuggingFace...")
        annotations = load_dataset(
            "parquet",
            data_files="hf://datasets/c-i-ber/Nova/data/nova-v1.parquet",
            split="train"
        )
        # Convert to list format
        annotations = [annotations[i] for i in range(len(annotations))]

    # Load images
    print("  Loading images from HuggingFace...")
    images = load_dataset("c-i-ber/Nova", split="train")

    # Filter to samples with annotations
    if isinstance(annotations[0], dict) and 'filename' in annotations[0]:
        # Using preprocessed annotations
        valid_annotations = [ann for ann in annotations if ann.get('gt_boxes')]
    else:
        # Using raw parquet annotations
        valid_annotations = []
        for ann in annotations:
            gt_boxes = []
            for bbox in ann.get('bboxes', []):
                if bbox.get('source') == 'gold':
                    x1, y1 = bbox['x'], bbox['y']
                    x2, y2 = x1 + bbox['width'], y1 + bbox['height']
                    gt_boxes.append([x1, y1, x2, y2])

            if gt_boxes:
                valid_annotations.append({
                    'filename': ann['filename'],
                    'gt_boxes': gt_boxes,
                    'image_index': None  # Will be set below
                })

    # Create filename to image index mapping
    ann_filenames = [ann['filename'] for ann in valid_annotations]
    sorted_filenames = sorted(set(ann_filenames))
    filename_to_idx = {fn: i for i, fn in enumerate(sorted_filenames)}

    # Update image indices
    for ann in valid_annotations:
        ann['image_index'] = filename_to_idx[ann['filename']]

    # Limit samples if requested
    if n_samples:
        valid_annotations = valid_annotations[:n_samples]

    # Extract data
    query_images = []
    gt_boxes_list = []
    image_ids = []

    for ann in tqdm(valid_annotations, desc="Processing images"):
        img_idx = ann['image_index']
        img = images[img_idx]['image']

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        query_images.append(np.array(img.convert('RGB')))
        gt_boxes_list.append(ann['gt_boxes'])
        image_ids.append(ann.get('filename', f'image_{img_idx}'))

    print(f"  ✓ Loaded {len(query_images)} images with annotations")
    return query_images, gt_boxes_list, image_ids


def load_healthy_references(
    dataset: str,
    data_dir: Path,
    n_refs: int = 50
) -> List[np.ndarray]:
    """Load healthy reference images."""
    refs = []

    if dataset == "nova":
        # Try to load from healthy indices
        healthy_file = data_dir.parent / "healthy_references" / "nova_healthy_indices.json"

        if healthy_file.exists():
            print(f"Loading healthy references from {healthy_file}...")
            with open(healthy_file) as f:
                healthy_indices = json.load(f)

            images = load_dataset("c-i-ber/Nova", split="train")
            for idx in healthy_indices[:n_refs]:
                img = images[idx]['image']
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                refs.append(np.array(img.convert('RGB')))

        else:
            print("Warning: Healthy reference indices not found.")
            print("Using first N images as references (not ideal).")
            images = load_dataset("c-i-ber/Nova", split="train")
            for i in range(min(n_refs, 100)):
                img = images[i]['image']
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                refs.append(np.array(img.convert('RGB')))

    elif dataset == "cxr":
        # Load "No Finding" cases as references
        ann_file = data_dir / "annotations" / "annotations_test.csv"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"CXR annotations not found at {ann_file}. "
                "Please download VinDr-CXR dataset first."
            )

        import csv
        no_finding_ids = []
        with open(ann_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class_name'] == 'No finding':
                    no_finding_ids.append(row['image_id'])

        # Load corresponding images (requires NPZ files)
        imgs_dir = data_dir / "images" / "test"
        for img_id in list(set(no_finding_ids))[:n_refs]:
            npz_file = imgs_dir / f"{img_id}.npz"
            if npz_file.exists():
                img_data = np.load(npz_file)['image']
                # Convert to RGB
                img_rgb = (img_data * 255).astype(np.uint8)
                img_rgb = np.stack([img_rgb] * 3, axis=-1)
                refs.append(img_rgb)

    print(f"  ✓ Loaded {len(refs)} healthy references")
    return refs


class WALDOWithVLM(WALDO):
    """Extended WALDO class with VLM client implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = None,
        **kwargs
    ):
        """Initialize with API key."""
        if not HAS_OPENAI:
            raise ImportError("openai library required: pip install openai")

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.api_model = model

        super().__init__(vlm_client=self.client, model=model, **kwargs)

    def _call_vlm(self, images: List[np.ndarray], prompt: str) -> str:
        """Call VLM API with images and prompt."""
        # Convert images to base64
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Add images
        for img_array in images:
            img_pil = Image.fromarray(img_array.astype(np.uint8))
            img_b64 = image_to_base64(img_pil)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }
            })

        # Call API with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.0,
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Error calling VLM: {e}")
                    return '{"boxes": [], "description": "API error"}'


def run_evaluation(
    query_images: List[np.ndarray],
    gt_boxes_list: List[List],
    reference_pool: List[np.ndarray],
    waldo: WALDOWithVLM,
    modality: str,
    image_ids: List[str]
) -> List[Dict]:
    """Run WALDO inference and evaluation."""
    results = []

    for i, (query_img, gt_boxes, img_id) in enumerate(
        tqdm(
            zip(query_images, gt_boxes_list, image_ids),
            total=len(query_images),
            desc="Running inference"
        )
    ):
        # Run WALDO
        output = waldo.localize(
            query_image=query_img,
            reference_pool=reference_pool,
            modality=modality
        )

        pred_boxes = output['boxes']

        # Compute metrics
        if pred_boxes and gt_boxes:
            iou = compute_iou(pred_boxes, gt_boxes)
        else:
            iou = 0.0

        hit_30 = iou >= 0.30
        hit_50 = iou >= 0.50

        results.append({
            'image_id': img_id,
            'iou': float(iou),
            'hit_30': hit_30,
            'hit_50': hit_50,
            'n_pred': len(pred_boxes),
            'n_gt': len(gt_boxes),
            'pred_boxes': [[float(c) for c in box] for box in pred_boxes],
            'gt_boxes': [[float(c) for c in box] for box in gt_boxes],
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run WALDO inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', type=str, choices=['nova', 'cxr'],
                        default='nova', help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='VLM model name')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--openrouter', action='store_true',
                        help='Use OpenRouter API')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of samples to evaluate')
    parser.add_argument('--n-views', type=int, default=5,
                        help='Number of self-consistency views')
    parser.add_argument('--n-references', type=int, default=3,
                        help='Number of references per view')
    parser.add_argument('--n-ref-pool', type=int, default=50,
                        help='Size of reference pool')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing results')
    parser.add_argument('--results', type=str, default=None,
                        help='Results file to evaluate (with --eval-only)')
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key and not args.eval_only:
        raise ValueError("API key required. Set --api-key or OPENAI_API_KEY env var")

    # Set base URL for OpenRouter
    base_url = "https://openrouter.ai/api/v1" if args.openrouter else None

    if args.eval_only:
        # Evaluate existing results
        if not args.results:
            raise ValueError("--results required with --eval-only")

        print(f"Loading results from {args.results}...")
        with open(args.results) as f:
            data = json.load(f)
            results = data.get('results', data.get('detailed_results', []))

        metrics = compute_map(results)
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"Samples: {len(results)}")
        print(f"mAP@30:  {metrics['map_30']:.1%}")
        print(f"mAP@50:  {metrics['map_50']:.1%}")
        print(f"Avg IoU: {metrics['avg_iou']:.1%}")
        print("=" * 80)
        return

    # Load dataset
    data_dir = Path(args.data_dir) / args.dataset

    if args.dataset == "nova":
        query_images, gt_boxes_list, image_ids = load_nova_dataset(
            data_dir, n_samples=args.n_samples
        )
        modality = "mri"
    else:
        raise NotImplementedError("CXR dataset loading not yet implemented")

    # Load healthy references
    reference_pool = load_healthy_references(
        args.dataset, data_dir, n_refs=args.n_ref_pool
    )

    # Initialize WALDO
    print("\nInitializing WALDO...")
    waldo = WALDOWithVLM(
        api_key=api_key,
        model=args.model,
        base_url=base_url,
        n_views=args.n_views,
        n_references=args.n_references,
    )
    print(f"  Model: {args.model}")
    print(f"  Views: {args.n_views}")
    print(f"  References per view: {args.n_references}")

    # Run evaluation
    print("\nRunning WALDO inference...")
    results = run_evaluation(
        query_images,
        gt_boxes_list,
        reference_pool,
        waldo,
        modality,
        image_ids
    )

    # Compute metrics
    metrics = compute_map(results)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(results)}")
    print(f"mAP@30:  {metrics['map_30']:.1%}")
    print(f"mAP@50:  {metrics['map_50']:.1%}")
    print(f"Avg IoU: {metrics['avg_iou']:.1%}")
    print("=" * 80)

    # Save results
    if args.output:
        output_file = args.output
    else:
        output_file = f"results/{args.dataset}_waldo_{args.model}_{len(results)}samples.json"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'config': {
            'dataset': args.dataset,
            'model': args.model,
            'n_samples': len(results),
            'n_views': args.n_views,
            'n_references': args.n_references,
        },
        'metrics': metrics,
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
