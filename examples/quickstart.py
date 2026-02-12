#!/usr/bin/env python3
"""
WALDO Quickstart Example

This script demonstrates basic WALDO usage for anomaly localization.
It shows how to:
1. Load a dataset
2. Select healthy references
3. Run WALDO inference
4. Visualize results

Run with:
    python examples/quickstart.py --api-key YOUR_KEY
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw
import json

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    print("Warning: datasets not installed. Install with: pip install datasets")
    HAS_DATASETS = False

from waldo.reference_selector import WassersteinReferenceSelector


def load_sample_data():
    """Load a sample NOVA image with annotations."""
    if not HAS_DATASETS:
        raise ImportError("datasets library required")

    print("Loading NOVA dataset (this may take a minute on first run)...")

    # Load annotations
    annotations = load_dataset(
        "parquet",
        data_files="hf://datasets/c-i-ber/Nova/data/nova-v1.parquet",
        split="train"
    )

    # Load images
    images = load_dataset("c-i-ber/Nova", split="train")

    # Find a sample with annotation
    sample_idx = None
    for i in range(len(annotations)):
        ann = annotations[i]
        bboxes = ann.get('bboxes', [])
        gold_boxes = [b for b in bboxes if b.get('source') == 'gold']
        if gold_boxes:
            sample_idx = i
            break

    if sample_idx is None:
        raise ValueError("No annotated samples found")

    # Get sample annotation
    ann = annotations[sample_idx]
    filename = ann['filename']

    # Extract GT boxes
    gt_boxes = []
    for bbox in ann.get('bboxes', []):
        if bbox.get('source') == 'gold':
            x1, y1 = bbox['x'], bbox['y']
            x2, y2 = x1 + bbox['width'], y1 + bbox['height']
            gt_boxes.append([x1, y1, x2, y2])

    # Get image (need to find index from filename)
    ann_filenames = [annotations[i]['filename'] for i in range(len(annotations))]
    sorted_filenames = sorted(set(ann_filenames))
    img_idx = sorted_filenames.index(filename)

    img = images[img_idx]['image']
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    query_image = np.array(img.convert('RGB'))

    # Get healthy references (samples without annotations)
    print("Loading healthy reference images...")
    healthy_refs = []
    for i in range(len(annotations)):
        ann = annotations[i]
        bboxes = ann.get('bboxes', [])
        if not bboxes or len(bboxes) == 0:
            # This is a healthy sample
            fn = ann['filename']
            try:
                ref_img_idx = sorted_filenames.index(fn)
                ref_img = images[ref_img_idx]['image']
                if not isinstance(ref_img, Image.Image):
                    ref_img = Image.fromarray(ref_img)
                healthy_refs.append(np.array(ref_img.convert('RGB')))

                if len(healthy_refs) >= 20:
                    break
            except (ValueError, IndexError):
                continue

    print(f"✓ Loaded sample image: {filename}")
    print(f"  GT boxes: {len(gt_boxes)}")
    print(f"  Healthy references: {len(healthy_refs)}")

    return query_image, gt_boxes, healthy_refs, filename


def visualize_results(
    image: np.ndarray,
    gt_boxes: list,
    pred_boxes: list,
    output_path: str
):
    """Visualize predicted and GT boxes."""
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    # Draw GT boxes (green)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='green', width=3)

    # Draw predicted boxes (red, dashed)
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        # Normalize if needed
        if max(box) <= 1000 and min(box) >= 0:
            # Assuming 0-1000 normalized coordinates
            h, w = image.shape[:2]
            x1, y1, x2, y2 = x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000

        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

    img_pil.save(output_path)
    print(f"✓ Saved visualization to {output_path}")


def demo_reference_selection(query_image, healthy_refs):
    """Demonstrate Wasserstein reference selection."""
    print("\n" + "=" * 80)
    print("Demo: Wasserstein Reference Selection")
    print("=" * 80)

    # Initialize selector
    selector = WassersteinReferenceSelector(device="cuda")

    # Select references
    print("Selecting optimal references using Sliced Wasserstein Distance...")
    ref_indices = selector.select_references(
        query=query_image,
        reference_pool=healthy_refs,
        n_references=3,
        use_entropy_weighting=True
    )

    print(f"✓ Selected reference indices: {ref_indices}")
    print("\nThese references are the most anatomically similar to the query,")
    print("making them ideal for differential analysis.")

    return [healthy_refs[i] for i in ref_indices]


def demo_vlm_call_simulation(query_image, references):
    """
    Simulate a VLM call (for demo without API key).

    In practice, this would call the actual VLM API.
    """
    print("\n" + "=" * 80)
    print("Demo: VLM Differential Prompting (Simulated)")
    print("=" * 80)

    prompt = """You are a medical imaging expert. Compare the QUERY image (first) with the REFERENCE images (subsequent).

Task: Identify regions in the QUERY that appear DIFFERENT from the healthy reference.

Instructions:
1. Look for intensity differences, mass effects, or abnormal structures
2. The reference shows normal anatomy - use it to identify deviations
3. Return bounding boxes in normalized 0-1000 coordinates

Return JSON: {"boxes": [[x1, y1, x2, y2], ...], "description": "brief finding"}"""

    print("Prompt Template:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    print(f"\nImages sent to VLM:")
    print(f"  - 1 query image ({query_image.shape})")
    print(f"  - {len(references)} reference images")
    print("\nNote: Use run_inference.py with --api-key to run actual VLM inference")

    # Return simulated boxes for visualization
    h, w = query_image.shape[:2]
    simulated_boxes = [
        [200, 150, 350, 300],  # Simulated detection
    ]

    return simulated_boxes


def main():
    parser = argparse.ArgumentParser(description='WALDO Quickstart Demo')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for actual inference (optional for demo)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for visualizations')
    args = parser.parse_args()

    print("=" * 80)
    print("WALDO Quickstart Demo")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load sample data
    query_image, gt_boxes, healthy_refs, filename = load_sample_data()

    # Demo 1: Reference selection
    selected_refs = demo_reference_selection(query_image, healthy_refs)

    # Demo 2: VLM call (simulated)
    pred_boxes = demo_vlm_call_simulation(query_image, selected_refs)

    # Visualize
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    vis_path = output_dir / f"demo_{filename.replace('.png', '_result.png')}"
    visualize_results(query_image, gt_boxes, pred_boxes, str(vis_path))

    # Summary
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Get an API key from OpenAI or OpenRouter")
    print("2. Run full inference:")
    print("   python scripts/run_inference.py --api-key YOUR_KEY --n-samples 10")
    print("3. Analyze results:")
    print("   python scripts/read_results.py --dataset nova")
    print("\nFor more details, see USAGE.md")


if __name__ == '__main__':
    main()
