#!/usr/bin/env python3
"""
Download and prepare datasets for WALDO experiments.

This script automates downloading NOVA and VinDr-CXR datasets with proper
preprocessing and organization.

Usage:
    python scripts/download_datasets.py --dataset nova --output-dir data/nova
    python scripts/download_datasets.py --dataset cxr --output-dir data/cxr
    python scripts/download_datasets.py --all
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import json

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets library not installed. Run: pip install datasets")

def download_nova(output_dir: Path, cache_dir: Optional[Path] = None) -> Tuple[int, int]:
    """
    Download NOVA brain MRI dataset with annotations.

    The NOVA dataset contains brain MRI scans with bounding box annotations
    for various neurological findings. Images are downloaded from HuggingFace
    and annotations are extracted from the parquet file.

    Args:
        output_dir: Directory to save downloaded data
        cache_dir: Optional cache directory for HuggingFace downloads

    Returns:
        Tuple of (num_images, num_annotations)
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required. Install with: pip install datasets")

    print("=" * 80)
    print("Downloading NOVA Brain MRI Dataset")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = output_dir / "annotations"
    images_dir = output_dir / "images"
    annotations_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Set cache directory
    if cache_dir:
        os.environ['HF_HOME'] = str(cache_dir)

    # Download annotations (contains bboxes and metadata)
    print("\n[1/3] Downloading annotations from parquet...")
    annotations = load_dataset(
        "parquet",
        data_files="hf://datasets/c-i-ber/Nova/data/nova-v1.parquet",
        split="train",
        cache_dir=cache_dir
    )
    print(f"  ✓ Loaded {len(annotations)} annotations")

    # Download images
    print("\n[2/3] Downloading images...")
    images = load_dataset(
        "c-i-ber/Nova",
        split="train",
        cache_dir=cache_dir
    )
    print(f"  ✓ Loaded {len(images)} images")

    # Create filename to image index mapping (critical for correct alignment!)
    # IMPORTANT: Annotations are NOT sorted by filename, but images ARE!
    ann_filenames = [annotations[i]['filename'] for i in range(len(annotations))]
    sorted_filenames = sorted(set(ann_filenames))
    filename_to_img_idx = {fn: i for i, fn in enumerate(sorted_filenames)}

    # Save annotations as JSON with proper image alignment
    print("\n[3/3] Processing and saving annotations...")
    processed_annotations = []

    for i in range(len(annotations)):
        ann = annotations[i]
        filename = ann['filename']

        # Get GT bboxes (gold standard only). NOTE: some NOVA rows have bboxes=None
        # (key present but null), so `or []` is required, not just a default.
        gt_boxes = []
        for bbox in (ann.get('bboxes') or []):
            if bbox.get('source') == 'gold':
                x1, y1 = bbox['x'], bbox['y']
                x2, y2 = x1 + bbox['width'], y1 + bbox['height']
                gt_boxes.append([x1, y1, x2, y2])

        # Get metadata
        meta = ann.get('meta', {}) or {}
        diagnosis = meta.get('final_diagnosis', 'Unknown')

        # Get corresponding image index
        img_idx = filename_to_img_idx.get(filename)

        processed_annotations.append({
            'filename': filename,
            'image_index': img_idx,
            'diagnosis': diagnosis,
            'gt_boxes': gt_boxes,
            'image_size': [ann.get('width', 512), ann.get('height', 512)]
        })

    # Save annotations
    annotations_file = annotations_dir / "nova_annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(processed_annotations, f, indent=2)
    print(f"  ✓ Saved annotations to {annotations_file}")

    # Save filename mapping for reference
    mapping_file = annotations_dir / "filename_to_index.json"
    with open(mapping_file, 'w') as f:
        json.dump(filename_to_img_idx, f, indent=2)
    print(f"  ✓ Saved filename mapping to {mapping_file}")

    # Save sample images for quick access
    print("\n[Optional] Saving sample images (first 10)...")
    for i in range(min(10, len(images))):
        img = images[i]['image']
        if isinstance(img, Image.Image):
            img_path = images_dir / f"sample_{i:03d}.png"
            img.save(img_path)
    print(f"  ✓ Saved {min(10, len(images))} sample images to {images_dir}/")

    # Create README
    readme_content = f"""# NOVA Brain MRI Dataset

Downloaded: {len(images)} images
Annotations: {len(annotations)} samples

## Structure:
- annotations/nova_annotations.json: All annotations with GT boxes and diagnoses
- annotations/filename_to_index.json: Mapping from filenames to image indices
- images/: Sample images (first 10 for reference)

## Important Notes:
1. Images and annotations have DIFFERENT orderings in the original dataset
2. Use filename_to_index.json to correctly align annotations with images
3. GT boxes are in pixel coordinates (typically 512x512 images)
4. Only 'gold' standard annotations are included

## Access Full Images:
The full dataset is cached by HuggingFace and can be accessed programmatically:

```python
from datasets import load_dataset
images = load_dataset("c-i-ber/Nova", split="train")
img = images[image_index]['image']  # Use image_index from annotations
```

## Citation:
@misc{{nova2024,
  title={{NOVA: A Large-Scale Medical Imaging Dataset for Neurological Condition Assessment}},
  author={{...}},
  year={{2024}},
  publisher={{HuggingFace}}
}}
"""

    readme_file = output_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"\n✓ Created README at {readme_file}")

    print("\n" + "=" * 80)
    print(f"✓ NOVA dataset downloaded successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    print("=" * 80)

    return len(images), len(annotations)


def download_vindr_cxr(output_dir: Path, subset: str = "test") -> int:
    """
    Download VinDr-CXR chest X-ray dataset.

    Note: VinDr-CXR requires PhysioNet credentials and cannot be automatically
    downloaded. This function provides instructions and helper utilities.

    Args:
        output_dir: Directory to save dataset info
        subset: Which subset to prepare ("test", "train", or "all")

    Returns:
        0 (dataset must be manually downloaded)
    """
    print("=" * 80)
    print("VinDr-CXR Dataset Setup")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    instructions = f"""# VinDr-CXR Chest X-Ray Dataset

## Manual Download Required

VinDr-CXR is hosted on PhysioNet and requires credentialed access:

1. Create account at: https://physionet.org/register/
2. Complete CITI training: https://physionet.org/about/citi-course/
3. Request access: https://physionet.org/content/vindr-cxr/1.0.0/
4. Download using wget:
   ```bash
   wget -r -N -c -np --user YOUR_USERNAME --ask-password \\
     https://physionet.org/files/vindr-cxr/1.0.0/
   ```

## Expected Structure:
```
{output_dir}/
├── annotations/
│   ├── annotations_train.csv
│   └── annotations_test.csv
├── images/
│   ├── train/  (preprocessed NPZ files)
│   └── test/   (preprocessed NPZ files)
└── dicom/      (optional, original DICOMs)
```

## Preprocessing CXR Images:

Once downloaded, preprocess DICOM files to NPZ:

```python
import numpy as np
import pydicom
from pathlib import Path

def preprocess_dicom_to_npz(dicom_path, output_path, target_size=256):
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(float)

    # Normalize to 0-1
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Resize if needed
    if img.shape != (target_size, target_size):
        from PIL import Image
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((target_size, target_size))
        img = np.array(img_pil) / 255.0

    np.savez_compressed(output_path, image=img)
```

## Dataset Statistics (Test Set):
- Images: ~3,000 chest X-rays
- Annotations: ~15,000 bounding boxes
- Findings: 14 different pathologies
- "No Finding" cases: ~10,606 (used as healthy references)

## Citation:
@article{{nguyen2022vindr,
  title={{VinDr-CXR: An open dataset of chest X-rays with radiologist annotations}},
  author={{Nguyen, Ha Q and Lam, Khanh and ...}},
  journal={{Scientific Data}},
  year={{2022}}
}}
"""

    readme_file = output_dir / "README_DOWNLOAD.md"
    with open(readme_file, 'w') as f:
        f.write(instructions)

    print(f"\n✓ Created download instructions at {readme_file}")
    print("\nIMPORTANT: VinDr-CXR requires manual download from PhysioNet.")
    print(f"Please follow instructions in {readme_file}")
    print("=" * 80)

    return 0


def create_healthy_references(dataset: str, data_dir: Path, output_dir: Path, n_samples: int = 50):
    """
    Extract healthy reference images from dataset.

    Args:
        dataset: "nova" or "cxr"
        data_dir: Directory containing the dataset
        output_dir: Directory to save healthy references
        n_samples: Number of healthy samples to extract
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "nova":
        print(f"\n[NOVA] Extracting {n_samples} healthy references...")

        # Load annotations
        ann_file = data_dir / "annotations" / "nova_annotations.json"
        if not ann_file.exists():
            print(f"  ✗ Annotations not found at {ann_file}")
            return

        with open(ann_file) as f:
            annotations = json.load(f)

        # Find samples with no findings (empty gt_boxes)
        healthy_samples = [
            ann for ann in annotations
            if not ann.get('gt_boxes') or len(ann.get('gt_boxes', [])) == 0
        ]

        print(f"  Found {len(healthy_samples)} healthy samples")

        # Save list of healthy sample indices
        healthy_indices = [ann['image_index'] for ann in healthy_samples[:n_samples]]
        output_file = output_dir / "nova_healthy_indices.json"
        with open(output_file, 'w') as f:
            json.dump(healthy_indices, f, indent=2)

        print(f"  ✓ Saved healthy reference indices to {output_file}")

    elif dataset == "cxr":
        print(f"\n[CXR] Instructions for extracting healthy references...")
        print("  1. Filter annotations CSV for 'No finding' class")
        print("  2. Randomly sample images from those cases")
        print("  3. Save image IDs to healthy_references.json")
        print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare datasets for WALDO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download NOVA only
  python scripts/download_datasets.py --dataset nova

  # Download all datasets
  python scripts/download_datasets.py --all

  # Specify custom cache and output directories
  python scripts/download_datasets.py --dataset nova \\
    --output-dir /data/nova \\
    --cache-dir /cache/huggingface
        """
    )
    parser.add_argument('--dataset', type=str, choices=['nova', 'cxr', 'all'],
                        default='all', help='Dataset to download')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for datasets')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache directory for HuggingFace downloads')
    parser.add_argument('--extract-healthy', action='store_true',
                        help='Extract healthy reference samples')
    parser.add_argument('--n-healthy', type=int, default=50,
                        help='Number of healthy samples to extract')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.dataset in ['nova', 'all']:
        nova_dir = output_dir / 'nova'
        download_nova(nova_dir, cache_dir)

        if args.extract_healthy:
            healthy_dir = output_dir / 'healthy_references'
            create_healthy_references('nova', nova_dir, healthy_dir, args.n_healthy)

    if args.dataset in ['cxr', 'all']:
        cxr_dir = output_dir / 'cxr'
        download_vindr_cxr(cxr_dir)

        if args.extract_healthy:
            healthy_dir = output_dir / 'healthy_references'
            print("\nCXR healthy reference extraction requires manual annotation filtering.")
            print(f"See instructions in {cxr_dir}/README_DOWNLOAD.md")

    print("\n✓ Dataset preparation complete!")


if __name__ == '__main__':
    main()
