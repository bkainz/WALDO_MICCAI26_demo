"""
Data loading utilities for WALDO.

This module provides comprehensive dataset loaders for NOVA and VinDr-CXR
with proper preprocessing, caching, and batch handling.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass
import csv
from PIL import Image
from tqdm import tqdm

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class DataSample:
    """Container for a single data sample."""
    image_id: str
    image: np.ndarray  # RGB numpy array (H, W, 3)
    gt_boxes: List[List[float]]  # Ground truth boxes [[x1, y1, x2, y2], ...]
    metadata: Dict  # Additional metadata (diagnosis, modality, etc.)

    def __post_init__(self):
        """Validate data after initialization."""
        assert self.image.ndim == 3, f"Image must be 3D, got shape {self.image.shape}"
        assert self.image.shape[2] == 3, f"Image must be RGB, got {self.image.shape[2]} channels"


class NOVADataLoader:
    """
    DataLoader for NOVA brain MRI dataset.

    Handles proper annotation-image alignment, which is critical because:
    - Parquet annotations are NOT sorted by filename
    - Images from HuggingFace ARE sorted alphabetically by filename
    - Without correct mapping, GT boxes will be paired with wrong images!
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        load_images_on_init: bool = False
    ):
        """
        Initialize NOVA data loader.

        Args:
            cache_dir: HuggingFace cache directory
            load_images_on_init: Whether to load all images immediately (uses more memory)
        """
        self.cache_dir = cache_dir
        self._annotations = None
        self._images_dataset = None
        self._filename_to_idx = None
        # Cache processed annotations PER filter_empty value: calling get_sample
        # (filter_empty=True) then get_healthy_samples (filter_empty=False) must not
        # share a cache, or the healthy-reference pool would come back empty.
        self._processed_cache = {}

        if load_images_on_init:
            self._load_datasets()

    def _load_datasets(self):
        """Load datasets from HuggingFace."""
        if not HAS_DATASETS:
            raise ImportError("datasets library required: pip install datasets")

        if self._annotations is None:
            print("Loading NOVA annotations...")
            self._annotations = load_dataset(
                "parquet",
                data_files="hf://datasets/c-i-ber/Nova/data/nova-v1.parquet",
                split="train",
                cache_dir=self.cache_dir
            )

        if self._images_dataset is None:
            print("Loading NOVA images...")
            self._images_dataset = load_dataset(
                "c-i-ber/Nova",
                split="train",
                cache_dir=self.cache_dir
            )

        # Create filename to image index mapping (CRITICAL!)
        if self._filename_to_idx is None:
            ann_filenames = [self._annotations[i]['filename'] for i in range(len(self._annotations))]
            sorted_filenames = sorted(set(ann_filenames))
            self._filename_to_idx = {fn: i for i, fn in enumerate(sorted_filenames)}

    def _process_annotations(self, filter_empty: bool = True) -> List[Dict]:
        """
        Process annotations and create aligned dataset.

        Args:
            filter_empty: If True, only include samples with GT boxes

        Returns:
            List of processed annotation dictionaries
        """
        if filter_empty in self._processed_cache:
            return self._processed_cache[filter_empty]

        self._load_datasets()

        processed = []
        for i in range(len(self._annotations)):
            ann = self._annotations[i]
            filename = ann['filename']

            # Extract GT boxes (gold standard only). Some NOVA rows have bboxes=None
            # (key present but null), so `or []` is required, not just a default.
            gt_boxes = []
            for bbox in (ann.get('bboxes') or []):
                if bbox.get('source') == 'gold':
                    x1, y1 = bbox['x'], bbox['y']
                    x2, y2 = x1 + bbox['width'], y1 + bbox['height']
                    gt_boxes.append([x1, y1, x2, y2])

            # Skip if no boxes and filtering enabled
            if filter_empty and not gt_boxes:
                continue

            # Get metadata
            meta = ann.get('meta', {}) or {}

            processed.append({
                'filename': filename,
                'image_index': self._filename_to_idx[filename],
                'gt_boxes': gt_boxes,
                'diagnosis': meta.get('final_diagnosis', 'Unknown'),
                'modality': meta.get('modality', 'MRI'),
                'image_size': [ann.get('width', 512), ann.get('height', 512)]
            })

        self._processed_cache[filter_empty] = processed
        return processed

    def get_sample(self, index: int, filter_empty: bool = True) -> DataSample:
        """
        Get a single sample by index.

        Args:
            index: Index in processed annotations
            filter_empty: Whether to filter empty annotations

        Returns:
            DataSample object
        """
        annotations = self._process_annotations(filter_empty=filter_empty)
        ann = annotations[index]

        # Load image
        img_idx = ann['image_index']
        img = self._images_dataset[img_idx]['image']

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img_array = np.array(img.convert('RGB'))

        return DataSample(
            image_id=ann['filename'],
            image=img_array,
            gt_boxes=ann['gt_boxes'],
            metadata={
                'diagnosis': ann['diagnosis'],
                'modality': ann['modality'],
                'image_size': ann['image_size']
            }
        )

    def get_healthy_samples(self, n_samples: int = 50) -> List[DataSample]:
        """
        Get healthy reference samples (no findings).

        Args:
            n_samples: Number of healthy samples to return

        Returns:
            List of DataSample objects with no GT boxes
        """
        # Process without filtering
        annotations = self._process_annotations(filter_empty=False)

        healthy_samples = []
        for ann in annotations:
            if not ann['gt_boxes']:
                sample = DataSample(
                    image_id=ann['filename'],
                    image=self._get_image_array(ann['image_index']),
                    gt_boxes=[],
                    metadata={'diagnosis': 'Healthy', 'modality': ann['modality']}
                )
                healthy_samples.append(sample)

                if len(healthy_samples) >= n_samples:
                    break

        return healthy_samples

    def _get_image_array(self, img_idx: int) -> np.ndarray:
        """Helper to get image as numpy array."""
        img = self._images_dataset[img_idx]['image']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return np.array(img.convert('RGB'))

    def __len__(self) -> int:
        """Get number of samples (with annotations)."""
        return len(self._process_annotations(filter_empty=True))

    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over all samples."""
        for i in range(len(self)):
            yield self.get_sample(i)


class VinDrCXRDataLoader:
    """
    DataLoader for VinDr-CXR chest X-ray dataset.

    Handles loading from preprocessed NPZ files and CSV annotations.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "test",
        load_dicom: bool = False
    ):
        """
        Initialize VinDr-CXR data loader.

        Args:
            data_dir: Root directory containing VinDr-CXR data
            split: Which split to load ("train" or "test")
            load_dicom: Whether to load from DICOM (slower) or NPZ (faster)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.load_dicom = load_dicom

        # Paths
        self.annotations_file = self.data_dir / "annotations" / f"annotations_{split}.csv"
        self.images_dir = self.data_dir / "images" / split

        if not self.annotations_file.exists():
            raise FileNotFoundError(
                f"Annotations not found at {self.annotations_file}. "
                "Please download VinDr-CXR dataset first."
            )

        # Load annotations
        self._load_annotations()

    def _load_annotations(self):
        """Load and process annotations from CSV."""
        print(f"Loading VinDr-CXR annotations from {self.annotations_file}...")

        # Group annotations by image_id
        self.annotations_by_image = {}

        with open(self.annotations_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row['image_id']

                if img_id not in self.annotations_by_image:
                    self.annotations_by_image[img_id] = {
                        'image_id': img_id,
                        'boxes': [],
                        'classes': [],
                        'rad_id': row.get('rad_id', ''),
                    }

                # Only include if there's a finding (not "No finding")
                if row['class_name'] != 'No finding':
                    # Parse box coordinates
                    x1 = float(row['x_min'])
                    y1 = float(row['y_min'])
                    x2 = float(row['x_max'])
                    y2 = float(row['y_max'])

                    self.annotations_by_image[img_id]['boxes'].append([x1, y1, x2, y2])
                    self.annotations_by_image[img_id]['classes'].append(row['class_name'])

        # Filter to only images with findings
        self.image_ids_with_findings = [
            img_id for img_id, ann in self.annotations_by_image.items()
            if ann['boxes']
        ]

        print(f"  ✓ Loaded {len(self.image_ids_with_findings)} images with findings")

    def get_sample(self, index: int) -> DataSample:
        """
        Get a single sample by index.

        Args:
            index: Index in the dataset

        Returns:
            DataSample object
        """
        img_id = self.image_ids_with_findings[index]
        ann = self.annotations_by_image[img_id]

        # Load image
        img_array = self._load_image(img_id)

        return DataSample(
            image_id=img_id,
            image=img_array,
            gt_boxes=ann['boxes'],
            metadata={
                'classes': ann['classes'],
                'modality': 'CXR',
                'rad_id': ann['rad_id']
            }
        )

    def _load_image(self, img_id: str) -> np.ndarray:
        """Load image from NPZ or DICOM."""
        if self.load_dicom:
            # Load from DICOM
            dicom_path = self.images_dir.parent.parent / "dicom" / self.split / f"{img_id}.dicom"
            if not dicom_path.exists():
                raise FileNotFoundError(f"DICOM not found: {dicom_path}")

            import pydicom
            dcm = pydicom.dcmread(dicom_path)
            img = dcm.pixel_array.astype(float)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        else:
            # Load from preprocessed NPZ
            npz_path = self.images_dir / f"{img_id}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"NPZ not found: {npz_path}. "
                    "Please preprocess DICOMs to NPZ format first."
                )

            img = np.load(npz_path)['image']

        # Convert to RGB
        img_rgb = (img * 255).astype(np.uint8)
        img_rgb = np.stack([img_rgb] * 3, axis=-1)

        return img_rgb

    def get_healthy_samples(self, n_samples: int = 50) -> List[DataSample]:
        """
        Get healthy reference samples ("No finding" cases).

        Args:
            n_samples: Number of healthy samples to return

        Returns:
            List of DataSample objects
        """
        # Find "No finding" cases
        no_finding_ids = []
        with open(self.annotations_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class_name'] == 'No finding':
                    no_finding_ids.append(row['image_id'])

        # Remove duplicates and limit
        no_finding_ids = list(set(no_finding_ids))[:n_samples]

        healthy_samples = []
        for img_id in tqdm(no_finding_ids, desc="Loading healthy references"):
            try:
                img_array = self._load_image(img_id)
                healthy_samples.append(DataSample(
                    image_id=img_id,
                    image=img_array,
                    gt_boxes=[],
                    metadata={'diagnosis': 'No finding', 'modality': 'CXR'}
                ))
            except FileNotFoundError:
                continue

        return healthy_samples

    def __len__(self) -> int:
        """Get number of samples with findings."""
        return len(self.image_ids_with_findings)

    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over all samples."""
        for i in range(len(self)):
            yield self.get_sample(i)


def get_dataloader(
    dataset: str,
    data_dir: Optional[Path] = None,
    **kwargs
) -> object:
    """
    Factory function to get appropriate dataloader.

    Args:
        dataset: Dataset name ("nova" or "cxr")
        data_dir: Data directory (optional for NOVA, required for CXR)
        **kwargs: Additional arguments passed to dataloader

    Returns:
        Dataset loader instance
    """
    if dataset.lower() == "nova":
        return NOVADataLoader(**kwargs)
    elif dataset.lower() in ["cxr", "vindr", "vindr-cxr"]:
        if data_dir is None:
            raise ValueError("data_dir required for VinDr-CXR dataset")
        return VinDrCXRDataLoader(data_dir=data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
