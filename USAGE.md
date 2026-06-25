<!-- This enhanced usage guide. WALDO demo repository -->

# WALDO Usage Guide

Complete guide for using WALDO for zero-shot medical anomaly localization.

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Running Inference](#running-inference)
- [Evaluating Results](#evaluating-results)
- [API Configuration](#api-configuration)
- [Advanced Usage](#advanced-usage)

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Download NOVA dataset
python scripts/download_datasets.py --dataset nova

# 3. Run inference (requires API key)
export OPENAI_API_KEY="your-key-here"
python scripts/run_inference.py --dataset nova --model gpt-4o --n-samples 10

# 4. Analyze results
python scripts/read_results.py --dataset nova
```

## Installation

### Basic Installation

```bash
# Clone repository
git clone https://github.com/bkainz/WALDO_MICCAI26_demo.git
cd WALDO_MICCAI26_demo

# Install with pip
pip install -e .
```

### With Optional Dependencies

```bash
# For visualization
pip install -e ".[viz]"

# For CXR DICOM preprocessing
pip install -e ".[cxr]"

# For development
pip install -e ".[dev]"

# All extras
pip install -e ".[dev,viz,cxr]"
```

### Manual Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### NOVA Brain MRI Dataset

NOVA is automatically downloaded from HuggingFace:

```bash
# Download dataset and extract healthy references
python scripts/download_datasets.py --dataset nova --extract-healthy

# Specify custom directories
python scripts/download_datasets.py --dataset nova \\
    --output-dir /data/nova \\
    --cache-dir /cache/huggingface
```

**Dataset Structure:**
```
data/nova/
├── annotations/
│   ├── nova_annotations.json          # All annotations with GT boxes
│   └── filename_to_index.json         # Image index mapping
├── images/
│   └── sample_*.png                   # Sample images
└── README.md
```

**Important Notes:**
- Images and annotations have different orderings in the original dataset
- Use `filename_to_index.json` for correct alignment
- Only 'gold' standard annotations are included
- GT boxes are in pixel coordinates (typically 512x512)

### VinDr-CXR Dataset

VinDr-CXR requires manual download from PhysioNet:

```bash
# Generate download instructions
python scripts/download_datasets.py --dataset cxr
# Follow instructions in data/cxr/README_DOWNLOAD.md
```

**Steps:**
1. Create PhysioNet account: https://physionet.org/register/
2. Complete CITI training
3. Request access to VinDr-CXR
4. Download using provided wget command
5. Preprocess DICOMs to NPZ format

## Running Inference

### Basic Usage

```bash
# With GPT-4o
python scripts/run_inference.py \\
    --dataset nova \\
    --model gpt-4o \\
    --api-key YOUR_KEY \\
    --n-samples 10

# With Qwen via OpenRouter
python scripts/run_inference.py \\
    --dataset nova \\
    --model qwen/qwen-2.5-72b-instruct \\
    --api-key YOUR_OPENROUTER_KEY \\
    --openrouter \\
    --n-samples 10
```

### Advanced Configuration

```bash
python scripts/run_inference.py \\
    --dataset nova \\
    --model gpt-4o \\
    --n-samples 100 \\
    --n-references 5 \\         # K references per query (paper: 5 = 3 Stage 1 + 2 Stage 2)
    --n-ref-pool 100 \\         # Larger healthy reference pool
    --output results/my_experiment.json
```

### Parameters

- `--dataset`: Dataset to use (`nova` or `cxr`)
- `--model`: VLM model name
- `--api-key`: API key (or set `OPENAI_API_KEY` env var)
- `--openrouter`: Use OpenRouter API instead of OpenAI
- `--n-samples`: Number of test samples
- `--n-references`: K references per query (default: 5; split 3 Stage 1 + 2 Stage 2)
- `--n-ref-pool`: Size of healthy reference pool (default: 50)
- `--device`: Device for the DINOv3 backbone (`cuda` or `cpu`)
- `--output`: Output JSON file path

## Evaluating Results

### Analyze Pre-computed Results

```bash
# Analyze all results
python scripts/read_results.py --dataset all

# Analyze specific dataset
python scripts/read_results.py --dataset nova
python scripts/read_results.py --dataset cxr

# Specify custom results directory
python scripts/read_results.py --results-dir my_results/
```

### Evaluate Custom Results File

```bash
python scripts/run_inference.py \\
    --eval-only \\
    --results results/my_experiment.json
```

### Metrics Computed

- **mAP@30**: Mean Average Precision at IoU=0.30
- **mAP@50**: Mean Average Precision at IoU=0.50
- **Avg IoU**: Average Intersection over Union
- **95% CI**: Confidence intervals for all metrics

## API Configuration

### OpenAI API

```bash
export OPENAI_API_KEY="sk-..."
python scripts/run_inference.py --model gpt-4o
```

### OpenRouter API

```bash
export OPENAI_API_KEY="sk-or-..."
python scripts/run_inference.py \\
    --model qwen/qwen-2.5-72b-instruct \\
    --openrouter
```

### Custom API Endpoint

Modify the `base_url` in `run_inference.py`:

```python
waldo = WALDOWithVLM(
    api_key=api_key,
    model=args.model,
    base_url="https://your-custom-endpoint.com/v1"
)
```

## Advanced Usage

### Using WALDO Programmatically

```python
from waldo import WALDO
from waldo.reference_selector import WassersteinReferenceSelector
import numpy as np

# Initialize
waldo = WALDO(
    vlm_client=your_vlm_client,
    model="gpt-4o",
    n_views=5,
    n_references=3,
)

# Prepare data
query_image = np.array(...)  # Shape: (H, W, 3)
reference_pool = [np.array(...), ...]  # List of reference images

# Run inference
result = waldo.localize(
    query_image=query_image,
    reference_pool=reference_pool,
    modality="mri"  # or "cxr"
)

# Extract results
predicted_boxes = result['boxes']  # [[x1, y1, x2, y2], ...]
confidences = result['confidences']
```

### Custom Reference Selection

```python
from waldo.reference_selector import WassersteinReferenceSelector

# Initialize selector (DINOv3-ViT-B/16 + entropy-weighted Sliced Wasserstein)
selector = WassersteinReferenceSelector(device="cuda")

# Select K diverse Goldilocks-zone references
ref_indices = selector.select_references(
    query_image=query_image,
    reference_images=reference_pool,
    n_references=5,
)

# Or get the per-reference SW distance to the query as well:
scored = selector.select_references_with_scores(query_image, reference_pool, n_references=5)

selected_refs = [reference_pool[i] for i in ref_indices]
```

### Custom VLM Integration

Extend the `WALDO` class and implement `_call_vlm`:

```python
class CustomWALDO(WALDO):
    def _call_vlm(self, images: List[np.ndarray], prompt: str) -> str:
        # Your VLM API call here
        response = your_vlm_api(images, prompt)
        return response
```

### Batch Processing

```python
import json
from pathlib import Path
from tqdm import tqdm
from waldo.metrics import compute_best_iou  # best IoU between any pred and any GT box

results = []
for i, (query, gt_boxes) in enumerate(tqdm(dataset)):
    output = waldo.localize(query, reference_pool, modality="mri")

    results.append({
        'image_id': i,
        'pred_boxes': output['boxes'],
        'gt_boxes': gt_boxes,
        'iou': compute_best_iou(output['boxes'], gt_boxes)
    })

    # Save checkpoint every 100 samples
    if (i + 1) % 100 == 0:
        with open(f'checkpoint_{i+1}.json', 'w') as f:
            json.dump(results, f)
```

## Troubleshooting

### Dataset Download Issues

**Problem**: HuggingFace dataset download fails

**Solution**:
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets/c-i-ber___nova
python scripts/download_datasets.py --dataset nova
```

### API Rate Limiting

**Problem**: API rate limit exceeded

**Solution**:
- Reduce `n_samples` for testing
- Add delays between requests
- Use batch processing with checkpoints

### Memory Issues

**Problem**: Out of memory with large reference pools

**Solution**:
- Reduce `n_ref_pool`
- Use smaller images (resize before processing)
- Process in batches

### DINOv3 backbone issues

**Problem**: DINOv3 fails to load (it is gated on the HuggingFace Hub) or runs out of GPU memory.

**Solution**:
```bash
# Accept the licence at huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m, then:
huggingface-cli login
```
```python
# Force CPU for the backbone:
waldo = WALDO(..., device="cpu")
```
If DINOv3 cannot be loaded, the selector automatically falls back to `facebook/dinov2-base`
with a printed warning.

## Citation

If you use WALDO in your research, please cite:

```bibtex
@inproceedings{kainz2026waldo,
  title     = {Wasserstein-Aligned Localisation for VLM-Based Distributional OOD
               Detection in Medical Imaging},
  author    = {Kainz, Bernhard and Mueller, Johanna P. and Baugh, Matthew M. G.
               and Bercea, Cosmin I.},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

## Support

- **Issues**: https://github.com/bkainz/WALDO_MICCAI26_demo/issues
- **Disclaimer**: see [DISCLAIMER.md](DISCLAIMER.md) (AI-assisted distillation; intended use)
