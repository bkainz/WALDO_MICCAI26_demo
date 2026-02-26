# WALDO: Wasserstein-Aligned Localisation via Differential Observations

A training-free framework for zero-shot medical anomaly localisation using vision-language models with optimal transport-based reference selection.

<p align="center">
  <img src="figures/waldo_method_overview.png" width="100%">
</p>

## Key Features

- **Zero-shot localisation**: No training on target domain required
- **Differential prompting**: Compare query images against healthy references to identify anomalies
- **Optimal transport reference selection**: Use Sliced Wasserstein Distance on DINOv2 embeddings to select diverse, relevant references
- **Self-consistency aggregation**: Aggregate predictions across multiple reference sets using weighted NMS

## Results

### NOVA Brain MRI

| Method | mAP@30 | mAP@50 | Avg IoU |
|--------|--------|--------|---------|
| Zero-shot (Qwen3-32B) | 53.3% | 30.0% | 32.9% |
| **WALDO (Qwen3-32B)** | **63.3%** | **43.3%** | **39.2%** |
| Improvement | +10.0% | +13.3% | +6.3% |

### VinDr-CXR (n=949)

| Method | mAP@30 | mAP@50 | Avg IoU |
|--------|--------|--------|---------|
| Zero-shot (Qwen3-32B) | 12.8% | 4.3% | 8.9% |
| **WALDO (Qwen3-32B)** | **34.1%** | **10.7%** | **22.2%** |
| Improvement | +21.3% | +6.4% | +13.3% |

<p align="center">
  <img src="figures/nova_analysis_violin.png" width="48%">
</p>

*IoU distribution analysis by lesion size (left panels) and disease type (right panels). Left two: NOVA, Right two: VinDr-CXR*

## Getting Started

### Quick Installation

```bash
# Clone repository
git clone https://github.com/bkainz/WALDO_MICCAI26_demo.git
cd WALDO_MICCAI26_demo

# Install with pip
pip install -e .
```

### Quick Start (3 commands)

```bash
# 1. Download NOVA dataset
python scripts/download_datasets.py --dataset nova --extract-healthy

# 2. Run inference (requires API key)
export OPENAI_API_KEY="your-key-here"
python scripts/run_inference.py --dataset nova --model gpt-4o --n-samples 10

# 3. Analyze results
python scripts/read_results.py --dataset nova
```

### Try the Demo (No API Key Required)

```bash
# Run quickstart demo
python examples/quickstart.py

# This demonstrates:
# - Dataset loading
# - Wasserstein reference selection
# - Differential prompting workflow
```

For detailed instructions, see [USAGE.md](USAGE.md).


## Installation

### Option 1: Install as Package (Recommended)

```bash
pip install -e .

# With optional dependencies
pip install -e ".[viz,cxr]"  # For visualization and CXR preprocessing
```

### Option 2: Manual Requirements

```bash
pip install -r requirements.txt
```

## Utilities and Scripts

### Dataset Management

**`scripts/download_datasets.py`** - Automated dataset downloading and preprocessing
```bash
# Download NOVA with healthy reference extraction
python scripts/download_datasets.py --dataset nova --extract-healthy --n-healthy 50

# Get VinDr-CXR download instructions
python scripts/download_datasets.py --dataset cxr

# Custom directories
python scripts/download_datasets.py --dataset nova \\
    --output-dir /data/nova --cache-dir /cache/hf
```

Features:
- Automatic NOVA download from HuggingFace
- Proper annotation-image alignment (critical for NOVA)
- Healthy reference extraction
- VinDr-CXR setup instructions

### Inference

**`scripts/run_inference.py`** - Complete end-to-end inference pipeline
```bash
# Run WALDO with GPT-4o
python scripts/run_inference.py --dataset nova --model gpt-4o --api-key YOUR_KEY

# Use Qwen via OpenRouter
python scripts/run_inference.py --dataset nova \\
    --model qwen/qwen-2.5-72b-instruct --openrouter --api-key YOUR_KEY

# Advanced configuration
python scripts/run_inference.py --dataset nova --model gpt-4o \\
    --n-samples 100 --n-views 7 --n-references 5 --n-ref-pool 100
```

Features:
- Automated dataset loading
- Wasserstein reference selection
- VLM API integration (OpenAI, OpenRouter)
- Evaluation and metrics
- Checkpoint saving

### Analysis

**`scripts/read_results.py`** - Comprehensive results analysis
```bash
# Analyze all pre-computed results
python scripts/read_results.py --dataset all

# Analyze specific dataset
python scripts/read_results.py --dataset nova --results-dir custom_results/
```

Features:
- mAP@30, mAP@50, Average IoU
- 95% confidence intervals
- Formatted result tables

### Examples

**`examples/quickstart.py`** - Interactive demo (no API key required)
```bash
python examples/quickstart.py --output-dir outputs/
```

Features:
- Dataset loading demonstration
- Wasserstein reference selection walkthrough
- Visualization generation
- Educational workflow explanation

## Quick Start

### Reading Pre-computed Results

The `results/` directory contains raw predictions for all experiments in the paper:

```bash
# Analyze all results
python scripts/read_results.py --dataset all

# Analyze specific dataset
python scripts/read_results.py --dataset nova
python scripts/read_results.py --dataset cxr
```

### Using WALDO

```python
from waldo import WALDO, WassersteinReferenceSelector

# Initialize with your VLM client
waldo = WALDO(
    vlm_client=your_openai_client,
    model="qwen2.5-vl-72b",
    n_views=5,
    n_references=3,
)

# Localize anomalies
result = waldo.localize(
    query_image=query_rgb,
    reference_pool=healthy_references,
    modality="mri"  # or "cxr"
)

print(f"Found {len(result['boxes'])} anomaly regions")
```

## Repository Structure

```
waldo-demo/
├── waldo/                         # Core WALDO implementation
│   ├── __init__.py
│   ├── waldo.py                   # Main WALDO class
│   ├── reference_selector.py      # SWD-based reference selection
│   └── metrics.py                 # Evaluation metrics
├── scripts/
│   └── read_results.py            # Results reader and analyzer
├── results/                        # Raw experimental results
│   ├── nova/                      # NOVA brain MRI results (2 files)
│   └── cxr/                       # VinDr-CXR results (6 files)
├── figures/
│   ├── waldo_method_overview.png  # Method diagram
│   ├── nova_analysis_violin.png   # NOVA IoU analysis
│   ├── cxr_analysis_violin.png    # CXR IoU analysis
│   ├── disease_examples_grid.pdf  # Disease type examples
│   ├── prompts/                   # Prompt examples (3 files)
│   └── qualitative_samples/       # Localisation examples
│       ├── nova/                  # 30 brain MRI samples
│       └── cxr/                   # 30 chest X-ray samples
└── requirements.txt
```

## Results File Format

Each JSON results file contains:

```json
{
  "results": [
    {
      "image_id": "...",
      "iou": 0.52,
      "hit_30": true,
      "hit_50": true,
      "n_pred": 2,
      "n_gt": 3,
      "pred_boxes": [[x1, y1, x2, y2], ...],
      "gt_boxes": [[x1, y1, x2, y2], ...]
    },
    ...
  ]
}
```

## Method Overview

WALDO consists of three stages:

1. **Reference Selection**: Use entropy-weighted Sliced Wasserstein Distance on DINOv2 patch embeddings to select diverse, anatomically-aligned healthy references

2. **Differential Prompting**: Query a VLM with the patient image and selected references, asking it to identify regions that differ from normal anatomy

3. **Self-Consistency Aggregation**: Repeat with different reference subsets and aggregate predictions using weighted NMS

## Citation

```bibtex
@inproceedings{waldo2026,
  title={WALDO: Wasserstein-Aligned Localisation via Differential Observations for Zero-Shot Medical Anomaly Detection},
  author={...},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2026}
}
```

## License

MIT License
