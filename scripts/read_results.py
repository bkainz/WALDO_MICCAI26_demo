#!/usr/bin/env python3
"""
Read and analyze WALDO evaluation results.

This script loads the pre-computed results from the paper and computes
evaluation metrics.

Usage:
    python scripts/read_results.py --dataset nova
    python scripts/read_results.py --dataset cxr
    python scripts/read_results.py --all
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waldo.metrics import compute_map, compute_confidence_interval


def load_results(results_path: str) -> Dict:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def summarize_results(results: List[Dict], name: str) -> Dict:
    """Compute summary statistics for results."""
    metrics = compute_map(results)

    # Compute confidence intervals
    hits_30 = [1 if r.get('hit_30', r.get('iou', 0) >= 0.3) else 0 for r in results]
    hits_50 = [1 if r.get('hit_50', r.get('iou', 0) >= 0.5) else 0 for r in results]
    ious = [r.get('iou', 0) for r in results]

    mean_30, ci_low_30, ci_high_30 = compute_confidence_interval(hits_30)
    mean_50, ci_low_50, ci_high_50 = compute_confidence_interval(hits_50)
    mean_iou, ci_low_iou, ci_high_iou = compute_confidence_interval(ious)

    return {
        'name': name,
        'n_samples': len(results),
        'mAP@30': mean_30 * 100,
        'mAP@30_CI': [ci_low_30 * 100, ci_high_30 * 100],
        'mAP@50': mean_50 * 100,
        'mAP@50_CI': [ci_low_50 * 100, ci_high_50 * 100],
        'avg_iou': mean_iou * 100,
        'avg_iou_CI': [ci_low_iou * 100, ci_high_iou * 100],
    }


def print_results_table(summaries: List[Dict]):
    """Print results as formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Method':<30} {'n':>6} {'mAP@30':>12} {'mAP@50':>12} {'Avg IoU':>12}")
    print("-" * 80)

    for s in summaries:
        ci_30 = f"[{s['mAP@30_CI'][0]:.1f}, {s['mAP@30_CI'][1]:.1f}]"
        print(f"{s['name']:<30} {s['n_samples']:>6} "
              f"{s['mAP@30']:>5.1f}% {ci_30:<6} "
              f"{s['mAP@50']:>5.1f}% "
              f"{s['avg_iou']:>5.1f}%")

    print("=" * 80)


def analyze_nova(results_dir: Path):
    """Analyze NOVA brain MRI results."""
    print("\n" + "=" * 80)
    print("NOVA Brain MRI Results")
    print("=" * 80)

    summaries = []

    # Load WALDO results
    waldo_files = list(results_dir.glob("nova_waldo_*.json"))
    for f in sorted(waldo_files):
        data = load_results(f)
        model = f.stem.replace('nova_waldo_', '')

        # Handle different result formats
        if 'results' in data:
            results = data['results']
        elif 'detailed_results' in data:
            # NOVA format has methods as keys
            detailed = data['detailed_results']
            if isinstance(detailed, dict):
                # Use waldo_complete if available
                if 'waldo_complete' in detailed:
                    results = detailed['waldo_complete']
                else:
                    # Use first available method
                    results = list(detailed.values())[0] if detailed else []
            else:
                results = detailed
        else:
            results = []

        if results:
            summary = summarize_results(results, f"WALDO ({model})")
            summaries.append(summary)

    # Load zero-shot results
    zs_files = list(results_dir.glob("nova_zeroshot_*.json"))
    for f in sorted(zs_files):
        data = load_results(f)
        results = data.get('results', [])
        model = f.stem.replace('nova_zeroshot_', '')
        if results:
            summary = summarize_results(results, f"Zero-shot ({model})")
            summaries.append(summary)

    if summaries:
        print_results_table(summaries)
    else:
        print("No NOVA results found. Please ensure results are in results/nova/")


def analyze_cxr(results_dir: Path):
    """Analyze VinDr-CXR results."""
    print("\n" + "=" * 80)
    print("VinDr-CXR Results")
    print("=" * 80)

    summaries = []

    # Load WALDO results
    waldo_files = list(results_dir.glob("cxr_waldo_*.json"))
    for f in sorted(waldo_files):
        data = load_results(f)
        results = data.get('results', [])
        model = f.stem.replace('cxr_waldo_', '')
        summary = summarize_results(results, f"WALDO ({model})")
        summaries.append(summary)

    # Load zero-shot results
    zs_files = list(results_dir.glob("cxr_zeroshot_*.json"))
    for f in sorted(zs_files):
        data = load_results(f)
        results = data.get('results', [])
        model = f.stem.replace('cxr_zeroshot_', '')
        summary = summarize_results(results, f"Zero-shot ({model})")
        summaries.append(summary)

    if summaries:
        print_results_table(summaries)
    else:
        print("No CXR results found. Please ensure results are in results/cxr/")


def main():
    parser = argparse.ArgumentParser(description='Read and analyze WALDO results')
    parser.add_argument('--dataset', type=str, choices=['nova', 'cxr', 'all'],
                        default='all', help='Dataset to analyze')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing results')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.dataset in ['nova', 'all']:
        analyze_nova(results_dir / 'nova')

    if args.dataset in ['cxr', 'all']:
        analyze_cxr(results_dir / 'cxr')


if __name__ == '__main__':
    main()
