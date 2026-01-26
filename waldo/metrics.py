"""
Evaluation metrics for anomaly localisation.

Implements mAP@IoU and related metrics following the NOVA benchmark protocol.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union between two boxes.

    Args:
        box1: [x1, y1, x2, y2] first box
        box2: [x1, y1, x2, y2] second box

    Returns:
        IoU score in [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_best_iou(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
) -> float:
    """
    Compute best IoU between any predicted and ground truth box.

    Args:
        pred_boxes: List of predicted boxes
        gt_boxes: List of ground truth boxes

    Returns:
        Maximum IoU across all pred-gt pairs
    """
    if not pred_boxes or not gt_boxes:
        return 0.0

    best_iou = 0.0
    for pred in pred_boxes:
        for gt in gt_boxes:
            iou = compute_iou(pred, gt)
            best_iou = max(best_iou, iou)

    return best_iou


def compute_hit_rate(
    results: List[Dict],
    iou_threshold: float = 0.3,
) -> float:
    """
    Compute hit rate (mAP@IoU) at a given threshold.

    Args:
        results: List of result dicts with 'iou' key
        iou_threshold: IoU threshold for counting a hit

    Returns:
        Hit rate as fraction of samples with IoU >= threshold
    """
    if not results:
        return 0.0

    hits = sum(1 for r in results if r.get('iou', 0) >= iou_threshold)
    return hits / len(results)


def compute_map(
    results: List[Dict],
    thresholds: List[float] = [0.3, 0.5],
) -> Dict[str, float]:
    """
    Compute mAP at multiple IoU thresholds.

    Args:
        results: List of result dicts with 'iou' key
        thresholds: List of IoU thresholds

    Returns:
        Dictionary with mAP at each threshold
    """
    metrics = {}
    for thresh in thresholds:
        key = f"mAP@{int(thresh*100)}"
        metrics[key] = compute_hit_rate(results, thresh)

    # Also compute average IoU
    if results:
        metrics["avg_iou"] = np.mean([r.get('iou', 0) for r in results])
    else:
        metrics["avg_iou"] = 0.0

    return metrics


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: List of metric values
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    if not values:
        return 0.0, 0.0, 0.0

    values = np.array(values)
    n = len(values)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return float(np.mean(values)), float(lower), float(upper)


def evaluate_predictions(
    predictions: List[Dict],
    ground_truths: List[Dict],
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth.

    Args:
        predictions: List of prediction dicts with 'image_id' and 'boxes' keys
        ground_truths: List of ground truth dicts with 'image_id' and 'boxes' keys

    Returns:
        Evaluation metrics dictionary
    """
    # Create GT lookup
    gt_lookup = {gt['image_id']: gt['boxes'] for gt in ground_truths}

    results = []
    for pred in predictions:
        image_id = pred['image_id']
        pred_boxes = pred.get('boxes', [])
        gt_boxes = gt_lookup.get(image_id, [])

        iou = compute_best_iou(pred_boxes, gt_boxes)
        results.append({
            'image_id': image_id,
            'iou': iou,
            'hit_30': iou >= 0.3,
            'hit_50': iou >= 0.5,
        })

    # Compute metrics
    metrics = compute_map(results)

    # Add confidence intervals
    hits_30 = [1 if r['hit_30'] else 0 for r in results]
    hits_50 = [1 if r['hit_50'] else 0 for r in results]

    _, ci_low_30, ci_high_30 = compute_confidence_interval(hits_30)
    _, ci_low_50, ci_high_50 = compute_confidence_interval(hits_50)

    metrics['mAP@30_CI'] = [ci_low_30, ci_high_30]
    metrics['mAP@50_CI'] = [ci_low_50, ci_high_50]
    metrics['n_samples'] = len(results)

    return metrics
