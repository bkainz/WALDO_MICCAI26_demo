"""
Batch inference utilities for WALDO.

Provides tools for processing large datasets with checkpointing,
parallel processing, and progress tracking.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np


@dataclass
class BatchConfig:
    """Configuration for batch inference."""
    batch_size: int = 1
    checkpoint_every: int = 10
    max_retries: int = 3
    retry_delay: float = 2.0
    save_intermediate: bool = True
    resume_from_checkpoint: bool = True


class CheckpointManager:
    """Manage checkpoints during batch processing."""

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        results: List[Dict],
        index: int,
        metadata: Optional[Dict] = None
    ):
        """
        Save checkpoint.

        Args:
            results: Results processed so far
            index: Current index in dataset
            metadata: Additional metadata
        """
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{index}.json"

        checkpoint_data = {
            "index": index,
            "n_results": len(results),
            "results": results,
            "metadata": metadata or {},
            "timestamp": time.time()
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        Load latest checkpoint.

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            return None

        # Get latest by index
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))

        with open(latest) as f:
            return json.load(f)

    def clear_checkpoints(self):
        """Clear all checkpoints."""
        for checkpoint in self.checkpoint_dir.glob("checkpoint_*.json"):
            checkpoint.unlink()


class BatchProcessor:
    """Process dataset in batches with checkpointing and error handling."""

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize batch processor.

        Args:
            config: Batch configuration
            checkpoint_dir: Directory for checkpoints
        """
        self.config = config or BatchConfig()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir or Path("checkpoints")
        ) if self.config.save_intermediate else None

    def process_dataset(
        self,
        dataset: List[Any],
        process_fn: Callable[[Any], Dict],
        desc: str = "Processing"
    ) -> List[Dict]:
        """
        Process dataset with checkpointing.

        Args:
            dataset: List of items to process
            process_fn: Function to process each item
            desc: Description for progress bar

        Returns:
            List of results
        """
        # Try to resume from checkpoint
        start_idx = 0
        results = []

        if self.config.resume_from_checkpoint and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            if checkpoint:
                start_idx = checkpoint['index']
                results = checkpoint['results']
                print(f"Resuming from checkpoint at index {start_idx}")

        # Process items
        for i in tqdm(range(start_idx, len(dataset)), desc=desc, initial=start_idx, total=len(dataset)):
            item = dataset[i]

            # Process with retries
            result = None
            for attempt in range(self.config.max_retries):
                try:
                    result = process_fn(item)
                    break
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        print(f"  Retry {attempt + 1}/{self.config.max_retries} after error: {e}")
                        time.sleep(self.config.retry_delay * (2 ** attempt))
                    else:
                        print(f"  Failed after {self.config.max_retries} attempts: {e}")
                        result = {"error": str(e), "item_index": i}

            if result:
                results.append(result)

            # Save checkpoint
            if self.checkpoint_manager and (i + 1) % self.config.checkpoint_every == 0:
                self.checkpoint_manager.save_checkpoint(
                    results,
                    i + 1,
                    metadata={"total": len(dataset)}
                )

        # Clear checkpoints after successful completion
        if self.checkpoint_manager and self.config.save_intermediate:
            self.checkpoint_manager.clear_checkpoints()

        return results


class ExperimentTracker:
    """Track and log experiments with results aggregation."""

    def __init__(self, experiment_dir: Path):
        """
        Initialize experiment tracker.

        Args:
            experiment_dir: Directory to save experiment logs
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []

    def log_experiment(
        self,
        name: str,
        config: Dict,
        results: List[Dict],
        metrics: Dict
    ):
        """
        Log an experiment.

        Args:
            name: Experiment name
            config: Configuration dict
            results: List of results
            metrics: Computed metrics
        """
        experiment_data = {
            "name": name,
            "timestamp": time.time(),
            "config": config,
            "results": results,
            "metrics": metrics,
            "n_samples": len(results)
        }

        # Save individual experiment
        exp_file = self.experiment_dir / f"{name}_{int(time.time())}.json"
        with open(exp_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)

        self.experiments.append(experiment_data)

        print(f"Logged experiment: {name}")
        print(f"  Samples: {len(results)}")
        print(f"  Metrics: {metrics}")

    def load_experiments(self) -> List[Dict]:
        """
        Load all experiments.

        Returns:
            List of experiment data
        """
        experiments = []
        for exp_file in sorted(self.experiment_dir.glob("*.json")):
            with open(exp_file) as f:
                experiments.append(json.load(f))
        return experiments

    def compare_experiments(
        self,
        metric: str = "avg_iou"
    ) -> Dict[str, float]:
        """
        Compare all experiments by a metric.

        Args:
            metric: Metric to compare

        Returns:
            Dict mapping experiment names to metric values
        """
        experiments = self.load_experiments()
        comparison = {}

        for exp in experiments:
            name = exp['name']
            value = exp['metrics'].get(metric, 0.0)
            comparison[name] = value

        # Sort by metric
        comparison = dict(sorted(comparison.items(), key=lambda x: x[1], reverse=True))

        return comparison


class ProgressiveEvaluator:
    """
    Evaluate model progressively on increasing dataset sizes.

    Useful for estimating performance on large datasets without
    processing everything.
    """

    def __init__(
        self,
        sample_sizes: Optional[List[int]] = None
    ):
        """
        Initialize progressive evaluator.

        Args:
            sample_sizes: List of sample sizes to evaluate at
        """
        self.sample_sizes = sample_sizes or [10, 25, 50, 100, 250, 500, 1000]

    def evaluate_progressive(
        self,
        dataset: List[Any],
        process_fn: Callable[[Any], Dict],
        metric_fn: Callable[[List[Dict]], Dict]
    ) -> Dict[int, Dict]:
        """
        Evaluate at increasing sample sizes.

        Args:
            dataset: Full dataset
            process_fn: Function to process each sample
            metric_fn: Function to compute metrics from results

        Returns:
            Dict mapping sample sizes to metrics
        """
        results_by_size = {}
        all_results = []

        for size in self.sample_sizes:
            if size > len(dataset):
                break

            print(f"\nEvaluating on {size} samples...")

            # Process only new samples
            start_idx = len(all_results)
            for i in tqdm(range(start_idx, size), desc=f"Processing"):
                result = process_fn(dataset[i])
                all_results.append(result)

            # Compute metrics on all results so far
            metrics = metric_fn(all_results)
            results_by_size[size] = metrics

            print(f"  Results: {metrics}")

        return results_by_size


def parallel_process(
    dataset: List[Any],
    process_fn: Callable[[Any], Dict],
    n_workers: int = 4,
    desc: str = "Processing"
) -> List[Dict]:
    """
    Process dataset in parallel (for non-API batch processing).

    Args:
        dataset: List of items to process
        process_fn: Function to process each item
        n_workers: Number of parallel workers
        desc: Description for progress bar

    Returns:
        List of results in original order
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(dataset)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_fn, item): i
            for i, item in enumerate(dataset)
        }

        # Collect results as they complete
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(dataset),
            desc=desc
        ):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                results[idx] = {"error": str(e), "item_index": idx}

    return results


class RateLimiter:
    """
    Rate limiter for API calls.

    Ensures compliance with API rate limits.
    """

    def __init__(
        self,
        max_calls_per_minute: int = 60,
        max_tokens_per_minute: int = 10000
    ):
        """
        Initialize rate limiter.

        Args:
            max_calls_per_minute: Maximum API calls per minute
            max_tokens_per_minute: Maximum tokens per minute (if applicable)
        """
        self.max_calls_per_minute = max_calls_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.call_times = []
        self.tokens_used = []

    def wait_if_needed(self, estimated_tokens: int = 1000):
        """
        Wait if rate limit would be exceeded.

        Args:
            estimated_tokens: Estimated tokens for next request
        """
        current_time = time.time()

        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if current_time - t < 60]
        self.tokens_used = [
            (t, tokens) for t, tokens in self.tokens_used
            if current_time - t < 60
        ]

        # Check if we would exceed limits
        if len(self.call_times) >= self.max_calls_per_minute:
            # Wait until oldest call is > 1 minute old
            wait_time = 60 - (current_time - self.call_times[0]) + 0.1
            if wait_time > 0:
                print(f"Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

        total_tokens = sum(tokens for _, tokens in self.tokens_used)
        if total_tokens + estimated_tokens > self.max_tokens_per_minute:
            # Wait until we're under the token limit
            wait_time = 60 - (current_time - self.tokens_used[0][0]) + 0.1
            if wait_time > 0:
                print(f"Token limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

        # Record this call
        self.call_times.append(current_time)
        self.tokens_used.append((current_time, estimated_tokens))
