"""eval_metrics — scoring and annotation helpers extracted from benchmark_pipeline."""
from __future__ import annotations

import json
from pathlib import Path


def read_gt_label(ann_dir: Path, img_path: Path) -> str | None:
    """Return ground-truth label for img_path from a TDUS annotation JSON.

    Expected path: ann_dir / (img_path.name + ".json")
    Label comes from ann["tags"][0]["value"].
    Returns None when the annotation is missing or malformed.
    """
    ann_path = ann_dir / (img_path.name + ".json")
    if not ann_path.exists():
        return None
    try:
        with open(ann_path) as f:
            ann = json.load(f)
        return ann["tags"][0]["value"]
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


def score_batch(
    batch_paths: list,
    predictions: list,
    batch_n_boxes: list,
    gt_labels_map: dict,
    n_total: int,
    n_correct: int,
    n_no_detection: int,
    per_class: dict,
    log_fn,
    throughput_only: bool = False,
) -> tuple:
    """Accumulate per-image accuracy stats and emit one log line per batch.

    When throughput_only=True, skip accuracy bookkeeping and only count images.
    Returns (n_total, n_correct, n_no_detection, per_class).
    """
    if throughput_only:
        n_total += len(batch_paths)
        log_fn(f"batch n={len(batch_paths)}  running_total={n_total}")
        return n_total, n_correct, n_no_detection, per_class

    n_batch_correct = 0
    for path, (species, confidence), n_boxes in zip(batch_paths, predictions, batch_n_boxes):
        gt_label = gt_labels_map[path]
        n_total += 1
        if n_boxes == 0:
            n_no_detection += 1
        per_class[gt_label]["total"] += 1
        if species == gt_label:
            n_correct      += 1
            n_batch_correct += 1
            per_class[gt_label]["correct"] += 1

    top1_running = n_correct / n_total if n_total > 0 else 0.0
    log_fn(
        f"batch n={len(batch_paths)}  correct={n_batch_correct}/{len(batch_paths)}"
        f"  running_top1={top1_running:.3f}  no_det={n_no_detection}"
    )
    return n_total, n_correct, n_no_detection, per_class
