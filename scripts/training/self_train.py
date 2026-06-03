"""Iterative self-training loop (Phase D).

Each iteration k:
  1. YOLO inference on all frames with confidence threshold get_score_threshold(k)
  2. Intersection filter: keep YOLO boxes with IoU > 0.5 against cached GDINO labels
  3. Write filtered labels → rebuild dataset → re-train from previous weights
  4. Evaluate on manual val set (if available)
  5. Check convergence (2 consecutive rounds without mAP improvement)

Usage:
  python scripts/self_train.py \
    --weights runs/detect/tree_yolo26s_opt/weights/best.pt \
    --frames-base data/frames \
    --gdino-labels-base data/pseudo_labels \
    --manual-val-data data/manual_labels/val/data.yaml \
    --max-iter 5
"""
import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

from yolo_io import load_yolo_xyxy, write_yolo_labels

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

VIDEOS = ["eastbound_20240319", "westbound_20240319",
          "eastbound_20240530", "westbound_20240530"]


# ── Threshold strategy ────────────────────────────────────────────────────────

def get_score_threshold(k: int, base_thr: float = 0.35,
                        step: float = 0.05, max_thr: float = 0.55) -> float:
    """Confidence threshold for iteration k (escalates each round)."""
    return min(base_thr + (k - 1) * step, max_thr)


# ── BBox intersection filter ──────────────────────────────────────────────────

def compute_intersection(yolo_boxes: torch.Tensor, gdino_boxes: torch.Tensor,
                         iou_thr: float = 0.5) -> torch.Tensor:
    """Return rows of yolo_boxes where IoU with any gdino_box > iou_thr.

    Args:
        yolo_boxes:  (N, 4) xyxy float
        gdino_boxes: (M, 4) xyxy float
        iou_thr:     minimum IoU to accept

    Returns:
        (K, 4) accepted boxes (subset of yolo_boxes)
    """
    if len(yolo_boxes) == 0 or len(gdino_boxes) == 0:
        return yolo_boxes.new_zeros((0, 4))
    iou = box_iou(yolo_boxes.float(), gdino_boxes.float())  # (N, M)
    keep = (iou.max(dim=1).values > iou_thr)
    return yolo_boxes[keep]



# ── Core iteration step ───────────────────────────────────────────────────────

def run_inference_and_filter(weights: str, frames_base: Path,
                             gdino_labels_base: Path, out_base: Path,
                             score_thr: float, iou_thr: float,
                             device: str) -> dict:
    """Run YOLO on ALL frames, compare with GDINO labels, write filtered labels.

    YOLO is run on every frame regardless of GDINO output, so we can measure
    the divergence between the two models as self-training progresses.

    Label assignment:
      Case A — GDINO∅, YOLO∅  : empty label (both agree: background)
      Case B — GDINO∅, YOLO has boxes : empty label (trust GDINO; YOLO false-pos)
      Case C — GDINO has boxes, intersection accepted ≥1 : intersection boxes
      Case D — GDINO has boxes, YOLO∅ or no overlap : fall back to GDINO labels

    Stats logged per case let you track model alignment across iterations.
    """
    model = YOLO(weights)
    stats = {
        "total": 0,
        "case_a": 0,   # GDINO∅ ∩ YOLO∅   → background consensus
        "case_b": 0,   # GDINO∅ ∩ YOLO+   → YOLO false positives
        "case_c": 0,   # both+ ∩ overlap   → high-confidence boxes
        "case_d": 0,   # GDINO+ ∩ YOLO∅/no overlap → GDINO fallback
        "boxes_accepted": 0,
        "boxes_fallback": 0,
    }

    for video in VIDEOS:
        frame_dir = frames_base / video
        gdino_lbl_dir = gdino_labels_base / video / "labels" / "all"
        out_img_dir = out_base / video / "images" / "all"
        out_lbl_dir = out_base / video / "labels" / "all"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not frame_dir.exists():
            log.warning(f"Frame dir not found, skipping: {frame_dir}")
            continue

        for img_path in sorted(frame_dir.glob("*.jpg")):
            stats["total"] += 1

            dst_img = out_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # YOLO inference on every frame
            res = model.predict(str(img_path), imgsz=1280, conf=score_thr,
                                device=device, verbose=False)[0]
            img_w, img_h = res.orig_shape[1], res.orig_shape[0]

            yolo_boxes  = res.boxes.xyxy.cpu() if len(res.boxes) else torch.zeros(0, 4)
            gdino_boxes = load_yolo_xyxy(
                gdino_lbl_dir / f"{img_path.stem}.txt", img_w, img_h)

            has_gdino = len(gdino_boxes) > 0
            has_yolo  = len(yolo_boxes)  > 0

            if not has_gdino and not has_yolo:
                # Case A: consensus background
                stats["case_a"] += 1
                final_boxes = torch.zeros(0, 4)

            elif not has_gdino and has_yolo:
                # Case B: YOLO false positives — trust GDINO (empty)
                stats["case_b"] += 1
                final_boxes = torch.zeros(0, 4)

            else:
                # GDINO has boxes (cases C / D)
                accepted = compute_intersection(yolo_boxes, gdino_boxes, iou_thr)
                if len(accepted) > 0:
                    # Case C: intersection agreed
                    stats["case_c"] += 1
                    stats["boxes_accepted"] += len(accepted)
                    final_boxes = accepted
                else:
                    # Case D: YOLO missed or no overlap — fall back to GDINO
                    stats["case_d"] += 1
                    stats["boxes_fallback"] += len(gdino_boxes)
                    final_boxes = gdino_boxes

            write_yolo_labels(out_lbl_dir / f"{img_path.stem}.txt",
                                 final_boxes, img_w, img_h)

    log.info(
        f"  frames={stats['total']}  "
        f"A(both∅)={stats['case_a']}  "
        f"B(GDINO∅,YOLO+)={stats['case_b']}  "
        f"C(intersection)={stats['case_c']}(boxes={stats['boxes_accepted']})  "
        f"D(fallback)={stats['case_d']}(boxes={stats['boxes_fallback']})"
    )
    return stats


def rebuild_and_train(iter_out_base: Path, weights: str, run_name: str,
                      epochs: int, device: str) -> str:
    """Re-split dataset and fine-tune; return path to best.pt."""
    split_out = iter_out_base / "all_4videos"
    subprocess.run([
        sys.executable, "scripts/split_dataset.py",
        "--pseudo-labels-base", str(iter_out_base),
        "--out-dir", str(split_out),
    ], check=True)

    subprocess.run([
        sys.executable, "scripts/train_yolo26.py",
        "--data", str(split_out / "data.yaml"),
        "--model", weights,
        "--epochs", str(epochs),
        "--run-name", run_name,
        "--device", device,
    ], check=True)

    best = Path("runs/detect").resolve() / run_name / "weights" / "best.pt"
    return str(best)


def evaluate(weights: str, val_data: str, device: str) -> float:
    """Return mAP@0.5 on manual val set; -1 if val_data missing."""
    if not Path(val_data).exists():
        log.warning(f"Manual val data not found: {val_data} — skipping eval")
        return -1.0
    model = YOLO(weights)
    metrics = model.val(data=val_data, imgsz=1280, device=device, verbose=False)
    return float(metrics.box.map50)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_self_training(args) -> None:
    best_weights = args.weights
    history: list[float] = []
    no_improve = 0

    for k in range(1, args.max_iter + 1):
        score_thr = get_score_threshold(k)
        log.info(f"=== Iteration {k}/{args.max_iter}  score_thr={score_thr:.2f} ===")

        iter_out = Path(args.iter_base) / f"iter{k}"
        stats = run_inference_and_filter(
            best_weights,
            frames_base=Path(args.frames_base),
            gdino_labels_base=Path(args.gdino_labels_base),
            out_base=iter_out,
            score_thr=score_thr,
            iou_thr=args.iou_thr,
            device=args.device,
        )
        log.info(
            f"  A(both∅)={stats['case_a']}  B(GDINO∅,YOLO+)={stats['case_b']}  "
            f"C(intersect)={stats['case_c']}(boxes={stats['boxes_accepted']})  "
            f"D(fallback)={stats['case_d']}(boxes={stats['boxes_fallback']})  "
            f"total={stats['total']}"
        )

        run_name = f"tree_yolo26s_iter{k}"
        best_weights = rebuild_and_train(
            iter_out, best_weights, run_name, args.epochs, args.device)
        log.info(f"  new weights: {best_weights}")

        map50 = evaluate(best_weights, args.manual_val_data, args.device)
        if map50 >= 0:
            log.info(f"  mAP@0.5 = {map50:.4f}")
            if history and map50 <= max(history):
                no_improve += 1
                log.info(f"  no improvement ({no_improve}/2)")
                if no_improve >= 2:
                    log.info("Converged — stopping early.")
                    break
            else:
                no_improve = 0
            history.append(map50)
        else:
            log.info("  eval skipped (no manual val set)")

    log.info(f"Self-training done. Final weights: {best_weights}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative self-training (Phase D)")
    parser.add_argument("--weights", required=True,
                        help="Initial best.pt from Phase C")
    parser.add_argument("--frames-base", default="data/frames")
    parser.add_argument("--gdino-labels-base", default="data/pseudo_labels")
    parser.add_argument("--manual-val-data", default="data/manual_labels/val/data.yaml")
    parser.add_argument("--iter-base", default="data/pseudo_labels_self",
                        help="Root dir for per-iteration label outputs")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per iteration")
    parser.add_argument("--iou-thr", type=float, default=0.5,
                        help="Min IoU for YOLO/GDINO intersection")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run_self_training(args)


if __name__ == "__main__":
    main()
