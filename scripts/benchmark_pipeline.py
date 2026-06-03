"""
benchmark_pipeline.py

Batch evaluation script:
  1. Walk --test-dir for all images (including subdirectories)
  2. Run batched predict_pipeline inference (DataLoader + GPU batch)
  3. Read ground-truth labels from --ann-dir (TDUS format: *.jpg.json, tags[0].value)
  4. Compute top-1 accuracy and write a JSON report

Usage:
    .venv/bin/python scripts/benchmark_pipeline.py \\
        --detector gdino \\
        --test-dir tdus_data/test/img \\
        --ann-dir  tdus_data/test/ann \\
        --svm-model gdino_dinov2_svm/results/svm_model.joblib \\
        --output   results/phase10_gdino_benchmark.json

    .venv/bin/python scripts/benchmark_pipeline.py \\
        --detector yolo \\
        --test-dir tdus_data/test/img \\
        --ann-dir  tdus_data/test/ann \\
        --svm-model gdino_dinov2_svm/results/svm_model.joblib \\
        --yolo-checkpoint runs/detect/tree_yolo26s_halfres/weights/best.pt \\
        --batch-size 16 --workers 4 \\
        --output   results/bench_yolo_halfres.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from _path_setup import setup as _setup; _setup()  # noqa: E402

from cli_common import add_detector_args  # noqa: E402
from eval_metrics import read_gt_label, score_batch  # noqa: E402
from predict_pipeline import (  # noqa: E402
    gdino_forward_batch,
    gdino_preprocess_batch,
    detect_yolo_batch,
    make_crops_gpu_batch,
    predict_species_batch,
)

# Per-detector batch size defaults tuned to ~12 GB VRAM
_DEFAULT_BATCH = {"gdino": 8, "yolo": 64, "rf-detr": 64}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _adjust_image_count(images: list, batch_size: int) -> list:
    """Return a list whose length is a multiple of batch_size.

    If too many: truncate to the largest multiple.
    If too few:  repeat items to fill exactly one batch.
    """
    n = len(images)
    if n >= batch_size:
        return images[:(n // batch_size) * batch_size]
    factor = (batch_size + n - 1) // n
    return (images * factor)[:batch_size]


def _find_images(test_dir: Path, limit: int | None = None) -> list[Path]:
    """Recursively find all image files under test_dir, up to limit images."""
    images = []
    for p in sorted(test_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
            if limit is not None and len(images) >= limit:
                break
    return images


# _read_gt_label removed — use eval_metrics.read_gt_label instead


# ── DataLoader dataset ────────────────────────────────────────────────────────

class _ImageDataset:
    """Map-style dataset: loads, resizes, and returns CHW uint8 tensors in workers.

    Returning torch.Tensor (not PIL) enables pin_memory to actually work,
    so the DataLoader can DMA-copy to page-locked memory and the H2D transfer
    can run asynchronously alongside GPU kernels on the main thread.
    """
    def __init__(self, paths: list[Path], max_edge: int = 1280):
        self.paths    = paths
        self.max_edge = max_edge

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        import numpy as np
        import torch
        from PIL import Image
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        w, h = img.size
        scale = self.max_edge / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        arr    = np.array(img)                             # (H, W, 3) uint8
        tensor = torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W) uint8, pinnable
        return tensor, str(path)


def _collate_tensor(batch):
    """Return (list[Tensor], list[str]) — variable sizes kept as a list (not stacked).

    DataLoader's pin_memory machinery recursively pins each Tensor in the list,
    enabling truly async non_blocking H2D transfers.
    """
    tensors, paths = zip(*batch)
    return list(tensors), list(paths)


# _score_batch removed — use eval_metrics.score_batch instead


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _tensor_to_pil(t: "torch.Tensor") -> "Image.Image":
    """CHW uint8 tensor → PIL Image."""
    from PIL import Image as _PIL
    return _PIL.fromarray(t.permute(1, 2, 0).numpy())


def _run_detector_loop(
    detector_obj,
    loader,
    device: str,
    dinov2,
    clf,
    gt_labels_map: dict,
    throughput_only: bool,
    crop_size: int = 448,
) -> "tuple[int, int, int, dict]":
    """Run batched detect → crop → embed → predict → score over a DataLoader.

    Uses the BaseDetector double-buffer interface when detector_obj.supports_pipeline
    is True (gdino, rf-detr), and a simple batch loop for yolo.
    Returns (n_total, n_correct, n_no_detection, per_class).
    """
    import torch
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    n_total        = 0
    n_correct      = 0
    n_no_detection = 0
    per_class: dict = defaultdict(lambda: {"correct": 0, "total": 0})

    def _process_batch(tensors, paths, batch_boxes):
        nonlocal n_total, n_correct, n_no_detection, per_class
        imgs_gpu     = [t.float().div(255.0).to(device, non_blocking=True) for t in tensors]
        crops_tensor = make_crops_gpu_batch(imgs_gpu, batch_boxes, target_size=crop_size)
        with torch.no_grad():
            feat = dinov2.forward_features(crops_tensor)
        embeddings   = feat["x_norm_clstoken"].cpu().float().numpy()
        predictions  = predict_species_batch(embeddings, clf)
        batch_n_boxes = [len(b) for b in batch_boxes]
        n_total, n_correct, n_no_detection, per_class = score_batch(
            paths, predictions, batch_n_boxes, gt_labels_map,
            n_total, n_correct, n_no_detection, per_class, log.info,
            throughput_only=throughput_only,
        )

    if detector_obj.supports_pipeline:
        # Generic double-buffer: CPU preprocess in thread, GPU forward on main thread.
        with ThreadPoolExecutor(max_workers=1) as pool:
            loader_iter = iter(loader)
            cur_tensors, cur_paths = next(loader_iter)
            cur_pil    = [_tensor_to_pil(t) for t in cur_tensors]
            cur_future = pool.submit(detector_obj.preprocess_cpu, cur_pil)

            for nxt_tensors, nxt_paths in loader_iter:
                nxt_pil    = [_tensor_to_pil(t) for t in nxt_tensors]
                nxt_future = pool.submit(detector_obj.preprocess_cpu, nxt_pil)

                batch_boxes = detector_obj.forward_preprocessed(cur_future.result())
                _process_batch(cur_tensors, cur_paths, batch_boxes)

                cur_tensors, cur_paths = nxt_tensors, nxt_paths
                cur_future             = nxt_future

            batch_boxes = detector_obj.forward_preprocessed(cur_future.result())
            _process_batch(cur_tensors, cur_paths, batch_boxes)

    else:  # yolo — Ultralytics handles its own batching
        for batch_tensors, batch_paths in loader:
            pil_imgs    = [_tensor_to_pil(t) for t in batch_tensors]
            batch_boxes = detector_obj.detect_batch(pil_imgs)
            _process_batch(batch_tensors, batch_paths, batch_boxes)

    return n_total, n_correct, n_no_detection, per_class


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch benchmark for the predict_pipeline."
    )
    add_detector_args(parser)
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Directory of test images (searched recursively)",
    )
    parser.add_argument(
        "--ann-dir",
        default=None,
        help="Directory of TDUS annotation JSON files (omit with --throughput-only)",
    )
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="Skip annotation loading and accuracy scoring; measure raw throughput only",
    )
    parser.add_argument(
        "--svm-model",
        required=True,
        help="Path to trained SVM model (.joblib)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output JSON report path (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="GPU inference batch size (default: 8 gdino, 64 rf-detr, 64 yolo)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker processes for image loading (default: 4)",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=1280,
        help="Worker pre-resize: longest edge of images sent to all detectors and DINOv2 (default: 1280)",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=None,
        help="YOLO inference resolution, independent of --max-edge (default: same as --max-edge). "
             "Allows high-res DINOv2 crops with smaller YOLO imgsz, or vice versa.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total images processed (useful for quick validation runs)",
    )
    parser.add_argument(
        "--timed-bench",
        action="store_true",
        help="Pre-load images, then loop detector+DINOv2+SVM for --duration seconds "
             "(excludes IO/preprocessing from measurement)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Measurement duration in seconds for --timed-bench (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="Warmup duration in seconds before counting starts (default: 5)",
    )
    parser.add_argument(
        "--crop-size",
        dest="crop_size",
        type=int,
        default=448,
        help="DINOv2 input crop size in pixels (default: 448). "
             "Must match the SVM model's training crop size.",
    )
    return parser


# ── Timed-bench helpers ───────────────────────────────────────────────────────

def _preload_batches(
    images: list,
    batch_size: int,
    detector_obj,
    device: str,
    max_edge: int,
) -> list[dict]:
    """Load and preprocess images entirely into CPU memory (outside timing).

    Returns a list of batch dicts. When detector_obj.supports_pipeline is True,
    each dict contains 'prep' (preprocess_cpu output) for zero-copy GPU forward.
    All dicts contain 'imgs_cpu' (CHW uint8 tensors) for the DINOv2 crop path.
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    images = _adjust_image_count(images, batch_size)

    pil_images: list = []
    tensors_uint8: list = []
    for path in images:
        img = PILImage.open(path).convert("RGB")
        w, h = img.size
        scale = max_edge / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), PILImage.BILINEAR)
        pil_images.append(img)
        arr = np.array(img)
        tensors_uint8.append(torch.from_numpy(arr).permute(2, 0, 1))  # CHW uint8

    batches: list[dict] = []
    for i in range(0, len(tensors_uint8), batch_size):
        batch_t   = tensors_uint8[i : i + batch_size]
        batch_pil = pil_images[i : i + batch_size]
        imgs_cpu  = batch_t
        if detector_obj.supports_pipeline:
            prep = detector_obj.preprocess_cpu(batch_pil)
            batches.append({"prep": prep, "pil_imgs": batch_pil,
                            "imgs_cpu": imgs_cpu, "_batch_size": len(batch_t)})
        else:
            # HWC numpy arrays: skip Ultralytics' internal PIL→numpy conversion (~15ms/img)
            batch_np = [t.permute(1, 2, 0).contiguous().numpy() for t in batch_t]
            batches.append({"batch_np": batch_np, "pil_imgs": batch_pil,
                            "imgs_cpu": imgs_cpu, "_batch_size": len(batch_t)})
    return batches


def _run_timed_bench(
    preloaded: list[dict],
    detector_obj,
    device: str,
    dinov2,
    clf,
    duration: float,
    warmup: float,
    log_fn,
    crop_size: int = 448,
) -> tuple[int, int, float, float]:
    """Loop over preloaded batches for warmup+duration seconds.

    Returns (n_total, n_warmup, elapsed_measuring_s, throughput_img_per_s).
    Only images processed after warmup are counted in n_total / elapsed.
    """
    import time
    import torch

    torch.cuda.empty_cache()

    n_batches  = len(preloaded)
    t0         = time.perf_counter()
    t_warmup_end = t0 + warmup
    t_end        = t0 + warmup + duration

    n_total    = 0
    n_warmup   = 0
    measuring  = warmup <= 0
    t_measure_start = t0 if measuring else None
    batch_idx  = 0

    while time.perf_counter() < t_end:
        t_now = time.perf_counter()
        if not measuring and t_now >= t_warmup_end:
            measuring       = True
            t_measure_start = t_now
            log_fn("warmup done — starting measurement")

        batch      = preloaded[batch_idx % n_batches]
        batch_size = batch["_batch_size"]

        if detector_obj.supports_pipeline:
            batch_boxes = detector_obj.forward_preprocessed(batch["prep"])
        else:
            batch_boxes = detector_obj.detect_batch(batch["batch_np"])

        # H2D after detection: YOLO activations released before image tensors live on GPU
        imgs_gpu = [t.float().div(255.0).to(device, non_blocking=True)
                    for t in batch["imgs_cpu"]]
        crops_tensor = make_crops_gpu_batch(imgs_gpu, batch_boxes, target_size=crop_size)
        del imgs_gpu

        with torch.no_grad():
            feat = dinov2.forward_features(crops_tensor)
        del crops_tensor

        embeddings = feat["x_norm_clstoken"].cpu().float().numpy()
        del feat

        predict_species_batch(embeddings, clf)

        if measuring:
            n_total += batch_size
            if n_total % (batch_size * 10) == 0:
                elapsed_so_far = time.perf_counter() - t_measure_start
                log_fn(f"[measuring] n={n_total}  {n_total/elapsed_so_far:.1f} img/s")
        else:
            n_warmup += batch_size

        batch_idx += 1

    elapsed  = time.perf_counter() - (t_measure_start or t0)
    throughput = n_total / elapsed if elapsed > 0 else 0.0
    return n_total, n_warmup, elapsed, throughput


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Validate detector-specific requirements early
    if args.detector == "rf-detr" and not args.rf_detr_checkpoint:
        parser.error("--rf-detr-checkpoint is required when --detector rf-detr")
    if not args.throughput_only and not args.ann_dir:
        parser.error("--ann-dir is required unless --throughput-only is set")
    if args.detector == "yolo" and not args.yolo_checkpoint:
        parser.error("--yolo-checkpoint is required when --detector yolo")

    if args.batch_size is None:
        args.batch_size = _DEFAULT_BATCH[args.detector]
        log.info(f"batch_size not specified; using detector default: {args.batch_size}")

    # YOLO inference resolution: default to --max-edge for backward compatibility
    yolo_imgsz = args.yolo_imgsz if args.yolo_imgsz is not None else args.max_edge
    if args.detector == "yolo":
        log.info(f"YOLO imgsz: {yolo_imgsz}  (max_edge={args.max_edge})")

    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import torch
    from torch.utils.data import DataLoader
    from predict_pipeline import load_models

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}  batch_size: {args.batch_size}  workers: {args.workers}")

    test_dir = Path(args.test_dir)
    all_images = _find_images(test_dir, limit=args.limit)
    log.info(f"Found {len(all_images)} images in {test_dir}")

    if args.throughput_only:
        valid_images  = all_images
        gt_labels_map = {}
        n_no_ann      = 0
        log.info("Throughput-only mode: skipping annotation filter")
    else:
        ann_dir = Path(args.ann_dir)
        valid_images  = []
        gt_labels_map = {}
        n_no_ann = 0
        for p in all_images:
            lbl = read_gt_label(ann_dir, p)
            if lbl is None:
                log.warning(f"No annotation for {p.name}, skipping")
                n_no_ann += 1
            else:
                valid_images.append(p)
                gt_labels_map[str(p)] = lbl
        log.info(f"Annotated: {len(valid_images)}  skipped (no ann): {n_no_ann}")

    # Load models once
    log.info("Loading models...")
    models = load_models(
        detector          = args.detector,
        device            = device,
        yolo_checkpoint   = args.yolo_checkpoint,
        svm_model_path    = args.svm_model,
        rf_detr_checkpoint= args.rf_detr_checkpoint,
        yolo_imgsz        = yolo_imgsz,
        crop_size         = args.crop_size,
    )
    log.info("Models loaded.")

    # Eagerly initialise detector model weights (lazy detectors load on first call)
    from PIL import Image as _PILImage
    _dummy = _PILImage.new("RGB", (64, 64))
    models["detector"].detect_batch([_dummy])
    log.info("Detector model weights initialised.")

    # ── Timed-bench mode: preload → warmup → measure (no IO in critical path) ──
    if args.timed_bench:
        import torch as _torch
        _torch.cuda.empty_cache()   # release activation cache from warmup run
        log.info(
            f"Timed-bench mode: preloading images "
            f"(warmup={args.warmup}s  duration={args.duration}s)"
        )
        preloaded = _preload_batches(
            valid_images, args.batch_size, models["detector"],
            device, args.max_edge,
        )
        n_imgs_preloaded = sum(b["_batch_size"] for b in preloaded)
        log.info(
            f"Preloaded {n_imgs_preloaded} images in {len(preloaded)} batches "
            f"(from {len(valid_images)} source images)"
        )
        n_total, n_warmup, elapsed, throughput = _run_timed_bench(
            preloaded, models["detector"], device,
            models["dinov2"], models["clf"],
            args.duration, args.warmup, log.info,
            crop_size=args.crop_size,
        )
        log.info(
            f"Timed-bench results: {n_total} imgs measured  "
            f"warmup={n_warmup}  {throughput:.1f} img/s  ({elapsed:.1f}s)"
        )
        report = {
            "mode":           "timed_bench",
            "detector":       args.detector,
            "batch_size":     args.batch_size,
            "n_preloaded":    n_imgs_preloaded,
            "n_batches":      len(preloaded),
            "warmup_s":       args.warmup,
            "duration_s":     args.duration,
            "n_measured":     n_total,
            "n_warmup":       n_warmup,
            "elapsed_s":      round(elapsed, 2),
            "throughput_img_per_s": round(throughput, 2),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info(f"Report written to {out_path}")
        return

    dataset = _ImageDataset(valid_images, max_edge=args.max_edge)
    loader  = DataLoader(
        dataset,
        batch_size             = args.batch_size,
        num_workers            = args.workers,
        collate_fn             = _collate_tensor,
        # 'spawn' avoids inheriting the main process's CUDA context into workers.
        # fork + pre-initialized CUDA → silent worker crash / deadlock on Linux.
        multiprocessing_context = "spawn" if args.workers > 0 else None,
        pin_memory             = (device == "cuda") and args.workers > 0,
        persistent_workers     = (args.workers > 0),
    )

    t0 = time.perf_counter()

    n_total, n_correct, n_no_detection, per_class = _run_detector_loop(
        models["detector"], loader, device,
        models["dinov2"], models["clf"],
        gt_labels_map, args.throughput_only,
        crop_size=args.crop_size,
    )

    elapsed = time.perf_counter() - t0
    throughput = n_total / elapsed if elapsed > 0 else 0.0
    top1 = round(n_correct / n_total, 4) if n_total > 0 else 0.0

    # Build per-class summary
    per_class_summary = {}
    for species, stats in sorted(per_class.items()):
        total   = stats["total"]
        correct = stats["correct"]
        per_class_summary[species] = {
            "total":    total,
            "correct":  correct,
            "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        }

    report = {
        "detector":       args.detector,
        "batch_size":     args.batch_size,
        "workers":        args.workers,
        "n_total":        n_total,
        "n_correct":      n_correct,
        "top1_accuracy":  top1,
        "n_no_detection": n_no_detection,
        "n_no_ann":       n_no_ann,
        "elapsed_s":      round(elapsed, 2),
        "throughput_img_per_s": round(throughput, 2),
        "per_class":      per_class_summary,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info(
        f"Results: {n_correct}/{n_total}  top1={top1:.4f}  "
        f"no_det={n_no_detection}  {throughput:.1f} img/s  ({elapsed:.1f}s total)"
    )
    log.info(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
