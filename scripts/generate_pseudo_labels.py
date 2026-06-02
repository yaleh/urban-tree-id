"""Generate YOLO-format pseudo-labels for video frames using Grounding DINO.

Stages B1 + B2 of the plan:
  - B1: batch GDINO inference with brightness filter
  - B2: NMS + score threshold → YOLO txt output

Output layout per video:
  data/pseudo_labels/{video_name}/images/all/*.jpg   (symlinked from data/frames/)
  data/pseudo_labels/{video_name}/labels/all/*.txt
"""
import argparse
import logging
import shutil
from pathlib import Path
from typing import Generator, Iterator

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from cli_common import add_gdino_threshold_args  # noqa: E402
from gdino_utils import gdino_preprocess  # noqa: E402  (scripts/ on sys.path)
from yolo_io import xyxy_to_yolo, write_yolo_labels

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "tree."


def collate_fn(batch):
    tensors, paths, pil_imgs = zip(*batch)
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded, masks = [], []
    for t in tensors:
        h, w = t.shape[1], t.shape[2]
        padded.append(F.pad(t, (0, max_w - w, 0, max_h - h)))
        m = torch.zeros(max_h, max_w, dtype=torch.long)
        m[:h, :w] = 1
        masks.append(m)
    return torch.stack(padded), torch.stack(masks), list(paths), list(pil_imgs)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FrameDataset(Dataset):
    def __init__(self, frame_dir: Path, brightness_thr: float = 50.0):
        self.paths = sorted(frame_dir.glob("*.jpg"))
        self.brightness_thr = brightness_thr

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = Image.open(p).convert("RGB")
        return gdino_preprocess(pil), str(p), pil


# ── Filtering ─────────────────────────────────────────────────────────────────

def is_too_dark(pil_img: Image.Image, thr: float = 50.0) -> bool:
    import numpy as np
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    return gray.mean() < thr


def filter_boxes(boxes_xyxy: torch.Tensor, scores: torch.Tensor,
                 score_thr: float = 0.35, iou_thr: float = 0.45):
    """Three-level filter: score threshold → NMS."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores
    mask = scores >= score_thr
    boxes, scores = boxes_xyxy[mask], scores[mask]
    if len(boxes) == 0:
        return boxes, scores
    keep = nms(boxes.float(), scores.float(), iou_thr)
    return boxes[keep], scores[keep]


# ── Batch inference kernel ────────────────────────────────────────────────────

def _run_gdino_batch(
    dataset: Dataset,
    processor,
    model,
    device: str,
    batch_size: int,
    box_threshold: float,
    text_threshold: float,
    num_workers: int = 0,
) -> Iterator[tuple[list[str], list[Image.Image], list[torch.Tensor], list[torch.Tensor]]]:
    """批量运行 GDino 推断，逐批 yield (paths, pil_imgs, boxes_list, scores_list)。

    num_workers 由调用方传入以保留原有行为差异：
    - run_inference 传 num_workers=2（视频帧场景，I/O 密集）
    - run_inference_flat 传 num_workers=0（平铺目录场景）

    每次 yield 一个 batch，boxes_list 和 scores_list 中每个元素对应一张图片，
    均已通过 post_process_grounded_object_detection 解析，但尚未经过 filter_boxes。
    """
    # Pre-tokenise text prompt once
    text_enc = processor.tokenizer(TEXT_PROMPT, return_tensors="pt", padding=True)
    input_ids_1      = text_enc["input_ids"]
    attn_mask_1      = text_enc["attention_mask"]
    token_type_ids_1 = text_enc.get("token_type_ids", torch.zeros_like(input_ids_1))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(num_workers > 0),
    )

    for pixel_values, pixel_mask, paths, pil_imgs in loader:
        B = pixel_values.shape[0]

        input_ids      = input_ids_1.repeat(B, 1).to(device)
        attn_mask      = attn_mask_1.repeat(B, 1).to(device)
        token_type_ids = token_type_ids_1.repeat(B, 1).to(device)

        with torch.no_grad():
            out = model(
                pixel_values=pixel_values.to(device),
                pixel_mask=pixel_mask.to(device),
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids,
            )

        orig_sizes = [(pil.size[1], pil.size[0]) for pil in pil_imgs]
        results = processor.post_process_grounded_object_detection(
            out, input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=orig_sizes,
        )

        boxes_list  = [res["boxes"].cpu() for res in results]
        scores_list = [res["scores"].cpu() for res in results]

        yield paths, pil_imgs, boxes_list, scores_list


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(frame_dir: Path, out_label_dir: Path, out_image_dir: Path,
                  box_threshold: float = 0.30, text_threshold: float = 0.25,
                  score_thr: float = 0.35, iou_thr: float = 0.45,
                  brightness_thr: float = 50.0, batch_size: int = 4,
                  device: str = "cuda") -> dict:

    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device).eval()

    dataset = FrameDataset(frame_dir, brightness_thr)
    log.info(f"Processing {len(dataset)} frames in {frame_dir}")
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(dataset), "dark_skipped": 0, "with_bbox": 0}

    batch_iter = _run_gdino_batch(
        dataset, processor, model, device, batch_size,
        box_threshold, text_threshold, num_workers=2,
    )
    for paths, pil_imgs, boxes_list, scores_list in tqdm(batch_iter, desc=frame_dir.parent.name):
        for p, pil, boxes, scores in zip(paths, pil_imgs, boxes_list, scores_list):
            src = Path(p)
            img_w, img_h = pil.size

            dst_img = out_image_dir / src.name
            if not dst_img.exists():
                shutil.copy2(src, dst_img)

            label_path = out_label_dir / f"{src.stem}.txt"

            if is_too_dark(pil, brightness_thr):
                stats["dark_skipped"] += 1
                write_yolo_labels(label_path, torch.zeros(0, 4), img_w, img_h)
                continue

            boxes, scores = filter_boxes(boxes, scores, score_thr, iou_thr)
            write_yolo_labels(label_path, boxes, img_w, img_h)
            if len(boxes) > 0:
                stats["with_bbox"] += 1

    coverage = stats["with_bbox"] / max(1, stats["total"] - stats["dark_skipped"])
    log.info(f"  total={stats['total']}, dark_skipped={stats['dark_skipped']}, "
             f"with_bbox={stats['with_bbox']}, coverage={coverage:.1%}")
    return stats


# ── Flat-directory inference ──────────────────────────────────────────────────

def run_inference_flat(img_dir: Path, out_label_dir: Path,
                       box_threshold: float = 0.30, text_threshold: float = 0.25,
                       score_thr: float = 0.35, iou_thr: float = 0.45,
                       batch_size: int = 4, device: str = "cuda") -> dict:
    """Run GDino on a flat image directory (no per-video organisation).

    Output: one <stem>.txt per image in out_label_dir.
    Images with zero detections are also listed in out_label_dir/detection_failures.txt.
    """
    out_label_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device).eval()

    dataset = FrameDataset(img_dir)
    log.info(f"Flat-dir: processing {len(dataset)} images in {img_dir}")

    stats = {"total": len(dataset), "with_bbox": 0, "no_bbox": 0}
    failure_paths: list[str] = []

    batch_iter = _run_gdino_batch(
        dataset, processor, model, device, batch_size,
        box_threshold, text_threshold, num_workers=0,
    )
    for paths, pil_imgs, boxes_list, scores_list in tqdm(batch_iter, desc=img_dir.name):
        for p, pil, boxes, scores in zip(paths, pil_imgs, boxes_list, scores_list):
            src = Path(p)
            img_w, img_h = pil.size
            boxes, scores = filter_boxes(boxes, scores, score_thr, iou_thr)

            label_path = out_label_dir / f"{src.stem}.txt"
            write_yolo_labels(label_path, boxes, img_w, img_h)

            if len(boxes) == 0:
                stats["no_bbox"] += 1
                failure_paths.append(str(src))
            else:
                stats["with_bbox"] += 1

    if failure_paths:
        failures_file = out_label_dir / "detection_failures.txt"
        failures_file.write_text("\n".join(failure_paths) + "\n")
        log.info(f"  {len(failure_paths)} images with no detection → {failures_file}")

    coverage = stats["with_bbox"] / max(1, stats["total"])
    log.info(f"  total={stats['total']}, with_bbox={stats['with_bbox']}, "
             f"no_bbox={stats['no_bbox']}, coverage={coverage:.1%}")
    return stats


def parse_args_flat(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO pseudo-labels (flat-dir mode)")
    parser.add_argument("--flat-dir", type=Path, required=True,
                        help="Flat directory of input images")
    parser.add_argument("--flat-out", type=Path, required=True,
                        help="Output directory for YOLO txt labels")
    add_gdino_threshold_args(parser)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate YOLO pseudo-labels via GDINO")
    parser.add_argument("--frames-base", default="data/frames",
                        help="Root dir containing per-video frame subdirs")
    parser.add_argument("--out-base", default="data/pseudo_labels",
                        help="Output root for per-video datasets")
    parser.add_argument("--videos", nargs="+",
                        default=["eastbound_20240319", "eastbound_20240530",
                                 "westbound_20240319", "westbound_20240530"])
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--iou-thr", type=float, default=0.45)
    parser.add_argument("--brightness-thr", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Flat-dir mode (mutually exclusive with video-based mode)
    parser.add_argument("--flat-dir", type=Path, default=None,
                        help="Flat image directory (overrides --frames-base / --videos)")
    parser.add_argument("--flat-out", type=Path, default=None,
                        help="Output label directory for flat-dir mode")
    args = parser.parse_args()

    if args.flat_dir is not None:
        if args.flat_out is None:
            parser.error("--flat-out is required when --flat-dir is used")
        run_inference_flat(args.flat_dir, args.flat_out,
                           args.box_threshold, args.text_threshold,
                           args.score_thr, args.iou_thr,
                           args.batch_size, args.device)
        log.info("Done.")
        return

    frames_base = Path(args.frames_base)
    out_base    = Path(args.out_base)

    for video_name in args.videos:
        frame_dir     = frames_base / video_name
        out_label_dir = out_base / video_name / "labels" / "all"
        out_image_dir = out_base / video_name / "images" / "all"
        if not frame_dir.exists():
            log.warning(f"Frame dir not found, skipping: {frame_dir}")
            continue
        run_inference(frame_dir, out_label_dir, out_image_dir,
                      args.box_threshold, args.text_threshold,
                      args.score_thr, args.iou_thr, args.brightness_thr,
                      args.batch_size, args.device)

    log.info("Done.")


if __name__ == "__main__":
    main()
