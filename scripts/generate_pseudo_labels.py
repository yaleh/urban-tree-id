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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

GDINO_MODEL   = "IDEA-Research/grounding-dino-tiny"
SHORTEST_EDGE = 800
LONGEST_EDGE  = 1333
GDINO_MEAN    = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
GDINO_STD     = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TEXT_PROMPT   = "tree."


# ── Preprocessing (reused from 03_extract_embeddings.py) ──────────────────────

def gdino_preprocess(pil_img):
    w, h = pil_img.size
    scale = SHORTEST_EDGE / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    if max(new_h, new_w) > LONGEST_EDGE:
        scale = LONGEST_EDGE / max(new_h, new_w)
        new_h, new_w = int(round(new_h * scale)), int(round(new_w * scale))
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    t = torch.as_tensor(np.array(resized), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return (t - GDINO_MEAN) / GDINO_STD


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


def xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h):
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w  = (x1 - x0) / img_w
    h  = (y1 - y0) / img_h
    return cx, cy, w, h


def write_label_file(label_path: Path, boxes_xyxy: torch.Tensor,
                     img_w: int, img_h: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if len(boxes_xyxy) == 0:
        label_path.write_text("")
        return
    lines = []
    for box in boxes_xyxy.tolist():
        x0, y0, x1, y1 = box
        cx, cy, w, h = xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n")


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

    # Pre-tokenise text prompt once
    text_enc = processor.tokenizer(TEXT_PROMPT, return_tensors="pt", padding=True)
    input_ids_1      = text_enc["input_ids"]
    attn_mask_1      = text_enc["attention_mask"]
    token_type_ids_1 = text_enc.get("token_type_ids", torch.zeros_like(input_ids_1))

    stats = {"total": len(dataset), "dark_skipped": 0, "with_bbox": 0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)

    for pixel_values, pixel_mask, paths, pil_imgs in tqdm(loader, desc=frame_dir.parent.name):
        B = pixel_values.shape[0]

        # Brightness filter (per image in batch)
        keep_mask = [not is_too_dark(pil, brightness_thr) for pil in pil_imgs]
        stats["dark_skipped"] += sum(1 for k in keep_mask if not k)

        # Run inference only for non-dark images; fill dark ones with empty
        input_ids      = input_ids_1.repeat(B, 1).to(device)
        attn_mask      = attn_mask_1.repeat(B, 1).to(device)
        token_type_ids = token_type_ids_1.repeat(B, 1).to(device)

        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
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

        for i, (res, pil, p, keep) in enumerate(zip(results, pil_imgs, paths, keep_mask)):
            src = Path(p)
            stem = src.stem
            img_w, img_h = pil.size

            # Copy image to dataset directory
            dst_img = out_image_dir / src.name
            if not dst_img.exists():
                shutil.copy2(src, dst_img)

            label_path = out_label_dir / f"{stem}.txt"

            if not keep:
                write_label_file(label_path, torch.zeros(0, 4), img_w, img_h)
                continue

            boxes  = res["boxes"].cpu()
            scores = res["scores"].cpu()
            boxes, scores = filter_boxes(boxes, scores, score_thr, iou_thr)

            write_label_file(label_path, boxes, img_w, img_h)
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

    text_enc = processor.tokenizer(TEXT_PROMPT, return_tensors="pt", padding=True)
    input_ids_1      = text_enc["input_ids"]
    attn_mask_1      = text_enc["attention_mask"]
    token_type_ids_1 = text_enc.get("token_type_ids", torch.zeros_like(input_ids_1))

    stats = {"total": len(dataset), "with_bbox": 0, "no_bbox": 0}
    failure_paths: list[str] = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    for pixel_values, pixel_mask, paths, pil_imgs in tqdm(loader, desc=img_dir.name):
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

        for res, pil, p in zip(results, pil_imgs, paths):
            src = Path(p)
            img_w, img_h = pil.size
            boxes  = res["boxes"].cpu()
            scores = res["scores"].cpu()
            boxes, scores = filter_boxes(boxes, scores, score_thr, iou_thr)

            label_path = out_label_dir / f"{src.stem}.txt"
            write_label_file(label_path, boxes, img_w, img_h)

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
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--iou-thr", type=float, default=0.45)
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
