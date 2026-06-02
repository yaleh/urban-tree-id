"""Draw GDino vs RF-DETR detection boxes on sample images for visual comparison.

Output: output/detector_comparison/<image_name>.jpg
  - Blue boxes  = GDino (pseudo-label settings: box_thr=0.30, text_thr=0.25, score_thr=0.35, NMS iou=0.45)
  - Red  boxes  = RF-DETR (threshold=0.3)
  - Green boxes = GT pseudo-labels (from val/labels/*.txt)

Usage:
    .venv/bin/python scripts/visualize_detector_comparison.py \
        --img-dir  data/tdus_resized/val/img \
        --lbl-dir  data/tdus_resized/val/labels \
        --rf-detr-checkpoint models/checkpoint_best_regular.pth \
        --n-samples 10 \
        --output-dir output/detector_comparison
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))


def draw_boxes(draw, boxes, color, label_prefix="", scores=None, width=3):
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [float(v) for v in box]
        for w in range(width):
            draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=color)
        tag = f"{label_prefix}{i+1}"
        if scores is not None and i < len(scores):
            tag += f" {scores[i]:.2f}"
        draw.text((x0 + 4, y0 + 2), tag, fill=color, font=font)


def load_gt_boxes(lbl_path: Path, img_w: int, img_h: int) -> list:
    if not lbl_path.exists():
        return []
    boxes = []
    for line in lbl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, cx, cy, bw, bh = map(float, line.split())
        x0 = (cx - bw / 2) * img_w
        y0 = (cy - bh / 2) * img_h
        x1 = (cx + bw / 2) * img_w
        y1 = (cy + bh / 2) * img_h
        boxes.append([x0, y0, x1, y1])
    return boxes


def pick_samples(img_dir: Path, n: int) -> list[Path]:
    """Pick n images spanning as many species as possible."""
    all_imgs = sorted(img_dir.glob("*.jpg"))
    if len(all_imgs) <= n:
        return all_imgs

    # Group by species prefix (first part of filename before "_tree_")
    from collections import defaultdict
    by_species: dict[str, list] = defaultdict(list)
    for p in all_imgs:
        species = p.stem.split("_tree_")[0] if "_tree_" in p.stem else p.stem[:20]
        by_species[species].append(p)

    # Round-robin pick from each species
    selected = []
    buckets = list(by_species.values())
    idx = 0
    while len(selected) < n:
        bucket = buckets[idx % len(buckets)]
        if bucket:
            selected.append(bucket.pop(0))
        idx += 1
    return selected[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir",  default="data/tdus_resized/val/img")
    parser.add_argument("--lbl-dir",  default="data/tdus_resized/val/labels")
    parser.add_argument("--rf-detr-checkpoint", default="models/checkpoint_best_regular.pth")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--rf-thr",    type=float, default=0.3)
    parser.add_argument("--output-dir", default="output/detector_comparison")
    args = parser.parse_args()

    import torch
    from PIL import Image, ImageDraw
    from predict_pipeline import detect_gdino, load_models
    from rf_detr_detector import RFDETRDetector

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run both models on CPU to avoid OOM when loading them simultaneously
    device = "cpu"
    print(f"Device: {device}  (CPU to avoid OOM with two large models)")

    print("Loading GDino...")
    gdino_models = load_models("gdino", device, svm_model_path=None)

    print("Loading RF-DETR...")
    rfdet = RFDETRDetector(checkpoint=args.rf_detr_checkpoint,
                           threshold=args.rf_thr, device=device)

    samples = pick_samples(Path(args.img_dir), args.n_samples)
    print(f"Processing {len(samples)} images → {out_dir}\n")

    summary_rows = []
    for p in samples:
        img = Image.open(p).convert("RGB")
        W, H = img.size

        gdino_boxes = detect_gdino(img, device,
                                   proc=gdino_models["gdino_proc"],
                                   model=gdino_models["gdino_model"])
        rfdetr_boxes = rfdet.detect(img).tolist()
        gt_boxes = load_gt_boxes(
            Path(args.lbl_dir) / f"{p.stem}.txt", W, H
        )

        # Compose side-by-side: original | annotated
        canvas = img.copy()
        draw = ImageDraw.Draw(canvas)

        draw_boxes(draw, gt_boxes,    color=(0, 200, 0),   label_prefix="GT")
        draw_boxes(draw, gdino_boxes, color=(30, 100, 255), label_prefix="G")
        draw_boxes(draw, rfdetr_boxes, color=(220, 50, 50), label_prefix="R")

        # Legend
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        legend = [
            ((0, 200, 0),    f"GT (pseudo-label): {len(gt_boxes)} boxes"),
            ((30, 100, 255), f"GDino (fixed):     {len(gdino_boxes)} boxes"),
            ((220, 50, 50),  f"RF-DETR thr={args.rf_thr}: {len(rfdetr_boxes)} boxes"),
        ]
        for row_i, (color, text) in enumerate(legend):
            y = H - 90 + row_i * 28
            draw.rectangle([8, y - 2, 14 + len(text) * 13, y + 24],
                           fill=(0, 0, 0, 180))
            draw.text((12, y), text, fill=color, font=font)

        out_path = out_dir / f"{p.stem}_compare.jpg"
        canvas.save(out_path, quality=92)

        summary_rows.append((p.name, len(gt_boxes), len(gdino_boxes), len(rfdetr_boxes)))
        print(f"  {p.name}: GT={len(gt_boxes)} GDino={len(gdino_boxes)} RF-DETR={len(rfdetr_boxes)}  → {out_path.name}")

    print(f"\nSaved {len(samples)} images to {out_dir}/")
    print(f"\n{'Image':<45} {'GT':>4} {'GDino':>7} {'RF-DETR':>8}")
    print("-" * 70)
    for row in summary_rows:
        print(f"{row[0]:<45} {row[1]:>4} {row[2]:>7} {row[3]:>8}")


if __name__ == "__main__":
    main()
