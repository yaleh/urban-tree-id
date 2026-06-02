"""Compare GDINO pseudo-labels vs YOLO inference and visualize largest divergences.

Divergence metric per frame:
  score = |n_gdino - n_yolo| + missed + false_pos
  where:
    missed     = GDINO boxes with no matching YOLO box (IoU > 0.3)
    false_pos  = YOLO boxes with no matching GDINO box (IoU > 0.3)

Output: annotated images with GDINO boxes (green) and YOLO boxes (red),
        sorted by divergence score descending.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

VIDEOS = ["eastbound_20240319", "westbound_20240319",
          "eastbound_20240530", "westbound_20240530"]


def load_yolo_xyxy(label_path: Path, img_w: int, img_h: int) -> torch.Tensor:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return torch.zeros(0, 4)
    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = map(float, parts)
        x0 = (cx - w / 2) * img_w
        y0 = (cy - h / 2) * img_h
        x1 = (cx + w / 2) * img_w
        y1 = (cy + h / 2) * img_h
        rows.append([x0, y0, x1, y1])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, 4)


def divergence_score(gdino: torch.Tensor, yolo: torch.Tensor, iou_thr: float = 0.3) -> tuple:
    n_g, n_y = len(gdino), len(yolo)
    if n_g == 0 and n_y == 0:
        return 0, 0, 0
    if n_g == 0:
        return n_y, 0, n_y   # all YOLO boxes are false positives
    if n_y == 0:
        return n_g, n_g, 0   # all GDINO boxes are missed

    iou = box_iou(gdino.float(), yolo.float())  # (G, Y)
    missed = int((iou.max(dim=1).values <= iou_thr).sum())
    false_pos = int((iou.max(dim=0).values <= iou_thr).sum())
    score = abs(n_g - n_y) + missed + false_pos
    return score, missed, false_pos


def draw_boxes(img, boxes_xyxy, color, label_prefix):
    for i, (x0, y0, x1, y1) in enumerate(boxes_xyxy.tolist()):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, f"{label_prefix}{i}", (x0, max(y0 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--frames-base", default="data/frames")
    parser.add_argument("--gdino-labels-base", default="data/pseudo_labels")
    parser.add_argument("--out-dir", default="runs/diff_gdino_yolo")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.3)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    model = YOLO(args.weights)
    frames_base = Path(args.frames_base)
    gdino_base = Path(args.gdino_labels_base)

    records = []  # (score, missed, false_pos, img_path, gdino_boxes, yolo_boxes)

    print("Running inference on all frames...")
    for video in VIDEOS:
        frame_dir = frames_base / video
        gdino_lbl_dir = gdino_base / video / "labels" / "all"
        if not frame_dir.exists():
            print(f"  skip (no frames): {video}")
            continue

        img_paths = sorted(frame_dir.glob("*.jpg"))
        print(f"  {video}: {len(img_paths)} frames")
        for img_path in img_paths:
            res = model.predict(str(img_path), imgsz=1280, conf=args.conf,
                                device=args.device, verbose=False)[0]
            img_w, img_h = res.orig_shape[1], res.orig_shape[0]

            yolo_boxes = res.boxes.xyxy.cpu() if len(res.boxes) else torch.zeros(0, 4)
            gdino_boxes = load_yolo_xyxy(
                gdino_lbl_dir / f"{img_path.stem}.txt", img_w, img_h)

            score, missed, false_pos = divergence_score(gdino_boxes, yolo_boxes, args.iou_thr)
            records.append((score, missed, false_pos, img_path, gdino_boxes, yolo_boxes))

    records.sort(key=lambda r: r[0], reverse=True)
    top = records[:args.top_n]

    print(f"\nSaving top {len(top)} divergent frames → {out_dir}")
    for rank, (score, missed, false_pos, img_path, gdino_boxes, yolo_boxes) in enumerate(top, 1):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        draw_boxes(img, gdino_boxes, color=(0, 200, 0), label_prefix="G")    # green
        draw_boxes(img, yolo_boxes,  color=(0, 0, 220), label_prefix="Y")    # red

        n_g, n_y = len(gdino_boxes), len(yolo_boxes)
        text = f"score={score} | GDINO={n_g} missed={missed} | YOLO={n_y} fp={false_pos}"
        cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"#{rank} {img_path.parent.name}/{img_path.name}",
                    (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        small = cv2.resize(img, (w // 2, h // 2))
        out_name = f"{rank:03d}_score{score}_{img_path.parent.name}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_dir / out_name), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  #{rank:3d}  score={score:3d}  G={n_g} miss={missed}  Y={n_y} fp={false_pos}"
              f"  {img_path.parent.name}/{img_path.name}")

    print(f"\nDone. {len(top)} images saved to {out_dir}/")
    print("Legend: GREEN = GDINO boxes, RED = YOLO boxes")


if __name__ == "__main__":
    main()
