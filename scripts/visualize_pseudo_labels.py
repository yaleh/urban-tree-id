"""Visualize pseudo-labels: draw bboxes on sampled frames for quality inspection."""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def draw_labels(img_path: Path, label_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read {img_path}")
    h, w = img.shape[:2]

    if not label_path.exists() or label_path.stat().st_size == 0:
        return img

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x0 = int((cx - bw / 2) * w)
        y0 = int((cy - bh / 2) * h)
        x1 = int((cx + bw / 2) * w)
        y1 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize pseudo-labels for quality check")
    parser.add_argument("--pseudo-labels-base", default="data/pseudo_labels")
    parser.add_argument("--videos", nargs="+",
                        default=["eastbound_20240319", "eastbound_20240530",
                                 "westbound_20240319", "westbound_20240530"])
    parser.add_argument("--n-per-video", type=int, default=6,
                        help="Frames to sample per video")
    parser.add_argument("--out-dir", default="data/pseudo_labels/visualizations")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Scale factor for saved images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for video_name in args.videos:
        img_dir = Path(args.pseudo_labels_base) / video_name / "images" / "all"
        lbl_dir = Path(args.pseudo_labels_base) / video_name / "labels" / "all"
        if not img_dir.exists():
            print(f"Skipping {video_name}: {img_dir} not found")
            continue

        all_imgs = sorted(img_dir.glob("*.jpg"))
        sample = random.sample(all_imgs, min(args.n_per_video, len(all_imgs)))

        n_bbox = 0
        for img_path in sample:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            vis = draw_labels(img_path, lbl_path)

            h, w = vis.shape[:2]
            small = cv2.resize(vis, (int(w * args.scale), int(h * args.scale)))

            bbox_count = 0
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                bbox_count = len(lbl_path.read_text().strip().splitlines())
            n_bbox += bbox_count

            out_name = f"{video_name}__{img_path.stem}_bbox{bbox_count}.jpg"
            cv2.imwrite(str(out_dir / out_name), small)

        print(f"{video_name}: {len(sample)} frames visualized, "
              f"avg {n_bbox/max(1,len(sample)):.1f} bbox/frame → {out_dir}")


if __name__ == "__main__":
    main()
