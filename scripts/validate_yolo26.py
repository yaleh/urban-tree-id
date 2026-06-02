"""Run val + optional visual inference check on trained YOLO26 weights (Stage C3)."""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def run_val(weights: str, data: str, imgsz: int, device: str) -> dict:
    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=device, verbose=True)
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    print(f"\nval mAP@0.5={map50:.4f}  mAP@0.5:0.95={map50_95:.4f}")
    if map50 < 0.25:
        print("WARNING: mAP@0.5 < 0.25 — quality gate NOT met")
    else:
        print("Quality gate PASSED: mAP@0.5 >= 0.25")
    return {"map50": map50, "map50_95": map50_95}


def visualize_predictions(weights: str, img_dir: str, out_dir: str,
                           n_samples: int, imgsz: int, device: str,
                           conf: float = 0.25) -> None:
    model = YOLO(weights)
    img_paths = sorted(Path(img_dir).glob("*.jpg"))
    if not img_paths:
        print(f"No .jpg found in {img_dir}")
        return
    sample = random.sample(img_paths, min(n_samples, len(img_paths)))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for img_path in sample:
        results = model.predict(str(img_path), imgsz=imgsz, conf=conf,
                                device=device, verbose=False)
        annotated = results[0].plot()
        h, w = annotated.shape[:2]
        small = cv2.resize(annotated, (w // 4, h // 4))
        cv2.imwrite(str(out / img_path.name), small)
        n_det = len(results[0].boxes)
        print(f"  {img_path.name}: {n_det} detections")

    print(f"\nSaved {len(sample)} annotated frames → {out}")


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO26 and visualize predictions")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data", default="data/pseudo_labels/all_4videos/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--predict-dir", default=None,
                        help="Dir of images to run inference on for visual check")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--vis-out-dir", default="runs/detect/visual_check")
    args = parser.parse_args()

    run_val(args.weights, args.data, args.imgsz, args.device)

    if args.predict_dir:
        visualize_predictions(args.weights, args.predict_dir, args.vis_out_dir,
                              args.n_samples, args.imgsz, args.device)


if __name__ == "__main__":
    main()
