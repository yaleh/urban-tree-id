"""
classify_multi_tree.py

Detect ALL trees in one or more images with GroundingDINO,
crop each bbox, embed with DINOv2 ViT-S/14, classify with an RBF-SVM
trained on TDUS GDino-crop embeddings.

Output per image (in --out-dir):
  <stem>_annotated.jpg  — original with coloured bboxes and species labels
  results.txt           — bbox coords + species + score for all images

Usage:
    python classify_multi_tree.py path/to/img1.jpg path/to/img2.jpg
    python classify_multi_tree.py --glob "tdus_data/test/img/*.jpg" --n 12
"""

import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from _path_setup import setup as _setup; _setup()

import argparse
import glob as glob_mod
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from sklearn import svm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from gdino_utils import gdino_preprocess, gdino_preprocess_with_size
from image_utils import make_crop_norm as make_crop, make_crop_xyxy, DEFAULT_CROP_SIZE
from predict_pipeline import detect_gdino, detect_yolo
from base_detector import BaseDetector

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SHORTEST_EDGE  = 800
LONGEST_EDGE   = 1333
GDINO_MEAN     = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
GDINO_STD      = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CROP_TARGET    = DEFAULT_CROP_SIZE

DINOV2_TRANSFORM = T.Compose([
    T.Resize(CROP_TARGET),
    T.CenterCrop(CROP_TARGET),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

COLORS = [
    (220,  50,  50),
    ( 50, 180,  50),
    ( 50,  80, 220),
    (220, 160,   0),
    (160,   0, 220),
    (  0, 190, 190),
    (220, 100,   0),
    (  0, 140,  80),
]


# ── Annotation ─────────────────────────────────────────────────────────────────

def annotate_image(pil_img, detections, max_long_edge=1600):
    """Draw coloured bboxes + species labels on a downscaled copy.
    detections: list of (box_norm, species, gdino_score, svm_conf)
    """
    W, H = pil_img.size
    scale = min(1.0, max_long_edge / max(W, H))
    disp = pil_img.resize((int(W * scale), int(H * scale)), Image.LANCZOS).copy()
    draw = ImageDraw.Draw(disp)
    dW, dH = disp.size

    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font_label = font_small = ImageFont.load_default()

    for idx, (box_norm, species, gdino_score, svm_conf) in enumerate(detections):
        color = COLORS[idx % len(COLORS)]
        x0 = box_norm[0] * dW
        y0 = box_norm[1] * dH
        x1 = box_norm[2] * dW
        y1 = box_norm[3] * dH

        # Bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Label text: abbreviated species + scores
        sp_short = species.replace("_", " ")
        label = f"{sp_short} ({svm_conf:.0%})"
        bbox_text = draw.textbbox((0, 0), label, font=font_label)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1] + 4

        # Place label above bbox if possible, else below
        label_y = y0 - th - 2 if y0 - th - 2 >= 0 else y1 + 2
        draw.rectangle([x0, label_y, x0 + tw + 6, label_y + th], fill=color)
        draw.text((x0 + 3, label_y + 2), label, fill=(255, 255, 255), font=font_label)

        # Detection index
        draw.text((x0 + 4, y0 + 4), f"#{idx+1}", fill=color, font=font_small)

    return disp


# ── Arg parser ────────────────────────────────────────────────────────────────

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*",
                        help="Image file paths")
    parser.add_argument("--glob", default=None,
                        help="Glob pattern to collect images (e.g. 'tdus_data/test/img/*.jpg')")
    parser.add_argument("--n",            type=int,   default=None,
                        help="Randomly sample N images from --glob")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--emb-dir",      default="gdino_crop_maxarea_embeddings",
                        help="Directory with cutout_train.npz for SVM fitting")
    parser.add_argument("--out-dir",      default="multi_tree_results")
    parser.add_argument("--text",         default="tree.")
    parser.add_argument("--score-thr",    type=float, default=0.3)
    parser.add_argument("--detector",     choices=["gdino", "yolo"], default="gdino",
                        help="Detection backend to use")
    parser.add_argument("--yolo-checkpoint", dest="yolo_checkpoint",
                        default="runs/detect/tree_yolo26s_halfres/weights/best.pt",
                        help="Path to YOLO .pt weights file")
    parser.add_argument("--yolo-model", dest="yolo_checkpoint",
                        help="[已废弃] 请使用 --yolo-checkpoint")
    parser.add_argument("--yolo-conf",    type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--yolo-imgsz",   type=int,   default=1280,
                        help="YOLO inference image size")
    return parser


# ── YOLO box extraction ────────────────────────────────────────────────────────

def yolo_boxes_from_results(result):
    """Extract (conf, (x0n,y0n,x1n,y1n)) tuples from a single ultralytics Results object.

    Returns [] if no boxes or all boxes are too small (width or height < 0.005).
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    out = []
    xyxyn = result.boxes.xyxyn.cpu()
    conf  = result.boxes.conf.cpu()
    for i in range(len(result.boxes)):
        x0n, y0n, x1n, y1n = xyxyn[i].tolist()
        c = float(conf[i].item())
        if x1n - x0n < 0.005 or y1n - y0n < 0.005:
            continue
        out.append((c, (x0n, y0n, x1n, y1n)))
    return out


# ── Detection + classification ─────────────────────────────────────────────────

def classify_detections(pil_img, boxes_xyxy, dinov2, clf, device):
    """Accept xyxy pixel-coordinate boxes, crop + embed + classify each one.

    Returns a list of result dicts with keys: box_norm, species, svm_conf.
    box_norm contains the normalised [x0, y0, x1, y1] coordinates derived
    by dividing the pixel coordinates by the image width/height.
    """
    if not boxes_xyxy:
        return []

    w, h = pil_img.size
    crops = []
    boxes_norm = []
    for box in boxes_xyxy:
        x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        crop = make_crop_xyxy(pil_img, (x0, y0, x1, y1))
        crops.append(DINOV2_TRANSFORM(crop))
        boxes_norm.append([x0 / w, y0 / h, x1 / w, y1 / h])

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        feats = dinov2.forward_features(batch)["x_norm_clstoken"].cpu().numpy()

    species_preds = clf.predict(feats)
    dec = clf.decision_function(feats)                    # (N, n_classes)
    exp_dec = np.exp(dec - dec.max(axis=1, keepdims=True))
    svm_confs = exp_dec / exp_dec.sum(axis=1, keepdims=True)
    svm_confs_max = svm_confs.max(axis=1)

    results = []
    for box_norm, species, conf in zip(boxes_norm, species_preds, svm_confs_max):
        results.append({
            "box_norm": box_norm,
            "species":  species,
            "svm_conf": float(conf),
        })
    return results


def detect_and_classify(img_path, detector: BaseDetector, dinov2, clf, device):
    """Detect trees with any BaseDetector, classify crops with DINOv2 + SVM.

    Returns list of dicts with keys: box_norm, species, det_score, svm_conf.
    Replaces the old gdino-specific and yolo-specific variants.
    """
    pil       = Image.open(img_path).convert("RGB")
    boxes_xyxy = detector.detect(pil)
    if not boxes_xyxy:
        return []
    return [
        {
            "box_norm":  det["box_norm"],
            "species":   det["species"],
            "det_score": 0.0,
            "svm_conf":  det["svm_conf"],
        }
        for det in classify_detections(pil, boxes_xyxy, dinov2, clf, device)
    ]


# ── Legacy shims (kept for backward compatibility with older call-sites) ────────

def detect_and_classify_gdino(img_path, gdino_model, gdino_processor,
                               dinov2, clf, device, text=None, score_thr=None):
    """Deprecated: use detect_and_classify(img_path, GDinoDetector(...), ...) instead."""
    from gdino_detector import GDinoDetector
    det = GDinoDetector(device=str(device))
    det._proc  = gdino_processor
    det._model = gdino_model
    return detect_and_classify(img_path, det, dinov2, clf, device)


def detect_and_classify_yolo(img_path, yolo_model, dinov2, clf, device,
                              conf=0.25, imgsz=1280):
    """Deprecated: use detect_and_classify(img_path, YOLODetector(...), ...) instead."""
    from yolo_detector import YOLODetector
    det = YOLODetector(checkpoint="", imgsz=imgsz)
    det._model = yolo_model
    return detect_and_classify(img_path, det, dinov2, clf, device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = build_arg_parser().parse_args()

    # Collect image paths
    img_paths = list(args.images)
    if args.glob:
        candidates = sorted(glob_mod.glob(args.glob))
        if args.n:
            random.seed(args.seed)
            candidates = random.sample(candidates, min(args.n, len(candidates)))
        img_paths.extend(candidates)
    if not img_paths:
        build_arg_parser().error("Provide image paths or --glob.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  {len(img_paths)} image(s)  |  detector={args.detector}")

    # Load detector via factory
    from detector_factory import make_detector
    if args.detector == "gdino":
        print(f"Loading GDino ({GDINO_MODEL_ID})...")
        detector_obj = make_detector("gdino", device=str(device))
    else:
        print(f"Loading YOLO ({args.yolo_checkpoint})...")
        detector_obj = make_detector(
            "yolo", device=str(device),
            checkpoint=args.yolo_checkpoint, imgsz=args.yolo_imgsz,
        )

    print("Loading DINOv2 vits14...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

    # Fit SVM on training embeddings
    print(f"Fitting SVM from {args.emb_dir}/cutout_train.npz ...")
    d = np.load(f"{args.emb_dir}/cutout_train.npz", allow_pickle=True)
    X_train, y_train = d["embeddings"].astype(np.float32), d["labels"].astype(str)
    clf = svm.SVC(kernel="rbf", gamma="scale", class_weight="balanced",
                  decision_function_shape="ovr")
    clf.fit(X_train, y_train)
    print(f"  SVM ready ({len(np.unique(y_train))} classes, {len(X_train)} training samples)")

    os.makedirs(args.out_dir, exist_ok=True)
    results_lines = []

    for img_path in img_paths:
        img_path = str(img_path)
        stem = Path(img_path).stem
        print(f"\n{'─'*60}\n{stem}")

        detections = detect_and_classify(img_path, detector_obj, dinov2, clf, device)

        if not detections:
            print("  No trees detected above threshold.")
            results_lines.append(f"{stem}: NO DETECTIONS\n")
            continue

        print(f"  {len(detections)} tree(s) detected:")
        lines = [f"{stem}  ({len(detections)} detection(s))\n"]
        for i, det in enumerate(detections):
            bn = det["box_norm"]
            sp = det["species"].replace("_", " ")
            line = (f"  #{i+1}  {sp:<45}  "
                    f"det={det['det_score']:.3f}  "
                    f"svm={det['svm_conf']:.1%}  "
                    f"box=[{bn[0]:.3f},{bn[1]:.3f},{bn[2]:.3f},{bn[3]:.3f}]")
            print(line)
            lines.append(line + "\n")
        results_lines.extend(lines + ["\n"])

        # Save annotated image
        pil = Image.open(img_path).convert("RGB")
        det_tuples = [(d["box_norm"], d["species"],
                       d["det_score"], d["svm_conf"]) for d in detections]
        annotated = annotate_image(pil, det_tuples)
        out_path = Path(args.out_dir) / f"{stem}_annotated.jpg"
        annotated.save(str(out_path), "JPEG", quality=90)
        print(f"  Saved: {out_path}")

    # Write summary
    txt_path = Path(args.out_dir) / "results.txt"
    with open(txt_path, "w") as f:
        f.writelines(results_lines)
    print(f"\nSummary written to {txt_path}")


if __name__ == "__main__":
    main()
