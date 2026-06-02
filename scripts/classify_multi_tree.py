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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import glob as glob_mod
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from sklearn import svm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from gdino_utils import gdino_preprocess, gdino_preprocess_with_size

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SHORTEST_EDGE  = 800
LONGEST_EDGE   = 1333
GDINO_MEAN     = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
GDINO_STD      = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CROP_TARGET    = 448

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


# ── Crop utility ───────────────────────────────────────────────────────────────

def make_crop(pil_img, box_norm, pad_frac=0.05):
    """box_norm: (x0,y0,x1,y1) normalized [0,1] in original image space."""
    W, H = pil_img.size
    x0 = box_norm[0] * W
    y0 = box_norm[1] * H
    x1 = box_norm[2] * W
    y1 = box_norm[3] * H
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, x0 - bw * pad_frac)
    y0 = max(0, y0 - bh * pad_frac)
    x1 = min(W, x1 + bw * pad_frac)
    y1 = min(H, y1 + bh * pad_frac)
    crop = pil_img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    scale = CROP_TARGET / max(cw, ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    scaled = crop.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (CROP_TARGET, CROP_TARGET), (255, 255, 255))
    canvas.paste(scaled, ((CROP_TARGET - nw) // 2, (CROP_TARGET - nh) // 2))
    return canvas


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
    parser.add_argument("--yolo-model",   default="runs/detect/tree_yolo26s_halfres/weights/best.pt",
                        help="Path to YOLO .pt weights file")
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

def detect_and_classify(img_path, gdino_model, gdino_processor,
                        dinov2, clf, device, text, score_thr):
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size

    # GDino forward
    tensor, prep_h, prep_w = gdino_preprocess_with_size(pil)
    pixel_values = tensor.unsqueeze(0)                          # (1, 3, H, W)
    pixel_mask   = torch.ones(1, prep_h, prep_w, dtype=torch.long)
    # Pad to square for single-image batch (no-op if already same size)
    max_H, max_W = prep_h, prep_w

    text_enc = gdino_processor.tokenizer(text, return_tensors="pt", padding=True)
    input_ids      = text_enc["input_ids"].to(device)
    attn_mask      = text_enc["attention_mask"].to(device)
    token_type_ids = text_enc.get("token_type_ids",
                                  torch.zeros_like(text_enc["input_ids"])).to(device)

    with torch.no_grad():
        out = gdino_model(
            pixel_values=pixel_values.to(device),
            pixel_mask=pixel_mask.to(device),
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )

    scores     = out.logits[0].sigmoid().max(dim=-1).values.cpu()  # (900,)
    pred_boxes = out.pred_boxes[0].cpu()                           # (900, 4) cx,cy,w,h norm

    # Collect all detections above threshold
    above = (scores >= score_thr).nonzero(as_tuple=True)[0]
    if above.numel() == 0:
        return []

    # Convert to xyxy normalized in original image space
    # (proportional resize → normalized coords are invariant)
    detections_raw = []
    for q in above.tolist():
        s = scores[q].item()
        cx, cy, bw, bh = pred_boxes[q].tolist()
        x0_n = max(0.0, cx - bw / 2)
        y0_n = max(0.0, cy - bh / 2)
        x1_n = min(1.0, cx + bw / 2)
        y1_n = min(1.0, cy + bh / 2)
        if x1_n - x0_n < 0.005 or y1_n - y0_n < 0.005:
            continue
        area = (x1_n - x0_n) * (y1_n - y0_n)
        detections_raw.append((s, area, (x0_n, y0_n, x1_n, y1_n)))

    # NMS: suppress boxes with IoU > 0.5 (keep higher-score box)
    detections_raw.sort(key=lambda x: -x[0])
    kept = []
    for det in detections_raw:
        box = det[2]
        suppress = False
        for k in kept:
            kbox = k[2]
            ix0 = max(box[0], kbox[0]); iy0 = max(box[1], kbox[1])
            ix1 = min(box[2], kbox[2]); iy1 = min(box[3], kbox[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = ((box[2]-box[0])*(box[3]-box[1]) +
                     (kbox[2]-kbox[0])*(kbox[3]-kbox[1]) - inter)
            if union > 0 and inter / union > 0.5:
                suppress = True
                break
        if not suppress:
            kept.append(det)

    # Classify each kept box
    results = []
    crops = []
    boxes_norm = []
    gdino_scores = []
    for s, area, box_norm in kept:
        crop = make_crop(pil, box_norm)
        crops.append(DINOV2_TRANSFORM(crop))
        boxes_norm.append(box_norm)
        gdino_scores.append(s)

    if not crops:
        return []

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        feats = dinov2.forward_features(batch)["x_norm_clstoken"].cpu().numpy()

    species_preds = clf.predict(feats)
    # Use decision_function for per-class confidence (softmax of distances)
    dec = clf.decision_function(feats)                   # (N, n_classes)
    exp_dec = np.exp(dec - dec.max(axis=1, keepdims=True))
    svm_confs = exp_dec / exp_dec.sum(axis=1, keepdims=True)
    svm_confs_max = svm_confs.max(axis=1)

    for box_norm, gdino_score, species, conf in zip(
            boxes_norm, gdino_scores, species_preds, svm_confs_max):
        results.append({
            "box_norm":  box_norm,
            "species":   species,
            "det_score": gdino_score,
            "svm_conf":  float(conf),
        })

    return results


def detect_and_classify_yolo(img_path, yolo_model, dinov2, clf, device, conf, imgsz):
    """Detect trees with YOLO, classify crops with DINOv2 + SVM.

    Returns list of dicts with keys: box_norm, species, det_score, svm_conf.
    """
    pil = Image.open(img_path).convert("RGB")

    results_list = yolo_model.predict(pil, conf=conf, imgsz=imgsz, verbose=False)
    boxes = yolo_boxes_from_results(results_list[0])
    if not boxes:
        return []

    crops = []
    boxes_norm = []
    det_scores = []
    for det_score, box_norm in boxes:
        crop = make_crop(pil, box_norm)
        crops.append(DINOV2_TRANSFORM(crop))
        boxes_norm.append(box_norm)
        det_scores.append(det_score)

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        feats = dinov2.forward_features(batch)["x_norm_clstoken"].cpu().numpy()

    species_preds = clf.predict(feats)
    dec = clf.decision_function(feats)
    exp_dec = np.exp(dec - dec.max(axis=1, keepdims=True))
    svm_confs = exp_dec / exp_dec.sum(axis=1, keepdims=True)
    svm_confs_max = svm_confs.max(axis=1)

    output = []
    for box_norm, det_score, species, svm_conf in zip(
            boxes_norm, det_scores, species_preds, svm_confs_max):
        output.append({
            "box_norm":  box_norm,
            "species":   species,
            "det_score": det_score,
            "svm_conf":  float(svm_conf),
        })
    return output


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

    # Load models based on detector choice
    if args.detector == "gdino":
        print(f"Loading GDino ({GDINO_MODEL_ID})...")
        gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID)
        gdino_model = gdino_model.to(device).eval()
    else:
        from ultralytics import YOLO
        print(f"Loading YOLO ({args.yolo_model})...")
        yolo_model = YOLO(args.yolo_model)

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

        if args.detector == "gdino":
            detections = detect_and_classify(
                img_path, gdino_model, gdino_processor,
                dinov2, clf, device, args.text, args.score_thr,
            )
        else:
            detections = detect_and_classify_yolo(
                img_path, yolo_model, dinov2, clf, device,
                args.yolo_conf, args.yolo_imgsz,
            )

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
