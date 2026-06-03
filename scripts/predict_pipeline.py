"""
predict_pipeline.py

Single-image inference pipeline:
  1. Detect tree bbox with the chosen detector (gdino / yolo / rf-detr)
  2. Crop largest-area bbox (make_crop logic from 03_extract_embeddings.py)
  3. Extract 384-dim CLS embedding with DINOv2 ViT-S/14
  4. Predict species with a pre-trained SVM
  5. Print result as JSON to stdout

Usage:
    .venv/bin/python scripts/predict_pipeline.py \\
        --image path/to/image.jpg \\
        --detector gdino \\
        --svm-model gdino_dinov2_svm/results/svm_model.joblib

    .venv/bin/python scripts/predict_pipeline.py \\
        --image path/to/image.jpg \\
        --detector rf-detr \\
        --svm-model gdino_dinov2_svm/results/svm_model.joblib \\
        --rf-detr-checkpoint models/rf_detr_tdus_best.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from _path_setup import setup as _setup; _setup()  # noqa: E402

from cli_common import add_detector_args  # noqa: E402
from gdino_utils import gdino_preprocess  # noqa: E402
from image_utils import make_crop_xyxy as make_crop, DEFAULT_CROP_SIZE  # noqa: E402

# ── Constants (mirrored from 03_extract_embeddings.py) ────────────────────────

GDINO_MODEL   = "IDEA-Research/grounding-dino-tiny"
GDINO_TEXT    = "tree."
GDINO_BOX_THR   = 0.30   # matches generate_pseudo_labels.py defaults
GDINO_TEXT_THR  = 0.25
GDINO_SCORE_THR = 0.35   # post-NMS score filter (was 0.3, now matches pseudo-label gen)
GDINO_IOU_THR   = 0.45   # NMS IoU threshold

DINOV2_MODEL  = "dinov2_vits14"
TARGET_SIZE   = DEFAULT_CROP_SIZE


# ── Preprocessing (copied verbatim from 03_extract_embeddings.py) ─────────────

def _make_transforms(size: int = TARGET_SIZE):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Detectors ─────────────────────────────────────────────────────────────────

def detect_gdino(pil_img, device, proc=None, model=None):
    """Return list of [x0,y0,x1,y1] boxes (float) using GDino-tiny.

    Thresholds match generate_pseudo_labels.py defaults so pipeline behavior
    is consistent with the pseudo-labels used for training.
    """
    import numpy as np
    import torch
    from torchvision.ops import nms as tv_nms
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    if proc is None:
        proc = AutoProcessor.from_pretrained(GDINO_MODEL)
    if model is None:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL)
        model = model.to(device).eval()

    pixel_values = gdino_preprocess(pil_img).unsqueeze(0)  # (1, C, H, W)
    H_orig, W_orig = pil_img.size[1], pil_img.size[0]
    H_new, W_new = pixel_values.shape[2], pixel_values.shape[3]
    pixel_mask = torch.ones(1, H_new, W_new, dtype=torch.long)

    text_enc       = proc.tokenizer(GDINO_TEXT, return_tensors="pt", padding=True)
    input_ids      = text_enc["input_ids"]
    attn_mask      = text_enc["attention_mask"]
    token_type_ids = text_enc.get("token_type_ids", torch.zeros_like(input_ids))

    with torch.no_grad():
        out = model(
            pixel_values=pixel_values.to(device),
            pixel_mask=pixel_mask.to(device),
            input_ids=input_ids.to(device),
            attention_mask=attn_mask.to(device),
            token_type_ids=token_type_ids.to(device),
        )

    results = proc.post_process_grounded_object_detection(
        out,
        input_ids,
        threshold=GDINO_BOX_THR,
        text_threshold=GDINO_TEXT_THR,
        target_sizes=[(H_orig, W_orig)],
    )
    boxes_t  = results[0]["boxes"].cpu()
    scores_t = results[0]["scores"].cpu()

    # Post-NMS score filter + NMS — mirrors generate_pseudo_labels.filter_boxes()
    if len(boxes_t) > 0:
        mask = scores_t >= GDINO_SCORE_THR
        boxes_t, scores_t = boxes_t[mask], scores_t[mask]
    if len(boxes_t) > 0:
        keep = tv_nms(boxes_t.float(), scores_t.float(), GDINO_IOU_THR)
        boxes_t = boxes_t[keep]

    return boxes_t.tolist()


def detect_yolo(pil_img, checkpoint: str = None, model=None, imgsz: int = 1280):
    """Return list of [x0,y0,x1,y1] boxes using YOLO.
    imgsz=1280 matches training imgsz for both tree_yolo26s_halfres and
    tree_yolo26s_unified_halfres_tdus. Pass imgsz explicitly to override.
    """
    from ultralytics import YOLO
    if model is None:
        model = YOLO(checkpoint)
    results = model(pil_img, verbose=False, imgsz=imgsz)
    boxes = []
    for r in results:
        for box in r.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            boxes.append([x0, y0, x1, y1])
    return boxes


def detect_rf_detr(pil_img, checkpoint: str = None, device: str = "cpu", detector=None):
    """Return list of [x0,y0,x1,y1] boxes using RF-DETR."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from rf_detr_detector import RFDETRDetector
    if detector is None:
        detector = RFDETRDetector(checkpoint=checkpoint, threshold=0.3, device=device)
    return detector.detect(pil_img)  # already returns list[list[float]]


# ── DINOv2 embedding ──────────────────────────────────────────────────────────

def extract_embedding(crop_pil, device, dinov2=None, transform=None):
    """Return 384-dim float32 numpy array (CLS token from DINOv2 ViT-S/14)."""
    import torch
    if dinov2 is None:
        dinov2 = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL).to(device).eval()
    if transform is None:
        transform = _make_transforms()
    tensor = transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = dinov2.forward_features(tensor)
    cls = feat["x_norm_clstoken"].cpu().float().numpy()[0]  # (384,)
    return cls


# ── SVM inference ─────────────────────────────────────────────────────────────

def predict_species(embedding, svm_model_path: str = None, clf=None):
    """Return (species: str, confidence: float) using the saved SVM."""
    import joblib
    import numpy as np
    if clf is None:
        clf = joblib.load(svm_model_path)
    emb = embedding.reshape(1, -1)
    species = clf.predict(emb)[0]
    # Use decision_function for a proxy confidence score when probability=False
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(emb)[0]
        confidence = float(proba.max())
    else:
        decision = clf.decision_function(emb)[0]
        # Normalise via softmax over classes for a [0,1] proxy
        if decision.ndim == 0:
            confidence = 1.0  # binary single-score → always "confident"
        else:
            import numpy as _np
            exp = _np.exp(decision - decision.max())
            proba = exp / exp.sum()
            confidence = float(proba.max())
    return species, confidence


# ── Batch inference helpers ───────────────────────────────────────────────────

def gdino_preprocess_batch(pil_imgs: list, proc) -> dict:
    """CPU-only step: run HuggingFace processor on a list of PIL images.

    Safe to call from a background thread (no GPU work).
    Returns the inputs dict (tensors on CPU).
    """
    n = len(pil_imgs)
    return proc(
        images=pil_imgs,
        text=[GDINO_TEXT] * n,
        return_tensors="pt",
        padding=True,
    )


def gdino_forward_batch(
    inputs: dict,
    target_sizes: list,
    device: str,
    proc,
    model,
) -> list[list]:
    """GPU step: H2D transfer + model forward + postprocess + NMS.

    inputs: CPU dict returned by gdino_preprocess_batch.
    target_sizes: list of (H, W) tuples, one per image.
    Returns list[list[list[float]]]: one box list per image.
    """
    import torch
    from torchvision.ops import nms as tv_nms

    inputs_gpu = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs_gpu)

    results = proc.post_process_grounded_object_detection(
        out,
        inputs_gpu["input_ids"],
        threshold=GDINO_BOX_THR,
        text_threshold=GDINO_TEXT_THR,
        target_sizes=target_sizes,
    )

    batch_boxes: list[list] = []
    for r in results:
        boxes_t  = r["boxes"].cpu()
        scores_t = r["scores"].cpu()
        if len(boxes_t) > 0:
            mask = scores_t >= GDINO_SCORE_THR
            boxes_t, scores_t = boxes_t[mask], scores_t[mask]
        if len(boxes_t) > 0:
            keep = tv_nms(boxes_t.float(), scores_t.float(), GDINO_IOU_THR)
            boxes_t = boxes_t[keep]
        batch_boxes.append(boxes_t.tolist())
    return batch_boxes


def detect_gdino_batch(pil_imgs: list, device: str, proc=None, model=None) -> list[list]:
    """Batch GDino detection: one GPU forward for all images in the batch.

    Uses the HuggingFace processor's native batch API (padding to max size).
    Returns list[list[list[float]]]: one box list per input image.
    Same thresholds and NMS as detect_gdino().
    Internally delegates to gdino_preprocess_batch + gdino_forward_batch.
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    if proc is None:
        proc = AutoProcessor.from_pretrained(GDINO_MODEL)
    if model is None:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL)
        model = model.to(device).eval()

    target_sizes = [(img.size[1], img.size[0]) for img in pil_imgs]
    inputs = gdino_preprocess_batch(pil_imgs, proc)
    return gdino_forward_batch(inputs, target_sizes, device, proc, model)


def detect_yolo_batch(imgs: list, model, imgsz: int = 1280) -> list[list]:
    """Run YOLO on a list of PIL images or numpy HWC uint8 arrays.
    Numpy arrays skip Ultralytics' internal PIL→numpy conversion (~15ms/img).
    imgsz should match the worker max_edge to avoid a second resize inside YOLO.
    Returns list of box lists ([x0,y0,x1,y1] in original image coordinates).
    """
    results = model(imgs, verbose=False, imgsz=imgsz)
    batch_boxes = []
    for r in results:
        boxes = [box.xyxy[0].tolist() for box in r.boxes]
        batch_boxes.append(boxes)
    return batch_boxes


def extract_embedding_batch(crop_pils: list, device: str, dinov2, transform) -> "np.ndarray":
    """Return (N, 384) float32 array for a batch of crop PIL images."""
    import numpy as np
    import torch
    tensors = torch.stack([transform(c) for c in crop_pils]).to(device)  # (N, 3, 448, 448)
    with torch.no_grad():
        feat = dinov2.forward_features(tensors)
    return feat["x_norm_clstoken"].cpu().float().numpy()  # (N, 384)


_DINOV2_MEAN = [0.485, 0.456, 0.406]
_DINOV2_STD  = [0.229, 0.224, 0.225]


def make_crops_gpu_batch(
    imgs_gpu: "list[torch.Tensor]",  # list of (3, H, W) float32 [0,1] on device (variable sizes)
    batch_boxes: list,                # list of list of [x0,y0,x1,y1]
    target_size: int = TARGET_SIZE,
    pad_frac: float = 0.05,
) -> "torch.Tensor":                  # (N, 3, target_size, target_size) normalized, on device
    """
    GPU-native crop + aspect-ratio-preserving resize + white-canvas pad + normalize.
    Accepts a list of variable-size tensors (handles portrait/landscape mixes).
    Replaces the CPU make_crop + transform loop (~637ms → ~5ms for batch=16).
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    device = imgs_gpu[0].device
    mean = torch.tensor(_DINOV2_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(_DINOV2_STD,  device=device).view(1, 3, 1, 1)

    crops = []
    for img, boxes in zip(imgs_gpu, batch_boxes):
        # img: (3, H, W)
        H, W = img.shape[1], img.shape[2]
        if boxes:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            x0, y0, x1, y1 = boxes[int(np.argmax(areas))]
            bw, bh = x1 - x0, y1 - y0
            x0 = max(0.0, x0 - bw * pad_frac)
            y0 = max(0.0, y0 - bh * pad_frac)
            x1 = min(W, x1 + bw * pad_frac)
            y1 = min(H, y1 + bh * pad_frac)
            crop = img[:, int(y0):int(y1), int(x0):int(x1)]
        else:
            crop = img

        ch, cw = crop.shape[1], crop.shape[2]
        if ch == 0 or cw == 0:
            crop = img
            ch, cw = crop.shape[1], crop.shape[2]

        # Resize to fit within target_size, maintaining aspect ratio
        scale = target_size / max(ch, cw)
        new_h, new_w = max(1, int(ch * scale)), max(1, int(cw * scale))
        crop = F.interpolate(crop.unsqueeze(0), size=(new_h, new_w),
                             mode="bilinear", align_corners=False)  # (1,3,new_h,new_w)

        # Pad to target_size × target_size with white (1.0)
        ph, pw = target_size - new_h, target_size - new_w
        crop = F.pad(crop, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=1.0)
        crops.append(crop)

    batch = torch.cat(crops, dim=0)          # (N, 3, target_size, target_size)
    return (batch - mean) / std              # normalize in-place on GPU


def predict_species_batch(embeddings: "np.ndarray", clf) -> list[tuple[str, float]]:
    """Return list of (species, confidence) for a (N, 384) embeddings array."""
    import numpy as np
    species_list = clf.predict(embeddings)
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(embeddings)
        confidences = probas.max(axis=1).tolist()
    else:
        decisions = clf.decision_function(embeddings)
        exp = np.exp(decisions - decisions.max(axis=1, keepdims=True))
        probas = exp / exp.sum(axis=1, keepdims=True)
        confidences = probas.max(axis=1).tolist()
    return list(zip(species_list, [float(c) for c in confidences]))


# ── Model loader (call once, reuse across images) ─────────────────────────────

def load_models(detector: str, device: str,
                yolo_checkpoint: str | None = None,
                svm_model_path: str | None = None,
                rf_detr_checkpoint: str | None = None,
                yolo_imgsz: int = 1280,
                crop_size: int = TARGET_SIZE) -> dict:
    """Load all models once and return a dict.

    Pass the returned dict to run_predict() to avoid reloading on every image.
    The "detector" key holds a BaseDetector instance; dinov2 and clf are stored
    separately since they are shared across all detector types.
    """
    import joblib
    import torch
    from detector_factory import make_detector

    models: dict = {"device": device}

    checkpoint = yolo_checkpoint or rf_detr_checkpoint
    models["detector"] = make_detector(detector, device=device, checkpoint=checkpoint,
                                       imgsz=yolo_imgsz)

    models["dinov2"]    = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL).to(device).eval()
    models["transform"] = _make_transforms(crop_size)
    models["crop_size"] = crop_size
    if svm_model_path:
        models["clf"] = joblib.load(svm_model_path)

    return models


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_predict(image_path: str, detector: str, svm_model_path: str,
                yolo_checkpoint: str | None = None,
                rf_detr_checkpoint: str | None = None,
                device: str | None = None,
                models: dict | None = None) -> dict:
    """
    Full inference pipeline for one image.
    Returns a dict with keys: image, species, confidence, n_boxes.

    Pass a pre-loaded `models` dict (from load_models()) to avoid reloading
    models on every call — critical for batch throughput.
    Crop + normalize + DINOv2 embedding run on GPU via make_crops_gpu_batch.
    """
    import numpy as np
    import torch
    import torchvision.transforms.functional as tvf
    from PIL import Image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _device = models["device"] if models else device

    pil_img = Image.open(image_path).convert("RGB")

    # --- Detection ---
    if models and "detector" in models:
        det_obj = models["detector"]
    else:
        from detector_factory import make_detector
        checkpoint = yolo_checkpoint or rf_detr_checkpoint
        det_obj = make_detector(detector, device=_device, checkpoint=checkpoint)
    boxes = det_obj.detect(pil_img)

    n_boxes = len(boxes)

    # --- GPU crop + normalize + embedding ---
    if models and "dinov2" in models:
        dinov2 = models["dinov2"]
    else:
        dinov2 = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL).to(_device).eval()

    img_tensor = tvf.to_tensor(pil_img).to(_device)          # (3, H, W) float32 [0,1]
    crops = make_crops_gpu_batch([img_tensor], [boxes])       # (1, 3, 448, 448) normalized
    with torch.no_grad():
        feat = dinov2.forward_features(crops)
    embedding = feat["x_norm_clstoken"].cpu().float().numpy()[0]  # (384,)

    # --- SVM prediction ---
    species, confidence = predict_species(
        embedding,
        svm_model_path=svm_model_path,
        clf=models.get("clf") if models else None,
    )

    return {
        "image":      image_path,
        "species":    species,
        "confidence": round(float(confidence), 4),
        "n_boxes":    n_boxes,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-image tree species prediction pipeline."
    )
    parser.add_argument("--image",   required=True, help="Path to input image")
    add_detector_args(parser)
    parser.add_argument(
        "--svm-model",
        required=True,
        help="Path to trained SVM model (.joblib)",
    )
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Validate detector-specific requirements early (before loading heavy models)
    if args.detector == "rf-detr" and not args.rf_detr_checkpoint:
        parser.error("--rf-detr-checkpoint is required when --detector rf-detr")
    if args.detector == "yolo" and not args.yolo_checkpoint:
        parser.error("--yolo-checkpoint is required when --detector yolo")

    result = run_predict(
        image_path        = args.image,
        detector          = args.detector,
        svm_model_path    = args.svm_model,
        yolo_checkpoint   = args.yolo_checkpoint,
        rf_detr_checkpoint= args.rf_detr_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
