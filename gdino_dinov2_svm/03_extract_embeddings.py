"""
03_extract_embeddings.py

Pipeline per image:
  1. GroundingDINO-tiny: detect "tree." → pick max-area bbox above score_thr
  2. Crop + pad to 448×448 on white canvas
  3. DINOv2 ViT-S/14: extract CLS token (384-dim)

Supports two dataset layouts:
  --dataset-format urban  (default): data_dir/tree/{split}/{species}/*.jpg
  --dataset-format tdus:             data_dir/{split}/img/*.jpg
                                     data_dir/{split}/ann/*.jpg.json
                                     (label from tags[0].value in JSON)

Output: embeddings/{train,val,test}.npz  (embeddings: N×384, labels: N str)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision.transforms as T
import sys
from pathlib import Path

_scripts = Path(__file__).parent.parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from gdino_utils import gdino_preprocess  # noqa: E402
from image_utils import make_crop_xyxy as make_crop, DEFAULT_CROP_SIZE  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

GDINO_MODEL   = "IDEA-Research/grounding-dino-tiny"
DINOV2_MODEL  = "dinov2_vits14"
TARGET_SIZE   = DEFAULT_CROP_SIZE
SHORTEST_EDGE = 800
LONGEST_EDGE  = 1333
GDINO_MEAN    = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
GDINO_STD     = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DINOV2_TRANSFORM = T.Compose([
    T.Resize(TARGET_SIZE),
    T.CenterCrop(TARGET_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def collate_fn(batch):
    tensors, labels, pil_imgs = zip(*batch)
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded, masks = [], []
    for t in tensors:
        h, w = t.shape[1], t.shape[2]
        padded.append(F.pad(t, (0, max_w - w, 0, max_h - h)))
        m = torch.zeros(max_h, max_w, dtype=torch.long)
        m[:h, :w] = 1
        masks.append(m)
    return torch.stack(padded), torch.stack(masks), list(labels), list(pil_imgs)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TreeDataset(Dataset):
    """Reads data_dir/tree/{split}/{species}/*.jpg"""

    def __init__(self, data_dir, split):
        self.samples = []
        root = Path(data_dir) / "tree" / split
        for species_dir in sorted(root.iterdir()):
            if not species_dir.is_dir():
                continue
            label = species_dir.name
            for p in sorted(species_dir.glob("*.jpg")):
                self.samples.append((str(p), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil = Image.open(path).convert("RGB")
        return gdino_preprocess(pil), label, pil


class TDUSDataset(Dataset):
    """Reads data_dir/{split}/img/*.jpg with labels from {split}/ann/*.jpg.json"""

    def __init__(self, data_dir, split):
        self.samples = []
        img_dir = Path(data_dir) / split / "img"
        ann_dir = Path(data_dir) / split / "ann"
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if "_thumb" in img_path.name:
                continue
            ann_path = ann_dir / (img_path.name + ".json")
            with open(ann_path) as f:
                ann = json.load(f)
            label = ann["tags"][0]["value"]
            self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil = Image.open(path).convert("RGB")
        return gdino_preprocess(pil), label, pil


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_split(split, data_dir, out_dir, gdino_proc, gdino_model,
                  dinov2, device, batch_size, num_workers, text, score_thr,
                  dataset_format="urban", crop_size=448):
    out_path = Path(out_dir) / f"{split}.npz"
    if out_path.exists():
        log.info(f"{split}: {out_path} exists, skipping")
        return

    dataset = TDUSDataset(data_dir, split) if dataset_format == "tdus" else TreeDataset(data_dir, split)
    log.info(f"{split}: {len(dataset)} images")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn,
                        pin_memory=True, prefetch_factor=2)

    # Pre-tokenize text prompt once
    text_enc = gdino_proc.tokenizer(text, return_tensors="pt", padding=True)
    input_ids_1      = text_enc["input_ids"]
    attn_mask_1      = text_enc["attention_mask"]
    token_type_ids_1 = text_enc.get("token_type_ids", torch.zeros_like(input_ids_1))

    all_emb, all_labels = [], []
    n_fallback = 0

    for pixel_values, pixel_mask, labels, pil_imgs in tqdm(loader, desc=split):
        B = pixel_values.shape[0]
        input_ids      = input_ids_1.repeat(B, 1).to(device)
        attn_mask      = attn_mask_1.repeat(B, 1).to(device)
        token_type_ids = token_type_ids_1.repeat(B, 1).to(device)

        with torch.no_grad():
            gdino_out = gdino_model(
                pixel_values=pixel_values.to(device, non_blocking=True),
                pixel_mask=pixel_mask.to(device, non_blocking=True),
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids,
            )

        # Post-process bboxes in pixel coords for each original image
        orig_sizes = [(pil.size[1], pil.size[0]) for pil in pil_imgs]  # (H, W)
        results = gdino_proc.post_process_grounded_object_detection(
            gdino_out,
            input_ids,
            box_threshold=score_thr,
            text_threshold=score_thr,
            target_sizes=orig_sizes,
        )

        dinov2_transform = T.Compose([
            T.Resize(crop_size),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        crops = []
        for i, (res, pil) in enumerate(zip(results, pil_imgs)):
            boxes = res["boxes"].cpu().tolist()
            if boxes:
                # pick largest-area box (matches notebook behaviour)
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                best_idx = int(np.argmax(areas))
                crop = make_crop(pil, boxes[best_idx], target_size=crop_size)
            else:
                n_fallback += 1
                crop = pil.resize((crop_size, crop_size), Image.LANCZOS)
            crops.append(dinov2_transform(crop))

        crop_batch = torch.stack(crops).to(device, non_blocking=True)
        with torch.no_grad():
            feat = dinov2.forward_features(crop_batch)
        cls = feat["x_norm_clstoken"].cpu().float().numpy()  # (B, 384)

        all_emb.append(cls)
        all_labels.extend(labels)

    if n_fallback:
        log.warning(f"{split}: {n_fallback} images had no detection, used full image")

    emb    = np.vstack(all_emb).astype(np.float32)
    labels = np.array(all_labels)
    np.savez(out_path, embeddings=emb, labels=labels)
    species, counts = np.unique(labels, return_counts=True)
    log.info(f"{split}: saved {out_path} — N={len(labels)}, dim={emb.shape[1]}, classes={len(species)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",        default="dataset")
    parser.add_argument("--out-dir",         default="embeddings")
    parser.add_argument("--splits",          nargs="+", default=["train", "val", "test"])
    parser.add_argument("--text",            default="tree.")
    parser.add_argument("--score-thr",       type=float, default=0.3)
    parser.add_argument("--batch-size",      type=int, default=8)
    parser.add_argument("--num-workers",     type=int, default=4)
    parser.add_argument("--dataset-format",  default="urban", choices=["urban", "tdus"])
    parser.add_argument("--crop-size",       type=int, default=448,
                        help="DINOv2 input crop size (default: 448)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info(f"Loading {GDINO_MODEL}...")
    gdino_proc  = AutoProcessor.from_pretrained(GDINO_MODEL)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL)
    gdino_model = gdino_model.to(device).eval()

    log.info(f"Loading DINOv2 {DINOV2_MODEL}...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL).to(device).eval()

    log.info(f"Crop size: {args.crop_size}px")
    for split in args.splits:
        extract_split(split, args.data_dir, args.out_dir,
                      gdino_proc, gdino_model, dinov2, device,
                      args.batch_size, args.num_workers, args.text, args.score_thr,
                      args.dataset_format, crop_size=args.crop_size)
        torch.cuda.empty_cache()

    log.info("Done.")


if __name__ == "__main__":
    main()
