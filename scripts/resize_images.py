"""Resize images so the longest edge equals --max-edge (default 1280).

Usage:
    .venv/bin/python scripts/resize_images.py \
        --src tdus_data/train/img \
        --dst data/tdus_resized/train/img \
        --max-edge 1280
"""
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def resize_dir(src: Path, dst: Path, max_edge: int, quality: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    paths = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS]
    for src_path in tqdm(paths, desc=f"{src.name}→{dst.name}"):
        dst_path = dst / src_path.name
        if dst_path.exists():
            continue
        img = Image.open(src_path).convert("RGB")
        w, h = img.size
        scale = max_edge / max(w, h)
        if scale < 1.0:
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=quality)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize images to max-edge length")
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--max-edge", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    if not args.src.is_dir():
        raise FileNotFoundError(f"src not found: {args.src}")

    resize_dir(args.src, args.dst, args.max_edge, args.quality)
    out_count = sum(1 for p in args.dst.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"Done: {out_count} images in {args.dst}")


if __name__ == "__main__":
    main()
