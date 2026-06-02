"""
image_utils.py

Unified crop utilities for the urban-tree-id pipeline.

Provides two public functions:
  make_crop_xyxy  — crop from pixel-coordinate xyxy box
  make_crop_norm  — crop from normalised [0,1] xyxy box (delegates to make_crop_xyxy)

Both pad the box by pad_frac, resize aspect-ratio-preservingly, and centre the
result on a white canvas of size target_size × target_size.
"""

from PIL import Image

DEFAULT_CROP_SIZE = 448


def make_crop_xyxy(
    pil_img: Image.Image,
    box_xyxy,
    pad_frac: float = 0.05,
    target_size: int = DEFAULT_CROP_SIZE,
) -> Image.Image:
    """Crop a PIL image using pixel-coordinate xyxy box, with padding.

    Pads the bbox by pad_frac * box_dimension on each side, clamps to image
    boundaries, resizes the crop to fit within target_size while preserving
    aspect ratio, and centres the result on a white canvas of
    target_size × target_size.

    Args:
        pil_img: Source PIL image (RGB or any mode).
        box_xyxy: Sequence (x0, y0, x1, y1) in pixel coordinates.
        pad_frac: Fractional padding to add around the box (default 0.05).
        target_size: Side length in pixels of the output square canvas.

    Returns:
        PIL Image of size (target_size, target_size).
    """
    W, H = pil_img.size
    x0, y0, x1, y1 = box_xyxy
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, x0 - bw * pad_frac)
    y0 = max(0, y0 - bh * pad_frac)
    x1 = min(W, x1 + bw * pad_frac)
    y1 = min(H, y1 + bh * pad_frac)
    crop = pil_img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    scale = target_size / max(cw, ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    scaled = crop.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    canvas.paste(scaled, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return canvas


def make_crop_norm(
    pil_img: Image.Image,
    box_norm,
    pad_frac: float = 0.05,
    target_size: int = DEFAULT_CROP_SIZE,
) -> Image.Image:
    """Crop a PIL image using normalised [0, 1] xyxy box coordinates.

    Converts normalised coordinates to pixel coordinates and delegates to
    make_crop_xyxy.

    Args:
        pil_img: Source PIL image.
        box_norm: Sequence (x0, y0, x1, y1) normalised to [0, 1].
        pad_frac: Fractional padding (default 0.05).
        target_size: Output canvas side length in pixels.

    Returns:
        PIL Image of size (target_size, target_size).
    """
    w, h = pil_img.size
    x0 = box_norm[0] * w
    y0 = box_norm[1] * h
    x1 = box_norm[2] * w
    y1 = box_norm[3] * h
    return make_crop_xyxy(pil_img, (x0, y0, x1, y1), pad_frac, target_size)
