"""Shared GDino preprocessing utilities.

Previously duplicated verbatim between generate_pseudo_labels.py and
predict_pipeline.py; extracted here so both import from a single source.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

SHORTEST_EDGE = 800
LONGEST_EDGE  = 1333

_GDINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_GDINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def gdino_preprocess(pil_img) -> torch.Tensor:
    """Resize + ImageNet-normalise a PIL image for GDino inference.

    Applies the SHORTEST_EDGE=800 / LONGEST_EDGE=1333 two-step resize used by
    Grounding DINO (mirrors 03_extract_embeddings.py).
    Returns a (3, H, W) float32 tensor.
    """
    w, h = pil_img.size
    scale = SHORTEST_EDGE / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    if max(new_h, new_w) > LONGEST_EDGE:
        scale = LONGEST_EDGE / max(new_h, new_w)
        new_h, new_w = int(round(new_h * scale)), int(round(new_w * scale))
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    t = torch.as_tensor(np.array(resized), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return (t - _GDINO_MEAN) / _GDINO_STD
