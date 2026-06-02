"""TDD tests for scripts/gdino_utils.py — shared GDino preprocessing.

These tests must fail before gdino_utils.py exists (RED), then pass after
the implementation is written (GREEN).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _make_pil(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), color=(120, 80, 60))


class TestGdinoPreprocess:
    """gdino_utils.gdino_preprocess must replicate the resize+normalise logic
    previously duplicated in generate_pseudo_labels and predict_pipeline."""

    def test_returns_tensor(self):
        from gdino_utils import gdino_preprocess
        out = gdino_preprocess(_make_pil(640, 480))
        assert isinstance(out, torch.Tensor)

    def test_output_is_3hw(self):
        from gdino_utils import gdino_preprocess
        out = gdino_preprocess(_make_pil(640, 480))
        assert out.ndim == 3
        assert out.shape[0] == 3

    def test_shortest_edge_set_for_large_image(self):
        """min(H,W) after resize must equal SHORTEST_EDGE when image is large enough."""
        from gdino_utils import gdino_preprocess, SHORTEST_EDGE
        # 1920×1080: min edge = 1080 → scale = 800/1080 → max edge = 1920*(800/1080) ≈ 1422 > 1333
        # so longest-edge clamp kicks in; use an image where it doesn't: 1200×900
        # min=900 → scale=800/900, new max=1200*(800/900)≈1067 ≤ 1333 → no clamp
        out = gdino_preprocess(_make_pil(1200, 900))
        h, w = out.shape[1], out.shape[2]
        assert min(h, w) == SHORTEST_EDGE

    def test_longest_edge_capped(self):
        """Very wide image: max(H,W) after resize must be ≤ LONGEST_EDGE."""
        from gdino_utils import gdino_preprocess, LONGEST_EDGE
        # 4000×600: min=600 → scale=800/600≈1.33 → new max=4000*1.33=5333 > 1333 → clamped
        out = gdino_preprocess(_make_pil(4000, 600))
        h, w = out.shape[1], out.shape[2]
        assert max(h, w) <= LONGEST_EDGE

    def test_values_are_normalized(self):
        """Output must be ImageNet-normalised (values outside [0, 1] expected)."""
        from gdino_utils import gdino_preprocess
        out = gdino_preprocess(_make_pil(320, 240))
        assert (out < 0).any() or (out > 1).any()

    def test_output_dtype_float32(self):
        from gdino_utils import gdino_preprocess
        out = gdino_preprocess(_make_pil(320, 240))
        assert out.dtype == torch.float32

    def test_matches_generate_pseudo_labels_reference(self):
        """Must produce the same tensor as the original generate_pseudo_labels impl."""
        from gdino_utils import gdino_preprocess, SHORTEST_EDGE, LONGEST_EDGE

        pil = _make_pil(800, 600)

        # Reference: replicate generate_pseudo_labels implementation exactly
        GDINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        GDINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        w, h = pil.size
        scale = SHORTEST_EDGE / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        if max(new_h, new_w) > LONGEST_EDGE:
            scale = LONGEST_EDGE / max(new_h, new_w)
            new_h, new_w = int(round(new_h * scale)), int(round(new_w * scale))
        resized = pil.resize((new_w, new_h), Image.BILINEAR)
        t_ref = torch.as_tensor(np.array(resized), dtype=torch.float32).permute(2, 0, 1) / 255.0
        t_ref = (t_ref - GDINO_MEAN) / GDINO_STD

        t_out = gdino_preprocess(pil)
        assert torch.allclose(t_out, t_ref, atol=1e-6), "Output differs from reference"

    def test_square_image_both_edges_equal(self):
        """Square image: H == W after resize, both equal to SHORTEST_EDGE (if ≤ LONGEST_EDGE)."""
        from gdino_utils import gdino_preprocess, SHORTEST_EDGE
        out = gdino_preprocess(_make_pil(900, 900))
        h, w = out.shape[1], out.shape[2]
        assert h == w == SHORTEST_EDGE


class TestGdinoUtils:
    """gdino_utils must export the expected constants."""

    def test_shortest_edge_is_800(self):
        from gdino_utils import SHORTEST_EDGE
        assert SHORTEST_EDGE == 800

    def test_longest_edge_is_1333(self):
        from gdino_utils import LONGEST_EDGE
        assert LONGEST_EDGE == 1333
