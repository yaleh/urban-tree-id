"""
tests/test_image_utils.py

TDD tests for scripts/image_utils.py — make_crop_xyxy and make_crop_norm.
"""
import sys
from pathlib import Path

# Ensure scripts/ is importable (mirrors conftest.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
from PIL import Image

from image_utils import make_crop_xyxy, make_crop_norm, DEFAULT_CROP_SIZE


def _make_img(w=200, h=150):
    """Create a small solid-color test image."""
    return Image.new("RGB", (w, h), color=(100, 150, 200))


# ── Equivalence ───────────────────────────────────────────────────────────────

class TestEquivalence:
    """make_crop_xyxy and make_crop_norm must produce pixel-identical results
    when given coordinates that represent the same region."""

    def test_same_output_for_equivalent_boxes(self):
        img = _make_img(200, 150)
        # xyxy pixel coords
        box_xyxy = (20, 10, 120, 90)
        # equivalent normalized coords
        box_norm = (20/200, 10/150, 120/200, 90/150)

        out_xyxy = make_crop_xyxy(img, box_xyxy)
        out_norm = make_crop_norm(img, box_norm)

        # Both must be the same size
        assert out_xyxy.size == out_norm.size

        import numpy as np
        arr_xyxy = np.array(out_xyxy)
        arr_norm = np.array(out_norm)
        assert (arr_xyxy == arr_norm).all(), "Pixel values differ between xyxy and norm variants"


# ── Output size ───────────────────────────────────────────────────────────────

class TestOutputSize:
    def test_default_target_size(self):
        img = _make_img()
        box = (10, 10, 100, 80)
        out = make_crop_xyxy(img, box)
        assert isinstance(out, Image.Image)
        assert out.size == (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE)

    def test_custom_target_size(self):
        img = _make_img()
        box = (10, 10, 100, 80)
        out = make_crop_xyxy(img, box, target_size=64)
        assert out.size == (64, 64)

    def test_norm_default_target_size(self):
        img = _make_img()
        box_norm = (0.05, 0.05, 0.8, 0.8)
        out = make_crop_norm(img, box_norm)
        assert out.size == (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE)


# ── Edge truncation ───────────────────────────────────────────────────────────

class TestEdgeTruncation:
    def test_box_at_top_left_corner(self):
        """Box starting at (0, 0) should not raise and produce correct output size."""
        img = _make_img(200, 150)
        box = (0, 0, 80, 60)
        out = make_crop_xyxy(img, box)
        assert out.size == (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE)

    def test_box_at_bottom_right_corner(self):
        """Box ending at image boundary should not raise."""
        img = _make_img(200, 150)
        box = (120, 90, 200, 150)
        out = make_crop_xyxy(img, box)
        assert out.size == (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE)

    def test_box_larger_than_image_clamped(self):
        """Coordinates outside image boundary should be clamped without error."""
        img = _make_img(200, 150)
        box = (-10, -10, 210, 160)
        out = make_crop_xyxy(img, box)
        assert out.size == (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE)


# ── Tiny box protection ───────────────────────────────────────────────────────

class TestTinyBoxProtection:
    def test_sub_pixel_width(self):
        """Width < 1px: max(1, ...) guard must prevent zero-size crop error."""
        img = _make_img(200, 150)
        # Very thin box: width = 0.5px
        box = (50.0, 50.0, 50.5, 80.0)
        out = make_crop_xyxy(img, box, pad_frac=0.0)
        assert out.size == (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE)

    def test_sub_pixel_height(self):
        """Height < 1px: max(1, ...) guard must prevent zero-size crop error."""
        img = _make_img(200, 150)
        box = (50.0, 50.0, 100.0, 50.3)
        out = make_crop_xyxy(img, box, pad_frac=0.0)
        assert out.size == (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE)


# ── pad_frac=0 ────────────────────────────────────────────────────────────────

class TestNoPadding:
    def test_pad_frac_zero_no_expansion(self):
        """With pad_frac=0, the crop region equals the original box exactly."""
        import numpy as np

        img = _make_img(200, 150)
        # Use a box well away from all edges so no clamping occurs
        x0, y0, x1, y1 = 40, 30, 120, 100
        box = (x0, y0, x1, y1)

        # Manual reference: crop exactly the box region, then scale
        ref_crop = img.crop(box)
        cw, ch = ref_crop.size
        scale = DEFAULT_CROP_SIZE / max(cw, ch)
        nw = max(1, int(cw * scale))
        nh = max(1, int(ch * scale))
        scaled = ref_crop.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE), (255, 255, 255))
        canvas.paste(scaled, ((DEFAULT_CROP_SIZE - nw) // 2, (DEFAULT_CROP_SIZE - nh) // 2))

        out = make_crop_xyxy(img, box, pad_frac=0.0)

        arr_ref = np.array(canvas)
        arr_out = np.array(out)
        assert (arr_ref == arr_out).all(), "pad_frac=0 result differs from direct crop"
