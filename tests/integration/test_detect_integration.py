"""
Integration tests for the detection pipeline with real model checkpoints.

Marked with @pytest.mark.integration so they can be excluded from CI:
    pytest -m "not integration"

Or run explicitly:
    pytest tests/integration/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_YOLO_N      = _PROJECT_ROOT / "yolo26n.pt"
_YOLO_S      = _PROJECT_ROOT / "yolo26s.pt"


def _yolo_checkpoint():
    """Return the smallest available YOLO checkpoint."""
    for p in [_YOLO_N, _YOLO_S]:
        if p.exists():
            return str(p)
    return None


def _make_synthetic_image(width=640, height=480):
    """Create a synthetic RGB image that looks vaguely natural (green/brown gradient)."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    # Sky gradient
    for y in range(height // 3):
        blue = int(150 + y * 0.5)
        draw.line([(0, y), (width, y)], fill=(100, 130, blue))
    # Ground
    for y in range(height // 3, height):
        green = int(100 - (y - height // 3) * 0.1)
        draw.rectangle([(0, y), (width, y)], fill=(50, max(green, 40), 30))
    # Draw a rough "tree" blob
    cx, cy = width // 2, height // 2
    draw.ellipse([cx - 80, cy - 120, cx + 80, cy + 20], fill=(40, 100, 30))
    draw.rectangle([cx - 10, cy + 20, cx + 10, cy + 80], fill=(80, 50, 20))
    return img


@pytest.mark.integration
@pytest.mark.skipif(_yolo_checkpoint() is None, reason="No YOLO checkpoint found")
class TestYOLODetectorIntegration:
    """Real YOLO inference on a synthetic image."""

    @pytest.fixture(scope="class")
    def detector(self):
        from yolo_detector import YOLODetector
        return YOLODetector(checkpoint=_yolo_checkpoint(), imgsz=640)

    @pytest.fixture(scope="class")
    def synthetic_image(self):
        return _make_synthetic_image()

    def test_detect_returns_list(self, detector, synthetic_image):
        result = detector.detect(synthetic_image)
        assert isinstance(result, list)

    def test_detect_each_box_has_four_coords(self, detector, synthetic_image):
        result = detector.detect(synthetic_image)
        for box in result:
            assert len(box) == 4, f"Each box must have 4 coords, got {len(box)}"

    def test_detect_coords_are_finite_floats(self, detector, synthetic_image):
        result = detector.detect(synthetic_image)
        for box in result:
            for coord in box:
                assert isinstance(coord, float) and np.isfinite(coord)

    def test_detect_batch_length_matches_input(self, detector, synthetic_image):
        imgs = [synthetic_image, synthetic_image.crop((0, 0, 320, 240))]
        result = detector.detect_batch(imgs)
        assert len(result) == 2

    def test_detect_batch_each_element_is_list(self, detector, synthetic_image):
        imgs = [synthetic_image]
        result = detector.detect_batch(imgs)
        assert isinstance(result[0], list)

    def test_coords_within_image_bounds(self, detector, synthetic_image):
        W, H = synthetic_image.size
        for box in detector.detect(synthetic_image):
            x0, y0, x1, y1 = box
            assert x0 >= 0 and y0 >= 0
            assert x1 <= W * 1.01 and y1 <= H * 1.01  # 1% tolerance for fp
            assert x0 < x1 and y0 < y1

    def test_no_exception_on_blank_image(self, detector):
        blank = Image.new("RGB", (320, 240), color=(200, 200, 200))
        result = detector.detect(blank)
        assert isinstance(result, list)


@pytest.mark.integration
@pytest.mark.skipif(_yolo_checkpoint() is None, reason="No YOLO checkpoint found")
class TestYOLODetectorFromFactory:
    """Verify make_detector('yolo', ...) returns a working detector."""

    def test_factory_creates_working_detector(self):
        from detector_factory import make_detector
        det = make_detector("yolo", device="cpu", checkpoint=_yolo_checkpoint(), imgsz=640)
        img = _make_synthetic_image()
        result = det.detect(img)
        assert isinstance(result, list)

    def test_is_base_detector_instance(self):
        from detector_factory import make_detector
        from base_detector import BaseDetector
        det = make_detector("yolo", device="cpu", checkpoint=_yolo_checkpoint())
        assert isinstance(det, BaseDetector)
