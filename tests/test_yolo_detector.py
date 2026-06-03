"""TDD tests for YOLODetector — written before implementation."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _make_detector(checkpoint="fake.pt", imgsz=1280):
    from yolo_detector import YOLODetector
    det = YOLODetector(checkpoint=checkpoint, imgsz=imgsz)
    det._model = MagicMock()
    return det


class TestYOLODetector:
    def test_supports_pipeline_false(self):
        assert _make_detector().supports_pipeline is False

    def test_checkpoint_and_imgsz_stored(self):
        from yolo_detector import YOLODetector
        det = YOLODetector(checkpoint="my.pt", imgsz=640)
        assert det.checkpoint == "my.pt"
        assert det.imgsz == 640

    def test_detect_batch_delegates_with_imgsz(self):
        det = _make_detector(imgsz=640)
        pil_imgs = [Image.new("RGB", (100, 100))]
        expected = [[[0.0, 0.0, 50.0, 50.0]]]
        with patch("yolo_detector.detect_yolo_batch", return_value=expected) as fn:
            result = det.detect_batch(pil_imgs)
        fn.assert_called_once_with(pil_imgs, det._model, imgsz=640)
        assert result == expected

    def test_detect_single_image(self):
        det = _make_detector()
        expected_batch = [[[1.0, 2.0, 3.0, 4.0]]]
        with patch("yolo_detector.detect_yolo_batch", return_value=expected_batch):
            result = det.detect(Image.new("RGB", (50, 50)))
        assert result == [[1.0, 2.0, 3.0, 4.0]]

    def test_detect_empty_returns_empty(self):
        det = _make_detector()
        with patch("yolo_detector.detect_yolo_batch", return_value=[[]]):
            result = det.detect(Image.new("RGB", (50, 50)))
        assert result == []

    def test_preprocess_cpu_raises(self):
        det = _make_detector()
        with pytest.raises(NotImplementedError):
            det.preprocess_cpu([Image.new("RGB", (10, 10))])
