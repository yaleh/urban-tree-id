"""TDD tests for GDinoDetector — written before implementation."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _make_detector():
    from gdino_detector import GDinoDetector
    det = GDinoDetector(device="cpu")
    det._proc  = MagicMock()
    det._model = MagicMock()
    return det


class TestGDinoDetector:
    def test_supports_pipeline(self):
        assert _make_detector().supports_pipeline is True

    def test_detect_batch_delegates(self):
        det = _make_detector()
        pil_imgs = [Image.new("RGB", (50, 50))]
        expected = [[[10.0, 20.0, 30.0, 40.0]]]
        with patch("gdino_detector.detect_gdino_batch", return_value=expected) as fn:
            result = det.detect_batch(pil_imgs)
        fn.assert_called_once_with(pil_imgs, det.device, det._proc, det._model)
        assert result == expected

    def test_preprocess_cpu_returns_inputs_and_hw_sizes(self):
        det = _make_detector()
        pil_imgs = [Image.new("RGB", (100, 80)), Image.new("RGB", (60, 40))]
        mock_inputs = {"pixel_values": MagicMock()}
        with patch("gdino_detector.gdino_preprocess_batch", return_value=mock_inputs):
            inputs, sizes = det.preprocess_cpu(pil_imgs)
        assert inputs is mock_inputs
        # PIL.size is (W, H); target_sizes must be (H, W)
        assert sizes == [(80, 100), (40, 60)]

    def test_forward_preprocessed_delegates(self):
        det = _make_detector()
        mock_inputs  = MagicMock()
        target_sizes = [(80, 100)]
        expected     = [[[1.0, 2.0, 3.0, 4.0]]]
        with patch("gdino_detector.gdino_forward_batch", return_value=expected) as fn:
            result = det.forward_preprocessed((mock_inputs, target_sizes))
        fn.assert_called_once_with(mock_inputs, target_sizes, det.device, det._proc, det._model)
        assert result == expected

    def test_detect_single_image(self):
        det = _make_detector()
        expected_batch = [[[5.0, 10.0, 15.0, 20.0]]]
        with patch("gdino_detector.detect_gdino_batch", return_value=expected_batch):
            result = det.detect(Image.new("RGB", (50, 50)))
        assert result == [[5.0, 10.0, 15.0, 20.0]]

    def test_preprocess_cpu_loads_proc_if_missing(self):
        from gdino_detector import GDinoDetector
        det = GDinoDetector(device="cpu")
        # _proc is None — preprocess_cpu should load it via AutoProcessor
        mock_proc = MagicMock()
        pil_imgs  = [Image.new("RGB", (50, 50))]
        with patch("gdino_detector.AutoProcessor") as MockAP, \
             patch("gdino_detector.gdino_preprocess_batch", return_value={}):
            MockAP.from_pretrained.return_value = mock_proc
            det.preprocess_cpu(pil_imgs)
        MockAP.from_pretrained.assert_called_once()
        assert det._proc is mock_proc
