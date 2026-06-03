"""TDD tests for detector_factory.make_detector."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestMakeDetector:
    def test_gdino_returns_gdino_detector(self):
        from detector_factory import make_detector
        from gdino_detector import GDinoDetector
        det = make_detector("gdino", device="cpu")
        assert isinstance(det, GDinoDetector)
        assert det.device == "cpu"

    def test_yolo_returns_yolo_detector(self):
        from detector_factory import make_detector
        from yolo_detector import YOLODetector
        det = make_detector("yolo", device="cpu", checkpoint="model.pt", imgsz=640)
        assert isinstance(det, YOLODetector)
        assert det.checkpoint == "model.pt"
        assert det.imgsz == 640

    def test_yolo_requires_checkpoint(self):
        from detector_factory import make_detector
        with pytest.raises(ValueError, match="checkpoint"):
            make_detector("yolo")

    def test_rfdetr_requires_checkpoint(self):
        from detector_factory import make_detector
        with pytest.raises(ValueError, match="checkpoint"):
            make_detector("rf-detr")

    def test_unknown_detector_raises(self):
        from detector_factory import make_detector
        with pytest.raises(ValueError, match="Unknown detector"):
            make_detector("ssd")

    def test_rfdetr_returns_rfdetr_detector(self):
        from detector_factory import make_detector
        from rf_detr_detector import RFDETRDetector
        mock_rfdetr_base = MagicMock()
        with patch.dict("sys.modules", {"rfdetr": MagicMock(RFDETRBase=mock_rfdetr_base)}):
            det = make_detector("rf-detr", device="cpu", checkpoint="model.pth")
        assert isinstance(det, RFDETRDetector)
        assert det.threshold == 0.3

    def test_rfdetr_forwards_threshold(self):
        from detector_factory import make_detector
        from rf_detr_detector import RFDETRDetector
        mock_rfdetr_base = MagicMock()
        with patch.dict("sys.modules", {"rfdetr": MagicMock(RFDETRBase=mock_rfdetr_base)}):
            det = make_detector("rf-detr", device="cpu", checkpoint="m.pth", threshold=0.5)
        assert det.threshold == 0.5
