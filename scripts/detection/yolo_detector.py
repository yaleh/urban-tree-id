"""YOLODetector — Ultralytics YOLO detector."""
from __future__ import annotations

from base_detector import BaseDetector
from predict_pipeline import detect_yolo_batch


class YOLODetector(BaseDetector):
    """Wraps an Ultralytics YOLO checkpoint.

    Ultralytics handles its own internal batching; the double-buffer
    pipeline interface is not used.
    """

    def __init__(self, checkpoint: str, imgsz: int = 1280) -> None:
        self.checkpoint = checkpoint
        self.imgsz      = imgsz
        self._model     = None

    def _load(self) -> None:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.checkpoint)

    def detect_batch(self, pil_imgs: list) -> list[list]:
        self._load()
        return detect_yolo_batch(pil_imgs, self._model, imgsz=self.imgsz)
