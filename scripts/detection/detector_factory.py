"""Detector factory — create a BaseDetector by name."""
from __future__ import annotations

from base_detector import BaseDetector


def make_detector(name: str, device: str = "cpu", **kwargs) -> BaseDetector:
    """Create and return a detector by name.

    Args:
        name:     "gdino", "yolo", or "rf-detr".
        device:   compute device string ("cpu", "cuda", ...).
        **kwargs: forwarded to the detector constructor:
                  yolo    → checkpoint (required), imgsz (default 1280)
                  rf-detr → checkpoint (required), threshold (default 0.3)
    """
    if name == "gdino":
        from gdino_detector import GDinoDetector
        return GDinoDetector(device=device)

    if name == "yolo":
        from yolo_detector import YOLODetector
        checkpoint = kwargs.get("checkpoint")
        if not checkpoint:
            raise ValueError("yolo detector requires checkpoint=<path>")
        return YOLODetector(checkpoint=checkpoint, imgsz=kwargs.get("imgsz", 1280))

    if name == "rf-detr":
        from rf_detr_detector import RFDETRDetector
        checkpoint = kwargs.get("checkpoint")
        if not checkpoint:
            raise ValueError("rf-detr detector requires checkpoint=<path>")
        return RFDETRDetector(
            checkpoint=checkpoint,
            threshold=kwargs.get("threshold", 0.3),
            device=device,
        )

    raise ValueError(f"Unknown detector: {name!r}. Choose gdino, yolo, or rf-detr.")
