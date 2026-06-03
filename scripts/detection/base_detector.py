"""BaseDetector — unified detection interface for the inference pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDetector(ABC):
    """Abstract base for all tree detectors.

    Canonical API: detect_batch() / detect().
    For throughput-critical paths, callers that need double-buffer preprocessing
    can use preprocess_cpu() + forward_preprocessed() when supports_pipeline is True.
    """

    @abstractmethod
    def detect_batch(self, pil_imgs: list) -> list[list]:
        """Detect objects in a batch of PIL images.

        Returns one list per image; each list contains [x0, y0, x1, y1] float
        rows in absolute pixel coordinates.
        """

    def detect(self, pil_img) -> list:
        """Single-image convenience wrapper."""
        return self.detect_batch([pil_img])[0]

    # ── Double-buffer interface ──────────────────────────────────────────────────

    @property
    def supports_pipeline(self) -> bool:
        """True when preprocess_cpu / forward_preprocessed are implemented."""
        return False

    def preprocess_cpu(self, pil_imgs: list) -> Any:
        """CPU-only preprocessing, safe to run in a background thread.

        Pair with forward_preprocessed() for double-buffer throughput.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support pipeline mode")

    def forward_preprocessed(self, preprocessed: Any) -> list[list]:
        """GPU forward pass given preprocess_cpu() output."""
        raise NotImplementedError(f"{type(self).__name__} does not support pipeline mode")
