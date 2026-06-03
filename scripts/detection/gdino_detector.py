"""GDinoDetector — Grounding DINO detector with double-buffer support."""
from __future__ import annotations

from base_detector import BaseDetector
from predict_pipeline import detect_gdino_batch, gdino_preprocess_batch, gdino_forward_batch

try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None  # type: ignore

GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"


class GDinoDetector(BaseDetector):
    """Wraps Grounding DINO-tiny; supports the preprocess_cpu / forward_preprocessed
    double-buffer interface for throughput-critical benchmark loops."""

    def __init__(self, device: str = "cpu", model_id: str = GDINO_MODEL) -> None:
        self.device   = device
        self.model_id = model_id
        self._proc    = None
        self._model   = None

    # ── Lazy model loading ───────────────────────────────────────────────────────

    def _load(self) -> None:
        from transformers import (
            AutoProcessor as AP,
            AutoModelForZeroShotObjectDetection as AZOD,
        )
        if self._proc is None:
            self._proc = AP.from_pretrained(self.model_id)
        if self._model is None:
            self._model = AZOD.from_pretrained(self.model_id).to(self.device).eval()

    def _load_proc(self) -> None:
        if self._proc is None:
            self._proc = AutoProcessor.from_pretrained(self.model_id)

    # ── BaseDetector implementation ──────────────────────────────────────────────

    def detect_batch(self, pil_imgs: list) -> list[list]:
        self._load()
        return detect_gdino_batch(pil_imgs, self.device, self._proc, self._model)

    @property
    def supports_pipeline(self) -> bool:
        return True

    def preprocess_cpu(self, pil_imgs: list) -> tuple:
        """CPU preprocessing: returns (inputs_dict, target_sizes).

        target_sizes is [(H, W), ...] — the convention expected by
        gdino_forward_batch.
        """
        self._load_proc()
        inputs       = gdino_preprocess_batch(pil_imgs, self._proc)
        target_sizes = [(img.size[1], img.size[0]) for img in pil_imgs]
        return inputs, target_sizes

    def forward_preprocessed(self, preprocessed: tuple) -> list[list]:
        inputs, target_sizes = preprocessed
        self._load()
        return gdino_forward_batch(inputs, target_sizes, self.device, self._proc, self._model)
