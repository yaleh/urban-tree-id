"""
tests/test_rf_detr_detector.py

TDD tests for scripts/rf_detr_detector.py.

Mock strategy: patch 'rfdetr.RFDETRBase' so no real model file is needed.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_mock_detections(xyxy: np.ndarray) -> MagicMock:
    """Return a mock Detections object whose .xyxy attribute is *xyxy*."""
    det = MagicMock()
    det.xyxy = xyxy
    return det


def _patch_rfdetr(xyxy: np.ndarray):
    """
    Context manager that patches rfdetr.RFDETRBase so that:
      - instantiation returns a mock model
      - model.predict(...) returns a mock Detections with the given xyxy array
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_mock_detections(xyxy)

    mock_cls = MagicMock(return_value=mock_model)
    return patch.dict(
        "sys.modules",
        {"rfdetr": MagicMock(RFDETRBase=mock_cls)},
    )


# ── tests ─────────────────────────────────────────────────────────────────────

class TestRFDETRDetector:

    def test_detect_returns_ndarray(self):
        """detect() must return a numpy ndarray with shape (N, 4) and dtype float32."""
        boxes = np.array([[10, 20, 50, 80], [5, 15, 30, 60]], dtype=np.float32)

        with _patch_rfdetr(boxes):
            from rf_detr_detector import RFDETRDetector
            det = RFDETRDetector(checkpoint="fake.pth", device="cpu")
            result = det.detect("fake_image.jpg")

        assert isinstance(result, np.ndarray), (
            f"Expected np.ndarray, got {type(result)}"
        )
        assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_detect_empty_returns_zeros(self):
        """When no detections exist, detect() must return shape (0, 4) float32 array."""
        empty = np.zeros((0, 4), dtype=np.float32)

        with _patch_rfdetr(empty):
            from rf_detr_detector import RFDETRDetector
            det = RFDETRDetector(checkpoint="fake.pth", device="cpu")
            result = det.detect("fake_image.jpg")

        assert isinstance(result, np.ndarray), (
            f"Expected np.ndarray, got {type(result)}"
        )
        assert result.shape == (0, 4), f"Expected shape (0, 4), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_detect_xyxy_format(self):
        """Each row in the result must satisfy x1 < x2 and y1 < y2."""
        boxes = np.array(
            [[10.0, 20.0, 50.0, 80.0],
             [5.0,  15.0, 30.0, 60.0]],
            dtype=np.float32,
        )

        with _patch_rfdetr(boxes):
            from rf_detr_detector import RFDETRDetector
            det = RFDETRDetector(checkpoint="fake.pth", device="cpu")
            result = det.detect("fake_image.jpg")

        for i, row in enumerate(result):
            x1, y1, x2, y2 = row
            assert x1 < x2, f"Row {i}: x1={x1} must be < x2={x2}"
            assert y1 < y2, f"Row {i}: y1={y1} must be < y2={y2}"

    def test_detect_batch_length(self):
        """detect_batch(imgs) must return a list whose length equals len(imgs)."""
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        images = ["img1.jpg", "img2.jpg", "img3.jpg"]

        with _patch_rfdetr(boxes):
            from rf_detr_detector import RFDETRDetector
            det = RFDETRDetector(checkpoint="fake.pth", device="cpu")
            results = det.detect_batch(images)

        assert isinstance(results, list), (
            f"Expected list, got {type(results)}"
        )
        assert len(results) == len(images), (
            f"Expected {len(images)} results, got {len(results)}"
        )
        for r in results:
            assert isinstance(r, np.ndarray), (
                f"Each element must be np.ndarray, got {type(r)}"
            )


# ── detect_batch_gpu tests ────────────────────────────────────────────────────

def _make_gpu_batch_detector(per_image_boxes: list[np.ndarray], threshold: float = 0.3):
    """
    Return an RFDETRDetector whose internals are mocked for detect_batch_gpu.

    Object hierarchy mirrors real rfdetr:
      RFDETRDetector.model          → mock_rfdetr_base  (RFDETRBase mock)
      RFDETRDetector.model.model    → mock_rf_model     (Model mock, has .resolution/.device/.postprocessors)
      RFDETRDetector.model.model.model → mock_inner     (nn.Module mock, called in forward)
    """
    import torch
    from unittest.mock import MagicMock

    post_results = []
    for boxes_np in per_image_boxes:
        nb = len(boxes_np)
        post_results.append({
            "scores": torch.full((nb,), 0.9),
            "boxes":  torch.as_tensor(boxes_np, dtype=torch.float32),
        })

    mock_inner    = MagicMock()
    mock_inner.return_value = {}

    mock_postproc = MagicMock(return_value=post_results)

    # rf_model: has .resolution, .device, .postprocessors, .model (nn.Module)
    mock_rf_model = MagicMock()
    mock_rf_model.model          = mock_inner
    mock_rf_model.resolution     = 560
    mock_rf_model.device         = "cpu"
    mock_rf_model.postprocessors = {"bbox": mock_postproc}

    # RFDETRBase: .model points to mock_rf_model
    mock_rfdetr_base = MagicMock()
    mock_rfdetr_base.model = mock_rf_model

    mock_cls = MagicMock(return_value=mock_rfdetr_base)

    with patch.dict("sys.modules", {"rfdetr": MagicMock(RFDETRBase=mock_cls)}):
        from rf_detr_detector import RFDETRDetector
        det = RFDETRDetector.__new__(RFDETRDetector)
    det.model     = mock_rfdetr_base
    det.threshold = threshold
    return det, mock_inner, mock_postproc


class TestRFDETRDetectorBatchGPU:

    def _make_pil(self, w=640, h=480):
        from PIL import Image as PILImage
        return PILImage.new("RGB", (w, h), color=(128, 64, 32))

    def test_batch_gpu_returns_list_of_correct_length(self):
        """detect_batch_gpu(imgs) returns list whose length == len(imgs)."""
        imgs = [self._make_pil() for _ in range(3)]
        per_img = [np.zeros((0, 4), dtype=np.float32)] * 3

        det, _, _ = _make_gpu_batch_detector(per_img)
        result = det.detect_batch_gpu(imgs)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_batch_gpu_each_element_is_list_of_boxes(self):
        """Each element of detect_batch_gpu result is a list of [x0,y0,x1,y1]."""
        boxes = np.array([[10.0, 20.0, 50.0, 80.0]], dtype=np.float32)
        imgs  = [self._make_pil(), self._make_pil()]
        per_img = [boxes, boxes]

        det, _, _ = _make_gpu_batch_detector(per_img)
        result = det.detect_batch_gpu(imgs)

        for item in result:
            assert isinstance(item, list), f"Expected list, got {type(item)}"
            for box in item:
                assert len(box) == 4, f"Each box must have 4 coords, got {len(box)}"

    def test_batch_gpu_empty_detection_yields_empty_list(self):
        """Images with no detections produce empty inner lists."""
        imgs    = [self._make_pil()]
        per_img = [np.zeros((0, 4), dtype=np.float32)]

        det, _, _ = _make_gpu_batch_detector(per_img)
        result = det.detect_batch_gpu(imgs)

        assert result == [[]], f"Expected [[]], got {result}"

    def test_batch_gpu_calls_forward_once(self):
        """detect_batch_gpu must call model.model.forward exactly once (true batching)."""
        imgs    = [self._make_pil() for _ in range(4)]
        per_img = [np.zeros((0, 4), dtype=np.float32)] * 4

        det, mock_inner, _ = _make_gpu_batch_detector(per_img)
        det.detect_batch_gpu(imgs)

        mock_inner.assert_called_once()

    def test_batch_gpu_threshold_filters_low_confidence(self):
        """Boxes whose scores are below threshold must be excluded."""
        import torch
        from unittest.mock import MagicMock

        boxes_np = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        post_results = [{"scores": torch.tensor([0.1]), "boxes": torch.as_tensor(boxes_np)}]

        mock_inner    = MagicMock(return_value={})
        mock_postproc = MagicMock(return_value=post_results)
        mock_rf_model = MagicMock()
        mock_rf_model.model          = mock_inner
        mock_rf_model.resolution     = 560
        mock_rf_model.device         = "cpu"
        mock_rf_model.postprocessors = {"bbox": mock_postproc}
        mock_rfdetr_base = MagicMock()
        mock_rfdetr_base.model = mock_rf_model

        mock_cls = MagicMock(return_value=mock_rfdetr_base)
        with patch.dict("sys.modules", {"rfdetr": MagicMock(RFDETRBase=mock_cls)}):
            from rf_detr_detector import RFDETRDetector
            det = RFDETRDetector.__new__(RFDETRDetector)
        det.model     = mock_rfdetr_base
        det.threshold = 0.3

        result = det.detect_batch_gpu([self._make_pil()])
        assert result == [[]], f"Expected no boxes after threshold filter, got {result}"


# ── preprocess_cpu / forward_from_cpu_tensors split ───────────────────────────

class TestRFDETRPreprocessCPU:
    """preprocess_cpu must be CPU-only (safe to thread)."""

    def _make_pil(self, w=320, h=240):
        from PIL import Image as PILImage
        return PILImage.new("RGB", (w, h), color=(64, 128, 192))

    def _make_chw_tensor(self, w=320, h=240):
        import torch
        return torch.randint(0, 256, (3, h, w), dtype=torch.uint8)

    def _make_det(self):
        import torch
        from unittest.mock import MagicMock
        mock_inner    = MagicMock(return_value={})
        mock_rf_model = MagicMock()
        mock_rf_model.model      = mock_inner
        mock_rf_model.resolution = 560
        mock_rf_model.device     = "cpu"
        mock_rf_model.postprocessors = {"bbox": MagicMock(return_value=[])}
        mock_base = MagicMock()
        mock_base.model = mock_rf_model
        mock_cls = MagicMock(return_value=mock_base)
        with patch.dict("sys.modules", {"rfdetr": MagicMock(RFDETRBase=mock_cls)}):
            from rf_detr_detector import RFDETRDetector
            det = RFDETRDetector.__new__(RFDETRDetector)
        det.model     = mock_base
        det.threshold = 0.3
        return det

    def test_preprocess_cpu_exists(self):
        from rf_detr_detector import RFDETRDetector
        assert hasattr(RFDETRDetector, "preprocess_cpu")
        assert callable(RFDETRDetector.preprocess_cpu)

    def test_returns_tuple_of_tensors_and_sizes(self):
        det = self._make_det()
        imgs = [self._make_pil(), self._make_pil()]
        tensors_cpu, orig_sizes = det.preprocess_cpu(imgs)
        assert isinstance(tensors_cpu, list)
        assert isinstance(orig_sizes, list)
        assert len(tensors_cpu) == 2
        assert len(orig_sizes) == 2

    def test_output_tensors_are_cpu(self):
        import torch
        det = self._make_det()
        imgs = [self._make_pil(320, 240)]
        tensors_cpu, _ = det.preprocess_cpu(imgs)
        assert tensors_cpu[0].device.type == "cpu", "preprocess_cpu must stay on CPU"

    def test_output_tensors_are_float(self):
        import torch
        det = self._make_det()
        imgs = [self._make_pil()]
        tensors_cpu, _ = det.preprocess_cpu(imgs)
        assert tensors_cpu[0].dtype == torch.float32

    def test_output_tensors_in_range_01(self):
        det = self._make_det()
        imgs = [self._make_pil()]
        tensors_cpu, _ = det.preprocess_cpu(imgs)
        t = tensors_cpu[0]
        assert t.min() >= 0.0 and t.max() <= 1.0, f"Expected [0,1], got [{t.min():.3f},{t.max():.3f}]"

    def test_orig_sizes_are_hw_tuples(self):
        det = self._make_det()
        imgs = [self._make_pil(w=320, h=240)]
        _, orig_sizes = det.preprocess_cpu(imgs)
        h, w = orig_sizes[0]
        assert h == 240 and w == 320

    def test_accepts_uint8_chw_tensor(self):
        import torch
        det = self._make_det()
        tensor = self._make_chw_tensor(w=320, h=240)
        tensors_cpu, orig_sizes = det.preprocess_cpu([tensor])
        assert len(tensors_cpu) == 1
        h, w = orig_sizes[0]
        assert h == 240 and w == 320


class TestRFDETRForwardFromCPUTensors:
    """forward_from_cpu_tensors must call GPU forward exactly once."""

    def _make_det_with_boxes(self, boxes_np, n_imgs=1):
        import torch
        from unittest.mock import MagicMock

        post_results = []
        for b in boxes_np:
            nb = len(b)
            post_results.append({
                "scores": torch.full((nb,), 0.9),
                "boxes":  torch.as_tensor(b, dtype=torch.float32) if nb else torch.zeros(0,4),
            })

        mock_inner    = MagicMock(return_value={})
        mock_postproc = MagicMock(return_value=post_results)
        mock_rf_model = MagicMock()
        mock_rf_model.model          = mock_inner
        mock_rf_model.resolution     = 560
        mock_rf_model.device         = "cpu"
        mock_rf_model.postprocessors = {"bbox": mock_postproc}
        mock_base = MagicMock()
        mock_base.model = mock_rf_model
        mock_cls = MagicMock(return_value=mock_base)
        with patch.dict("sys.modules", {"rfdetr": MagicMock(RFDETRBase=mock_cls)}):
            from rf_detr_detector import RFDETRDetector
            det = RFDETRDetector.__new__(RFDETRDetector)
        det.model     = mock_base
        det.threshold = 0.3
        return det, mock_inner

    def _cpu_tensors(self, n=2, h=240, w=320):
        import torch
        return [torch.rand(3, h, w) for _ in range(n)], [(h, w)] * n

    def test_forward_from_cpu_tensors_exists(self):
        from rf_detr_detector import RFDETRDetector
        assert callable(getattr(RFDETRDetector, "forward_from_cpu_tensors", None))

    def test_returns_list_of_correct_length(self):
        import numpy as np
        n = 3
        boxes_np = [np.zeros((0, 4), dtype=np.float32)] * n
        det, _ = self._make_det_with_boxes(boxes_np, n_imgs=n)
        tensors_cpu, orig_sizes = self._cpu_tensors(n)
        result = det.forward_from_cpu_tensors(tensors_cpu, orig_sizes)
        assert isinstance(result, list) and len(result) == n

    def test_calls_inner_forward_once(self):
        import numpy as np
        n = 4
        boxes_np = [np.zeros((0, 4), dtype=np.float32)] * n
        det, mock_inner = self._make_det_with_boxes(boxes_np, n_imgs=n)
        tensors_cpu, orig_sizes = self._cpu_tensors(n)
        det.forward_from_cpu_tensors(tensors_cpu, orig_sizes)
        mock_inner.assert_called_once()

    def test_detect_batch_gpu_still_works_as_before(self):
        """detect_batch_gpu must remain backward compatible after refactor."""
        import numpy as np
        from PIL import Image as PILImage
        n = 2
        boxes_np = [np.zeros((0, 4), dtype=np.float32)] * n
        det, _ = self._make_det_with_boxes(boxes_np, n_imgs=n)
        imgs = [PILImage.new("RGB", (320, 240)) for _ in range(n)]
        result = det.detect_batch_gpu(imgs)
        assert isinstance(result, list) and len(result) == n


# ── Method A: detect_batch_gpu accepts CHW uint8 Tensor input ─────────────────

class TestRFDETRDetectorBatchGPUTensorInput:
    """detect_batch_gpu must accept torch.Tensor (CHW uint8) in addition to PIL/path."""

    def _make_chw_tensor(self, w=640, h=480) -> "torch.Tensor":
        import torch
        return torch.randint(0, 256, (3, h, w), dtype=torch.uint8)

    def _make_detector_with_empty_boxes(self, n_imgs: int):
        """Return a detector whose postprocessor yields n_imgs empty box lists."""
        import torch
        from unittest.mock import MagicMock

        per_image_boxes = [np.zeros((0, 4), dtype=np.float32)] * n_imgs
        return _make_gpu_batch_detector(per_image_boxes)

    def test_tensor_input_returns_list(self):
        """detect_batch_gpu with tensor inputs must return a list."""
        import torch
        tensors = [self._make_chw_tensor() for _ in range(2)]
        det, _, _ = _make_gpu_batch_detector([np.zeros((0, 4), dtype=np.float32)] * 2)
        result = det.detect_batch_gpu(tensors)
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_tensor_input_correct_length(self):
        """Output list length must equal number of input tensors."""
        import torch
        n = 3
        tensors = [self._make_chw_tensor() for _ in range(n)]
        det, _, _ = _make_gpu_batch_detector([np.zeros((0, 4), dtype=np.float32)] * n)
        result = det.detect_batch_gpu(tensors)
        assert len(result) == n, f"Expected {n} results, got {len(result)}"

    def test_tensor_input_calls_forward_once(self):
        """Tensor batch input must still result in exactly one forward pass."""
        import torch
        tensors = [self._make_chw_tensor() for _ in range(3)]
        det, mock_inner, _ = _make_gpu_batch_detector(
            [np.zeros((0, 4), dtype=np.float32)] * 3
        )
        det.detect_batch_gpu(tensors)
        mock_inner.assert_called_once()

    def test_mixed_pil_and_tensor_not_required(self):
        """A pure-tensor batch (no PIL) must produce same-shaped output as a PIL batch."""
        import torch
        from PIL import Image as PILImage

        n = 2
        w, h = 320, 240
        tensors = [self._make_chw_tensor(w=w, h=h) for _ in range(n)]
        pil_imgs = [PILImage.new("RGB", (w, h)) for _ in range(n)]

        per_img = [np.zeros((0, 4), dtype=np.float32)] * n
        det_t, _, _ = _make_gpu_batch_detector(per_img)
        det_p, _, _ = _make_gpu_batch_detector(per_img)

        result_t = det_t.detect_batch_gpu(tensors)
        result_p = det_p.detect_batch_gpu(pil_imgs)

        assert len(result_t) == len(result_p)
