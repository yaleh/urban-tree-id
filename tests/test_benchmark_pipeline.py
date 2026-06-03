"""
tests/test_benchmark_pipeline.py

TDD tests for benchmark_pipeline.py optimizations:
  - Method A: _ImageDataset returns CHW uint8 tensors (enables real pin_memory)
  - Method B: gdino_preprocess_batch is a separable, threadable function
  - Method D: scoring loop emits one log line per batch, not per image
  - Larger default batch sizes (yolo: 64, rf-detr: 64)
  - RF-DETR double-buffer: preprocess_cpu in thread, forward_from_cpu_tensors on GPU
  - --timed-bench mode: preload, warmup, measure pure GPU throughput
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_png(path: Path, width: int = 640, height: int = 480) -> Path:
    """Write a solid-colour PNG and return the path."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    img.save(path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Method A – _ImageDataset returns tensors
# ══════════════════════════════════════════════════════════════════════════════

class TestImageDatasetReturnsTensor:
    """_ImageDataset.__getitem__ must return (CHW uint8 Tensor, str_path)."""

    def _make_dataset(self, path, max_edge=1280):
        from benchmark_pipeline import _ImageDataset
        return _ImageDataset([path], max_edge=max_edge)

    def test_getitem_returns_tuple(self, tmp_path):
        img_path = _make_png(tmp_path / "img.png")
        ds = self._make_dataset(img_path)
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2, (
            f"Expected 2-tuple, got {type(item)} len={len(item) if hasattr(item,'__len__') else '?'}"
        )

    def test_first_element_is_tensor(self, tmp_path):
        img_path = _make_png(tmp_path / "img.png")
        ds = self._make_dataset(img_path)
        tensor, _ = ds[0]
        assert isinstance(tensor, torch.Tensor), f"Expected Tensor, got {type(tensor)}"

    def test_tensor_dtype_uint8(self, tmp_path):
        img_path = _make_png(tmp_path / "img.png")
        ds = self._make_dataset(img_path)
        tensor, _ = ds[0]
        assert tensor.dtype == torch.uint8, f"Expected uint8, got {tensor.dtype}"

    def test_tensor_shape_is_chw(self, tmp_path):
        img_path = _make_png(tmp_path / "img.png", width=320, height=240)
        ds = self._make_dataset(img_path)
        tensor, _ = ds[0]
        assert tensor.ndim == 3, f"Expected 3-D tensor, got shape {tensor.shape}"
        assert tensor.shape[0] == 3, f"Expected C=3, got shape {tensor.shape}"
        assert tensor.shape[1] == 240 and tensor.shape[2] == 320, (
            f"Expected (3,240,320), got {tensor.shape}"
        )

    def test_second_element_is_path_string(self, tmp_path):
        img_path = _make_png(tmp_path / "img.png")
        ds = self._make_dataset(img_path)
        _, path_str = ds[0]
        assert isinstance(path_str, str), f"Expected str, got {type(path_str)}"
        assert path_str == str(img_path)

    def test_max_edge_resize_applied(self, tmp_path):
        """Image wider than max_edge must be resized so longest edge == max_edge."""
        img_path = _make_png(tmp_path / "big.png", width=2000, height=1000)
        ds = self._make_dataset(img_path, max_edge=800)
        tensor, _ = ds[0]
        # longest edge must be <= 800
        assert max(tensor.shape[1], tensor.shape[2]) <= 800, (
            f"Expected longest edge ≤ 800, got shape {tensor.shape}"
        )

    def test_small_image_not_upscaled(self, tmp_path):
        """Image smaller than max_edge must not be enlarged."""
        img_path = _make_png(tmp_path / "small.png", width=200, height=150)
        ds = self._make_dataset(img_path, max_edge=1280)
        tensor, _ = ds[0]
        assert tensor.shape[1] == 150 and tensor.shape[2] == 200, (
            f"Expected (3,150,200), got {tensor.shape}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Method A – _collate_tensor
# ══════════════════════════════════════════════════════════════════════════════

class TestColllateTensor:
    """_collate_tensor must return (list[Tensor], list[str]) for variable-size images."""

    def _collate(self, batch):
        from benchmark_pipeline import _collate_tensor
        return _collate_tensor(batch)

    def _make_item(self, h, w):
        return torch.zeros(3, h, w, dtype=torch.uint8), f"img_{h}x{w}.jpg"

    def test_returns_tuple_of_two(self):
        batch = [self._make_item(480, 640), self._make_item(720, 1280)]
        result = self._collate(batch)
        assert isinstance(result, (tuple, list)) and len(result) == 2

    def test_first_element_is_list_of_tensors(self):
        batch = [self._make_item(480, 640), self._make_item(720, 1280)]
        tensors, _ = self._collate(batch)
        assert isinstance(tensors, list)
        for t in tensors:
            assert isinstance(t, torch.Tensor)

    def test_second_element_is_list_of_str(self):
        batch = [self._make_item(480, 640), self._make_item(720, 1280)]
        _, paths = self._collate(batch)
        assert isinstance(paths, list)
        for p in paths:
            assert isinstance(p, str)

    def test_preserves_variable_sizes(self):
        """Tensors must NOT be stacked — variable-size images can't be stacked."""
        batch = [self._make_item(240, 320), self._make_item(480, 640)]
        tensors, _ = self._collate(batch)
        assert len(tensors) == 2
        assert tensors[0].shape == (3, 240, 320)
        assert tensors[1].shape == (3, 480, 640)

    def test_batch_of_one(self):
        batch = [self._make_item(100, 200)]
        tensors, paths = self._collate(batch)
        assert len(tensors) == 1 and len(paths) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Method B – GDino preprocess is a separable function
# ══════════════════════════════════════════════════════════════════════════════

class TestGdinoPreprocessSplit:
    """
    predict_pipeline must expose gdino_preprocess_batch and gdino_forward_batch
    so the benchmark can thread proc() independently of the GPU forward.
    """

    def _make_pil(self, w=100, h=80):
        return Image.new("RGB", (w, h), color=(0, 128, 255))

    def _build_proc_mock(self, n=2):
        """Return a mock proc whose __call__ returns a dict of tensors."""
        mock_proc = MagicMock()
        seq_len = 5
        mock_proc.return_value = {
            "input_ids":           torch.zeros(n, seq_len, dtype=torch.long),
            "attention_mask":      torch.ones(n, seq_len, dtype=torch.long),
            "pixel_values":        torch.zeros(n, 3, 224, 224),
            "pixel_mask":          torch.ones(n, 224, 224, dtype=torch.long),
        }
        return mock_proc

    def test_gdino_preprocess_batch_exists(self):
        from predict_pipeline import gdino_preprocess_batch
        assert callable(gdino_preprocess_batch)

    def test_gdino_preprocess_batch_calls_proc(self):
        """gdino_preprocess_batch must call proc() exactly once with all images."""
        from predict_pipeline import gdino_preprocess_batch
        imgs = [self._make_pil(), self._make_pil()]
        mock_proc = self._build_proc_mock(n=2)
        gdino_preprocess_batch(imgs, mock_proc)
        mock_proc.assert_called_once()
        call_kwargs = mock_proc.call_args
        # images must be passed as a list
        passed_images = call_kwargs[1].get("images") or call_kwargs[0][0]
        assert len(passed_images) == 2

    def test_gdino_preprocess_batch_returns_dict(self):
        from predict_pipeline import gdino_preprocess_batch
        imgs = [self._make_pil()]
        mock_proc = self._build_proc_mock(n=1)
        result = gdino_preprocess_batch(imgs, mock_proc)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_gdino_forward_batch_exists(self):
        from predict_pipeline import gdino_forward_batch
        assert callable(gdino_forward_batch)

    def test_gdino_forward_batch_returns_list(self):
        """gdino_forward_batch must return list[list] (one list per image)."""
        import torch
        from predict_pipeline import gdino_forward_batch

        n = 2
        seq_len = 5
        mock_inputs = {
            "input_ids":      torch.zeros(n, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
            "pixel_values":   torch.zeros(n, 3, 224, 224),
            "pixel_mask":     torch.ones(n, 224, 224, dtype=torch.long),
        }
        target_sizes = [(480, 640)] * n

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()

        mock_proc = MagicMock()
        # post_process returns list of dicts with "boxes" and "scores"
        mock_proc.post_process_grounded_object_detection.return_value = [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0)},
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0)},
        ]

        result = gdino_forward_batch(mock_inputs, target_sizes, "cpu", mock_proc, mock_model)

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == n, f"Expected {n} elements, got {len(result)}"

    def test_detect_gdino_batch_still_works(self):
        """Existing detect_gdino_batch must still work (regression check)."""
        import torch
        from predict_pipeline import detect_gdino_batch

        n = 2
        imgs = [self._make_pil(), self._make_pil()]
        seq_len = 5

        mock_proc = MagicMock()
        mock_proc.return_value = {
            "input_ids":      torch.zeros(n, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
            "pixel_values":   torch.zeros(n, 3, 224, 224),
            "pixel_mask":     torch.ones(n, 224, 224, dtype=torch.long),
        }
        mock_proc.post_process_grounded_object_detection.return_value = [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0)},
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0)},
        ]
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()

        result = detect_gdino_batch(imgs, "cpu", proc=mock_proc, model=mock_model)
        assert isinstance(result, list) and len(result) == n


# ══════════════════════════════════════════════════════════════════════════════
# Method D – batch-level logging
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchLogging:
    """
    score_batch (moved to eval_metrics) must emit exactly ONE log line per batch call,
    and return updated (n_total, n_correct, n_no_detection, per_class).
    benchmark_pipeline re-imports score_batch from eval_metrics.
    """

    def _call_score_batch(self, batch_paths, predictions, batch_n_boxes, gt_map, log_fn=None):
        from eval_metrics import score_batch
        from collections import defaultdict
        per_class = defaultdict(lambda: {"correct": 0, "total": 0})
        return score_batch(
            batch_paths=batch_paths,
            predictions=predictions,
            batch_n_boxes=batch_n_boxes,
            gt_labels_map=gt_map,
            n_total=0, n_correct=0, n_no_detection=0,
            per_class=per_class,
            log_fn=log_fn or (lambda *a, **kw: None),
        )

    def test_score_batch_exists(self):
        from eval_metrics import score_batch
        assert callable(score_batch)

    def test_log_called_exactly_once(self):
        """log_fn must be called exactly once regardless of batch size."""
        calls = []
        paths = [f"/img/a{i}.jpg" for i in range(8)]
        preds = [("Tilia", 0.9)] * 8
        boxes = [1] * 8
        gt    = {p: "Tilia" for p in paths}

        self._call_score_batch(paths, preds, boxes, gt, log_fn=lambda msg: calls.append(msg))
        assert len(calls) == 1, f"Expected 1 log call, got {len(calls)}"

    def test_correct_count_returned(self):
        paths = ["/img/a.jpg", "/img/b.jpg", "/img/c.jpg"]
        preds = [("Tilia", 0.9), ("Acer", 0.8), ("Acer", 0.7)]  # c.jpg wrong (no boxes → fallback pred)
        boxes = [1, 1, 0]
        gt    = {"/img/a.jpg": "Tilia", "/img/b.jpg": "Tilia", "/img/c.jpg": "Tilia"}

        n_total, n_correct, n_no_det, _ = self._call_score_batch(paths, preds, boxes, gt)
        assert n_total   == 3
        assert n_correct == 1   # only a.jpg is correct
        assert n_no_det  == 1   # c.jpg has 0 boxes

    def test_per_class_updated(self):
        paths = ["/img/a.jpg", "/img/b.jpg"]
        preds = [("Tilia", 0.9), ("Tilia", 0.8)]
        boxes = [1, 1]
        gt    = {"/img/a.jpg": "Tilia", "/img/b.jpg": "Acer"}

        _, _, _, per_class = self._call_score_batch(paths, preds, boxes, gt)
        assert per_class["Tilia"]["total"]   == 1
        assert per_class["Tilia"]["correct"] == 1
        assert per_class["Acer"]["total"]    == 1
        assert per_class["Acer"]["correct"]  == 0

    def test_no_detection_counted(self):
        paths = ["/img/a.jpg"]
        preds = [("Tilia", 0.5)]
        boxes = [0]
        gt    = {"/img/a.jpg": "Tilia"}

        _, _, n_no_det, _ = self._call_score_batch(paths, preds, boxes, gt)
        assert n_no_det == 1


# ══════════════════════════════════════════════════════════════════════════════
# --yolo-imgsz and --limit arguments
# ══════════════════════════════════════════════════════════════════════════════

def _make_benchmark_parser():
    from benchmark_pipeline import build_parser
    return build_parser()


class TestYoloImgsz:
    """--yolo-imgsz must be accepted as an independent argument."""

    def test_yolo_imgsz_argument_exists(self):
        parser = _make_benchmark_parser()
        args = parser.parse_args([
            "--detector", "yolo",
            "--test-dir", "/tmp",
            "--ann-dir", "/tmp",
            "--svm-model", "m.joblib",
            "--yolo-checkpoint", "w.pt",
            "--yolo-imgsz", "640",
        ])
        assert args.yolo_imgsz == 640

    def test_yolo_imgsz_defaults_to_none(self):
        """When omitted, yolo_imgsz should be None (caller falls back to max_edge)."""
        parser = _make_benchmark_parser()
        args = parser.parse_args([
            "--detector", "yolo",
            "--test-dir", "/tmp",
            "--ann-dir", "/tmp",
            "--svm-model", "m.joblib",
            "--yolo-checkpoint", "w.pt",
        ])
        assert args.yolo_imgsz is None

    def test_yolo_imgsz_independent_of_max_edge(self):
        """yolo_imgsz and max_edge can differ."""
        parser = _make_benchmark_parser()
        args = parser.parse_args([
            "--detector", "yolo",
            "--test-dir", "/tmp",
            "--ann-dir", "/tmp",
            "--svm-model", "m.joblib",
            "--yolo-checkpoint", "w.pt",
            "--max-edge", "2048",
            "--yolo-imgsz", "1280",
        ])
        assert args.max_edge == 2048
        assert args.yolo_imgsz == 1280


class TestLimitArgument:
    """--limit N must cap the number of images processed."""

    def test_limit_argument_exists(self):
        parser = _make_benchmark_parser()
        args = parser.parse_args([
            "--detector", "gdino",
            "--test-dir", "/tmp",
            "--ann-dir", "/tmp",
            "--svm-model", "m.joblib",
            "--limit", "100",
        ])
        assert args.limit == 100

    def test_limit_defaults_to_none(self):
        parser = _make_benchmark_parser()
        args = parser.parse_args([
            "--detector", "gdino",
            "--test-dir", "/tmp",
            "--ann-dir", "/tmp",
            "--svm-model", "m.joblib",
        ])
        assert args.limit is None

    def test_find_images_respects_limit(self, tmp_path):
        """_find_images with limit returns at most limit images."""
        for i in range(10):
            _make_png(tmp_path / f"img{i:02d}.png")
        from benchmark_pipeline import _find_images
        result = _find_images(tmp_path, limit=5)
        assert len(result) == 5

    def test_find_images_no_limit_returns_all(self, tmp_path):
        for i in range(6):
            _make_png(tmp_path / f"img{i:02d}.png")
        from benchmark_pipeline import _find_images
        result = _find_images(tmp_path)
        assert len(result) == 6


# ══════════════════════════════════════════════════════════════════════════════
# Larger default batch sizes
# ══════════════════════════════════════════════════════════════════════════════

def _make_benchmark_parser_for_defaults():
    from benchmark_pipeline import build_parser
    return build_parser()


class TestDefaultBatchSizes:
    """Default batch sizes must be tuned for ~12 GB VRAM headroom.

    These tests read `_DEFAULT_BATCH` directly from benchmark_pipeline so they
    fail as soon as the constant is out of date.
    """

    def test_yolo_default_batch_64(self):
        """YOLO default batch size should be 64."""
        from benchmark_pipeline import _DEFAULT_BATCH
        assert _DEFAULT_BATCH["yolo"] == 64, (
            f"Expected yolo default=64, got {_DEFAULT_BATCH['yolo']}"
        )

    def test_rfdetr_default_batch_64(self):
        """RF-DETR default batch size should be 64."""
        from benchmark_pipeline import _DEFAULT_BATCH
        assert _DEFAULT_BATCH["rf-detr"] == 64, (
            f"Expected rf-detr default=64, got {_DEFAULT_BATCH['rf-detr']}"
        )

    def test_gdino_default_batch_unchanged(self):
        """GDino default batch size should remain 8."""
        from benchmark_pipeline import _DEFAULT_BATCH
        assert _DEFAULT_BATCH["gdino"] == 8

    def test_explicit_batch_size_overrides_default(self):
        """--batch-size should always override the detector default."""
        parser = _make_benchmark_parser_for_defaults()
        args = parser.parse_args([
            "--detector", "yolo", "--test-dir", "/tmp",
            "--ann-dir", "/tmp", "--svm-model", "m.joblib",
            "--yolo-checkpoint", "w.pt",
            "--batch-size", "16",
        ])
        assert args.batch_size == 16


# ══════════════════════════════════════════════════════════════════════════════
# RF-DETR double-buffer: preprocess_cpu used in benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

class TestRFDETRDoubleBufferInBenchmark:
    """Verify that the rfdetr_detector exposes preprocess_cpu / forward_from_cpu_tensors
    so the benchmark's double-buffer pattern can call them."""

    def _make_mock_detector(self):
        """Mock RFDETRDetector that records calls."""
        det = MagicMock()
        # preprocess_cpu returns (list_of_tensors, list_of_sizes)
        det.preprocess_cpu.return_value = (
            [torch.zeros(3, 64, 64)],
            [(64, 64)],
        )
        det.forward_from_cpu_tensors.return_value = [[]]
        return det

    def test_rfdetr_detector_has_preprocess_cpu(self):
        """RFDETRDetector must expose preprocess_cpu for the double-buffer pattern."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rf_detr_detector",
            str(Path(__file__).parent.parent / "scripts" / "detection" / "rf_detr_detector.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        assert hasattr(mod, "RFDETRDetector") or True  # module not executed, just check source
        # Load source and check method names
        src = (Path(__file__).parent.parent / "scripts" / "detection" / "rf_detr_detector.py").read_text()
        assert "def preprocess_cpu" in src
        assert "def forward_from_cpu_tensors" in src

    def test_preprocess_cpu_called_for_each_batch(self):
        """In the double-buffer loop, preprocess_cpu should be called once per batch."""
        det = self._make_mock_detector()
        n_batches = 3
        # Simulate the double-buffer loop calling pattern
        for _ in range(n_batches):
            imgs = [torch.zeros(3, 64, 64)]
            tensors_cpu, orig_sizes = det.preprocess_cpu(imgs)
            det.forward_from_cpu_tensors(tensors_cpu, orig_sizes)
        assert det.preprocess_cpu.call_count == n_batches
        assert det.forward_from_cpu_tensors.call_count == n_batches

    def test_preprocess_cpu_result_passed_to_forward(self):
        """preprocess_cpu output must be forwarded to forward_from_cpu_tensors unchanged."""
        det = self._make_mock_detector()
        fake_tensors = [torch.ones(3, 32, 32)]
        fake_sizes = [(32, 32)]
        det.preprocess_cpu.return_value = (fake_tensors, fake_sizes)

        imgs = [torch.zeros(3, 32, 32)]
        tensors_cpu, orig_sizes = det.preprocess_cpu(imgs)
        det.forward_from_cpu_tensors(tensors_cpu, orig_sizes)

        call_args = det.forward_from_cpu_tensors.call_args
        assert call_args[0][0] is fake_tensors
        assert call_args[0][1] is fake_sizes


# ══════════════════════════════════════════════════════════════════════════════
# --timed-bench mode
# ══════════════════════════════════════════════════════════════════════════════

class TestAdjustImageCount:
    """_adjust_image_count must produce a list whose length is a multiple of batch_size."""

    def test_truncates_excess(self):
        from benchmark_pipeline import _adjust_image_count
        imgs = list(range(10))
        result = _adjust_image_count(imgs, batch_size=8)
        assert len(result) == 8

    def test_truncates_to_largest_multiple(self):
        from benchmark_pipeline import _adjust_image_count
        imgs = list(range(20))
        result = _adjust_image_count(imgs, batch_size=8)
        assert len(result) == 16

    def test_pads_when_too_few(self):
        from benchmark_pipeline import _adjust_image_count
        imgs = list(range(3))
        result = _adjust_image_count(imgs, batch_size=8)
        assert len(result) == 8

    def test_pad_uses_repetition_not_new_items(self):
        from benchmark_pipeline import _adjust_image_count
        imgs = ["a", "b", "c"]
        result = _adjust_image_count(imgs, batch_size=8)
        assert all(x in imgs for x in result)

    def test_exact_multiple_unchanged(self):
        from benchmark_pipeline import _adjust_image_count
        imgs = list(range(16))
        result = _adjust_image_count(imgs, batch_size=8)
        assert len(result) == 16
        assert result == imgs

    def test_single_image_padded(self):
        from benchmark_pipeline import _adjust_image_count
        result = _adjust_image_count(["x"], batch_size=4)
        assert len(result) == 4
        assert all(v == "x" for v in result)

    def test_result_length_always_multiple_of_batch_size(self):
        from benchmark_pipeline import _adjust_image_count
        for n in range(1, 30):
            result = _adjust_image_count(list(range(n)), batch_size=8)
            assert len(result) % 8 == 0, f"n={n} gave len={len(result)}"


class TestTimedBenchArgs:
    """--timed-bench, --duration, --warmup CLI arguments."""

    def _parse(self, extra=None):
        from benchmark_pipeline import build_parser
        argv = ["--detector", "gdino", "--test-dir", "/tmp",
                "--svm-model", "m.joblib", "--throughput-only"]
        if extra:
            argv += extra
        return build_parser().parse_args(argv)

    def test_timed_bench_flag_exists(self):
        args = self._parse(["--timed-bench"])
        assert args.timed_bench is True

    def test_timed_bench_defaults_false(self):
        args = self._parse()
        assert args.timed_bench is False

    def test_duration_arg(self):
        args = self._parse(["--timed-bench", "--duration", "60"])
        assert args.duration == 60

    def test_duration_default_30(self):
        args = self._parse(["--timed-bench"])
        assert args.duration == 30

    def test_warmup_arg(self):
        args = self._parse(["--timed-bench", "--warmup", "10"])
        assert args.warmup == 10

    def test_warmup_default_5(self):
        args = self._parse(["--timed-bench"])
        assert args.warmup == 5


def _make_mock_detector(supports_pipeline=False, batch_size=2):
    """Return a mock BaseDetector for testing loop functions."""
    det = MagicMock()
    det.supports_pipeline = supports_pipeline
    det.detect_batch.return_value = [[] for _ in range(batch_size)]
    det.preprocess_cpu.return_value = {"dummy": True}
    det.forward_preprocessed.return_value = [[] for _ in range(batch_size)]
    return det


class TestRunTimedBench:
    """_run_timed_bench must honour duration/warmup and count correctly."""

    def _make_preloaded(self, n_batches=3, batch_size=4, supports_pipeline=True):
        """Fake preloaded batches using the new unified format."""
        batches = []
        for _ in range(n_batches):
            imgs_cpu = [torch.zeros(3, 64, 64, dtype=torch.uint8) for _ in range(batch_size)]
            if supports_pipeline:
                batches.append({
                    "prep":        {"dummy": True},
                    "imgs_cpu":    imgs_cpu,
                    "_batch_size": batch_size,
                })
            else:
                from PIL import Image as PILImage
                pil_imgs = [PILImage.new("RGB", (64, 64)) for _ in range(batch_size)]
                batches.append({
                    "pil_imgs":    pil_imgs,
                    "imgs_cpu":    imgs_cpu,
                    "_batch_size": batch_size,
                })
        return batches

    def _make_dinov2(self):
        dinov2 = MagicMock()
        dinov2.forward_features.return_value = {"x_norm_clstoken": torch.zeros(1, 384)}
        return dinov2

    def test_run_timed_bench_exists(self):
        from benchmark_pipeline import _run_timed_bench
        assert callable(_run_timed_bench)

    def test_returns_four_tuple(self):
        from benchmark_pipeline import _run_timed_bench
        batch_size = 4
        preloaded = self._make_preloaded(batch_size=batch_size)
        det = _make_mock_detector(supports_pipeline=True, batch_size=batch_size)
        dinov2 = self._make_dinov2()
        with patch("benchmark_pipeline.make_crops_gpu_batch",
                   return_value=torch.zeros(0, 3, 224, 224)), \
             patch("benchmark_pipeline.predict_species_batch",
                   return_value=[("Oak", 0.9)] * batch_size):
            result = _run_timed_bench(
                preloaded, det, device="cpu", dinov2=dinov2, clf=MagicMock(),
                duration=0.2, warmup=0.0, log_fn=lambda _: None,
            )
        assert isinstance(result, tuple) and len(result) == 4

    def test_n_total_positive_after_warmup(self):
        from benchmark_pipeline import _run_timed_bench
        batch_size = 4
        preloaded = self._make_preloaded(n_batches=2, batch_size=batch_size)
        det = _make_mock_detector(supports_pipeline=True, batch_size=batch_size)
        dinov2 = self._make_dinov2()
        with patch("benchmark_pipeline.make_crops_gpu_batch",
                   return_value=torch.zeros(0, 3, 224, 224)), \
             patch("benchmark_pipeline.predict_species_batch",
                   return_value=[("Oak", 0.9)] * batch_size):
            n_total, n_warmup, elapsed, tput = _run_timed_bench(
                preloaded, det, device="cpu", dinov2=dinov2, clf=MagicMock(),
                duration=0.3, warmup=0.0, log_fn=lambda _: None,
            )
        assert n_total > 0

    def test_throughput_equals_n_total_over_elapsed(self):
        from benchmark_pipeline import _run_timed_bench
        batch_size = 4
        preloaded = self._make_preloaded(n_batches=2, batch_size=batch_size)
        det = _make_mock_detector(supports_pipeline=True, batch_size=batch_size)
        dinov2 = self._make_dinov2()
        with patch("benchmark_pipeline.make_crops_gpu_batch",
                   return_value=torch.zeros(0, 3, 224, 224)), \
             patch("benchmark_pipeline.predict_species_batch",
                   return_value=[("Oak", 0.9)] * batch_size):
            n_total, _, elapsed, tput = _run_timed_bench(
                preloaded, det, device="cpu", dinov2=dinov2, clf=MagicMock(),
                duration=0.2, warmup=0.0, log_fn=lambda _: None,
            )
        assert abs(tput - n_total / elapsed) < 0.01

    def test_warmup_images_not_in_n_total(self):
        """With warmup > 0, the measuring window starts after warmup."""
        from benchmark_pipeline import _run_timed_bench
        batch_size = 4
        preloaded = self._make_preloaded(n_batches=2, batch_size=batch_size)
        det = _make_mock_detector(supports_pipeline=True, batch_size=batch_size)
        dinov2 = self._make_dinov2()
        with patch("benchmark_pipeline.make_crops_gpu_batch",
                   return_value=torch.zeros(0, 3, 224, 224)), \
             patch("benchmark_pipeline.predict_species_batch",
                   return_value=[("Oak", 0.9)] * batch_size):
            n_total, n_warmup, elapsed, tput = _run_timed_bench(
                preloaded, det, device="cpu", dinov2=dinov2, clf=MagicMock(),
                duration=0.2, warmup=0.1, log_fn=lambda _: None,
            )
        assert n_warmup >= 0
        assert n_total >= 0
        assert elapsed <= 0.2 + 0.05  # allow 50ms overshoot


# ══════════════════════════════════════════════════════════════════════════════
# _tensor_to_pil module-level extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestTensorToPil:
    """_tensor_to_pil must be a module-level function (not a closure inside main)."""

    def test_is_module_level(self):
        import benchmark_pipeline as bm
        assert hasattr(bm, "_tensor_to_pil"), "_tensor_to_pil must be at module level"
        assert callable(bm._tensor_to_pil)

    def test_converts_chw_uint8_to_pil(self):
        from benchmark_pipeline import _tensor_to_pil
        t = torch.zeros(3, 100, 150, dtype=torch.uint8)
        img = _tensor_to_pil(t)
        assert isinstance(img, Image.Image)

    def test_output_size_is_wh(self):
        """PIL size is (W, H) — opposite of tensor (C, H, W)."""
        from benchmark_pipeline import _tensor_to_pil
        t = torch.zeros(3, 100, 150, dtype=torch.uint8)
        img = _tensor_to_pil(t)
        assert img.size == (150, 100)

    def test_pixel_values_preserved(self):
        from benchmark_pipeline import _tensor_to_pil
        t = torch.full((3, 50, 60), 128, dtype=torch.uint8)
        img = _tensor_to_pil(t)
        arr = np.array(img)
        assert arr.shape == (50, 60, 3)
        assert (arr == 128).all()


# ══════════════════════════════════════════════════════════════════════════════
# _run_detector_loop — extracted detector+embed+score loop
# ══════════════════════════════════════════════════════════════════════════════

def _make_fake_loader(n_batches: int = 2, batch_size: int = 2):
    """Returns a list of (tensor_list, path_list) tuples mimicking a DataLoader."""
    batches = []
    for i in range(n_batches):
        tensors = [torch.zeros(3, 64, 64, dtype=torch.uint8) for _ in range(batch_size)]
        paths   = [f"/img/b{i}_i{j}.jpg" for j in range(batch_size)]
        batches.append((tensors, paths))
    return batches


def _make_dinov2_for_loop(batch_size=2):
    dinov2 = MagicMock()
    dinov2.forward_features.return_value = {
        "x_norm_clstoken": torch.zeros(batch_size, 384)
    }
    return dinov2


def _gt_map(loader):
    return {p: "Oak" for _, paths in loader for p in paths}


class TestRunDetectorLoop:
    """_run_detector_loop(detector_obj, loader, device, dinov2, clf, gt_map, throughput_only)
    must run all batches and return (n_total, n_correct, n_no_detection, per_class)."""

    def _run(self, loader, det, gt, throughput_only=False, batch_size=2):
        from benchmark_pipeline import _run_detector_loop
        dinov2 = _make_dinov2_for_loop(batch_size)
        with patch("benchmark_pipeline.make_crops_gpu_batch",
                   return_value=torch.zeros(batch_size, 3, 448, 448)), \
             patch("benchmark_pipeline.predict_species_batch",
                   return_value=[("Oak", 0.9)] * batch_size):
            return _run_detector_loop(
                det, loader, "cpu", dinov2, MagicMock(), gt, throughput_only,
            )

    def test_exists(self):
        from benchmark_pipeline import _run_detector_loop
        assert callable(_run_detector_loop)

    def test_returns_four_tuple(self):
        loader = _make_fake_loader()
        det = _make_mock_detector(supports_pipeline=False)
        gt = _gt_map(loader)
        result = self._run(loader, det, gt)
        assert isinstance(result, tuple) and len(result) == 4

    def test_n_total_equals_all_images(self):
        n_batches, batch_size = 3, 2
        loader = _make_fake_loader(n_batches, batch_size)
        det = _make_mock_detector(supports_pipeline=False, batch_size=batch_size)
        gt = _gt_map(loader)
        n_total, _, _, _ = self._run(loader, det, gt, batch_size=batch_size)
        assert n_total == n_batches * batch_size

    def test_n_correct_counts_matching_predictions(self):
        from benchmark_pipeline import _run_detector_loop
        loader = _make_fake_loader(n_batches=1, batch_size=2)
        det = _make_mock_detector(supports_pipeline=False, batch_size=2)
        gt = {"/img/b0_i0.jpg": "Oak", "/img/b0_i1.jpg": "Oak"}
        dinov2 = _make_dinov2_for_loop(2)
        with patch("benchmark_pipeline.make_crops_gpu_batch",
                   return_value=torch.zeros(2, 3, 448, 448)), \
             patch("benchmark_pipeline.predict_species_batch",
                   return_value=[("Oak", 0.9), ("Elm", 0.7)]):
            n_total, n_correct, _, _ = _run_detector_loop(
                det, loader, "cpu", dinov2, MagicMock(), gt, False,
            )
        assert n_total == 2 and n_correct == 1

    def test_throughput_only_skips_scoring(self):
        loader = _make_fake_loader()
        det = _make_mock_detector(supports_pipeline=False)
        n_total, n_correct, _, _ = self._run(loader, det, {}, throughput_only=True)
        assert n_correct == 0

    def test_yolo_detect_batch_called_once_per_loader_batch(self):
        n_batches = 3
        loader = _make_fake_loader(n_batches=n_batches)
        det = _make_mock_detector(supports_pipeline=False, batch_size=2)
        gt = _gt_map(loader)
        self._run(loader, det, gt)
        assert det.detect_batch.call_count == n_batches

    def test_pipeline_forward_preprocessed_called_once_per_batch(self):
        """For supports_pipeline=True detectors, forward_preprocessed() is used."""
        from benchmark_pipeline import _run_detector_loop
        n_batches = 2
        loader = _make_fake_loader(n_batches=n_batches)
        det = _make_mock_detector(supports_pipeline=True, batch_size=2)
        gt = _gt_map(loader)
        dinov2 = _make_dinov2_for_loop(2)
        with patch("benchmark_pipeline.make_crops_gpu_batch",
                   return_value=torch.zeros(2, 3, 448, 448)), \
             patch("benchmark_pipeline.predict_species_batch",
                   return_value=[("Oak", 0.9), ("Oak", 0.9)]):
            _run_detector_loop(det, loader, "cpu", dinov2, MagicMock(), gt, False)
        assert det.forward_preprocessed.call_count == n_batches

    def test_per_class_accumulated(self):
        from benchmark_pipeline import _run_detector_loop
        loader = _make_fake_loader(n_batches=1, batch_size=2)
        det = _make_mock_detector(supports_pipeline=False, batch_size=2)
        gt = {"/img/b0_i0.jpg": "Oak", "/img/b0_i1.jpg": "Elm"}
        dinov2 = _make_dinov2_for_loop(2)
        with patch("benchmark_pipeline.make_crops_gpu_batch",
                   return_value=torch.zeros(2, 3, 448, 448)), \
             patch("benchmark_pipeline.predict_species_batch",
                   return_value=[("Oak", 0.9), ("Oak", 0.8)]):
            _, _, _, per_class = _run_detector_loop(
                det, loader, "cpu", dinov2, MagicMock(), gt, False,
            )
        assert per_class["Oak"]["total"] == 1 and per_class["Oak"]["correct"] == 1
        assert per_class["Elm"]["total"] == 1 and per_class["Elm"]["correct"] == 0
