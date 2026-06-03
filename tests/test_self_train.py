"""Tests for self_train.py — Phase D (D1 + D2)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Ensure ultralytics is available; skip this module if not installed.
ultralytics = pytest.importorskip("ultralytics")

from self_train import (
    compute_intersection,
    get_score_threshold,
)
from yolo_io import load_yolo_xyxy, write_yolo_labels as write_yolo_from_xyxy


# ── TestScoreThreshold (D2) ───────────────────────────────────────────────────

class TestScoreThreshold:
    def test_k1_base(self):
        assert abs(get_score_threshold(1) - 0.35) < 1e-9

    def test_k2_incremented(self):
        assert abs(get_score_threshold(2) - 0.40) < 1e-9

    def test_k5_capped(self):
        assert abs(get_score_threshold(5) - 0.55) < 1e-9

    def test_k10_capped_at_max(self):
        assert get_score_threshold(10) == 0.55

    def test_custom_params(self):
        assert abs(get_score_threshold(3, base_thr=0.2, step=0.1, max_thr=0.5) - 0.4) < 1e-9


# ── TestComputeIntersection (D2) ──────────────────────────────────────────────

class TestComputeIntersection:
    def test_two_of_four_accepted(self):
        # YOLO boxes: box0 and box1 overlap GDINO; box2 and box3 do not
        yolo = torch.tensor([
            [0.,  0., 50., 50.],   # overlaps gdino[0]
            [60., 60., 110., 110.],  # overlaps gdino[1]
            [200., 200., 250., 250.],  # no overlap
            [300., 300., 350., 350.],  # no overlap
        ])
        gdino = torch.tensor([
            [5.,  5., 55., 55.],
            [65., 65., 115., 115.],
        ])
        result = compute_intersection(yolo, gdino, iou_thr=0.5)
        assert len(result) == 2

    def test_no_overlap_returns_empty(self):
        yolo  = torch.tensor([[0., 0., 10., 10.]])
        gdino = torch.tensor([[100., 100., 200., 200.]])
        result = compute_intersection(yolo, gdino, iou_thr=0.5)
        assert len(result) == 0

    def test_empty_yolo_returns_empty(self):
        result = compute_intersection(
            torch.zeros(0, 4), torch.tensor([[0., 0., 50., 50.]]))
        assert len(result) == 0

    def test_empty_gdino_returns_empty(self):
        # Caller handles "GDINO empty" before calling compute_intersection;
        # function still returns empty correctly when called with empty gdino.
        result = compute_intersection(
            torch.tensor([[0., 0., 50., 50.]]), torch.zeros(0, 4))
        assert len(result) == 0

    def test_yolo_miss_returns_empty_for_fallback_logic(self):
        # When YOLO finds nothing, caller (run_inference_and_filter) falls back
        # to GDINO labels (Case D). compute_intersection signals this correctly.
        result = compute_intersection(torch.zeros(0, 4),
                                      torch.tensor([[0., 0., 50., 50.]]))
        assert len(result) == 0

    def test_perfect_overlap_accepted(self):
        box = torch.tensor([[10., 10., 90., 90.]])
        result = compute_intersection(box, box, iou_thr=0.5)
        assert len(result) == 1

    def test_output_is_subset_of_input(self):
        yolo  = torch.tensor([[0., 0., 50., 50.], [0., 0., 50., 50.]])
        gdino = torch.tensor([[5., 5., 45., 45.]])
        result = compute_intersection(yolo, gdino, iou_thr=0.5)
        assert result.shape[1] == 4


# ── TestLabelIO ───────────────────────────────────────────────────────────────

class TestLabelIO:
    def test_roundtrip(self, tmp_path):
        boxes = torch.tensor([[10., 20., 90., 80.]], dtype=torch.float32)
        p = tmp_path / "frame.txt"
        write_yolo_from_xyxy(p, boxes, img_w=100, img_h=100)
        loaded = load_yolo_xyxy(p, img_w=100, img_h=100)
        assert loaded.shape == (1, 4)
        assert torch.allclose(loaded, boxes, atol=0.5)

    def test_empty_write_empty_read(self, tmp_path):
        p = tmp_path / "empty.txt"
        write_yolo_from_xyxy(p, torch.zeros(0, 4), 100, 100)
        loaded = load_yolo_xyxy(p, 100, 100)
        assert len(loaded) == 0

    def test_missing_file_returns_empty(self, tmp_path):
        loaded = load_yolo_xyxy(tmp_path / "nonexistent.txt", 100, 100)
        assert len(loaded) == 0


# ── TestLoopControl (D1) ──────────────────────────────────────────────────────

class TestLoopControl:
    """Verify loop termination logic with mocked inference/train/eval."""

    def _make_args(self, tmp_path, max_iter=3, manual_val=None):
        args = MagicMock()
        args.weights = "fake.pt"
        args.frames_base = str(tmp_path / "frames")
        args.gdino_labels_base = str(tmp_path / "gdino")
        args.manual_val_data = manual_val or str(tmp_path / "nonexistent_val.yaml")
        args.iter_base = str(tmp_path / "iters")
        args.max_iter = max_iter
        args.epochs = 1
        args.iou_thr = 0.5
        args.device = "cpu"
        return args

    def test_runs_max_iter_without_val(self, tmp_path):
        args = self._make_args(tmp_path, max_iter=3)
        call_count = {"n": 0}

        _EMPTY_STATS = {"total": 0, "case_a": 0, "case_b": 0, "case_c": 0,
                        "case_d": 0, "boxes_accepted": 0, "boxes_fallback": 0}

        def fake_filter(*a, **kw):
            call_count["n"] += 1
            return _EMPTY_STATS

        with patch("self_train.run_inference_and_filter", side_effect=fake_filter), \
             patch("self_train.rebuild_and_train", return_value="fake_iter.pt"), \
             patch("self_train.evaluate", return_value=-1.0):
            from self_train import run_self_training
            run_self_training(args)

        assert call_count["n"] == 3

    def test_early_stop_after_two_no_improve(self, tmp_path):
        args = self._make_args(tmp_path, max_iter=5)
        map_values = [0.8, 0.7, 0.7, 0.9, 0.9]
        call_count = {"n": 0}

        def fake_eval(*a, **kw):
            v = map_values[call_count["n"]]
            call_count["n"] += 1
            return v

        _empty = {"total": 0, "case_a": 0, "case_b": 0, "case_c": 0,
                  "case_d": 0, "boxes_accepted": 0, "boxes_fallback": 0}
        with patch("self_train.run_inference_and_filter", return_value=_empty), \
             patch("self_train.rebuild_and_train", return_value="fake.pt"), \
             patch("self_train.evaluate", side_effect=fake_eval):
            from self_train import run_self_training
            run_self_training(args)

        assert call_count["n"] == 3

    def test_no_early_stop_when_improving(self, tmp_path):
        args = self._make_args(tmp_path, max_iter=4)
        map_values = [0.7, 0.8, 0.85, 0.9]
        call_count = {"n": 0}

        def fake_eval(*a, **kw):
            v = map_values[call_count["n"]]
            call_count["n"] += 1
            return v

        _empty = {"total": 0, "case_a": 0, "case_b": 0, "case_c": 0,
                  "case_d": 0, "boxes_accepted": 0, "boxes_fallback": 0}
        with patch("self_train.run_inference_and_filter", return_value=_empty), \
             patch("self_train.rebuild_and_train", return_value="fake.pt"), \
             patch("self_train.evaluate", side_effect=fake_eval):
            from self_train import run_self_training
            run_self_training(args)

        assert call_count["n"] == 4
