"""
Tests for classify_multi_tree.py — TDD suite for YOLO + DINOv2 + SVM pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock, patch

from classify_multi_tree import build_arg_parser, yolo_boxes_from_results, classify_detections


# ── TestArgParser ──────────────────────────────────────────────────────────────

class TestArgParser:
    def test_defaults(self):
        args = build_arg_parser().parse_args([])
        assert args.detector == "gdino"
        assert args.yolo_conf == 0.25
        assert args.yolo_imgsz == 1280
        assert "tree_yolo26s_halfres" in args.yolo_checkpoint

    def test_yolo_detector(self):
        args = build_arg_parser().parse_args(["--detector", "yolo"])
        assert args.detector == "yolo"

    def test_yolo_model_override(self):
        # --yolo-model is a deprecated alias for --yolo-checkpoint
        args = build_arg_parser().parse_args(["--yolo-model", "custom.pt"])
        assert args.yolo_checkpoint == "custom.pt"

    def test_yolo_checkpoint_new_name(self):
        # --yolo-checkpoint is the canonical new name
        args = build_arg_parser().parse_args(["--yolo-checkpoint", "new.pt"])
        assert args.yolo_checkpoint == "new.pt"

    def test_yolo_conf_override(self):
        args = build_arg_parser().parse_args(["--yolo-conf", "0.5"])
        assert args.yolo_conf == 0.5


# ── TestYoloBoxesFromResults ───────────────────────────────────────────────────

def _make_result(xyxyn_list, conf_list):
    """Build a mock ultralytics Results object with the given boxes."""
    result = MagicMock()
    if xyxyn_list is None:
        result.boxes = None
        return result

    mock_boxes = MagicMock()
    n = len(xyxyn_list)
    mock_boxes.__len__ = lambda self: n

    if n == 0:
        mock_boxes.xyxyn = torch.zeros(0, 4)
        mock_boxes.conf = torch.zeros(0)
    else:
        mock_boxes.xyxyn = torch.tensor(xyxyn_list, dtype=torch.float32)
        mock_boxes.conf = torch.tensor(conf_list, dtype=torch.float32)

    result.boxes = mock_boxes
    return result


class TestYoloBoxesFromResults:
    def test_no_boxes_none(self):
        result = _make_result(None, None)
        assert yolo_boxes_from_results(result) == []

    def test_no_boxes_empty(self):
        result = _make_result([], [])
        assert yolo_boxes_from_results(result) == []

    def test_single_box(self):
        result = _make_result([[0.1, 0.2, 0.8, 0.9]], [0.85])
        out = yolo_boxes_from_results(result)
        assert len(out) == 1
        conf, box = out[0]
        assert abs(conf - 0.85) < 1e-5
        assert len(box) == 4
        assert abs(box[0] - 0.1) < 1e-5
        assert abs(box[1] - 0.2) < 1e-5
        assert abs(box[2] - 0.8) < 1e-5
        assert abs(box[3] - 0.9) < 1e-5

    def test_multiple_boxes(self):
        result = _make_result(
            [[0.1, 0.2, 0.8, 0.9], [0.05, 0.05, 0.4, 0.5]],
            [0.9, 0.7],
        )
        out = yolo_boxes_from_results(result)
        assert len(out) == 2
        confs = [c for c, _ in out]
        assert abs(confs[0] - 0.9) < 1e-5
        assert abs(confs[1] - 0.7) < 1e-5

    def test_tiny_box_filtered(self):
        # Width = 0.003 (< 0.005 threshold) → should be filtered out
        result = _make_result([[0.1, 0.2, 0.103, 0.9]], [0.95])
        out = yolo_boxes_from_results(result)
        assert out == []


# ── TestClassifyDetections ─────────────────────────────────────────────────────

def _make_mock_dinov2():
    """Return a mock DINOv2 model that yields 384-dim zero embeddings (batch-aware)."""
    dinov2 = MagicMock()

    def _forward_features(batch_tensor):
        n = batch_tensor.shape[0]
        return {"x_norm_clstoken": torch.zeros(n, 384)}

    dinov2.forward_features.side_effect = _forward_features
    return dinov2


def _make_mock_clf(species="Quercus_robur"):
    """Return a mock sklearn SVM classifier (batch-aware)."""
    clf = MagicMock()

    def _predict(feats):
        n = feats.shape[0]
        return np.array([species] * n)

    def _decision_function(feats):
        n = feats.shape[0]
        return np.tile([1.0, 0.5, 0.2], (n, 1))

    clf.predict.side_effect = _predict
    clf.decision_function.side_effect = _decision_function
    return clf


class TestClassifyDetections:
    """Unit tests for classify_detections() — xyxy pixel → box_norm conversion."""

    def _run(self, img_size, boxes_xyxy):
        """Helper: run classify_detections with mocked models, return results."""
        W, H = img_size
        pil_img = Image.new("RGB", (W, H), color=(128, 128, 128))
        dinov2 = _make_mock_dinov2()
        clf = _make_mock_clf()
        device = "cpu"
        return classify_detections(pil_img, boxes_xyxy, dinov2, clf, device)

    def test_box_norm_values_correct(self):
        # img 200×150, box=[20, 30, 100, 90]
        # expected box_norm = [20/200, 30/150, 100/200, 90/150] = [0.1, 0.2, 0.5, 0.6]
        results = self._run((200, 150), [[20, 30, 100, 90]])
        assert len(results) == 1
        bn = results[0]["box_norm"]
        assert abs(bn[0] - 0.1) < 1e-6, f"x0_norm expected 0.1, got {bn[0]}"
        assert abs(bn[1] - 0.2) < 1e-6, f"y0_norm expected 0.2, got {bn[1]}"
        assert abs(bn[2] - 0.5) < 1e-6, f"x1_norm expected 0.5, got {bn[2]}"
        assert abs(bn[3] - 0.6) < 1e-6, f"y1_norm expected 0.6, got {bn[3]}"

    def test_box_norm_all_values_in_unit_range(self):
        # Use a box that doesn't perfectly hit round fractions
        results = self._run((320, 240), [[10, 15, 310, 230]])
        assert len(results) == 1
        for v in results[0]["box_norm"]:
            assert 0.0 <= v <= 1.0, f"box_norm value out of [0,1]: {v}"

    def test_empty_boxes_returns_empty_list(self):
        results = self._run((200, 150), [])
        assert results == []

    def test_result_dict_has_required_keys(self):
        results = self._run((200, 150), [[20, 30, 100, 90]])
        assert len(results) == 1
        d = results[0]
        assert "box_norm" in d
        assert "species" in d
        assert "svm_conf" in d

    def test_multiple_boxes(self):
        boxes = [[10, 10, 80, 80], [120, 50, 190, 130]]
        results = self._run((200, 200), boxes)
        assert len(results) == 2
        for r in results:
            for v in r["box_norm"]:
                assert 0.0 <= v <= 1.0
