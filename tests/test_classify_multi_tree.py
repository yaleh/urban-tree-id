"""
Tests for classify_multi_tree.py — TDD suite for YOLO + DINOv2 + SVM pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from unittest.mock import MagicMock

from classify_multi_tree import build_arg_parser, yolo_boxes_from_results


# ── TestArgParser ──────────────────────────────────────────────────────────────

class TestArgParser:
    def test_defaults(self):
        args = build_arg_parser().parse_args([])
        assert args.detector == "gdino"
        assert args.yolo_conf == 0.25
        assert args.yolo_imgsz == 1280
        assert "tree_yolo26s_halfres" in args.yolo_model

    def test_yolo_detector(self):
        args = build_arg_parser().parse_args(["--detector", "yolo"])
        assert args.detector == "yolo"

    def test_yolo_model_override(self):
        args = build_arg_parser().parse_args(["--yolo-model", "custom.pt"])
        assert args.yolo_model == "custom.pt"

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
