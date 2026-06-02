"""
tests/test_benchmark_yolo.py

TDD tests for YOLO detector support in benchmark_pipeline.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
import pytest

from benchmark_pipeline import build_arg_parser, read_yolo_speed_s


class TestArgParserYolo:
    """Tests for build_arg_parser() — YOLO-related arguments."""

    def test_default_detector(self):
        """Default detector should be 'gdino'."""
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.detector == "gdino"

    def test_yolo_detector(self):
        """--detector yolo should set args.detector to 'yolo'."""
        parser = build_arg_parser()
        args = parser.parse_args(["--detector", "yolo"])
        assert args.detector == "yolo"

    def test_yolo_model_default(self):
        """Default yolo_model should contain 'tree_yolo26s_halfres'."""
        parser = build_arg_parser()
        args = parser.parse_args(["--detector", "yolo"])
        assert "tree_yolo26s_halfres" in args.yolo_model

    def test_yolo_imgsz_default(self):
        """Default yolo_imgsz should be 1280."""
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.yolo_imgsz == 1280

    def test_yolo_imgsz_override(self):
        """--yolo-imgsz 640 should set args.yolo_imgsz to 640."""
        parser = build_arg_parser()
        args = parser.parse_args(["--yolo-imgsz", "640"])
        assert args.yolo_imgsz == 640


class TestReadYoloSpeedS:
    """Tests for read_yolo_speed_s(result)."""

    def test_speed_extracted_and_converted(self):
        """Should return (pre_s, inf_s, post_s) converted from ms to seconds."""
        mock_result = MagicMock()
        mock_result.speed = {"preprocess": 10.0, "inference": 50.0, "postprocess": 2.0}
        pre, inf, post = read_yolo_speed_s(mock_result)
        assert pytest.approx(pre)  == 0.010
        assert pytest.approx(inf)  == 0.050
        assert pytest.approx(post) == 0.002
