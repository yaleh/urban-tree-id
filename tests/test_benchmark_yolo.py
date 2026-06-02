"""
tests/test_benchmark_yolo.py

TDD tests for YOLO detector support in benchmark_pipeline.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
import pytest

from benchmark_pipeline import build_parser


_REQUIRED = ["--test-dir", ".", "--svm-model", "x.joblib"]


class TestArgParserYolo:
    """Tests for build_parser() — YOLO-related arguments."""

    def test_default_detector(self):
        """Default detector should be 'gdino'."""
        parser = build_parser()
        args = parser.parse_args(_REQUIRED)
        assert args.detector == "gdino"

    def test_yolo_detector(self):
        """--detector yolo should set args.detector to 'yolo'."""
        parser = build_parser()
        args = parser.parse_args(_REQUIRED + ["--detector", "yolo"])
        assert args.detector == "yolo"

    def test_yolo_imgsz_default(self):
        """Default yolo_imgsz should be None (inherits --max-edge at runtime)."""
        parser = build_parser()
        args = parser.parse_args(_REQUIRED)
        assert args.yolo_imgsz is None

    def test_yolo_imgsz_override(self):
        """--yolo-imgsz 640 should set args.yolo_imgsz to 640."""
        parser = build_parser()
        args = parser.parse_args(_REQUIRED + ["--yolo-imgsz", "640"])
        assert args.yolo_imgsz == 640
