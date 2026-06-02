"""Tests for yolo_io — unified YOLO label I/O module."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from yolo_io import load_yolo_xyxy, write_yolo_labels, xyxy_to_yolo


class TestRoundtrip:
    def test_roundtrip_single_box(self, tmp_path):
        boxes = torch.tensor([[10., 20., 90., 80.]], dtype=torch.float32)
        p = tmp_path / "frame.txt"
        write_yolo_labels(p, boxes, img_w=100, img_h=100)
        loaded = load_yolo_xyxy(p, img_w=100, img_h=100)
        assert loaded.shape == (1, 4)
        assert torch.allclose(loaded, boxes, atol=0.5)

    def test_roundtrip_multiple_boxes(self, tmp_path):
        boxes = torch.tensor([
            [0., 0., 50., 50.],
            [50., 50., 100., 100.],
            [10., 30., 70., 80.],
        ], dtype=torch.float32)
        p = tmp_path / "multi.txt"
        write_yolo_labels(p, boxes, img_w=100, img_h=100)
        loaded = load_yolo_xyxy(p, img_w=100, img_h=100)
        assert loaded.shape == (3, 4)
        assert torch.allclose(loaded, boxes, atol=0.5)


class TestEmptyHandling:
    def test_empty_tensor_writes_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        write_yolo_labels(p, torch.zeros(0, 4), img_w=100, img_h=100)
        assert p.exists()
        # File should be empty (0 bytes) or only whitespace/newlines
        content = p.read_text().strip()
        assert content == ""

    def test_empty_file_reads_empty_tensor(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        loaded = load_yolo_xyxy(p, img_w=100, img_h=100)
        assert loaded.shape == (0, 4)

    def test_missing_file_returns_empty_tensor(self, tmp_path):
        loaded = load_yolo_xyxy(tmp_path / "nonexistent.txt", img_w=100, img_h=100)
        assert loaded.shape == (0, 4)


class TestXyxyToYolo:
    def test_full_image_box(self):
        """A box covering the entire image should give cx=0.5, cy=0.5, w=1.0, h=1.0."""
        cx, cy, w, h = xyxy_to_yolo(0, 0, 100, 100, img_w=100, img_h=100)
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6

    def test_half_image_box(self):
        cx, cy, w, h = xyxy_to_yolo(0, 0, 50, 100, img_w=100, img_h=100)
        assert abs(cx - 0.25) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        assert abs(w - 0.5) < 1e-6
        assert abs(h - 1.0) < 1e-6


class TestFiveColumnFormat:
    def test_each_line_has_five_fields(self, tmp_path):
        boxes = torch.tensor([
            [10., 20., 90., 80.],
            [5., 5., 50., 50.],
        ], dtype=torch.float32)
        p = tmp_path / "labels.txt"
        write_yolo_labels(p, boxes, img_w=100, img_h=100)
        lines = [l for l in p.read_text().strip().splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            parts = line.split()
            assert len(parts) == 5, f"Expected 5 fields, got: {line!r}"

    def test_class_id_is_zero(self, tmp_path):
        boxes = torch.tensor([[10., 20., 90., 80.]], dtype=torch.float32)
        p = tmp_path / "labels.txt"
        write_yolo_labels(p, boxes, img_w=100, img_h=100)
        line = p.read_text().strip().splitlines()[0]
        class_id = int(line.split()[0])
        assert class_id == 0
