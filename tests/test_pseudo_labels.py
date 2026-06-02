"""Tests for generate_pseudo_labels.py and split_dataset.py (Stages B1-B3)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from generate_pseudo_labels import filter_boxes, is_too_dark
from yolo_io import xyxy_to_yolo, write_yolo_labels as write_label_file
from split_dataset import split_dataset, write_data_yaml, VIDEO_SPLITS


# ── TestInference (B1) ────────────────────────────────────────────────────────

class TestInference:
    def test_brightness_filter_dark(self):
        dark = Image.fromarray(np.full((100, 100, 3), 20, dtype=np.uint8))
        assert is_too_dark(dark, thr=50.0)

    def test_brightness_filter_bright(self):
        bright = Image.fromarray(np.full((100, 100, 3), 100, dtype=np.uint8))
        assert not is_too_dark(bright, thr=50.0)


# ── TestFilter (B2) ───────────────────────────────────────────────────────────

class TestFilter:
    def test_high_overlap_boxes_reduced(self):
        # Two boxes with IoU ≈ 0.8 → only one kept
        boxes = torch.tensor([[10., 10., 90., 90.],
                               [15., 15., 95., 95.]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8])
        kept_boxes, kept_scores = filter_boxes(boxes, scores, score_thr=0.35, iou_thr=0.45)
        assert len(kept_boxes) == 1

    def test_low_score_filtered(self):
        boxes = torch.tensor([[0., 0., 50., 50.],
                               [60., 60., 110., 110.]], dtype=torch.float32)
        scores = torch.tensor([0.2, 0.9])
        kept_boxes, _ = filter_boxes(boxes, scores, score_thr=0.35, iou_thr=0.45)
        assert len(kept_boxes) == 1  # only the 0.9-score box

    def test_empty_input(self):
        boxes = torch.zeros(0, 4)
        scores = torch.zeros(0)
        kb, ks = filter_boxes(boxes, scores)
        assert len(kb) == 0


# ── TestYoloFormat (B2) ───────────────────────────────────────────────────────

class TestYoloFormat:
    def test_center_box(self):
        # Box covers entire 100×100 image
        cx, cy, w, h = xyxy_to_yolo(0, 0, 100, 100, 100, 100)
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        assert abs(w  - 1.0) < 1e-6
        assert abs(h  - 1.0) < 1e-6

    def test_values_in_unit_range(self):
        cx, cy, w, h = xyxy_to_yolo(10, 20, 60, 80, 100, 100)
        for v in [cx, cy, w, h]:
            assert 0.0 <= v <= 1.0, f"Out of range: {v}"

    def test_known_values(self):
        cx, cy, w, h = xyxy_to_yolo(0, 0, 50, 50, 100, 100)
        assert abs(cx - 0.25) < 1e-6
        assert abs(cy - 0.25) < 1e-6
        assert abs(w  - 0.5)  < 1e-6
        assert abs(h  - 0.5)  < 1e-6

    def test_write_label_file_content(self, tmp_path):
        boxes = torch.tensor([[0., 0., 100., 100.]], dtype=torch.float32)
        path = tmp_path / "frame_000001.txt"
        write_label_file(path, boxes, img_w=100, img_h=100)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0"        # class id
        assert len(parts) == 5

    def test_write_label_file_empty(self, tmp_path):
        path = tmp_path / "empty.txt"
        write_label_file(path, torch.zeros(0, 4), img_w=100, img_h=100)
        assert path.exists()
        assert path.read_text() == ""

    def test_write_label_file_multi(self, tmp_path):
        boxes = torch.tensor([[0., 0., 50., 50.],
                               [50., 50., 100., 100.]], dtype=torch.float32)
        path = tmp_path / "multi.txt"
        write_label_file(path, boxes, img_w=100, img_h=100)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2


# ── TestSplit (B3) ────────────────────────────────────────────────────────────

def _make_pseudo_label_dir(base: Path, video_name: str, n_frames: int) -> None:
    img_dir = base / video_name / "images" / "all"
    lbl_dir = base / video_name / "labels" / "all"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(1, n_frames + 1):
        (img_dir / f"frame_{i:06d}.jpg").write_bytes(b"JPEG")
        (lbl_dir / f"frame_{i:06d}.txt").write_text(f"0 0.5 0.5 0.1 0.1\n")


class TestSplit:
    def test_cross_date_split(self, tmp_path):
        base = tmp_path / "pseudo"
        # 5 frames per video
        for v in ["eastbound_20240319", "eastbound_20240530",
                  "westbound_20240319", "westbound_20240530"]:
            _make_pseudo_label_dir(base, v, 5)

        out = tmp_path / "all_4videos"
        stats = split_dataset(base, out)

        # train: 2 videos × 5 = 10; val: 2 videos × 5 = 10
        assert stats["train"] == 10
        assert stats["val"]   == 10

    def test_train_only_has_319_videos(self, tmp_path):
        base = tmp_path / "pseudo"
        for v in ["eastbound_20240319", "eastbound_20240530",
                  "westbound_20240319", "westbound_20240530"]:
            _make_pseudo_label_dir(base, v, 3)

        out = tmp_path / "all_4videos"
        split_dataset(base, out)

        train_files = [f.name for f in (out / "images" / "train").glob("*.jpg")]
        for f in train_files:
            assert "20240319" in f, f"Non-train video in train: {f}"

    def test_no_naming_collision(self, tmp_path):
        base = tmp_path / "pseudo"
        for v in ["eastbound_20240319", "eastbound_20240530",
                  "westbound_20240319", "westbound_20240530"]:
            _make_pseudo_label_dir(base, v, 3)

        out = tmp_path / "all_4videos"
        split_dataset(base, out)

        # frame_000001.jpg exists in both eastbound and westbound → must be renamed
        train_files = [f.name for f in (out / "images" / "train").glob("*.jpg")]
        assert len(train_files) == len(set(train_files))  # no duplicates

    def test_data_yaml_absolute_path(self, tmp_path):
        base = tmp_path / "pseudo"
        for v in ["eastbound_20240319", "eastbound_20240530",
                  "westbound_20240319", "westbound_20240530"]:
            _make_pseudo_label_dir(base, v, 2)

        out = tmp_path / "all_4videos"
        split_dataset(base, out)
        yaml_path = write_data_yaml(out)

        data = yaml.safe_load(yaml_path.read_text())
        assert Path(data["path"]).is_absolute(), "path must be absolute"
        assert data["nc"] == 1
        assert data["names"] == ["tree"]
