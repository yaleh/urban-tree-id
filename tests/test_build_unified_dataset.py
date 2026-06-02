"""
tests/test_build_unified_dataset.py

TDD tests for build_unified_dataset.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from pathlib import Path

import pytest
import yaml

from build_unified_dataset import build_unified_dataset


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_road_dataset(base: Path, n_train: int = 5, n_val: int = 3) -> Path:
    """Create a minimal road-video YOLO dataset layout."""
    root = base / "road"
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n):
            (img_dir / f"frame_{i:06d}.jpg").write_bytes(b"JPEG")
            (lbl_dir / f"frame_{i:06d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    yaml_data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["tree"],
    }
    (root / "data.yaml").write_text(yaml.dump(yaml_data))
    return root


def _make_tdus_dataset(base: Path, n_train: int = 4, n_val: int = 2,
                       failures: list[str] | None = None) -> Path:
    """Create a minimal TDUS pre-scaled dataset layout."""
    root = base / "tdus"
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = root / split / "img"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n):
            (img_dir / f"Species_tree_{i}.jpg").write_bytes(b"JPEG")
            (lbl_dir / f"Species_tree_{i}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    if failures:
        fail_file = root / "train" / "labels" / "detection_failures.txt"
        fail_file.write_text("\n".join(failures) + "\n")
    return root


# ── tests ────────────────────────────────────────────────────────────────────

class TestBuildUnifiedDataset:
    def test_data_yaml_has_correct_fields(self, tmp_path):
        road = _make_road_dataset(tmp_path)
        tdus = _make_tdus_dataset(tmp_path)
        out  = tmp_path / "unified"

        build_unified_dataset(road, tdus, out)

        yaml_path = out / "data.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        assert data["nc"] == 1
        assert data["names"] == ["tree"]
        assert Path(data["path"]).is_absolute()

    def test_train_count_equals_road_plus_tdus(self, tmp_path):
        road = _make_road_dataset(tmp_path, n_train=5, n_val=3)
        tdus = _make_tdus_dataset(tmp_path, n_train=4, n_val=2)
        out  = tmp_path / "unified"

        build_unified_dataset(road, tdus, out)

        train_imgs = list((out / "images" / "train").glob("*.jpg"))
        assert len(train_imgs) == 5 + 4  # road + tdus

    def test_val_count_equals_road_plus_tdus(self, tmp_path):
        road = _make_road_dataset(tmp_path, n_train=5, n_val=3)
        tdus = _make_tdus_dataset(tmp_path, n_train=4, n_val=2)
        out  = tmp_path / "unified"

        build_unified_dataset(road, tdus, out)

        val_imgs = list((out / "images" / "val").glob("*.jpg"))
        assert len(val_imgs) == 3 + 2

    def test_each_image_has_label_file(self, tmp_path):
        road = _make_road_dataset(tmp_path, n_train=3, n_val=2)
        tdus = _make_tdus_dataset(tmp_path, n_train=2, n_val=1)
        out  = tmp_path / "unified"

        build_unified_dataset(road, tdus, out)

        for split in ("train", "val"):
            for img in (out / "images" / split).glob("*.jpg"):
                lbl = out / "labels" / split / (img.stem + ".txt")
                assert lbl.exists(), f"Missing label for {img.name}"

    def test_detection_failures_excluded(self, tmp_path):
        """Images listed in detection_failures.txt must not appear in unified dataset."""
        tdus = _make_tdus_dataset(tmp_path, n_train=4, n_val=2)
        # Mark first two TDUS train images as failures
        fail_names = [
            str(tdus / "train" / "img" / "Species_tree_0.jpg"),
            str(tdus / "train" / "img" / "Species_tree_1.jpg"),
        ]
        (tdus / "train" / "labels" / "detection_failures.txt").write_text(
            "\n".join(fail_names) + "\n"
        )
        road = _make_road_dataset(tmp_path, n_train=3, n_val=2)
        out  = tmp_path / "unified"

        build_unified_dataset(road, tdus, out,
                              tdus_failures=tdus / "train" / "labels" / "detection_failures.txt")

        train_imgs = {p.name for p in (out / "images" / "train").glob("*.jpg")}
        assert "Species_tree_0.jpg" not in train_imgs
        assert "Species_tree_1.jpg" not in train_imgs
        # road images and non-failed TDUS images must be present
        assert len(train_imgs) == 3 + 2  # 3 road + 2 remaining TDUS

    def test_tdus_test_set_never_included(self, tmp_path):
        """build_unified_dataset must never touch tdus_data/test/."""
        road = _make_road_dataset(tmp_path, n_train=2, n_val=1)
        tdus = _make_tdus_dataset(tmp_path, n_train=2, n_val=1)
        # Create a fake tdus test directory with an image
        test_img_dir = tdus / "test" / "img"
        test_img_dir.mkdir(parents=True)
        (test_img_dir / "secret_test.jpg").write_bytes(b"JPEG")
        out = tmp_path / "unified"

        build_unified_dataset(road, tdus, out)

        all_imgs = set(p.name for p in out.rglob("*.jpg"))
        assert "secret_test.jpg" not in all_imgs
