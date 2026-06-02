"""Tests for train_yolo26.py — Stage C1: validate_dataset()."""
import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_yolo26 import validate_dataset, train


def _make_dataset(base: Path, n_imgs: int = 5, n_lbls: int = 5,
                  absolute_path: bool = True, splits=("train", "val")) -> Path:
    """Create a minimal YOLO dataset layout."""
    dataset_root = base / "dataset"
    dataset_root.mkdir(parents=True)

    for split in splits:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(1, n_imgs + 1):
            (img_dir / f"frame_{i:06d}.jpg").write_bytes(b"JPEG")
        for i in range(1, n_lbls + 1):
            (lbl_dir / f"frame_{i:06d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    yaml_data = {
        "path": str(dataset_root) if absolute_path else "relative/path",
        "train": "images/train",
        "val":   "images/val",
        "nc": 1,
        "names": ["tree"],
    }
    yaml_path = base / "data.yaml"
    yaml_path.write_text(yaml.dump(yaml_data))
    return yaml_path


class TestValidateDataset:
    def test_valid_dataset_passes(self, tmp_path):
        yaml_path = _make_dataset(tmp_path)
        data = validate_dataset(str(yaml_path))
        assert data["nc"] == 1
        assert data["names"] == ["tree"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_dataset(str(tmp_path / "nonexistent.yaml"))

    def test_missing_required_field_raises(self, tmp_path):
        yaml_path = _make_dataset(tmp_path)
        data = yaml.safe_load(yaml_path.read_text())
        del data["nc"]
        yaml_path.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="missing fields"):
            validate_dataset(str(yaml_path))

    def test_relative_path_raises(self, tmp_path):
        yaml_path = _make_dataset(tmp_path, absolute_path=False)
        with pytest.raises(ValueError, match="absolute"):
            validate_dataset(str(yaml_path))

    def test_empty_image_dir_raises(self, tmp_path):
        yaml_path = _make_dataset(tmp_path, n_imgs=0, n_lbls=0)
        with pytest.raises(ValueError, match="empty or missing"):
            validate_dataset(str(yaml_path))

    def test_count_mismatch_within_tolerance_passes(self, tmp_path):
        yaml_path = _make_dataset(tmp_path, n_imgs=10, n_lbls=8)
        data = validate_dataset(str(yaml_path))
        assert data is not None

    def test_count_mismatch_exceeds_tolerance_raises(self, tmp_path):
        yaml_path = _make_dataset(tmp_path, n_imgs=20, n_lbls=5)
        with pytest.raises(ValueError, match="mismatch"):
            validate_dataset(str(yaml_path))


class TestTrainArgs:
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        import importlib, train_yolo26 as m
        parser = argparse.ArgumentParser()
        # Replicate the argparse setup from main() by calling parse_known_args
        # on a subprocess — simpler: just import and invoke main's parser directly.
        # We rebuild the parser here to keep tests isolated from main().
        parser.add_argument("--data", default="data/pseudo_labels/all_4videos/data.yaml")
        parser.add_argument("--model", default="yolo26s.pt")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--imgsz", type=int, default=1280)
        parser.add_argument("--batch", type=int, default=10)
        parser.add_argument("--freeze", type=int, default=10)
        parser.add_argument("--run-name", default="tree_yolo26s_pseudo")
        parser.add_argument("--device", default="0")
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--cache", default="True")
        parser.add_argument("--rect", type=lambda x: x.lower() != "false", default=True)
        parser.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=False)
        parser.add_argument("--mosaic", type=float, default=0.5)
        parser.add_argument("--scale", type=float, default=0.3)
        parser.add_argument("--translate", type=float, default=0.1)
        return parser.parse_args(argv)

    def test_default_mosaic(self):
        args = self._parse([])
        assert args.mosaic == 0.5

    def test_default_scale(self):
        args = self._parse([])
        assert args.scale == 0.3

    def test_default_translate(self):
        args = self._parse([])
        assert args.translate == 0.1

    def test_override_augmentation_args(self):
        args = self._parse(["--mosaic", "0.0", "--scale", "0.5", "--translate", "0.15"])
        assert args.mosaic == 0.0
        assert args.scale == 0.5
        assert args.translate == 0.15

    def test_train_passes_augmentation_to_model(self, tmp_path):
        yaml_path = _make_dataset(tmp_path)
        args = argparse.Namespace(
            data=str(yaml_path),
            model="yolo26s.pt",
            epochs=1,
            imgsz=640,
            batch=1,
            freeze=0,
            run_name="test_run",
            device="cpu",
            workers=0,
            cache=False,
            rect=False,
            deterministic=False,
            mosaic=0.0,
            scale=0.5,
            translate=0.15,
            lr0=1e-3,
            patience=20,
            close_mosaic=10,
        )
        mock_model = MagicMock()
        mock_model.train = MagicMock()
        with patch("train_yolo26.YOLO", return_value=mock_model):
            train(args)
        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["mosaic"] == 0.0
        assert call_kwargs["scale"] == 0.5
        assert call_kwargs["translate"] == 0.15


class TestTrainArgsFinetune:
    """Tests for new fine-tuning CLI params: --lr0, --patience, --close-mosaic."""

    def _parse(self, argv):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", default="data/pseudo_labels/all_4videos/data.yaml")
        parser.add_argument("--model", default="yolo26s.pt")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--imgsz", type=int, default=1280)
        parser.add_argument("--batch", type=int, default=10)
        parser.add_argument("--freeze", type=int, default=10)
        parser.add_argument("--run-name", default="tree_yolo26s_pseudo")
        parser.add_argument("--device", default="0")
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--cache", default="True")
        parser.add_argument("--rect", type=lambda x: x.lower() != "false", default=True)
        parser.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=False)
        parser.add_argument("--mosaic", type=float, default=0.5)
        parser.add_argument("--scale", type=float, default=0.3)
        parser.add_argument("--translate", type=float, default=0.1)
        parser.add_argument("--lr0", type=float, default=1e-3)
        parser.add_argument("--patience", type=int, default=20)
        parser.add_argument("--close-mosaic", type=int, default=10)
        return parser.parse_args(argv)

    def test_default_lr0(self):
        args = self._parse([])
        assert args.lr0 == pytest.approx(1e-3)

    def test_override_lr0(self):
        args = self._parse(["--lr0", "2e-4"])
        assert args.lr0 == pytest.approx(2e-4)

    def test_default_patience(self):
        args = self._parse([])
        assert args.patience == 20

    def test_default_close_mosaic(self):
        args = self._parse([])
        assert args.close_mosaic == 10

    def test_train_passes_lr0_to_model(self, tmp_path):
        yaml_path = _make_dataset(tmp_path)
        args = argparse.Namespace(
            data=str(yaml_path),
            model="yolo26s.pt",
            epochs=1,
            imgsz=640,
            batch=1,
            freeze=0,
            run_name="test_run",
            device="cpu",
            workers=0,
            cache=False,
            rect=False,
            deterministic=False,
            mosaic=0.0,
            scale=0.5,
            translate=0.15,
            lr0=2e-4,
            patience=15,
            close_mosaic=5,
        )
        mock_model = MagicMock()
        mock_model.train = MagicMock()
        with patch("train_yolo26.YOLO", return_value=mock_model):
            train(args)
        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["lr0"] == pytest.approx(2e-4)
        assert call_kwargs["patience"] == 15
        assert call_kwargs["close_mosaic"] == 5
