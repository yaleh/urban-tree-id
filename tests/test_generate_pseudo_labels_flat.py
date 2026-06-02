"""
tests/test_generate_pseudo_labels_flat.py

TDD tests for --flat-dir mode in generate_pseudo_labels.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from generate_pseudo_labels import run_inference_flat, parse_args_flat


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_img_dir(base: Path, n: int) -> Path:
    """Create n tiny JPEG files in base/img/."""
    img_dir = base / "img"
    img_dir.mkdir(parents=True)
    from PIL import Image
    for i in range(n):
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img.save(img_dir / f"tree_{i:03d}.jpg")
    return img_dir


def _mock_gdino(boxes_per_image: list[torch.Tensor], scores_per_image: list[torch.Tensor]):
    """Return (mock_model, mock_processor) whose post_process returns given boxes/scores."""
    mock_processor = MagicMock()
    mock_processor.tokenizer.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    mock_processor.tokenizer.side_effect = lambda *a, **kw: {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }

    call_count = [0]

    def _post_process(outputs, input_ids, box_threshold, text_threshold, target_sizes):
        results = []
        for i in range(len(target_sizes)):
            idx = call_count[0] % len(boxes_per_image)
            call_count[0] += 1
            results.append({
                "boxes": boxes_per_image[idx],
                "scores": scores_per_image[idx],
            })
        return results

    mock_processor.post_process_grounded_object_detection.side_effect = _post_process

    mock_model = MagicMock()
    mock_model.return_value = MagicMock()

    return mock_model, mock_processor


# ── tests ────────────────────────────────────────────────────────────────────

class TestArgsFlatDir:
    def test_flat_dir_parsed(self):
        args = parse_args_flat(["--flat-dir", "/some/dir", "--flat-out", "/out"])
        assert args.flat_dir == Path("/some/dir")
        assert args.flat_out == Path("/out")

    def test_defaults(self):
        args = parse_args_flat(["--flat-dir", "/some/dir", "--flat-out", "/out"])
        assert args.box_threshold == pytest.approx(0.30)
        assert args.score_thr == pytest.approx(0.35)
        assert args.batch_size == 4


class TestRunInferenceFlat:
    def test_output_count_equals_input(self, tmp_path):
        """Should produce one .txt per input image."""
        img_dir = _make_img_dir(tmp_path, n=3)
        out_dir = tmp_path / "labels"

        boxes   = torch.tensor([[10., 10., 50., 50.]])
        scores  = torch.tensor([0.9])
        mock_model, mock_proc = _mock_gdino([boxes] * 3, [scores] * 3)

        with patch("generate_pseudo_labels.AutoModelForZeroShotObjectDetection") as pm, \
             patch("generate_pseudo_labels.AutoProcessor") as pp:
            pm.from_pretrained.return_value = mock_model
            pp.from_pretrained.return_value = mock_proc
            run_inference_flat(img_dir, out_dir, device="cpu")

        txt_files = list(out_dir.glob("*.txt"))
        assert len(txt_files) == 3

    def test_empty_detection_writes_empty_txt_and_failure_log(self, tmp_path):
        """Images with no boxes → empty txt + entry in detection_failures.txt."""
        img_dir = _make_img_dir(tmp_path, n=2)
        out_dir = tmp_path / "labels"

        empty_boxes  = torch.zeros(0, 4)
        empty_scores = torch.zeros(0)
        mock_model, mock_proc = _mock_gdino([empty_boxes] * 2, [empty_scores] * 2)

        with patch("generate_pseudo_labels.AutoModelForZeroShotObjectDetection") as pm, \
             patch("generate_pseudo_labels.AutoProcessor") as pp:
            pm.from_pretrained.return_value = mock_model
            pp.from_pretrained.return_value = mock_proc
            run_inference_flat(img_dir, out_dir, device="cpu")

        failures_path = out_dir / "detection_failures.txt"
        assert failures_path.exists(), "detection_failures.txt should be created"
        failures = failures_path.read_text().splitlines()
        assert len(failures) == 2

    def test_failure_list_excludes_images_with_boxes(self, tmp_path):
        """Only images with zero boxes appear in detection_failures.txt."""
        img_dir = _make_img_dir(tmp_path, n=3)
        out_dir = tmp_path / "labels"

        boxes_yes   = torch.tensor([[10., 10., 50., 50.]])
        scores_yes  = torch.tensor([0.9])
        boxes_no    = torch.zeros(0, 4)
        scores_no   = torch.zeros(0)
        # image 0 → box, image 1 → no box, image 2 → box
        mock_model, mock_proc = _mock_gdino(
            [boxes_yes, boxes_no, boxes_yes],
            [scores_yes, scores_no, scores_yes],
        )

        with patch("generate_pseudo_labels.AutoModelForZeroShotObjectDetection") as pm, \
             patch("generate_pseudo_labels.AutoProcessor") as pp:
            pm.from_pretrained.return_value = mock_model
            pp.from_pretrained.return_value = mock_proc
            run_inference_flat(img_dir, out_dir, device="cpu")

        failures_path = out_dir / "detection_failures.txt"
        failures = failures_path.read_text().splitlines() if failures_path.exists() else []
        assert len(failures) == 1
