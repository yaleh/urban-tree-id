"""
tests/test_generate_pseudo_labels.py

TDD tests for _run_gdino_batch extracted from generate_pseudo_labels.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import torch
from PIL import Image

from generate_pseudo_labels import _run_gdino_batch, FrameDataset


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_img_dir(base: Path, n: int) -> Path:
    """Create n tiny JPEG files in base/img/."""
    img_dir = base / "img"
    img_dir.mkdir(parents=True)
    for i in range(n):
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img.save(img_dir / f"frame_{i:03d}.jpg")
    return img_dir


def _make_mock_processor(boxes_sequence: list[torch.Tensor], scores_sequence: list[torch.Tensor]):
    """Return a mock processor whose post_process returns boxes/scores in order."""
    mock_processor = MagicMock()
    mock_processor.tokenizer.side_effect = lambda *a, **kw: {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }

    call_idx = [0]

    def _post_process(outputs, input_ids, box_threshold, text_threshold, target_sizes):
        results = []
        for _ in range(len(target_sizes)):
            idx = call_idx[0] % len(boxes_sequence)
            call_idx[0] += 1
            results.append({
                "boxes": boxes_sequence[idx],
                "scores": scores_sequence[idx],
            })
        return results

    mock_processor.post_process_grounded_object_detection.side_effect = _post_process
    return mock_processor


def _make_mock_model():
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    return mock_model


# ── Tests for _run_gdino_batch ────────────────────────────────────────────────

class TestRunGdinoBatch:
    def test_yields_correct_tuple_structure(self, tmp_path):
        """Each yielded element must be a 4-tuple: (paths, pil_imgs, boxes_list, scores_list)."""
        img_dir = _make_img_dir(tmp_path, n=2)
        dataset = FrameDataset(img_dir)

        boxes = torch.tensor([[10., 10., 50., 50.]])
        scores = torch.tensor([0.9])
        mock_proc = _make_mock_processor([boxes] * 2, [scores] * 2)
        mock_model = _make_mock_model()

        results = list(_run_gdino_batch(
            dataset, mock_proc, mock_model,
            device="cpu", batch_size=4,
            box_threshold=0.30, text_threshold=0.25,
            num_workers=0,
        ))

        assert len(results) == 1, "2 images in one batch of size 4 → 1 yield"
        paths, pil_imgs, boxes_list, scores_list = results[0]
        assert len(paths) == 2
        assert len(pil_imgs) == 2
        assert len(boxes_list) == 2
        assert len(scores_list) == 2

    def test_yielded_paths_are_strings(self, tmp_path):
        """paths in each yield must be strings (compatible with Path(p))."""
        img_dir = _make_img_dir(tmp_path, n=3)
        dataset = FrameDataset(img_dir)

        boxes = torch.tensor([[0., 0., 32., 32.]])
        scores = torch.tensor([0.8])
        mock_proc = _make_mock_processor([boxes] * 3, [scores] * 3)
        mock_model = _make_mock_model()

        for paths, pil_imgs, boxes_list, scores_list in _run_gdino_batch(
            dataset, mock_proc, mock_model,
            device="cpu", batch_size=4,
            box_threshold=0.30, text_threshold=0.25,
            num_workers=0,
        ):
            for p in paths:
                assert isinstance(p, str), f"Expected str, got {type(p)}"

    def test_boxes_and_scores_are_cpu_tensors(self, tmp_path):
        """boxes_list and scores_list elements must be CPU tensors."""
        img_dir = _make_img_dir(tmp_path, n=2)
        dataset = FrameDataset(img_dir)

        boxes = torch.tensor([[5., 5., 60., 60.]])
        scores = torch.tensor([0.7])
        mock_proc = _make_mock_processor([boxes] * 2, [scores] * 2)
        mock_model = _make_mock_model()

        for _, _, boxes_list, scores_list in _run_gdino_batch(
            dataset, mock_proc, mock_model,
            device="cpu", batch_size=4,
            box_threshold=0.30, text_threshold=0.25,
            num_workers=0,
        ):
            for b, s in zip(boxes_list, scores_list):
                assert isinstance(b, torch.Tensor)
                assert isinstance(s, torch.Tensor)
                assert b.device.type == "cpu"
                assert s.device.type == "cpu"

    def test_total_images_covered_across_batches(self, tmp_path):
        """All dataset images must appear across all yielded batches."""
        n = 5
        img_dir = _make_img_dir(tmp_path, n=n)
        dataset = FrameDataset(img_dir)

        boxes = torch.zeros(0, 4)
        scores = torch.zeros(0)
        mock_proc = _make_mock_processor([boxes] * n, [scores] * n)
        mock_model = _make_mock_model()

        all_paths = []
        for paths, pil_imgs, boxes_list, scores_list in _run_gdino_batch(
            dataset, mock_proc, mock_model,
            device="cpu", batch_size=2,
            box_threshold=0.30, text_threshold=0.25,
            num_workers=0,
        ):
            all_paths.extend(paths)

        assert len(all_paths) == n, f"Expected {n} paths total, got {len(all_paths)}"

    def test_num_workers_zero_passed_to_dataloader(self, tmp_path):
        """Verify num_workers=0 is forwarded to DataLoader (flat-dir scenario)."""
        img_dir = _make_img_dir(tmp_path, n=1)
        dataset = FrameDataset(img_dir)

        mock_proc = _make_mock_processor(
            [torch.zeros(0, 4)], [torch.zeros(0)]
        )
        mock_model = _make_mock_model()

        with patch("generate_pseudo_labels.DataLoader") as mock_dl_cls:
            # DataLoader must return something iterable
            mock_dl_cls.return_value = iter([])
            list(_run_gdino_batch(
                dataset, mock_proc, mock_model,
                device="cpu", batch_size=4,
                box_threshold=0.30, text_threshold=0.25,
                num_workers=0,
            ))
            _, kwargs = mock_dl_cls.call_args
            assert kwargs.get("num_workers") == 0

    def test_num_workers_two_passed_to_dataloader(self, tmp_path):
        """Verify num_workers=2 is forwarded to DataLoader (video-frame scenario)."""
        img_dir = _make_img_dir(tmp_path, n=1)
        dataset = FrameDataset(img_dir)

        mock_proc = _make_mock_processor(
            [torch.zeros(0, 4)], [torch.zeros(0)]
        )
        mock_model = _make_mock_model()

        with patch("generate_pseudo_labels.DataLoader") as mock_dl_cls:
            mock_dl_cls.return_value = iter([])
            list(_run_gdino_batch(
                dataset, mock_proc, mock_model,
                device="cpu", batch_size=4,
                box_threshold=0.30, text_threshold=0.25,
                num_workers=2,
            ))
            _, kwargs = mock_dl_cls.call_args
            assert kwargs.get("num_workers") == 2

    def test_pil_imgs_are_pil_image(self, tmp_path):
        """pil_imgs in each batch must be PIL.Image.Image instances."""
        img_dir = _make_img_dir(tmp_path, n=2)
        dataset = FrameDataset(img_dir)

        boxes = torch.tensor([[0., 0., 64., 64.]])
        scores = torch.tensor([0.95])
        mock_proc = _make_mock_processor([boxes] * 2, [scores] * 2)
        mock_model = _make_mock_model()

        for _, pil_imgs, _, _ in _run_gdino_batch(
            dataset, mock_proc, mock_model,
            device="cpu", batch_size=4,
            box_threshold=0.30, text_threshold=0.25,
            num_workers=0,
        ):
            for img in pil_imgs:
                assert isinstance(img, Image.Image), f"Expected PIL.Image, got {type(img)}"
