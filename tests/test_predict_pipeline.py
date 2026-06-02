"""
tests/test_predict_pipeline.py

TDD tests for scripts/predict_pipeline.py — argparse-level only,
no real model inference is run.
"""
import sys
import pytest
from pathlib import Path

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _make_parser():
    """Import and return the argparse parser from predict_pipeline without side effects."""
    # Import lazily so heavy deps are not resolved at collection time.
    import importlib.util, types

    # Stub out all heavy imports so the module can be parsed without GPU/model deps
    heavy_stubs = [
        "torch", "torch.nn", "torch.nn.functional", "torchvision",
        "torchvision.transforms", "numpy", "PIL", "PIL.Image",
        "transformers", "joblib", "tqdm",
    ]
    originals = {}
    for mod in heavy_stubs:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
            originals[mod] = None  # sentinel: we inserted this
        else:
            originals[mod] = sys.modules[mod]  # already present, don't remove

    spec = importlib.util.spec_from_file_location(
        "predict_pipeline",
        Path(__file__).parent.parent / "scripts" / "predict_pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Restore stubs we added (leave pre-existing ones intact)
    for mod_name, orig in originals.items():
        if orig is None:
            del sys.modules[mod_name]

    return mod.build_parser()


# ── tests ─────────────────────────────────────────────────────────────────────

class TestPredictPipelineArgparse:

    def test_rf_detr_detector_option_accepted(self):
        """--detector rf-detr must be a valid choice (argparse should not raise)."""
        parser = _make_parser()
        args = parser.parse_args([
            "--image", "photo.jpg",
            "--detector", "rf-detr",
            "--svm-model", "model.joblib",
            "--rf-detr-checkpoint", "ckpt.pth",
        ])
        assert args.detector == "rf-detr"

    def test_rf_detr_requires_checkpoint(self):
        """--detector rf-detr without --rf-detr-checkpoint must exit with non-zero."""
        import subprocess, sys
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "scripts" / "predict_pipeline.py"),
                "--image", "photo.jpg",
                "--detector", "rf-detr",
                "--svm-model", "model.joblib",
                # intentionally omitting --rf-detr-checkpoint
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, (
            "Expected non-zero exit when --rf-detr-checkpoint is missing, "
            f"got returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_gdino_path_unchanged(self):
        """--detector gdino (default) must be parseable without extra arguments."""
        parser = _make_parser()
        args = parser.parse_args([
            "--image", "photo.jpg",
            "--svm-model", "model.joblib",
        ])
        assert args.detector == "gdino"


# ── detect_gdino_batch tests ──────────────────────────────────────────────────

def _make_pil_rgb(w=640, h=480):
    from PIL import Image
    return Image.new("RGB", (w, h), color=(100, 150, 200))


def _build_gdino_batch_mocks(per_image_boxes: list, per_image_scores: list):
    """
    Return (mock_proc, mock_model) for detect_gdino_batch.
    mock_proc.post_process_grounded_object_detection returns per-image results.
    """
    import torch
    from unittest.mock import MagicMock

    post_results = [
        {"boxes": torch.as_tensor(b, dtype=torch.float32),
         "scores": torch.as_tensor(s, dtype=torch.float32)}
        for b, s in zip(per_image_boxes, per_image_scores)
    ]

    mock_proc = MagicMock()
    mock_proc.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    mock_proc.post_process_grounded_object_detection.return_value = post_results

    mock_model = MagicMock()
    mock_model.return_value = MagicMock()

    return mock_proc, mock_model


class TestDetectGdinoBatch:

    def test_returns_list_of_correct_length(self):
        """detect_gdino_batch must return list whose length == len(pil_imgs)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import predict_pipeline as pp

        imgs = [_make_pil_rgb() for _ in range(3)]
        proc, model = _build_gdino_batch_mocks([[], [], []], [[], [], []])
        result = pp.detect_gdino_batch(imgs, device="cpu", proc=proc, model=model)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_each_element_is_list(self):
        """Each element of the result must be a list (possibly empty)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import predict_pipeline as pp

        imgs   = [_make_pil_rgb(), _make_pil_rgb()]
        boxes  = [[[10.0, 20.0, 50.0, 80.0]], []]
        scores = [[0.9], []]
        proc, model = _build_gdino_batch_mocks(boxes, scores)
        result = pp.detect_gdino_batch(imgs, device="cpu", proc=proc, model=model)

        for item in result:
            assert isinstance(item, list), f"Expected list, got {type(item)}"

    def test_score_threshold_filters_low_scores(self):
        """Boxes with score < GDINO_SCORE_THR (0.35) must be excluded."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import predict_pipeline as pp

        imgs   = [_make_pil_rgb()]
        boxes  = [[[10.0, 20.0, 50.0, 80.0]]]
        scores = [[0.20]]   # below GDINO_SCORE_THR=0.35
        proc, model = _build_gdino_batch_mocks(boxes, scores)
        result = pp.detect_gdino_batch(imgs, device="cpu", proc=proc, model=model)

        assert result == [[]], f"Low-score box must be filtered out, got {result}"

    def test_high_score_box_is_kept(self):
        """Boxes with score >= GDINO_SCORE_THR must survive."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import predict_pipeline as pp

        imgs   = [_make_pil_rgb()]
        boxes  = [[[10.0, 20.0, 50.0, 80.0]]]
        scores = [[0.90]]   # well above threshold
        proc, model = _build_gdino_batch_mocks(boxes, scores)
        result = pp.detect_gdino_batch(imgs, device="cpu", proc=proc, model=model)

        assert len(result[0]) == 1, f"Expected 1 box kept, got {result}"
