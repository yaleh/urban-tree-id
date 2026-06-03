"""
tests/test_prepare_rf_detr_dataset.py

TDD tests for scripts/prepare_rf_detr_dataset.py.

The script validates a TDUS fine-tune dataset and generates a data.yaml
suitable for RF-DETR training.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


# ── helpers ───────────────────────────────────────────────────────────────────

SCRIPT = Path(__file__).parent.parent / "scripts" / "training" / "prepare_rf_detr_dataset.py"
PYTHON = Path(__file__).parent.parent / ".venv" / "bin" / "python"


def _make_dataset(base: Path, n_train: int = 3, n_val: int = 2,
                  failures: list[str] | None = None,
                  skip_labels_dir: bool = False) -> Path:
    """
    Create a minimal TDUS-style dataset layout under *base*.

    Layout::

        base/
          train/
            img/          ← n_train .jpg files
            labels/       ← n_train .txt files  (skipped if skip_labels_dir)
              [detection_failures.txt]
          val/
            img/          ← n_val .jpg files
            labels/       ← n_val .txt files
    """
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = base / split / "img"
        img_dir.mkdir(parents=True)
        for i in range(n):
            (img_dir / f"tree_{i:03d}.jpg").write_bytes(b"JPEG")

        if split == "train" and skip_labels_dir:
            continue

        lbl_dir = base / split / "labels"
        lbl_dir.mkdir(parents=True)
        for i in range(n):
            (lbl_dir / f"tree_{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    if failures is not None:
        fail_path = base / "train" / "labels" / "detection_failures.txt"
        fail_path.write_text("\n".join(failures) + "\n")

    return base


def _run_script(tdus_data: Path, output_yaml: Path) -> subprocess.CompletedProcess:
    """Invoke the script as a subprocess and return the completed process."""
    python = str(PYTHON) if PYTHON.exists() else sys.executable
    return subprocess.run(
        [python, str(SCRIPT),
         "--tdus-data", str(tdus_data),
         "--output-yaml", str(output_yaml)],
        capture_output=True,
        text=True,
    )


# ── tests ─────────────────────────────────────────────────────────────────────

class TestPrepareRfDetrDataset:

    def test_yaml_has_correct_nc(self, tmp_path):
        """
        Generated data.yaml must declare nc=1, names=['tree'], and an absolute
        path pointing to the tdus-data root.
        """
        ds = _make_dataset(tmp_path / "ds")
        out_yaml = tmp_path / "data.yaml"

        result = _run_script(ds, out_yaml)
        assert result.returncode == 0, (
            f"Script exited with {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        assert out_yaml.exists(), "data.yaml was not created"
        data = yaml.safe_load(out_yaml.read_text())

        assert data["nc"] == 1, f"Expected nc=1, got {data.get('nc')}"
        assert isinstance(data.get("names"), list), "'names' should be a list"
        assert data["names"][0] == "tree", (
            f"Expected names[0]='tree', got {data['names'][0]!r}"
        )
        assert Path(data["path"]).is_absolute(), (
            f"'path' field must be absolute, got {data['path']!r}"
        )

    def test_detection_failures_excluded(self, tmp_path):
        """
        Images listed in detection_failures.txt must be subtracted from the
        valid train count reported on stdout.
        """
        n_train = 3
        # Mark one image as a failure (use its basename, as the script spec allows)
        failures = ["tree_001.jpg"]
        ds = _make_dataset(tmp_path / "ds", n_train=n_train, failures=failures)
        out_yaml = tmp_path / "data.yaml"

        result = _run_script(ds, out_yaml)
        assert result.returncode == 0, (
            f"Script exited with {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # The script must print a number that equals n_train - len(failures)
        expected_valid = n_train - len(failures)
        stdout = result.stdout
        assert str(expected_valid) in stdout, (
            f"Expected valid train count {expected_valid} in stdout.\n"
            f"stdout: {stdout!r}"
        )

    def test_missing_labels_dir_raises(self, tmp_path):
        """
        When train/labels/ does not exist the script must exit non-zero
        (SystemExit / FileNotFoundError propagated as returncode != 0).
        """
        ds = _make_dataset(tmp_path / "ds", skip_labels_dir=True)
        out_yaml = tmp_path / "data.yaml"

        result = _run_script(ds, out_yaml)
        assert result.returncode != 0, (
            "Expected a non-zero exit code when train/labels/ is missing, "
            f"but got returncode=0.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_no_test_split_in_output(self, tmp_path):
        """
        The generated data.yaml must not contain a 'test:' key — RF-DETR
        training uses only train and val splits.
        """
        ds = _make_dataset(tmp_path / "ds")
        out_yaml = tmp_path / "data.yaml"

        result = _run_script(ds, out_yaml)
        assert result.returncode == 0, (
            f"Script exited with {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        assert out_yaml.exists(), "data.yaml was not created"
        data = yaml.safe_load(out_yaml.read_text())
        assert "test" not in data, (
            f"data.yaml must not contain a 'test' key, but got keys: {list(data.keys())}"
        )
