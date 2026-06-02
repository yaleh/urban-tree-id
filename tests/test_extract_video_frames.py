"""Tests for scripts/extract_video_frames.py"""
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from extract_video_frames import extract_frames, extract_frames_uniform


def _make_video(path: Path, n_frames: int, fps: float = 20.0,
                width: int = 64, height: int = 36) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_video_with_scene_change(path: Path) -> None:
    """10 identical dark frames then 5 bright frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 20.0, (64, 36))
    for _ in range(10):
        writer.write(np.full((36, 64, 3), 30, dtype=np.uint8))
    for _ in range(5):
        writer.write(np.full((36, 64, 3), 200, dtype=np.uint8))
    writer.release()


class TestUniform:
    def test_frame_count(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_video(video, n_frames=10, fps=20.0)
        out = tmp_path / "out"
        # every_n=3 from 10 frames → frames 0,3,6,9 → 4 frames
        saved = extract_frames(video, out, every_n=3, strategy="uniform")
        assert saved == 4
        assert len(list(out.glob("*.jpg"))) == 4

    def test_naming_format(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_video(video, n_frames=10)
        out = tmp_path / "out"
        extract_frames(video, out, every_n=3, strategy="uniform")
        for p in out.glob("*.jpg"):
            assert re.match(r"frame_\d{6}\.jpg", p.name), f"Bad name: {p.name}"

    def test_auto_creates_output_dir(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_video(video, n_frames=5)
        out = tmp_path / "deep" / "nested" / "out"
        assert not out.exists()
        extract_frames(video, out, every_n=2, strategy="uniform")
        assert out.exists()

    def test_output_readable(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_video(video, n_frames=6, width=64, height=36)
        out = tmp_path / "out"
        extract_frames(video, out, every_n=3, strategy="uniform")
        for p in out.glob("*.jpg"):
            img = cv2.imread(str(p))
            assert img is not None
            assert img.shape[1] == 64  # width
            assert img.shape[0] == 36  # height

    def test_stem_subdir_via_cli(self, tmp_path):
        """CLI creates <out-base>/<video-stem>/ directory."""
        video = tmp_path / "myvideo.mp4"
        _make_video(video, n_frames=6)
        out_base = tmp_path / "frames"
        script = Path(__file__).parent.parent / "scripts" / "extract_video_frames.py"
        subprocess.run(
            [sys.executable, str(script),
             "--videos", str(video),
             "--out-base", str(out_base),
             "--every-n", "3"],
            check=True, capture_output=True,
        )
        assert (out_base / "myvideo").is_dir()


class TestDiff:
    def test_first_frame_always_saved(self, tmp_path):
        video = tmp_path / "static.mp4"
        _make_video_with_scene_change(video)
        out = tmp_path / "out"
        # impossibly high threshold → still saves first frame
        extract_frames(video, out, strategy="diff", diff_thr=255.0)
        frames = sorted(out.glob("*.jpg"))
        assert len(frames) >= 1
        assert frames[0].name == "frame_000001.jpg"  # ffmpeg output is 1-indexed

    def test_scene_change_detected(self, tmp_path):
        video = tmp_path / "scene.mp4"
        _make_video_with_scene_change(video)
        out = tmp_path / "out"
        saved = extract_frames(video, out, strategy="diff", diff_thr=5.0)
        assert saved >= 2

    def test_high_threshold_fewer_frames(self, tmp_path):
        video = tmp_path / "scene.mp4"
        _make_video_with_scene_change(video)
        out_lo, out_hi = tmp_path / "lo", tmp_path / "hi"
        saved_lo = extract_frames(video, out_lo, strategy="diff", diff_thr=1.0)
        saved_hi = extract_frames(video, out_hi, strategy="diff", diff_thr=50.0)
        assert saved_lo >= saved_hi

    def test_cli_invokable(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_video(video, n_frames=10)
        script = Path(__file__).parent.parent / "scripts" / "extract_video_frames.py"
        result = subprocess.run(
            [sys.executable, str(script),
             "--videos", str(video),
             "--out-base", str(tmp_path / "out"),
             "--every-n", "3"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr


class TestScale:
    def test_scale_resizes_output(self, tmp_path):
        """Frames are resized to (scale_w, scale_h) when both args provided."""
        video = tmp_path / "test.mp4"
        _make_video(video, n_frames=6, width=64, height=36)
        out = tmp_path / "out"
        extract_frames_uniform(video, out, every_n=3, scale_w=32, scale_h=18)
        for p in out.glob("*.jpg"):
            img = cv2.imread(str(p))
            assert img is not None
            assert img.shape[1] == 32
            assert img.shape[0] == 18

    def test_no_scale_keeps_original_size(self, tmp_path):
        """Omitting scale_w/scale_h preserves original dimensions."""
        video = tmp_path / "test.mp4"
        _make_video(video, n_frames=6, width=64, height=36)
        out = tmp_path / "out"
        extract_frames_uniform(video, out, every_n=3)
        for p in out.glob("*.jpg"):
            img = cv2.imread(str(p))
            assert img.shape[1] == 64
            assert img.shape[0] == 36

    def test_scale_via_cli(self, tmp_path):
        """--scale-w and --scale-h CLI args produce scaled output."""
        video = tmp_path / "test.mp4"
        _make_video(video, n_frames=6, width=64, height=36)
        out_base = tmp_path / "frames"
        script = Path(__file__).parent.parent / "scripts" / "extract_video_frames.py"
        result = subprocess.run(
            [sys.executable, str(script),
             "--videos", str(video),
             "--out-base", str(out_base),
             "--every-n", "3",
             "--scale-w", "32", "--scale-h", "18"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        frames = list((out_base / "test").glob("*.jpg"))
        assert len(frames) > 0
        img = cv2.imread(str(frames[0]))
        assert img.shape[1] == 32
        assert img.shape[0] == 18
