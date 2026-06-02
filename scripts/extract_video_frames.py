"""Extract representative frames from one or more driving videos.

Uses ffmpeg for robust decoding of multi-stream MP4 files that OpenCV
cannot reliably seek through (grabFrame packet read attempts issue).
"""
import argparse
import subprocess
import json
from pathlib import Path


def _probe(video_path: Path) -> tuple[float, float]:
    """Return (fps, duration_seconds) via ffprobe."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
         str(video_path)],
        stderr=subprocess.DEVNULL,
    )
    streams = json.loads(out)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    num, den = map(int, v["r_frame_rate"].split("/"))
    fps = num / den
    duration = float(v["duration"])
    return fps, duration


def extract_frames_uniform(video_path: Path, out_dir: Path, every_n: int = 60,
                           scale_w: int | None = None, scale_h: int | None = None) -> int:
    """Save every N-th frame using ffmpeg select filter.

    Output fps = native_fps / every_n.  Frame files are named
    frame_{second:06d}.jpg where second is the 0-based output frame index.
    When scale_w and scale_h are both provided, frames are resized to that
    resolution after selection.
    """
    fps, duration = _probe(video_path)
    print(f"{video_path.name}: FPS={fps:.1f}, duration={duration:.1f}s, "
          f"total_frames≈{int(fps*duration)}, saving every {every_n} frames "
          f"(≈{fps/every_n:.2f} fps output)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # select filter: emit frame when frame index mod N == 0
    select = f"select=not(mod(n\\,{every_n}))"
    if scale_w is not None and scale_h is not None:
        vf = f"{select},scale={scale_w}:{scale_h}"
    else:
        vf = select
    out_pattern = str(out_dir / "frame_%06d.jpg")

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", vf,
         "-vsync", "0",          # preserve selected frame pts, no duplication
         "-q:v", "2",            # JPEG quality ≈ 95
         out_pattern],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    saved = len(list(out_dir.glob("frame_*.jpg")))
    print(f"  Saved {saved} frames → {out_dir}")
    return saved


def extract_frames_diff(video_path: Path, out_dir: Path, diff_thr: float = 5.0) -> int:
    """Save frames where mean absolute grayscale diff from previous saved frame
    exceeds diff_thr.  Uses ffmpeg scene-detection filter as a first pass.

    scene=score<diff_thr/255 selects frames where the scene-change score
    (a value in [0,1] proportional to mean pixel diff / 255) exceeds the
    normalised threshold.
    """
    fps, duration = _probe(video_path)
    print(f"{video_path.name}: FPS={fps:.1f}, duration={duration:.1f}s, "
          f"diff strategy (thr={diff_thr})")
    out_dir.mkdir(parents=True, exist_ok=True)

    # scene filter: 0 = first frame always kept; threshold normalised to [0,1]
    norm_thr = diff_thr / 255.0
    # also always keep the very first frame (select on scene change OR n==0)
    vf = f"select='gt(scene,{norm_thr:.6f})+eq(n,0)',setpts=N/TB"
    out_pattern = str(out_dir / "frame_%06d.jpg")

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", vf,
         "-vsync", "0",
         "-q:v", "2",
         out_pattern],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    saved = len(list(out_dir.glob("frame_*.jpg")))
    print(f"  Saved {saved} frames → {out_dir}")
    return saved


def extract_frames(video_path: Path, out_dir: Path, every_n: int = 60,
                   strategy: str = "uniform", diff_thr: float = 5.0,
                   scale_w: int | None = None, scale_h: int | None = None) -> int:
    if strategy == "uniform":
        return extract_frames_uniform(video_path, out_dir, every_n, scale_w, scale_h)
    elif strategy == "diff":
        return extract_frames_diff(video_path, out_dir, diff_thr)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from driving videos")
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--out-base", default="data/frames",
                        help="Output root; frames go to <out-base>/<video-stem>/")
    parser.add_argument("--every-n", type=int, default=60,
                        help="Uniform: save every N-th frame (default 60 → 1fps from 60fps)")
    parser.add_argument("--strategy", choices=["uniform", "diff"], default="uniform")
    parser.add_argument("--diff-thr", type=float, default=5.0)
    parser.add_argument("--scale-w", type=int, default=None,
                        help="Resize output width (requires --scale-h)")
    parser.add_argument("--scale-h", type=int, default=None,
                        help="Resize output height (requires --scale-w)")
    args = parser.parse_args()

    out_base = Path(args.out_base)
    total = 0
    for video_path in args.videos:
        vp = Path(video_path)
        out_dir = out_base / vp.stem
        total += extract_frames(vp, out_dir, args.every_n, args.strategy, args.diff_thr,
                                args.scale_w, args.scale_h)

    print(f"\nDone. Total frames saved: {total}")


if __name__ == "__main__":
    main()
