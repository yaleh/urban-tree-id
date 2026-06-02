"""Assemble per-video pseudo-label datasets into a single YOLO dataset.

Cross-date split (Stage B3):
  train ← eastbound_20240319 + westbound_20240319
  val   ← eastbound_20240530 + westbound_20240530

File names get a video-prefix to avoid collisions:
  {video_name}__{frame_name}.jpg / .txt
"""
import argparse
import shutil
import yaml
from pathlib import Path

VIDEO_SPLITS = {
    "train": ["eastbound_20240319", "westbound_20240319"],
    "val":   ["eastbound_20240530", "westbound_20240530"],
}


def split_dataset(pseudo_labels_base: Path, out_dir: Path) -> dict:
    stats = {"train": 0, "val": 0}

    for split, video_names in VIDEO_SPLITS.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for video_name in video_names:
            src_img_dir = pseudo_labels_base / video_name / "images" / "all"
            src_lbl_dir = pseudo_labels_base / video_name / "labels" / "all"
            if not src_img_dir.exists():
                print(f"  Warning: {src_img_dir} not found, skipping")
                continue

            for src_img in sorted(src_img_dir.glob("*.jpg")):
                stem = src_img.stem
                dst_name = f"{video_name}__{src_img.name}"
                dst_img = img_dir / dst_name
                dst_lbl = lbl_dir / f"{video_name}__{stem}.txt"

                shutil.copy2(src_img, dst_img)

                src_lbl = src_lbl_dir / f"{stem}.txt"
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    dst_lbl.write_text("")  # negative sample

            n = len(list(src_img_dir.glob("*.jpg")))
            stats[split] += n
            print(f"  {split} ← {video_name}: {n} frames")

    return stats


def write_data_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "data.yaml"
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc": 1,
        "names": ["tree"],
    }
    yaml_path.write_text(yaml.dump(data, default_flow_style=False))
    print(f"  Written: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Assemble YOLO dataset from per-video pseudo-labels")
    parser.add_argument("--pseudo-labels-base", default="data/pseudo_labels")
    parser.add_argument("--out-dir", default="data/pseudo_labels/all_4videos")
    args = parser.parse_args()

    pseudo_labels_base = Path(args.pseudo_labels_base)
    out_dir = Path(args.out_dir)

    print("Splitting dataset...")
    stats = split_dataset(pseudo_labels_base, out_dir)
    print(f"  train: {stats['train']} frames, val: {stats['val']} frames")

    print("Writing data.yaml...")
    yaml_path = write_data_yaml(out_dir)
    print(f"Done. Dataset at: {out_dir}")
    return yaml_path


if __name__ == "__main__":
    main()
