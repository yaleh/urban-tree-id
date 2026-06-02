"""Build a unified YOLO dataset from road-video frames and TDUS pre-scaled images.

Road dataset layout  (data/pseudo_labels_half/all_4videos/):
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt
    data.yaml

TDUS dataset layout  (data/tdus_resized/):
    {train,val}/img/*.jpg
    {train,val}/labels/*.txt
    train/labels/detection_failures.txt   (optional)

Output (data/unified_halfres_tdus/):
    images/{train,val}/*.jpg   (copied)
    labels/{train,val}/*.txt   (copied)
    data.yaml
"""
import argparse
import shutil
from pathlib import Path

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _copy_files(src_dir: Path, dst_dir: Path, exclude: set[str] | None = None) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in src_dir.iterdir():
        if src.suffix.lower() not in IMG_EXTS and src.suffix.lower() != ".txt":
            continue
        if exclude and src.name in exclude:
            continue
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        count += 1
    return count


def build_unified_dataset(
    road_data: Path,
    tdus_data: Path,
    out: Path,
    tdus_failures: Path | None = None,
) -> None:
    """Merge road and TDUS datasets into out/."""
    out.mkdir(parents=True, exist_ok=True)

    # Load excluded filenames from detection_failures.txt
    excluded_train: set[str] = set()
    if tdus_failures and tdus_failures.exists():
        for line in tdus_failures.read_text().splitlines():
            line = line.strip()
            if line:
                excluded_train.add(Path(line).name)

    for split in ("train", "val"):
        out_img = out / "images" / split
        out_lbl = out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        # Road images + labels
        road_img_dir = road_data / "images" / split
        road_lbl_dir = road_data / "labels" / split
        if road_img_dir.exists():
            _copy_files(road_img_dir, out_img)
            _copy_files(road_lbl_dir, out_lbl)

        # TDUS images + labels (only train/val, never test)
        tdus_img_dir = tdus_data / split / "img"
        tdus_lbl_dir = tdus_data / split / "labels"
        exclude = excluded_train if split == "train" else None
        if tdus_img_dir.exists():
            _copy_files(tdus_img_dir, out_img, exclude=exclude)
        if tdus_lbl_dir.exists():
            # Also exclude matching label files and skip detection_failures.txt
            lbl_exclude: set[str] | None = None
            if exclude:
                lbl_exclude = {Path(n).stem + ".txt" for n in exclude}
                lbl_exclude.add("detection_failures.txt")
            else:
                lbl_exclude = {"detection_failures.txt"}
            _copy_files(tdus_lbl_dir, out_lbl, exclude=lbl_exclude)

    # Write data.yaml
    yaml_data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["tree"],
    }
    (out / "data.yaml").write_text(yaml.dump(yaml_data, default_flow_style=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified YOLO dataset")
    parser.add_argument("--road-data", type=Path, required=True,
                        help="Road dataset root (data/pseudo_labels_half/all_4videos/)")
    parser.add_argument("--tdus-data", type=Path, required=True,
                        help="TDUS pre-scaled dataset root (data/tdus_resized/)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory (data/unified_halfres_tdus/)")
    parser.add_argument("--tdus-failures", type=Path, default=None,
                        help="Path to detection_failures.txt from Stage 8.3")
    args = parser.parse_args()

    build_unified_dataset(args.road_data, args.tdus_data, args.out, args.tdus_failures)

    for split in ("train", "val"):
        n = sum(1 for _ in (args.out / "images" / split).glob("*.jpg"))
        print(f"  {split}: {n} images")
    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()
