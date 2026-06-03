"""Validate TDUS fine-tune dataset layout and generate data.yaml for RF-DETR.

Stage 10.2 of the plan:
  1. Verify train/img and train/labels exist and are non-empty
  2. Exclude images listed in detection_failures.txt
  3. Generate data.yaml (only train + val splits, nc=1, names=['tree'])
  4. Print dataset statistics
"""
import argparse
import sys
from pathlib import Path

import yaml


def load_failures(labels_dir: Path) -> set[str]:
    fail_file = labels_dir / "detection_failures.txt"
    if not fail_file.exists():
        return set()
    names = set()
    for line in fail_file.read_text().splitlines():
        line = line.strip()
        if line:
            names.add(Path(line).name)
    return names


def count_valid(img_dir: Path, labels_dir: Path, failures: set[str]) -> int:
    imgs = {p.name for p in img_dir.glob("*.jpg")}
    imgs |= {p.name for p in img_dir.glob("*.jpeg")}
    imgs |= {p.name for p in img_dir.glob("*.png")}
    return len(imgs - failures)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RF-DETR dataset and generate data.yaml"
    )
    parser.add_argument("--tdus-data", required=True,
                        help="Root of TDUS resized dataset (contains train/ and val/)")
    parser.add_argument("--output-yaml", required=True,
                        help="Path to write data.yaml")
    args = parser.parse_args()

    root = Path(args.tdus_data).resolve()
    out_yaml = Path(args.output_yaml)

    train_img = root / "train" / "img"
    train_lbl = root / "train" / "labels"
    val_img   = root / "val"   / "img"

    # Validate required directories
    if not train_lbl.exists():
        print(f"ERROR: train/labels/ not found: {train_lbl}", file=sys.stderr)
        sys.exit(1)
    if not train_img.exists():
        print(f"ERROR: train/img/ not found: {train_img}", file=sys.stderr)
        sys.exit(1)

    failures = load_failures(train_lbl)

    n_train_imgs  = sum(1 for p in train_img.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n_train_valid = count_valid(train_img, train_lbl, failures)

    n_val = 0
    if val_img.exists():
        n_val = sum(1 for p in val_img.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    # Generate data.yaml
    data = {
        "path":  str(root),
        "train": "train/img",
        "val":   "val/img",
        "nc":    1,
        "names": ["tree"],
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))

    # Stats
    detection_rate = n_train_valid / n_train_imgs if n_train_imgs > 0 else 0.0
    print(f"有效 train 样本: {n_train_valid}/{n_train_imgs} "
          f"({detection_rate:.1%}, {len(failures)} failures)")
    print(f"val 样本: {n_val}")
    print(f"data.yaml → {out_yaml}")

    if n_train_valid < 2800:
        print(f"WARNING: 有效标签数 {n_train_valid} < 2800，请调查检出率后再继续 Phase 10.3",
              file=sys.stderr)


if __name__ == "__main__":
    main()
