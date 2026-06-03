"""Fine-tune YOLO26s on pseudo-labeled video frames (Stage C1 + C2)."""
import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def validate_dataset(data_yaml_path: str) -> dict:
    """Validate YOLO dataset structure and data.yaml before training."""
    p = Path(data_yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"data.yaml not found: {p}")

    data = yaml.safe_load(p.read_text())

    required = {"path", "train", "val", "nc", "names"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"data.yaml missing fields: {missing}")

    if not Path(data["path"]).is_absolute():
        raise ValueError(f"data.yaml 'path' must be absolute, got: {data['path']}")

    dataset_root = Path(data["path"])
    for split in ("train", "val"):
        img_dir = dataset_root / data[split]
        if not img_dir.exists() or not any(img_dir.glob("*.jpg")):
            raise ValueError(f"images/{split}/ empty or missing: {img_dir}")

        lbl_dir = dataset_root / data[split].replace("images", "labels")
        if not lbl_dir.exists():
            raise ValueError(f"labels/{split}/ missing: {lbl_dir}")

        n_imgs = len(list(img_dir.glob("*.jpg")))
        n_lbls = len(list(lbl_dir.glob("*.txt")))
        if abs(n_imgs - n_lbls) > 5:
            raise ValueError(
                f"{split}: image/label count mismatch: {n_imgs} imgs vs {n_lbls} labels"
            )

    return data


def train(args) -> None:
    data = validate_dataset(args.data)
    print(f"Dataset validated: {data['nc']} class(es), "
          f"names={data['names']}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        optimizer="auto",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        save=True,
        project=str(Path("runs/detect").resolve()),
        name=args.run_name,
        mosaic=args.mosaic,
        mixup=0.0,
        degrees=10.0,
        translate=args.translate,
        scale=args.scale,
        flipud=0.0,
        fliplr=0.5,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        rect=args.rect,
        deterministic=args.deterministic,
    )

    best = Path("runs/detect").resolve() / args.run_name / "weights" / "best.pt"
    if best.exists():
        print(f"\nBest weights: {best}")
    else:
        print("\nWarning: best.pt not found", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26s on pseudo-labeled trees")
    parser.add_argument("--data", default="data/pseudo_labels/all_4videos/data.yaml")
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--run-name", default="tree_yolo26s_pseudo")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4,
                        help="Dataloader workers for background prefetching")
    parser.add_argument("--cache", default="True",
                        help="Cache images in RAM (True/False/disk)")
    parser.add_argument("--rect", type=lambda x: x.lower() != "false", default=True,
                        help="Rectangular training (aspect-ratio-preserving)")
    parser.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=False,
                        help="CuDNN deterministic mode (slower, reproducible)")
    parser.add_argument("--mosaic", type=float, default=0.5,
                        help="Mosaic augmentation probability (0.0 = disabled)")
    parser.add_argument("--scale", type=float, default=0.3,
                        help="Random scale augmentation range")
    parser.add_argument("--translate", type=float, default=0.1,
                        help="Random translate augmentation range")
    parser.add_argument("--lr0", type=float, default=1e-3,
                        help="Initial learning rate (use 2e-4 for fine-tuning)")
    parser.add_argument("--patience", type=int, default=20,
                        help="EarlyStopping patience epochs")
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="Disable mosaic for last N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
