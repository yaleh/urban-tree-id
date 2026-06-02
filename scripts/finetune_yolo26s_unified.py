"""
finetune_yolo26s_unified.py

本地微调脚本，对应 notebooks/finetune_yolo26s_colab.ipynb。

功能：
  1. 从 tree_yolo26s_halfres/weights/best.pt 微调
  2. 数据集：data/unified_halfres_tdus（road + TDUS，共 5247 train / 2477 val）
  3. 自动 resume：若 runs/detect/<run_name>/weights/last.pt 存在则续训
  4. 超参数与 Colab notebook 完全对齐（batch 改为 -1 autobatch 适配本地 12GB VRAM）

用法：
    .venv/bin/python scripts/finetune_yolo26s_unified.py
    .venv/bin/python scripts/finetune_yolo26s_unified.py --batch 8 --epochs 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


# ── 数据集校验（复用 train_yolo26.py 逻辑）────────────────────────────────────

def validate_dataset(data_yaml: Path) -> dict:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    data = yaml.safe_load(data_yaml.read_text())
    required = {"path", "train", "val", "nc", "names"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"data.yaml missing fields: {missing}")
    root = Path(data["path"])
    for split in ("train", "val"):
        img_dir = root / data[split]
        if not img_dir.exists() or not any(img_dir.glob("*.jpg")):
            raise ValueError(f"images/{split}/ empty or missing: {img_dir}")
        lbl_dir = root / data[split].replace("images", "labels")
        if not lbl_dir.exists():
            raise ValueError(f"labels/{split}/ missing: {lbl_dir}")
        n_imgs = len(list(img_dir.glob("*.jpg")))
        n_lbls = len(list(lbl_dir.glob("*.txt")))
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")
    return data


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    data_yaml  = Path(args.data).resolve()
    src_pt     = Path(args.model).resolve()
    resume_pt  = (Path("runs/detect") / args.run_name / "weights" / "last.pt").resolve()

    # 数据集校验
    print("Validating dataset...")
    validate_dataset(data_yaml)

    # Resume 判断
    if resume_pt.exists():
        print(f"Resuming from {resume_pt}")
        model = YOLO(str(resume_pt))
        model.train(resume=True)
    else:
        if not src_pt.exists():
            raise FileNotFoundError(f"Source checkpoint not found: {src_pt}")
        print(f"Fresh fine-tune from {src_pt}")
        model = YOLO(str(src_pt))
        model.train(
            data        = str(data_yaml),
            epochs      = args.epochs,
            imgsz       = args.imgsz,
            batch       = args.batch,      # -1 = autobatch（Colab 用 24，本地 12GB 用 autobatch）
            freeze      = args.freeze,
            lr0         = args.lr0,
            lrf         = 0.01,
            cos_lr      = True,
            patience    = args.patience,
            rect        = False,           # 跨域 mosaic 必须关闭 rect
            mosaic      = 0.5,
            scale       = 0.5,
            translate   = 0.15,
            degrees     = 10.0,
            fliplr      = 0.5,
            flipud      = 0.0,
            mixup       = 0.0,
            close_mosaic= args.close_mosaic,
            project     = str(Path("runs/detect").resolve()),
            name        = args.run_name,
            exist_ok    = True,
            device      = args.device,
            workers     = args.workers,
            cache       = True,
            deterministic = False,
            save        = True,
        )

    # 结果汇报
    best = (Path("runs/detect") / args.run_name / "weights" / "best.pt").resolve()
    results_csv = (Path("runs/detect") / args.run_name / "results.csv").resolve()
    print("\n" + "=" * 60)
    if best.exists():
        size_mb = best.stat().st_size / 1024 / 1024
        print(f"best.pt:     {best}  ({size_mb:.1f} MB)")
    else:
        print("WARNING: best.pt not found", file=sys.stderr)
    if results_csv.exists():
        # 打印最后一行 results.csv（最终 epoch 指标）
        lines = results_csv.read_text().strip().splitlines()
        if len(lines) >= 2:
            header = [h.strip() for h in lines[0].split(",")]
            values = [v.strip() for v in lines[-1].split(",")]
            metrics = dict(zip(header, values))
            map50 = metrics.get("metrics/mAP50(B)", "?")
            map50_95 = metrics.get("metrics/mAP50-95(B)", "?")
            print(f"Final mAP50: {map50}  mAP50-95: {map50_95}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO26s on unified road+TDUS dataset (local version of Colab notebook)"
    )
    parser.add_argument(
        "--data",
        default="data/unified_halfres_tdus/data.yaml",
        help="Path to data.yaml (default: data/unified_halfres_tdus/data.yaml)",
    )
    parser.add_argument(
        "--model",
        default="runs/detect/tree_yolo26s_halfres/weights/best.pt",
        help="Source checkpoint for fresh fine-tune (default: tree_yolo26s_halfres best.pt)",
    )
    parser.add_argument(
        "--run-name",
        default="tree_yolo26s_unified_halfres_tdus",
        help="Output run name under runs/detect/ (default: tree_yolo26s_unified_halfres_tdus)",
    )
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--imgsz",        type=int,   default=1280)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size; -1 = autobatch (default). Colab used 24 on larger VRAM.",
    )
    parser.add_argument("--freeze",       type=int,   default=5)
    parser.add_argument("--lr0",          type=float, default=2e-4)
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--close-mosaic", type=int,   default=10)
    parser.add_argument("--device",       default="0")
    parser.add_argument("--workers",      type=int,   default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
