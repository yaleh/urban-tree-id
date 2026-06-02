"""Unified YOLO label I/O.

All scripts that read or write YOLO-format label files should import from
this module instead of duplicating the logic locally.
"""
from pathlib import Path

import torch


def load_yolo_xyxy(label_path: Path, img_w: int, img_h: int) -> torch.Tensor:
    """Read a YOLO format label file; return shape (N, 4) xyxy pixel-coord tensor.

    If the file does not exist or is empty, returns a shape (0, 4) empty tensor.
    All scripts should import from this module — do not re-implement locally.
    """
    if not label_path.exists() or label_path.stat().st_size == 0:
        return torch.zeros(0, 4)
    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = map(float, parts)
        x0 = (cx - w / 2) * img_w
        y0 = (cy - h / 2) * img_h
        x1 = (cx + w / 2) * img_w
        y1 = (cy + h / 2) * img_h
        rows.append([x0, y0, x1, y1])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, 4)


def xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h) -> tuple:
    """Convert xyxy pixel coordinates to YOLO normalised (cx, cy, w, h).

    All scripts should import from this module — do not re-implement locally.
    """
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w  = (x1 - x0) / img_w
    h  = (y1 - y0) / img_h
    return cx, cy, w, h


def write_yolo_labels(label_path: Path, boxes_xyxy: torch.Tensor,
                      img_w: int, img_h: int) -> None:
    """Write an xyxy pixel-coord tensor to a YOLO format label file (class_id=0).

    Creates parent directories as needed.  An empty tensor produces an empty file.
    All scripts should import from this module — do not re-implement locally.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if len(boxes_xyxy) == 0:
        label_path.write_text("")
        return
    lines = []
    for box in boxes_xyxy.tolist():
        x0, y0, x1, y1 = box
        cx, cy, w, h = xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n")
