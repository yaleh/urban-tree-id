"""RF-DETR 推理封装，实现 BaseDetector 接口。

返回格式：list[list[float]]，每个内层 list 为 [x0, y0, x1, y1] 绝对像素坐标，
与 GDino / YOLO 路径的格式完全一致。
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from base_detector import BaseDetector

_RFDETR_MEANS = [0.485, 0.456, 0.406]
_RFDETR_STDS  = [0.229, 0.224, 0.225]


class RFDETRDetector(BaseDetector):
    def __init__(
        self,
        checkpoint: Union[str, Path],
        threshold: float = 0.3,
        device: str = "cuda",
    ):
        from rfdetr import RFDETRBase

        # num_classes=0: rfdetr checks shape[0] == num_classes+1; our checkpoint has shape [1]
        self.model = RFDETRBase(pretrain_weights=str(checkpoint), device=device, num_classes=0)
        self.threshold = threshold

    # ── BaseDetector interface ───────────────────────────────────────────────────

    @property
    def supports_pipeline(self) -> bool:
        return True

    def detect(self, image: Union[str, Path, "Image.Image"]) -> list:
        """运行推理，返回 [[x0, y0, x1, y1], ...] 绝对像素坐标列表。"""
        dets = self.model.predict(image, threshold=self.threshold)
        if len(dets.xyxy) == 0:
            return []
        return dets.xyxy.astype(np.float32).tolist()

    def detect_batch(self, images: list) -> list[list]:
        """Serial fallback: one GPU call per image. Use detect_batch_gpu for throughput."""
        return [self.detect(img) for img in images]

    def preprocess_cpu(self, images: list) -> tuple[list, list[tuple[int, int]]]:
        """CPU-only: decode images → list of float [0,1] CHW CPU tensors + orig_sizes (H,W).

        Safe to run in a background thread while the GPU processes the previous batch.
        """
        import torch

        orig_sizes: list[tuple[int, int]] = []
        tensors_cpu: list = []
        for img in images:
            if isinstance(img, torch.Tensor):
                arr = img.permute(1, 2, 0).contiguous().numpy()
                h, w = arr.shape[:2]
            elif isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
                w, h = img.size
                arr = np.array(img)
            else:
                w, h = img.size
                arr = np.array(img)
            orig_sizes.append((h, w))
            tensors_cpu.append(torch.from_numpy(arr).float().div(255.0).permute(2, 0, 1))
        return tensors_cpu, orig_sizes

    def forward_preprocessed(self, preprocessed: tuple) -> list[list]:
        """GPU forward pass given preprocess_cpu() output."""
        tensors_cpu, orig_sizes = preprocessed
        return self.forward_from_cpu_tensors(tensors_cpu, orig_sizes)

    # ── High-throughput GPU-batch helpers ────────────────────────────────────────

    def forward_from_cpu_tensors(
        self,
        tensors_cpu: list,
        orig_sizes: list[tuple[int, int]],
    ) -> list[list]:
        """GPU: H2D + normalize + resize + forward + postprocess.

        Accepts output of preprocess_cpu. Returns list[list[list[float]]].
        """
        import torch
        import torchvision.transforms.functional as F

        rf_model   = self.model.model
        inner      = rf_model.model
        resolution = rf_model.resolution
        device     = rf_model.device

        _mean = torch.tensor(_RFDETR_MEANS, device=device).view(3, 1, 1)
        _std  = torch.tensor(_RFDETR_STDS,  device=device).view(3, 1, 1)

        tensors_gpu = [
            F.resize(
                (t.to(device, non_blocking=True) - _mean) / _std,
                [resolution, resolution],
                antialias=True,
            )
            for t in tensors_cpu
        ]
        batch = torch.stack(tensors_gpu)

        inner.eval()
        with torch.inference_mode():
            predictions = inner(batch)
            results = rf_model.postprocessors["bbox"](
                predictions,
                target_sizes=torch.tensor(orig_sizes, device=device),
            )

        batch_boxes: list[list] = []
        for result in results:
            keep  = result["scores"] > self.threshold
            boxes = result["boxes"][keep].cpu().numpy().tolist()
            batch_boxes.append(boxes)
        return batch_boxes

    def detect_batch_gpu(self, images: list) -> list[list]:
        """True GPU batch via preprocess_cpu + forward_from_cpu_tensors.

        Returns list[list[list[float]]]: outer=images, inner=boxes ([x0,y0,x1,y1]).
        """
        tensors_cpu, orig_sizes = self.preprocess_cpu(images)
        return self.forward_from_cpu_tensors(tensors_cpu, orig_sizes)
