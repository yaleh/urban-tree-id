# GDino + DINOv2 + SVM — Accuracy & Throughput Results

## Dataset

TDUS (Taiwan Diverse Urban Species), 22 classes:
- train: 3168 images
- val:   395 images
- test:  386 images

## Accuracy vs DINOv2 Crop Size

Pipeline: GDino-tiny (tree detection) → DINOv2 ViT-S/14 (CLS embedding, 384-dim) → RBF-SVM

| Crop size | Val acc | Test acc | Val F1 (macro) | Test F1 (macro) | SVM weights |
|-----------|---------|----------|----------------|-----------------|-------------|
| 224px     | 89.1%   | 87.1%    | 88.96%         | 87.27%          | `results_224/svm_model.joblib` |
| **448px** | **97.2%** | **95.3%** | **97.29%**  | **95.27%**      | `results/svm_model.joblib` (default) |
| 518px     | 97.7%   | 96.9%    | 97.76%         | 96.95%          | `results_518/svm_model.joblib` |

Notes:
- 448px is the default (best accuracy/throughput trade-off; 518px adds +1.6pp test acc but forces batch=8 with GDino due to VRAM)
- DINOv2 was pre-trained at 224px; 448px = 4× patch count; 518px = paper-standard eval resolution
- All three runs use identical image sets (label arrays verified identical across crop sizes)

## End-to-End Pipeline Accuracy (TDUS test split, 386 images)

Pipeline accuracy depends on both the detector and SVM. Results below use `benchmark_pipeline.py`
on the TDUS test split with the stated YOLO checkpoint + SVM combination.

| YOLO checkpoint | SVM crop | Test acc | no_det | img/s | Notes |
|-----------------|----------|----------|--------|-------|-------|
| `tree_yolo26s_unified_halfres_tdus` | 448px | **94.82%** | **0** | 8.8 | recommended default |
| `tree_yolo26s_unified_halfres_tdus` | 518px | 96.1% | 0 | 8.5 | +1.3pp acc, −3% throughput |
| GDino-tiny (no YOLO) | 448px | 94.0% | 0 | 3.0 | GDino detector baseline |
| `tree_yolo26s_b10` | 518px | 71.2% | 35 | 7.8 | trained on video frames only; domain mismatch |

Notes:
- `tree_yolo26s_b10` accuracy is low due to training/test domain mismatch (trained on dashcam video
  frames; TDUS test images are portrait close-ups) and because the SVM was trained on GDino crops
  while inference uses YOLO crops.
- `tree_yolo26s_unified_halfres_tdus` resolves both issues by training on a combined dataset that
  includes TDUS images, achieving higher accuracy than GDino at 2.9× the throughput.

## Throughput

GPU: NVIDIA (12 GB VRAM)

### GDino detector (batch limited by GDino VRAM)

| Crop size | Batch | img/s | VRAM peak | Notes |
|-----------|-------|-------|-----------|-------|
| 448px     | 8     | 4.5   | ~10.3 GB  | stable |
| 448px     | 16    | OOM   | 11.9 GB → crash | GDino forward needs ~3 GB more |
| 518px     | 8     | 4.4   | 10.3 GB   | stable; essentially same as 448px |
| 518px     | 16    | OOM   | —         | same root cause |

GDino is the throughput bottleneck; DINOv2 crop size (448 vs 518px) has negligible impact on img/s.

### YOLO detector (1280px imgsz)

| YOLO checkpoint | Crop size | Batch | img/s | Notes |
|-----------------|-----------|-------|-------|-------|
| `tree_yolo26s_unified_halfres_tdus` | 518px | 32 | 8.5 | measured 2026-06-04 |
| `tree_yolo26s_b10` | 518px | 32 | 7.8 | measured 2026-06-04 |
| `tree_yolo26s_b10` | 448px | 64 | 25.4 | measured 2026-06-03, commit d4f14e8; imgsz=960 |
| `tree_yolo26s_b10` | 224px | 64 | 34.7 | imgsz=960; accuracy −8pp vs 448px |
| `tree_yolo26s_b10` | 518px | 64 | 20.1 | imgsz=960; measured 2026-06-03 |

**Recommended production configuration:** `tree_yolo26s_unified_halfres_tdus` + SVM 448px crop
(94.82% test acc, 8.8 img/s, zero missed detections on test split).
Use 518px SVM for +1.3pp accuracy at a minor throughput cost (8.5 img/s).
