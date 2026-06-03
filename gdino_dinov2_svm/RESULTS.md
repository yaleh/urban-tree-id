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

### YOLO detector (960px imgsz)

| Crop size | Batch | img/s | Notes |
|-----------|-------|-------|-------|
| 448px     | 64    | 25.4  | measured 2026-06-03, commit d4f14e8 |
| 224px     | 64    | 34.7  | +37% vs 448px; accuracy −8pp |
| 518px     | 64    | 20.1  | measured 2026-06-03; −21% vs 448px |

**Recommended production configuration:** YOLO 960px + DINOv2 448px crop + SVM (25.4 img/s, 95.3% test acc).
518px adds +1.6pp accuracy at −21% throughput (20.1 img/s); 224px saves +37% throughput but costs −8pp accuracy.
