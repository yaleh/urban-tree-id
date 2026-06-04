# Model Weights

All weights are hosted on HuggingFace at **[yaleh/urban-tree-id](https://huggingface.co/yaleh/urban-tree-id)**.
The `data/` directory is excluded from git; download weights with the commands below.

## Download

```python
from huggingface_hub import hf_hub_download

# YOLO detector (recommended)
hf_hub_download(
    repo_id="yaleh/urban-tree-id",
    filename="model_weights/tree_yolo26s_unified_halfres_tdus/best.pt",
    local_dir=".",
)

# SVM classifier (518px crop, highest accuracy)
hf_hub_download(
    repo_id="yaleh/urban-tree-id",
    filename="model_weights/svm/gdino_dinov2_518px/svm_model.joblib",
    local_dir=".",
)
```

Or download everything at once:

```bash
huggingface-cli download yaleh/urban-tree-id --local-dir .
```

## YOLO Checkpoints

| Checkpoint | Test acc (pipeline) | no_det / 386 | img/s | Training data |
|-----------|---------------------|--------------|-------|---------------|
| `tree_yolo26s_unified_halfres_tdus` | **96.1%** | **0** | 8.5 | unified: video frames + TDUS |
| `tree_yolo26s_b10` | 71.2% | 35 | 7.8 | video frame pseudo-labels |
| `tree_yolo26s_halfres` | — | — | — | video frame pseudo-labels (half-res) |

Pipeline accuracy measured on TDUS test split (386 images) with SVM `gdino_dinov2_518px`, crop_size=518.

**Recommended:** `tree_yolo26s_unified_halfres_tdus` — trained on combined video-frame and TDUS data,
zero missed detections on the test split.

## SVM Classifiers

All SVMs use GDino-tiny crops → DINOv2 ViT-S/14 CLS token (384-dim) → RBF-SVM, 22 species.
Accuracy measured on standalone SVM evaluation (GDino detector, TDUS splits).

| Checkpoint | Crop size | Val acc | Test acc | Notes |
|-----------|-----------|---------|----------|-------|
| `svm/gdino_dinov2_518px` | 518px | 97.7% | 96.9% | highest accuracy |
| `svm/gdino_dinov2_448px` | 448px | 97.2% | 95.3% | default; best accuracy/throughput trade-off |
| `svm/gdino_dinov2_224px` | 224px | 89.1% | 87.1% | fastest (+37% throughput vs 448px) |

Pass `--svm-model` and `--crop-size` together to `benchmark_pipeline.py` to match the SVM's training resolution.
