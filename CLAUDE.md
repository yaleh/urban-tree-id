# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Tests
```bash
pytest tests/                          # full suite
pytest tests/ -m 'not integration'     # skip tests that download real models
pytest tests/test_predict_pipeline.py -v  # single file
```

### Inference (single image)
```bash
PYTHONPATH=scripts:scripts/detection:scripts/eval:scripts/training:scripts/utils \
python3 scripts/predict_pipeline.py \
  --image path/to/image.jpg \
  --detector yolo \
  --yolo-checkpoint data/model_weights/tree_yolo26s_unified_halfres_tdus/best.pt \
  --svm-model gdino_dinov2_svm/results_518/svm_model.joblib
```

### Benchmark (batch evaluation with accuracy)
```bash
PYTHONPATH=scripts:scripts/detection:scripts/eval:scripts/training:scripts/utils \
python3 scripts/benchmark_pipeline.py \
  --detector yolo \
  --yolo-checkpoint data/model_weights/tree_yolo26s_unified_halfres_tdus/best.pt \
  --test-dir /path/to/tdus/test/img \
  --ann-dir /path/to/tdus/test/ann \
  --svm-model gdino_dinov2_svm/results_518/svm_model.joblib \
  --crop-size 518 --batch-size 32
```

Add `--timed-bench` to measure throughput only (pre-loads images, excludes I/O).
Add `--throughput-only` to skip annotation loading entirely.

### Pseudo-label generation (for YOLO training data)
```bash
PYTHONPATH=scripts:scripts/detection:scripts/eval:scripts/training:scripts/utils \
python3 scripts/training/generate_pseudo_labels.py \
  --flat-dir data/frames/video_name --flat-out data/pseudo_labels/video_name/labels
```

### Download model weights
```python
from huggingface_hub import hf_hub_download
hf_hub_download("yaleh/urban-tree-id",
    "model_weights/tree_yolo26s_unified_halfres_tdus/best.pt", local_dir=".")
```

See `data/model_weights/README.md` for the full weights inventory and accuracy table.

## Architecture

### Pipeline

Three-stage inference: **detect → embed → classify**

```
Image(s)
  → Detector (GDino / YOLO / RF-DETR)  →  bbox [x0,y0,x1,y1]
  → make_crops_gpu_batch()              →  (N,3,crop_size,crop_size) tensor
  → DINOv2 ViT-S/14 forward_features() →  384-dim CLS token
  → RBF-SVM                            →  {species, confidence}
```

`make_crops_gpu_batch()` in `scripts/predict_pipeline.py` does the crop entirely on GPU (aspect-ratio-preserving resize + white-canvas pad). The `--crop-size` passed to `benchmark_pipeline.py` **must match** the crop size used when training the SVM.

### Detector abstraction

`scripts/detection/base_detector.py` defines `BaseDetector`. All detectors return boxes as `list[list[float]]` in absolute `[x0, y0, x1, y1]` pixel coordinates.

Detectors that set `supports_pipeline = True` (GDino, RF-DETR) expose:
- `preprocess_cpu(images)` — safe to run in a background thread
- `forward_preprocessed(preprocessed)` — GPU-only forward pass

`benchmark_pipeline.py` uses these to double-buffer preprocessing and GPU work. YOLO (`supports_pipeline = False`) is batched directly by Ultralytics.

New detectors: subclass `BaseDetector`, implement `detect_batch`, and optionally the pipeline pair.

### Path setup

Scripts are **not installed as a package**. Each entry-point script calls:
```python
from _path_setup import setup; setup()
```
which adds `scripts/`, `scripts/detection/`, `scripts/eval/`, `scripts/training/`, `scripts/utils/` to `sys.path`. As a result, all intra-project imports are flat (e.g. `from base_detector import BaseDetector`, not `from detection.base_detector`). `conftest.py` calls the same setup for pytest.

When running scripts directly (not via pytest), set `PYTHONPATH` explicitly as shown in the commands above, or rely on `_path_setup` being called inside the script.

### Key files

| File | Role |
|------|------|
| `scripts/predict_pipeline.py` | Core inference logic: `gdino_forward_batch`, `make_crops_gpu_batch`, `predict_species_batch`, `run_pipeline` |
| `scripts/benchmark_pipeline.py` | Batched evaluation with DataLoader, double-buffering, accuracy scoring, JSON report |
| `scripts/detection/detector_factory.py` | Instantiates the right detector from CLI args |
| `scripts/utils/cli_common.py` | Shared argparse factories; defines canonical threshold defaults |
| `scripts/training/generate_pseudo_labels.py` | GDino batch inference → YOLO `.txt` labels |
| `gdino_dinov2_svm/RESULTS.md` | Accuracy and throughput benchmark table |
| `data/model_weights/README.md` | HuggingFace download instructions for all checkpoints |

### Annotation formats

**TDUS** (ground-truth evaluation):
`ann/{image}.jpg.json` → `{"tags": [{"value": "acer_palmatum"}]}`  species in `snake_case`.

**YOLO** (training labels):
`labels/{stem}.txt` — one line per box: `0 cx cy w h` (normalized).

### VRAM notes

- GDino: ~1.2 GB/image; batch=8 ≈ 10.3 GB peak on 12 GB GPU. Batch=16 OOMs.
- YOLO `tree_yolo26s_unified_halfres_tdus`: batch=32 at imgsz=1280 fits comfortably.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when fragmentation causes spurious OOM.
