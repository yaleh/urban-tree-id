# Proposal: Video-Based Tree Detection via Grounding DINO Pseudo-GT + YOLO26 Fine-Tuning

**Status**: Draft  
**Date**: 2026-05-30  
**视频数据源**（4 个完整视频，共 ~7.5 GB）：
- `assets/original/computervisie_2024/eastbound_20240319.MP4`（2.25 GB）
- `assets/original/computervisie_2024/eastbound_20240530.MP4`（2.01 GB）
- `assets/original/computervisie_2024/westbound_20240319.MP4`（1.48 GB）
- `assets/original/computervisie_2024/westbound_20240530.MP4`（1.73 GB）

---

## 背景

当前项目已积累了完整的 Grounding DINO + DINOv2 特征提取基础设施：

- **`tdus_gdino_dinov2_svm.ipynb`**：端到端 notebook，使用 Grounding DINO 对树木做 zero-shot bbox 检测，将 crop 送入 DINOv2 提取 CLS token，再经 SVM 分类树种。已在 TDUS 数据集上完整验证，覆盖从检测到分类的全流程。实际使用的推理参数为 `GDINO_SCORE_THR = 0.3`，文本提示 `"tree."`，模型为 `IDEA-Research/grounding-dino-tiny`（172M 参数，bf16 推理）。
- **`gdino_dinov2_svm/`**：上述 notebook 的模块化版本，包含 `01_install.sh`、`02_download_dataset.py`、`03_extract_embeddings.py`、`04_train_validate.py`，是可复用的流水线脚本集。其中 `03_extract_embeddings.py` 的推理逻辑（预处理、batch collate、后处理 bbox 坐标还原）可直接改造为伪标签生成器。
- **`extract_embeddings_gdino.py`**：已支持分片（`--shard N/M`）和批量大小扫描，可高效处理大规模图像集。
- **`validate_gdino_bbox_crop.py`**、**`gdino_bbox_validation/`**：针对 Grounding DINO bbox 质量的专项验证基础设施，可直接复用于伪 GT 抽检可视化。

此外，项目拥有真实行车视频素材：`assets/original/computervisie_2024/` 下有多段城市道路行车视频，包含 `eastbound` / `westbound` 两个方向、20240319 和 20240530 两个拍摄日期。本方案直接使用**全部 4 个完整视频**作为数据源（参见标题，共约 7.5 GB），最大化训练数据覆盖度和场景多样性。`eastbound_20240319_short_20fps_compressed.MP4`（60 秒压缩短片）仅用于脚本快速验证，不作为正式数据源。

> **注意**：4 个原始视频的实际帧率和时长未经确认（短片标注为 20fps，原始视频可能为 20fps 或 30fps）。帧提取前须先运行 `ffprobe` 确认各视频的 `fps` 和 `duration`，以确定正确的采样间隔 N。

**目前没有视频帧提取脚本**，需在本方案中新建。

当前流水线仅做**分类**（在已有 bbox crop 的基础上识别树种），尚未有专门针对**视频场景的树木目标检测模型**。要在实际行车视频上做树木检测，需要一个能实时运行的检测模型，而非零样本的 Grounding DINO（172M 参数，推理慢、不适合视频流）。

**已安装环境**：Ultralytics 8.4.56（已含 YOLO26 支持，`YOLO("yolo26n.pt")` 默认模型）；`.venv/bin/python` 为项目标准运行环境。

---

## 目标

1. 以最少人工标注的方式，从行车视频自动生成树木 bbox 伪标签（pseudo ground truth）。
2. 用伪标签 fine-tune 一个轻量目标检测器（YOLO26），使其能在视频帧上高效、准确地检测树木。
3. 构建可迭代的自训练框架，进一步提升检测质量。

---

## 方案设计

### 阶段零：视频帧提取脚本（新建）

**需新建脚本**：`scripts/extract_video_frames.py`

由于项目目前无视频处理脚本，需从零实现。参考实现骨架如下：

```python
# scripts/extract_video_frames.py
import cv2, argparse
from pathlib import Path

def extract_frames(video_path, out_dir, every_n=20, strategy="uniform"):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS={fps:.1f}, total_frames={total}, duration={total/fps:.1f}s")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx, saved = 0, 0
    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if strategy == "uniform":
            if idx % every_n == 0:
                cv2.imwrite(str(out_dir / f"frame_{idx:06d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
        elif strategy == "diff":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None or cv2.absdiff(gray, prev_gray).mean() >= args.diff_thr:
                cv2.imwrite(str(out_dir / f"frame_{idx:06d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                prev_gray = gray
        idx += 1
    cap.release()
    print(f"Saved {saved} frames → {out_dir}")
```

**注意**：`cv2` 已通过 `opencv-python` 在 `.venv` 中可用（`skimage` 依赖已安装）。JPEG 质量建议 ≥ 90，避免压缩伪影影响 Grounding DINO 的检测精度。

---

### 阶段一：视频帧采样

**输入**：4 个完整视频（见标题）

**前置步骤（必须）**：帧提取前运行 `ffprobe` 确认各视频的实际 fps 和时长：

```bash
for f in assets/original/computervisie_2024/eastbound_20240319.MP4 \
          assets/original/computervisie_2024/eastbound_20240530.MP4 \
          assets/original/computervisie_2024/westbound_20240319.MP4 \
          assets/original/computervisie_2024/westbound_20240530.MP4; do
  ffprobe -v quiet -print_format json -show_streams "$f" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); v=[s for s in d['streams'] if s['codec_type']=='video'][0]; print(f\"{f}: fps={eval(v['r_frame_rate']):.1f}, duration={float(v['duration']):.0f}s\")"
done
```

根据实际 fps 确定采样间隔 N（目标采样率：1 fps）：`N = round(fps)`（20fps → N=20，30fps → N=30）。

**目标**：从 4 个视频中提取信息量密集、冗余度低的代表性帧，覆盖双方向、双日期的行车场景。

**两种采样策略（可按需选择）**：

#### 1A. 等间隔采样（推荐初版）

- 每 N 帧取一帧（N 由上述 ffprobe 结果确定），对应采样率 ≈ 1 fps。
- 简单、确定性强、易于复现；1 fps 对应行车场景中约 10–15 米的位移，足以保证画面多样性。
- 实现：`cv2.VideoCapture` 逐帧读取，`frame_idx % N == 0` 时保存。

#### 1B. 基于画面差异的自适应采样（可选增强）

- 计算相邻帧之间的帧差均值（`cv2.absdiff` + `.mean()`），仅在变化超过阈值时保存当前帧。
- 优点：在慢速路段避免过密采样，在场景切换处保留更多帧。
- 推荐阈值：≥ 5.0（灰度帧差均值，对应约 2% 像素变化）；阈值过高（≥ 10.0）会在交通灯等短暂停顿时漏掉过多树木帧，需实验确认。
- 可选工具：`skimage.metrics.structural_similarity`（SSIM），但运算开销约为帧差法的 10×，不推荐在大量帧上使用。

**输出**：每个视频独立输出到对应子目录，以保留视频来源信息（用于跨视频划分）：

```
data/frames/
├── eastbound_20240319/   # JPEG 帧，命名 frame_{idx:06d}.jpg
├── eastbound_20240530/
├── westbound_20240319/
└── westbound_20240530/
```

**预估规模**（假设原始视频亦为 20fps，估算码率约 30 Mbps）：

| 视频 | 估算时长 | N=20 时帧数 |
|------|---------|------------|
| eastbound_20240319 | ~590 秒 | ~590 帧 |
| eastbound_20240530 | ~526 秒 | ~526 帧 |
| westbound_20240319 | ~387 秒 | ~387 帧 |
| westbound_20240530 | ~451 秒 | ~451 帧 |
| **合计** | **~1,954 秒（32.6 分钟）** | **~1,954 帧** |

> 实际帧数以 ffprobe 结果为准，上表为估算值（±50%）。GDINO 批量推理约 0.5–1 秒/帧，全量推理预计耗时 **15–30 分钟**。

---

### 阶段二：Grounding DINO 伪标签生成

**输入**：阶段一输出的帧图像集  
**工具**：Grounding DINO tiny（`IDEA-Research/grounding-dino-tiny`，与现有脚本一致）

> **重要**：本阶段需**新建**伪标签生成脚本（建议命名 `scripts/generate_pseudo_labels.py`），在现有 `gdino_dinov2_svm/03_extract_embeddings.py` 的 Grounding DINO 推理框架基础上改造。现有脚本已实现：`gdino_preprocess`（resize+normalize）、批量 pad/collate、`post_process_grounded_object_detection`（xyxy 坐标还原）。改造要点：将"取最高分 bbox"的逻辑替换为"保留所有高置信度 bbox + NMS 过滤 + 写 YOLO txt"。

**步骤**：

#### 2.1 文本提示设计

使用简单、宽泛的提示词：`"tree."` （Grounding DINO 标准格式，与现有代码一致）。

- 不推荐使用 `"tree . trees"` 等多词形式，Grounding DINO tiny 对多 token 提示的处理不一致，单词 `"tree."` 已被项目验证有效（TDUS 数据集 recall=100%，fallback=0）。
- 不推荐使用树种名称，会造成严重漏检。

#### 2.2 置信度阈值策略

现有代码在 `gdino_dinov2_svm/` 使用 `score_thr=0.2`（box 和 text 统一），`tdus_gdino_dinov2_svm.ipynb` 使用 `GDINO_SCORE_THR=0.3`，实验结果显示 0.3 已能覆盖全部 TDUS 树木（fallback=0）。

行车视频场景（小目标、遮挡、透视变形）与 TDUS 静态图有差异，建议分两级过滤：

| 参数 | 推荐值 | 作用 |
|------|--------|------|
| `box_threshold` | 0.30 | 候选框召回阈值（宽松，保证不漏） |
| `text_threshold` | 0.25 | 文本匹配阈值 |
| 伪 GT 写入阈值 | 0.35 | 仅将 score ≥ 0.35 的 bbox 写入 .txt 文件 |

> **说明**：将 `box_threshold` 设为 0.35、写入阈值设为 0.40 的双阈值设计（原文档）实际上等价于单阈值 0.40，不必要地丢弃了 0.35–0.40 的有效检测。建议统一使用上表的三级策略，初始写入阈值 0.35，根据质检结果上调。

#### 2.3 NMS 过滤

- IoU 阈值：0.45（比标准 0.5 稍严，行车视频中相邻帧树木密集，适度去除重叠更有利于标签质量）
- 实现：`torchvision.ops.nms`（已在 `.venv` 中可用）

```python
from torchvision.ops import nms
import torch

def filter_boxes(boxes_xyxy, scores, iou_thr=0.45, score_thr=0.35):
    """boxes_xyxy: tensor[N,4], scores: tensor[N]"""
    mask = scores >= score_thr
    boxes, scores = boxes_xyxy[mask], scores[mask]
    if len(boxes) == 0:
        return boxes, scores
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep], scores[keep]
```

#### 2.4 YOLO 格式标注输出

将过滤后的 bbox（xyxy，像素坐标）转换为 YOLO txt 格式，**必须归一化到 [0, 1]**：

```python
def xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h):
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w  = (x1 - x0) / img_w
    h  = (y1 - y0) / img_h
    return cx, cy, w, h  # 全部在 [0, 1]

# 输出到 {label_dir}/{stem}.txt，每行：
# 0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}
```

类别只有一类，`class_id = 0`（tree）。若某帧无有效检测，写空 .txt 文件（YOLO 训练要求标注文件存在，允许为空代表负样本）。

#### 2.5 数据集目录结构（YOLO 格式，必须严格遵守）

YOLO26 要求以下目录结构，图像和标签需对应放置：

```
data/pseudo_labels/all_4videos/
├── images/
│   ├── train/   # 训练帧 (.jpg)，来自 20240319 两个视频
│   └── val/     # 验证帧 (.jpg)，来自 20240530 两个视频
└── labels/
    ├── train/   # 对应的 YOLO .txt 标注
    └── val/
```

**建议**：先逐视频生成伪标签（各自放入 `data/pseudo_labels/{video_name}/images/all/` + `labels/all/`），质检通过后再按阶段三的跨视频划分策略组合成最终数据集。

#### 2.6 data.yaml（完整版）

```yaml
# data/pseudo_labels/all_4videos/data.yaml
path: /absolute/path/to/data/pseudo_labels/all_4videos
train: images/train
val:   images/val

nc: 1
names:
  - tree
```

> **注意**：`path` 必须填写**绝对路径**，YOLO26 的 `data.yaml` 解析器不支持相对路径。

#### 2.7 质量检查

随机抽取 20–30 张帧，可视化 Grounding DINO 原始输出与过滤后伪 GT 的差异，人工确认：

- bbox 大体覆盖树冠（允许包含少量背景，YOLO 对此有一定容忍度）
- 无将路灯、建筑物等错检为树的严重假阳性
- 行道树密集处未出现大量重叠 bbox（若有，降低 NMS IoU 阈值至 0.35）

可复用 `validate_gdino_bbox_crop.py` 的可视化逻辑（`annotate_image` + `make_crop`），或直接调用已有的 `gdino_bbox_validation/` 基础设施。

**输出**：各视频独立数据集 `data/pseudo_labels/{video_name}/`（images/all/ + labels/all/），以及合并后的 `data/pseudo_labels/all_4videos/`（images/train/ + images/val/ + data.yaml）。

---

### 阶段三：数据集划分

**方法：跨日期划分（cross-date split）**

| 划分 | 视频 | 理由 |
|------|------|------|
| **train** | `eastbound_20240319` + `westbound_20240319` | 20240319 拍摄（春季早期） |
| **val** | `eastbound_20240530` + `westbound_20240530` | 20240530 拍摄（春季晚期，叶片更茂盛） |

**选择理由**：
- 跨日期划分比单视频时间切割更能测试模型的**泛化性**——val set 来自不同拍摄日，场景分布与 train 有差异（不同日光角度、叶片状态），mAP 更有参考价值。
- 两个日期各提供双方向视角（eastbound + westbound），train 和 val 均覆盖两个行驶方向，避免单方向偏差。
- 估算帧数：train ~977 帧，val ~977 帧（各约 50%，近似平衡）。
- **严禁随机打乱后再划分**：相邻帧高度相似（仅几米位移），随机划分会造成帧间泄漏，val mAP 虚高。

**实现**（`scripts/split_dataset.py`）：

```python
import shutil
from pathlib import Path

VIDEO_SPLITS = {
    "train": ["eastbound_20240319", "westbound_20240319"],
    "val":   ["eastbound_20240530", "westbound_20240530"],
}
OUT_DIR = Path("data/pseudo_labels/all_4videos")

for split, video_names in VIDEO_SPLITS.items():
    for video_name in video_names:
        src_img_dir = Path(f"data/pseudo_labels/{video_name}/images/all")
        src_lbl_dir = Path(f"data/pseudo_labels/{video_name}/labels/all")
        for p in sorted(src_img_dir.glob("*.jpg")):
            stem = p.stem
            # 加入视频名前缀避免跨视频命名冲突
            dst_name = f"{video_name}__{p.name}"
            dst_img = OUT_DIR / "images" / split / dst_name
            dst_lbl = OUT_DIR / "labels" / split / f"{video_name}__{stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst_img)
            src_lbl = src_lbl_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy(src_lbl, dst_lbl)
            else:
                dst_lbl.touch()  # 空标注 = 负样本帧
```

---

### 阶段四：YOLO26 Fine-Tuning

**模型说明**：YOLO26 由 Ultralytics 于 2026 年 1 月 14 日发布，核心特性是**原生端到端推理（NMS-free）**，直接输出最终预测，无需后处理 NMS，推理更快、部署更简便，尤其适合嵌入式/边缘设备场景。官方文档：[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)。本项目已安装 Ultralytics 8.4.56，包含完整 YOLO26 支持。

**模型规格选择**：

| 规格 | 参数量 | 推荐场景 |
|------|--------|----------|
| `yolo26n.pt` | 最小 | 快速实验、边缘设备、伪标签质量不确定时 |
| `yolo26s.pt` | 小 | **推荐初版**：精度/速度均衡，伪标签阶段首选 |
| `yolo26m.pt` | 中 | 精度优先，GPU 资源充足且标签质量经验证后使用 |

**Fine-tuning 配置**：

```python
# 使用 .venv/bin/python 运行
from ultralytics import YOLO

model = YOLO("yolo26s.pt")   # 加载预训练权重（首次运行自动下载）
results = model.train(
    data="/absolute/path/to/data/pseudo_labels/all_4videos/data.yaml",
    epochs=100,               # 伪标签场景建议 80–150 epochs，配合 early stopping
    imgsz=1280,               # 与 GDINO 有效分辨率对齐（1333×749），避免小树标注丢失
    batch=2,                  # RTX 3060 12GB 实测：imgsz=1280 rect 训练安全值
    freeze=10,                # 冻结 backbone 前 10 层（层 0–9），只训练 neck+head
    optimizer="auto",         # auto：小数据集自动选 AdamW，大数据集选 MuSGD
    lr0=1e-3,                 # AdamW 默认起始 lr；若 optimizer=MuSGD 改为 lr0=0.01
    lrf=0.01,                 # 终止 lr 比例：lr_final = lr0 * lrf
    cos_lr=True,              # 余弦退火，防止后期震荡
    patience=20,              # early stopping：20 epochs 无改善则停止
    save=True,
    project="runs/detect",
    name="tree_yolo26s_pseudo",
    # 数据增强（伪标签下适度增强，避免过拟合伪标签噪声）
    mosaic=0.5,               # 默认 1.0；伪标签场景适度降低
    mixup=0.0,                # 关闭 mixup，伪标签混合无意义
    degrees=10.0,             # 轻微旋转
    translate=0.1,
    scale=0.3,
    flipud=0.0,               # 行车视频无需上下翻转
    fliplr=0.5,
)
```

**关键参数说明**：

- **`freeze=10`**：冻结 backbone 前 10 层，仅训练 neck + head。伪标签质量有限时首选，防止 backbone 特征被噪声标签破坏；标签质量经验证后可改为 `freeze=0`（全网络微调）。
- **`optimizer="auto"`**：官方建议值。对于本场景（迭代次数 < 10,000 时选 AdamW，否则选 MuSGD），auto 会根据数据量自动判断。若出现训练不稳定（loss spike），显式指定 `optimizer="AdamW"`, `lr0=1e-3`。
- **`epochs=100` + `patience=20`**：100 epochs 上限 + 20 epochs early stopping，比原文 50 epochs 更充分（伪标签任务通常需要更长训练才能收敛）。
- **`cos_lr=True`**：官方 YOLO26 fine-tune 指南推荐，配合 `lrf=0.01` 实现余弦衰减。
- **`mosaic=0.5`**：降低 mosaic 增强概率（默认 1.0），因行车视频帧已有足够场景多样性，过强的 mosaic 会破坏 bbox 语义。

**注意**：因 YOLO26 是 NMS-free 端到端模型，不需要在推理侧手动添加 NMS 后处理：

```python
# 推理示例（对单个视频目录批量推理）
model = YOLO("runs/detect/tree_yolo26s_pseudo/weights/best.pt")
for video_name in ["eastbound_20240319", "eastbound_20240530",
                   "westbound_20240319", "westbound_20240530"]:
    results = model.predict(f"data/frames/{video_name}/", conf=0.25, save=True)
```

---

### 阶段五（可选）：迭代自训练

**动机**：伪 GT 质量有限，直接 fine-tune 的模型可能继承 Grounding DINO 的偏差。迭代自训练可逐步提升标签质量。

**前提条件（必须在阶段四前完成）**：手工标注 20–50 张帧作为固定 val set，用于客观评估每轮迭代的 mAP，是判断自训练是否收敛的唯一可靠依据。若跳过此步骤，val mAP 将只反映模型对自身伪标签的拟合程度，无参考价值。

**流程**：

```
迭代 k=1, 2, ...:
  1. 用 YOLO_k 对全部帧重新推理，得到预测 bbox 集合 P_k（conf ≥ 0.25）
  2. 与 Grounding DINO 原始输出 G（conf ≥ 0.30）取交集（IoU > 0.5）：
       accepted = P_k ∩ G（两者均预测为树的 bbox 保留）
  3. 用 accepted 作为新一轮伪 GT，重新 fine-tune → YOLO_{k+1}
  4. 在手工标注 val set 上评估 mAP@0.5
  5. 若 mAP 不再提升（连续 2 轮无改善）则停止
```

**实践建议**：

- 每轮迭代保存模型（`runs/detect/tree_yolo26s_iter{k}/weights/best.pt`）和对应伪标签版本，便于回溯。
- 置信度阈值可随迭代轮次适度收紧：第 1 轮写入阈值 0.35，后续每轮 +0.05（上限 0.55），逐步纯化标签。
- 通常 2–3 轮后收益递减，超过 5 轮通常无意义。

---

## 权衡分析

### 伪 GT 质量 vs. 覆盖度

- **高置信度阈值**（≥ 0.50）：伪 GT 精确度高，但漏检率上升，训练数据稀少；适合树木密集、清晰的场景。
- **低置信度阈值**（≥ 0.30）：覆盖更多样本，但噪声标签增多，可能降低最终 mAP。
- **推荐**：0.35 作为起始写入阈值，在手工标注 val set 上观察 mAP 趋势后调整。

### 等间隔采样 vs. 差异采样

- 等间隔采样实现简单，结果确定，适合快速迭代；20 fps 视频以 N=20 采样，帧间距约 1 秒/15 米，已有足够空间多样性。
- 差异采样在路口、停车等低速场景下更有优势，可避免大量近似帧浪费计算资源；但阈值选择需要实验，初版不推荐作为默认策略。
- 推荐路径：先用等间隔采样完成全流程验证，再用差异采样优化数据集质量。

### YOLO26 NMS-free vs. 传统 YOLO

- YOLO26 端到端无 NMS，推理延迟更低、部署更简单，但训练时对超参数（尤其是 loss balancing）较敏感，官方已通过 Progressive Loss Balancing + STAL 改善了这一问题。
- 若 YOLO26 训练不稳定（val loss 持续震荡 > 10 epochs），可回退至 YOLO11s（更成熟的选择，Ultralytics 官方推荐用于大多数生产场景），其余流程不变。

### Fine-tuning 层冻结策略

- **冻结 backbone**（`freeze=10`）：只训练 neck + head，收敛快，训练数据少或伪标签质量不确定时防止过拟合；适合阶段四初版。
- **全网络微调**（`freeze=0`）：需要更多数据和更长训练，适合标签质量经手工验证的后期迭代轮次，或数据集扩充至 1000+ 帧后。
- 如需在两者之间做选择，可对比 `freeze=10` 和 `freeze=0` 在相同手工 val set 上的 mAP，选择更优者。

---

## 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 伪 GT 包含大量假阳性（非树 bbox，如路灯、行人） | 中 | 高 | 提高写入阈值至 0.40；人工抽检 30 张帧可视化；降低 NMS IoU 阈值至 0.35 |
| 行车视频场景变化剧烈（遮挡、运动模糊、夜间） | 高 | 中 | 仅对日间清晰帧生成伪标签（可按帧亮度过滤：`mean(gray) < 50` 则跳过）；加入数据增强（blur、brightness jitter） |
| 伪 GT 漏检远处小树 | 高 | 低 | 适当降低 box_threshold 至 0.25；使用 `imgsz=1280` 训练提升小目标检测；可补充差异采样保留场景切换帧 |
| YOLO26 训练不稳定（loss spike/diverge） | 低-中 | 高 | 显式指定 `optimizer="AdamW"`, `lr0=1e-3`；降低 `batch` 至 8；回退至 YOLO11s |
| 帧间标签泄漏（相似帧跨 train/val） | 高（若随机划分） | 高 | **必须**按时间顺序划分 train/val，禁止随机 shuffle 后再切割 |
| 训练 OOM（RTX 3060 12GB + imgsz=1280） | 低 | 低 | batch=2 已验证安全（实测 rect 1280×736 占用 ~7GB）；若仍 OOM 改用 `yolo26n` 或降至 `imgsz=640` |
| 无视频帧提取脚本（当前状态） | 确定 | 中 | 按阶段零新建 `scripts/extract_video_frames.py`，约 50 行代码 |
| 4 个视频 GDINO 推理耗时长（~15–30 分钟） | 确定 | 低 | 提前确认显存（bf16 推理约 4–6 GB），使用 `--batch` 参数最大化吞吐；可用 `--shard N/M` 并行 |
| 原始视频实际 fps 未知（可能非 20fps） | 中 | 中 | 帧提取前必须运行 ffprobe 确认，并相应调整采样间隔 N |
| 跨日期场景分布差异导致 val mAP 偏低 | 中 | 低 | 跨日期划分是有意为之（测试泛化性）；mAP 偏低时检查两个日期的标注质量是否一致 |
| data.yaml 路径错误导致训练失败 | 中 | 低 | `path` 字段使用绝对路径；训练前用 `model.val(data=...)` 预验证数据集格式 |

---

## 现有代码复用关系

| 现有组件 | 在本方案中的作用 | 改动幅度 |
|----------|-----------------|---------|
| `tdus_gdino_dinov2_svm.ipynb` | Grounding DINO 推理参数参考（score_thr=0.3、text="tree."、bf16 推理、OOM 自动 microbatch 降级） | 仅参考，无需修改 |
| `gdino_dinov2_svm/03_extract_embeddings.py` | **核心复用**：gdino_preprocess、collate_fn、post_process bbox 坐标还原逻辑，改造为伪标签生成器 | 中等改动（约 100 行新增逻辑）|
| `extract_embeddings_gdino.py` | 分片推理 (`--shard N/M`) 逻辑，适合大规模帧集并行处理 | 仅参考 |
| `validate_gdino_bbox_crop.py` | bbox 可视化与质量验证，直接复用于伪 GT 抽检（annotate_image + make_crop） | 无需修改 |
| `gdino_bbox_validation/` | 已有的 bbox 评估结果，提供 conf 阈值选择的经验参考（0.3 在 TDUS 上已能 100% 召回） | 无需修改 |

**需新建的文件**：

| 文件 | 用途 |
|------|------|
| `scripts/extract_video_frames.py` | 阶段零：视频帧提取（uniform/diff 两种策略） |
| `scripts/generate_pseudo_labels.py` | 阶段二：Grounding DINO 批量推理 → YOLO txt 标注 |
| `scripts/split_dataset.py` | 阶段三：按时间顺序划分 train/val，生成标准 YOLO 目录结构 |
| `scripts/train_yolo26.py` | 阶段四：YOLO26 fine-tune 入口 |
| `scripts/validate_yolo26.py` | 阶段四（验证）：加载 best.pt，输出 mAP/precision/recall；提供推理示例 |
| `scripts/visualize_pseudo_labels.py` | 阶段二（质检）：随机抽取 20–30 帧可视化伪 GT，复用 `validate_gdino_bbox_crop.py` |
| `data/pseudo_labels/.../data.yaml` | YOLO26 数据集配置（含绝对路径） |

---

## 参考资料

- [Ultralytics YOLO26 官方文档](https://docs.ultralytics.com/models/yolo26)
- [YOLO26 Fine-Tuning 指南 (Ultralytics)](https://docs.ultralytics.com/guides/finetuning-guide)
- [YOLO26 Training Recipe (Ultralytics)](https://docs.ultralytics.com/guides/yolo26-training-recipe)
- [Ultralytics Model Training 参数参考](https://docs.ultralytics.com/modes/train)
- [Why YOLO26 Removes NMS (Ultralytics Blog)](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment)
- [YOLO26 发布公告 (BusinessWire)](https://www.businesswire.com/news/home/20260114168538/en/Ultralytics-Launches-YOLO26-Setting-a-New-Global-Standard-for-Edge-First-Vision-AI)
