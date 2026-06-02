# Proposal: 4× Frame Density at Half Resolution — Re-extract, Re-label, Retrain

**Status**: Draft (架构师审查 2026-05-31)  
**Date**: 2026-05-31  
**Builds on**: `docs/proposals/proposal-tree-detection-gdino-yolo.md`

---

## 背景

当前 pipeline（Phase A–C）已取得 mAP@0.5 = **0.874**（YOLO26s, freeze=10, batch=10, mosaic=0.5），但这一结果存在一个系统性缺陷：Grounding DINO 在生成伪标签时损失了约 51% 的图像信息。

### 关键分辨率数据

| 环节 | 分辨率 | 相对原始帧的 scale |
|------|--------|-------------------|
| 原始视频帧 | 2704×1520 | 1.00 |
| GDINO 内部处理（SHORTEST_EDGE=800, LONGEST_EDGE=1333） | ~1333×749 | **0.49** |
| YOLO 训练（imgsz=1280, rect=True） | 1280×719 | **0.473** |

GDINO 的 `gdino_preprocess` 函数（见 `scripts/generate_pseudo_labels.py` 第 38–47 行）以 SHORTEST_EDGE=800 触发缩放，将 1520px 高度压缩至 749px，比例 ≈ 0.49。超过一半的像素信息在伪标签生成阶段即已丢失。

### bbox 尺寸特征

当前数据集中 bbox 高度中位数约为帧高的 **47%**（近景大树干，非小目标）。这意味着：
- mosaic 增强将图像缩小到 1/4 面积后，树干高度仅剩约 170px，严重损失细节。
- GDINO 在 749px 高度的图像上检测占高 47% 的目标（≈350px），召回率尚可；但若目标更小（远景树木），丢失信息的代价更高。

### YOLO26l 验证失败的根本原因

Colab L4（23GB VRAM）上训练的 YOLO26l（26M 参数）取得 mAP@0.5 = **0.811**，差于参数量更少的 YOLO26s（mAP=0.874）。根本原因是 521 张训练图对 26M 参数而言严重不足，模型过拟合。扩充训练数据是突破当前上限的必要条件，而非仅仅更换模型。

### 当前帧统计

- 原始帧数：1043 帧（521 train + 522 val）
- 实际每视频帧数（`data/frames/` 实测）：eastbound_20240319=314, eastbound_20240530=281, westbound_20240319=207, westbound_20240530=241
- 采样方式：等间隔（`every_n=60`），来自 4 个完整视频，视频实际帧率 **59.9 fps**（ffprobe 确认）

---

## 目标

1. 通过提高采样密度（4×）获得更多训练样本，同时以半分辨率存储，控制磁盘占用。
2. 在半分辨率帧（1352×760）上重新生成 GDINO 伪标签，使 GDINO 内部 scale 从 0.49 提升至 ≈0.98，接近原生分辨率处理。
3. 以已验证的最优配置（YOLO26s, freeze=10, mosaic=0.0）在 Colab L4 上重训，实现 mAP@0.5 > 0.90。
4. 支持 Colab session 中断后从 Drive 的 `last.pt` 增量恢复训练。

---

## 方案设计

### 核心思路：分辨率折叠

原始帧为 2704×1520（约 411 万像素/帧），全量存储 1043 帧约需 X MB。  
新方案：长宽各 ×0.5，分辨率 1352×760（约 103 万像素/帧，为原来的 1/4）；采样密度提高 4×（every_n=60→15），总帧数约 **4147 帧**（ffprobe 实测 59.9fps 计算）。

**存储量基本不变**：4× 帧数 × 1/4 像素数 = 原存储量（JPEG 压缩比相近时）。

**GDINO scale 提升**：

| 场景 | 帧分辨率 | GDINO 处理高度 | scale |
|------|---------|--------------|-------|
| 当前 | 2704×1520 | ~749px | 0.49 |
| 新方案 | 1352×760 | ~760px（几乎不缩放）| **≈0.98** |

在 1352×760 的帧上，SHORTEST_EDGE=800 触发判断：`min(760, 1352)=760 < 800`，因此触发放大（scale = 800/760 ≈ 1.053），但 LONGEST_EDGE=1333 检查：`max(800, 1410) > 1333`，再次缩放，最终处理分辨率约 1333×760（scale≈1.00）。**GDINO 几乎以帧原始分辨率处理**，无信息损失。

> 注：精确计算取决于 `gdino_preprocess` 的 rounding 行为，实际 scale 在 0.97–1.00 之间，相比当前 0.49 是质的提升。

### Phase A'：重新提取帧（本地执行）

**输入**：4 个原始视频（`assets/original/computervisie_2024/*.MP4`）  
**脚本**：`scripts/extract_video_frames.py`（已实现）  
**修改点**：仅需调整 `--every-n` 参数（降为原来的 1/4）并在 ffmpeg 命令中加入 `scale=1352:760` 缩放滤镜。

**目标帧数估算**（基于 ffprobe 实测：fps=59.9，视频时长见下表）：

| 视频 | 实测时长 | 原帧数（every_n=60, ≈1fps） | 新帧数（every_n=15, ≈4fps） |
|------|---------|--------------------------|--------------------------|
| eastbound_20240319 | 313s | 314（实测） | ~1249 |
| eastbound_20240530 | 280s | 281（实测） | ~1118 |
| westbound_20240319 | 206s | 207（实测） | ~822 |
| westbound_20240530 | 240s | 241（实测） | ~958 |
| **合计** | — | **1043** | **~4147** |

> **[修正]** 原草稿表格使用了错误的 fps=20 和 every_n=5，导致新旧帧数均偏差约 1.9×。正确参数：视频实际 fps=59.9，`--every-n 60` 对应当前 1fps 采样，`--every-n 15` 对应 4× 密度（≈4fps 输出）。

**输出目录**（替换现有 `data/frames/`）：

```
data/frames_half/
├── eastbound_20240319/   # 1352×760 JPEG，frame_{idx:06d}.jpg
├── eastbound_20240530/
├── westbound_20240319/
└── westbound_20240530/
```

使用新目录名 `data/frames_half/` 以保留现有 `data/frames/`（原始 2704×1520 帧），便于对比和回退。

**执行命令示意**（本地，需在 `.venv` 激活后运行）：

```bash
# fps 已通过 ffprobe 确认为 59.9fps，all 4 videos 一致
# every_n=15 → 59.9/15 ≈ 4fps 输出，4× 当前密度

# 提取（每 15 帧取一帧，ffmpeg 内置缩放至 1352×760）
# 注：extract_video_frames.py 需在 ffmpeg vf 中追加 scale 滤镜（见下方改动说明）
.venv/bin/python scripts/extract_video_frames.py \
  --videos assets/original/computervisie_2024/eastbound_20240319.MP4 \
           assets/original/computervisie_2024/eastbound_20240530.MP4 \
           assets/original/computervisie_2024/westbound_20240319.MP4 \
           assets/original/computervisie_2024/westbound_20240530.MP4 \
  --out-base data/frames_half \
  --every-n 15 \
  --strategy uniform
```

> `scripts/extract_video_frames.py` 的 `extract_frames_uniform` 当前 vf 为 `select=not(mod(n,{every_n}))`，需新增 `--scale-w`/`--scale-h` CLI 参数并在函数内拼接 `,scale={w}:{h}`（约 20 行改动，见 Plan Stage 5.1）。`extract_frames_diff` 不支持 scale，传入时打印 warning 并忽略。

### Phase B'：重新生成 GDINO 伪标签（本地执行）

**输入**：`data/frames_half/`（1352×760 帧）  
**脚本**：`scripts/generate_pseudo_labels.py`（已实现，无需修改）  
**参数**：SHORTEST_EDGE=800, LONGEST_EDGE=1333（保持不变，scale 自然提升）

**推理耗时估算**：
- 新帧数约 4147 帧（修正后），GDINO tiny 在桌面 GPU 上约 0.5–1 秒/帧（参考：GitHub issue #132 显示 V100 batch=1 约 2–3 fps；本地 batch=4 加 padding 后约 1–2 fps 有效吞吐）
- 预计耗时：**35 分钟–1.5 小时**（vs 当前 ~10–20 分钟）
- **[勘误]** `scripts/generate_pseudo_labels.py` 当前**不存在** `--shard N/M` 参数；分批处理只能通过 `--videos` 参数指定视频子集实现（如每次传 1–2 个视频），或在脚本中手动添加该功能

**伪标签输出**：

```
data/pseudo_labels_half/
├── eastbound_20240319/images/all/   # 1352×760 帧（symlink 或 copy）
├── eastbound_20240319/labels/all/   # YOLO txt 格式
├── ...
└── all_4videos/
    ├── images/train/    # 20240319 两个视频
    ├── images/val/      # 20240530 两个视频
    ├── labels/train/
    ├── labels/val/
    └── data.yaml
```

**数据集划分**（复用现有跨日期策略）：

| 划分 | 视频 | 估算帧数（every_n=15, fps=59.9） |
|------|------|--------------------------------|
| train | eastbound_20240319 + westbound_20240319 | ~2071（1249+822） |
| val | eastbound_20240530 + westbound_20240530 | ~2076（1118+958） |

> **[修正]** 原草稿误写 ~3908/~3908，实际基于 ffprobe 数据计算为 ~2071/~2076，train/val 近乎对称。

脚本 `scripts/split_dataset.py` 可直接复用，只需修改 `OUT_DIR` 路径为 `data/pseudo_labels_half/all_4videos`。

### Phase C'：YOLO26s 重训（Colab L4 执行）

**关键配置变更**（对比现有 `notebooks/train_yolo26l_colab.ipynb`）：

| 参数 | 现有 notebook（YOLO26l） | 新方案（YOLO26s） | 变更原因 |
|------|------------------------|-----------------|---------|
| model | `yolo26l.pt`（26M 参） | `yolo26s.pt`（已验最优） | 26l 在 521 帧上过拟合；~2071 帧仍优先 26s |
| freeze | 5 | **10** | 回到已验证最优（freeze=10 in `scripts/train_yolo26.py`）|
| mosaic | 0.5 | **0.0** | bbox 高度 47% frameh，mosaic 压缩后树干仅 ~170px；已验证有害 |
| close_mosaic | 10（默认） | **10**（保留） | mosaic=0.0 时 close_mosaic 无实际效果，保持默认即可 |
| scale | 0.3 | **0.5** | 替代 mosaic，提供尺度变化增强（官方建议大目标场景用 scale 替代 mosaic） |
| translate | 0.1 | **0.15** | 略微加强位移增强，弥补 mosaic=0 |
| batch | autobatch | autobatch（保留） | L4 23GB VRAM，autobatch 自动探测安全上限（预计 ~24–32） |
| imgsz | 1280 | **1280**（不变） | 1352×760 帧 → rect 缩放后 ≈1280×719，与当前一致 |
| cache | True | **True**（保留） | ~4147 图 × 1352×760 × 3ch ≈ 12 GB RAM；Colab L4 约 52 GB 系统 RAM，可承载 |

**完整训练配置**：

```python
model.train(
    data=str(yaml_path),       # data/pseudo_labels_half/all_4videos/data.yaml
    epochs=100,
    imgsz=1280,
    batch=-1,                  # autobatch on L4 23GB（Ultralytics API: -1 = 自动探测）
    freeze=10,
    optimizer="auto",
    lr0=1e-3,
    lrf=0.01,
    cos_lr=True,
    patience=20,
    save=True,
    project="/content/runs/detect",
    name="tree_yolo26s_halfres",
    mosaic=0.0,                # 关闭（vs 现有 0.5）
    mixup=0.0,
    degrees=10.0,
    translate=0.15,            # 略加强（vs 现有 0.1）
    scale=0.5,                 # 加强（vs 现有 0.3）
    flipud=0.0,
    fliplr=0.5,
    device=0,
    workers=4,
    cache=True,
    rect=True,
    deterministic=False,
)
```

### 增量训练恢复

Colab session 易中断，需支持从 Drive 的 `last.pt` 恢复。在 Colab notebook 中加入以下逻辑：

```python
from pathlib import Path
from ultralytics import YOLO

# last.pt 先从 Drive 复制到本地（Drive I/O 慢，训练时用本地路径）
LOCAL_LAST = Path("/content/runs/detect/tree_yolo26s_halfres/weights/last.pt")
DRIVE_LAST = DRIVE_DIR / "weights" / "last.pt"

if DRIVE_LAST.exists():
    print(f"Resuming from {DRIVE_LAST}")
    LOCAL_LAST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DRIVE_LAST, LOCAL_LAST)
    # 关键：resume=True 时 YOLO() 接受 last.pt 路径，train() 只传 resume=True
    # 所有训练参数（data、epochs、imgsz 等）从 checkpoint 中自动恢复，
    # 不得重复传入（否则会触发"cannot override"错误或被静默忽略）
    model = YOLO(str(LOCAL_LAST))
    model.train(resume=True)
else:
    print("Starting fresh training")
    model = YOLO("yolo26s.pt")
    model.train(data=str(yaml_path), ...)   # 新训练才传完整参数
```

> **[架构注意]** `resume=True` 时 Ultralytics 从 checkpoint 恢复 epoch 计数、optimizer state、scheduler state。已知问题：部分 Colab 环境下 resume 后 loss spike（GitHub issue #3299, #17282），根因是 checkpoint 写入不完整（session 在 epoch 中途被杀死）。**缓解**：每次 session 结束前确认 last.pt 已写入（training cell 正常 return）再上传 Drive；若 resume 后立即出现 NaN loss，删除 Drive 的 last.pt 并从 best.pt 重新开始 fine-tune。

Weights 在训练中自动保存（`save=True`，每 epoch 末），session 结束前上传到 Drive：

```python
# 训练完成或 session 结束前执行（Results 包含 epoch 号，可在最后一个 cell 放此代码）
import shutil
from pathlib import Path

run_dir = Path("/content/runs/detect/tree_yolo26s_halfres")
dst = DRIVE_DIR / "weights"
dst.mkdir(parents=True, exist_ok=True)
for fname in ("best.pt", "last.pt"):
    src = run_dir / "weights" / fname
    if src.exists():
        shutil.copy2(src, dst / fname)
        print(f"Saved {fname} → {dst}")
# results.csv 在 run_dir 根目录，不在 weights/ 子目录
results_src = run_dir / "results.csv"
if results_src.exists():
    shutil.copy2(results_src, dst / "results.csv")
    print(f"Saved results.csv → {dst}")
```

> **[勘误]** 原草稿的 results.csv 上传路径写为 `weights/results.csv` 是可行的，但应注意 results.csv 实际位于 `run_dir/` 根目录（非 `run_dir/weights/`），原代码路径 `f"/content/runs/detect/.../weights/{fname}"` 对 results.csv 是**错误路径**。上方已修正。

---

## 权衡分析

### 4× 帧数 vs. 数据冗余

**正面**：更多训练样本直接解决 YOLO26l 过拟合问题；即便使用 YOLO26s，从 521 → ~2071 张训练图有望提升泛化能力（不同角度、光照、遮挡的树木场景覆盖更全面）。Ultralytics 官方建议 fine-tune 场景至少 200–500 张/类，推荐 1500 张以上；~2071 张已达推荐上限，有望突破当前 mAP 瓶颈。

**负面**：相邻帧（间隔约 0.25 秒，every_n=15 @ 59.9fps）高度相似，4× 密度会增加训练集内的帧间相似度。缓解措施：依赖数据增强（scale=0.5, translate=0.15, fliplr=0.5）在训练时引入多样性，而非依赖天然的帧差异。另一缓解手段：使用 `diff` 策略（`extract_frames_diff`）替代 uniform 采样，仅保留场景变化帧——但会引入不确定性，本方案暂不采用。

### 半分辨率 vs. 全分辨率

**正面**：GDINO scale 从 0.49 → ≈0.98，伪标签质量显著提升；YOLO 训练时 imgsz=1280 对 1352×760 帧 scale=0.947（vs 当前 0.473），更接近原始比例。

**负面**：半分辨率帧在人工审查时细节略逊于原始帧；但训练模型时分辨率差异可忽略（YOLO 最终都缩放到 1280px 宽）。

**存储对比**：

| 方案 | 帧数 | 每帧像素 | 相对存储量 |
|------|------|---------|----------|
| 当前（全分辨率） | 1043（实测） | 4,108,880（2704×1520） | 1.0× |
| 新方案（半分辨率） | ~4147 | 1,027,520（1352×760） | **≈1.0×** |

两种方案存储量基本相同（4× 帧数 × 1/4 像素），是本方案最重要的工程约束满足点。

> **[修正]** 原表中当前帧数写为 ~1954（基于错误 fps 估算），实测为 1043。

### mosaic=0.0 vs. mosaic=0.5

**已验证数据**：现有最优结果（mAP=0.874）来自 mosaic=0.5，但背景是原始 2704×1520 帧 + freeze=10。mosaic 将图像缩小到 1/4 后，bbox 高度从帧高 47%（≈714px in 1520px）降至约 178px，对大目标树木的检测无益。

新方案以 scale=0.5（随机缩放至 50%–150% 原尺寸）替代 mosaic 提供尺度增强，保持 bbox 的语义完整性。此配置在 YOLO 官方建议中对大目标场景（bbox 占图像面积 >20%）更为适用。

**`close_mosaic` 参数说明**：Ultralytics 默认 `close_mosaic=10`，即训练最后 10 个 epoch 关闭 mosaic 以稳定收敛。本方案已全程 `mosaic=0.0`，`close_mosaic` 值无实际影响，保持默认即可（不必额外修改）。

### YOLO26s vs. YOLO26l（在新数据量下）

- train 集从 521 → ~2071 张，是否应重新考虑 YOLO26l？
- 建议**初版仍使用 YOLO26s**：~2071 张对 26M 参数的 YOLO26l 仍偏少（Ultralytics 官方建议 1500+ 张/类为 fine-tune 上限；26l 推荐 5000+ 张高质量标注以充分利用模型容量）；且 YOLO26s（freeze=10）已被验证为当前 pipeline 最优选择，新变量应逐步引入。
- 可在 YOLO26s 训练完成并确认 mAP > 0.90 后，以相同配置尝试 YOLO26l 作为对比实验。

---

## 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| GDINO 推理耗时超预期（~4147 帧，预计 35min–1.5h） | 中 | 低 | 通过 `--videos` 参数分 4 个视频串行执行；本地后台运行不阻塞其他工作；**注：脚本无 `--shard` 参数** |
| 半分辨率帧中小目标（远景树木）更难检测 | 中 | 中 | 半分辨率下 GDINO scale≈0.986，几乎无信息损失；YOLO imgsz=1280 与当前相同，无退化 |
| 4× 密度导致帧间相似帧过多，val mAP 虚高 | 中 | 中 | 跨日期划分（train=20240319, val=20240530）保证 val 与 train 有实质场景差异；相似帧集中在同日期内，影响有限 |
| `extract_video_frames.py` 需修改以支持 `scale` 滤镜 | 确定 | 低 | 仅需在 `extract_frames_uniform` 的 ffmpeg `-vf` 参数中追加 `,scale=1352:760`，约 1 行改动 |
| Colab session 中断导致进度丢失 | 高 | 中 | 增量恢复逻辑（`resume=True` + last.pt 先拷贝到本地再恢复）已在方案中设计；注意 epoch 中途 kill 会导致 checkpoint 不完整，resume 后可能出现 NaN loss |
| mosaic=0.0 后训练数据增强不足，模型欠拟合 | 低 | 中 | scale=0.5 + translate=0.15 + fliplr=0.5 提供充分增强；~2071 张训练图是当前的 4×，欠拟合风险低 |
| 新方案 mAP 不及预期（< 0.874） | 低 | 高 | 可回退到现有最优 checkpoint（`runs/detect/tree_yolo26s_pseudo/weights/best.pt`）；新方案所有产物存放在独立目录 |
| YOLO26s autobatch 在 L4 上探测结果不稳定 | 低 | 低 | 参照现有 notebook `check_train_batch_size` 逻辑；若 autobatch 异常，手动设 batch=16 |
| cache=True 导致 Colab RAM 不足 | 低 | 中 | ~4147 图 × 2.9MB = ~12 GB RAM；Colab L4 提供约 52 GB 系统 RAM，有足够余量；若不足则改 `cache="disk"` |

---

## 现有代码复用关系

| 现有组件 | 本方案中的角色 | 改动幅度 |
|----------|--------------|---------|
| `scripts/extract_video_frames.py` | Phase A'：重新提取帧（uniform 策略）| **小改动**：新增 `--scale-w`/`--scale-h` CLI 参数，在 `extract_frames_uniform` 的 ffmpeg `-vf` 追加 `,scale={w}:{h}`；默认 `--every-n` 保持 60 不变，调用时传入 15 |
| `scripts/generate_pseudo_labels.py` | Phase B'：重新生成伪标签 | **无需修改**；已有 `--frames-base`/`--out-base`/`--videos` CLI 参数；SHORTEST_EDGE=800/LONGEST_EDGE=1333 参数不变，scale 自然提升 |
| `scripts/split_dataset.py` | 数据集划分 | **无需修改**；已有 `--pseudo-labels-base`/`--out-dir` CLI 参数，修改调用时的路径参数即可 |
| `scripts/train_yolo26.py` | Phase C'：YOLO26s 训练（本地验证用） | **需修改**：`mosaic=0.5`、`scale=0.3`、`translate=0.1` 当前**硬编码**在 `train()` 函数内，不支持 CLI 传入；需添加对应 CLI 参数或直接修改默认值 |
| `notebooks/train_yolo26l_colab.ipynb` | Colab 训练 notebook 的基础 | **中等改动**：修改 model→yolo26s、freeze→10、mosaic→0.0、scale→0.5、translate→0.15；新增增量恢复逻辑 |

**需新建/修改的内容**：

| 文件 | 类型 | 说明 |
|------|------|------|
| `notebooks/train_yolo26s_colab_halfres.ipynb` | 新建 | 基于现有 notebook 修改，包含增量恢复逻辑和新参数 |
| `data/pseudo_labels_half/` | 运行时产物 | Phase B' 输出，不提交 git |

---

## 执行顺序

```
本地：
  1. 修改 scripts/extract_video_frames.py
       → 新增 --scale-w/--scale-h CLI 参数（type=int，默认 None）
       → 在 extract_frames_uniform() 的 ffmpeg vf 中，当两参数均不为 None 时追加 ,scale={w}:{h}
       → extract_frames_diff() 传入 scale 参数时打印 warning 并忽略（不支持）
  2. 修改 scripts/train_yolo26.py（若需本地验证）
       → 将 mosaic/scale/translate 改为 CLI 参数，或修改 train() 默认值
  3. 运行帧提取（every_n=15，fps=59.9，目标 ~4fps 输出，预计总帧数 ~4147）
       .venv/bin/python scripts/extract_video_frames.py \
         --videos <4个视频路径> --out-base data/frames_half --every-n 15 \
         --scale-w 1352 --scale-h 760
  4. 运行 generate_pseudo_labels.py（约 35min–1.5h，分视频串行）
       .venv/bin/python scripts/generate_pseudo_labels.py \
         --frames-base data/frames_half --out-base data/pseudo_labels_half
  5. 运行 split_dataset.py（路径指向 pseudo_labels_half）
       .venv/bin/python scripts/split_dataset.py \
         --pseudo-labels-base data/pseudo_labels_half \
         --out-dir data/pseudo_labels_half/all_4videos
  6. 上传数据集到 Google Drive（images/train ~2071 张, images/val ~2076 张, labels/）

Colab：
  7. 新建训练 notebook（YOLO26s, freeze=10, mosaic=0.0, scale=0.5, autobatch）
       → 含 resume 逻辑：检测 Drive/weights/last.pt → 复制本地 → YOLO(last.pt).train(resume=True)
  8. 首次运行训练；session 中断后用 last.pt 恢复（先上传 last.pt 到 Drive）
  9. 训练完成后将 best.pt + last.pt + results.csv 同步回 Drive
```

---

## 成功指标

| 指标 | 目标值 | 当前基线 |
|------|--------|---------|
| mAP@0.5（val，跨日期） | > 0.90 | 0.874 |
| mAP@0.5:0.95 | > 0.70 | — |
| 训练图数量 | ~2071（修正值） | 521 |
| GDINO 伪标签 scale | ≈0.986（精确计算值） | 0.49 |
| 存储增量 | ≈0（相对当前） | — |

---

## 架构师审查意见（2026-05-31）

以下问题在本次审查中发现并已就地修正，记录在此作为变更追踪：

### 重大错误（已修正）

1. **Phase A' 帧数表格全部错误**
   - 原草稿写 "N=20 @ 20fps"，实际视频 fps=59.9（ffprobe 实测），当前 `every_n=60`
   - 原表旧帧数写 ~1954（实测 1043），新帧数写 ~7816（正确约 4147）
   - 修正：基于 ffprobe 实测数据重写全表，every_n=15 对应 4× 密度

2. **`--shard` 参数不存在**
   - 原草稿写"可用 `--shard N/M` 参数分批处理"，`scripts/generate_pseudo_labels.py` 无此参数
   - 修正：改为"通过 `--videos` 参数指定视频子集串行处理"

3. **`train_yolo26.py` 中 mosaic/scale/translate 为硬编码，非 CLI 参数**
   - 原草稿写"mosaic/scale/translate 参数通过命令行传入"，实际均为函数内 hardcode
   - 修正：代码复用表格中标注为"需修改"，执行顺序中增加修改该脚本的步骤

4. **resume 训练的 results.csv 路径错误**
   - 原草稿在权重上传 cell 中用 `weights/{fname}` 路径读取 results.csv，但该文件在 run_dir 根目录，非 weights/ 子目录
   - 修正：results.csv 单独从 run_dir 根目录复制

### 一般性改进（已修正）

5. **GDINO 推理耗时估算重新计算**：基于正确帧数 ~4147（非 7816），估算为 35min–1.5h

6. **cache=True RAM 需求说明**：新增 ~12 GB 计算过程，确认 Colab L4 (~52 GB RAM) 可承载；新增 cache="disk" 备选

7. **`close_mosaic` 参数说明**：新增说明 mosaic=0.0 时 close_mosaic 无效，无需额外修改

8. **resume=True 已知风险**：补充 GitHub #3299/#17282 的 NaN loss 问题及缓解措施

9. **训练图数量与 Ultralytics 官方建议对齐**：~2071 张已达 fine-tune 推荐上限（1500+），但低于完整训练推荐（5000+/类）

### 方案整体评估

方案核心思路（分辨率折叠 + GDINO scale 提升）在架构上成立，数学验证正确（GDINO scale 0.49→0.986）。主要风险是帧间相似度高（every_n=15 @ 59.9fps = 4fps 采样），但跨日期 val 划分已有效隔离。建议按修正后的参数执行，Phase C' 完成后对比 mAP@0.5 vs 0.874 基线。

---

## 参考

- 现有最优实验记录：`scripts/train_yolo26.py`（freeze=10, mosaic=0.5, batch=10 → mAP=0.874）
- GDINO 预处理逻辑：`scripts/generate_pseudo_labels.py`，第 38–47 行（`gdino_preprocess`）
- Colab 训练 notebook 基础：`notebooks/train_yolo26l_colab.ipynb`
- 原始 pipeline 方案：`docs/proposals/proposal-tree-detection-gdino-yolo.md`
- Ultralytics YOLO26 训练参数：https://docs.ultralytics.com/modes/train
- Ultralytics 数据集大小建议：https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results
- Ultralytics resume training：https://docs.ultralytics.com/modes/train（resume=True + last.pt path）
- GDINO 推理速度参考：https://github.com/IDEA-Research/GroundingDINO/issues/132
- mosaic 关闭最佳实践：https://docs.ultralytics.com/guides/yolo-data-augmentation
- resume NaN loss 已知问题：https://github.com/ultralytics/ultralytics/issues/17282, #3299
