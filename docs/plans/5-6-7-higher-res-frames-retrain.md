# Plan: 4× Frame Density at Half Resolution — Re-extract, Re-label, Retrain

**状态**: Completed  
**日期**: 2026-05-31  
**对应 Proposal**: `docs/proposals/proposal-higher-res-frames-retrain.md`  
**Phase 编号**: 5–7（接续 `4-tree-detection-gdino-yolo.md`）

---

## 训练结果摘要

| 指标 | 值 | 备注 |
|------|----|------|
| **最终 mAP50** | **0.8859** | epoch 39，best.pt 基于 fitness |
| **最终 mAP50-95** | **0.6502** | epoch 57（best.pt 对应 epoch） |
| **停止 epoch** | 77 | EarlyStopping patience=20，best fitness at epoch 57 |
| vs Phase 4（26l 全分辨率）| +0.075 | 0.886 vs 0.811 |
| vs Phase C（26s 本地）| +0.012 | 0.886 vs 0.874 |
| train frames | 2078 | eastbound/westbound 20240319 |
| val frames | 2082 | eastbound/westbound 20240530 |
| results.csv | `docs/results/tree_yolo26s_halfres/results.csv` | 已提交 |
| best.pt | Google Drive `TreeLearn/tree_yolo26s_halfres/weights/best.pt` | 20 MB，不入 git |

---

## 总体架构

```
Phase 5（本地）: extract_video_frames.py 改造 — 半分辨率帧提取
    └── Stage 5.1: --scale-w/--scale-h CLI 参数 + vf 滤镜扩展
    └── Stage 5.2: 半分辨率帧提取执行（every_n=15，data/frames_half/）

Phase 6（本地）: 伪标签重生成 + 数据集重组
    └── Stage 6.1: GDINO 伪标签重生成（执行，无需修改脚本）
    └── Stage 6.2: 数据集划分执行 + data.yaml 重生成（执行，无需修改脚本）

Phase 7（Colab）: YOLO26s 重训 notebook — 增量恢复版
    └── Stage 7.1: train_yolo26.py 训练超参 CLI 化
    └── Stage 7.2: notebooks/train_yolo26s_colab_halfres.ipynb — 新建/改造
    └── Stage 7.3: Drive 增量恢复逻辑（resume=True + last.pt 往返）
```

---

## 依赖关系

```
Phase 5 → Phase 6（half-res 帧是伪标签生成的输入）
Phase 6 → Phase 7（重组后的 data.yaml 是 Colab 训练的输入）

Stage 5.1 → Stage 5.2（先改脚本，再执行提取）
Stage 5.2 → Stage 6.1（帧目录存在才能验证 generate_pseudo_labels.py 路径）
Stage 6.1 → Stage 6.2（推理完成后才能划分数据集）
Stage 7.1（本地，可与 6.1/6.2 并行） → Stage 7.2
Stage 7.2 → Stage 7.3（notebook 骨架就绪后再加 resume 逻辑）

Phase 4 的产物（data/frames/、data/pseudo_labels/、runs/detect/tree_yolo26s_pseudo/）
  在本方案中保留不动，作为回退基线。
```

---

## 测试策略

- **倾向 TDD**：有新增/修改逻辑的 Stage 先写测试，再写实现；覆盖率目标 ≥ 80%
- **单元测试**：新增 CLI 参数解析、vf 滤镜字符串拼接逻辑、data.yaml 路径参数替换逻辑
- **集成测试**：对 1 个短视频片段（≤ 10 秒）跑通 Stage 5.1 + 5.2，验证帧尺寸为 1352×760
- **测试工具**：`pytest`（`.venv/bin/pytest`）；测试文件复用/补充 `tests/` 目录现有文件
- **不需要单元测试的 Stage**：
  - Stage 5.2（纯执行步骤，由 Stage 5.1 的集成测试间接覆盖）
  - Stage 6.1（generate_pseudo_labels.py 无需修改，验证为手动执行）
  - Stage 7.3（notebook 中的 Drive I/O 逻辑，在 Colab 上手动验证）
- **Notebook 测试策略**（Stage 7.2/7.3）：Colab 环境无法本地 pytest；改用 cell-level
  断言（`assert isinstance(model, YOLO)`、`assert LOCAL_LAST.exists()`）替代 pytest

---

## Phase 5：半分辨率帧提取

**目标**：改造 `scripts/extract_video_frames.py`，支持在提取时对帧做缩放，以 `every_n=15`
提取 4× 密度的 1352×760 帧到 `data/frames_half/`。  
**执行环境**：本地  
**依赖**：无（Phase 5 为起点）  
**代码量估计**：≤ 80 行（修改量）  
**涉及文件**：
- `scripts/extract_video_frames.py`（修改）
- `tests/test_extract_video_frames.py`（补充测试）

---

### Stage 5.1：--scale-w/--scale-h CLI 参数 + vf 滤镜扩展

**目标**：在 `extract_frames_uniform` 中支持可选的 ffmpeg `scale` 滤镜，并通过 CLI
参数 `--scale-w` / `--scale-h` 传入目标分辨率。

**具体任务**：

1. 修改 `extract_frames_uniform(video_path, out_dir, every_n, scale_w=None, scale_h=None)`：
   - 当 `scale_w` 和 `scale_h` 均不为 None 时，在 select 滤镜后追加
     `,scale={scale_w}:{scale_h}` 生成完整 vf 字符串
   - 当任一参数为 None 时，保持原有行为（不追加 scale 滤镜）
   - 修改量：函数签名 1 行 + vf 拼接 3 行 + docstring 更新 ≤ 10 行
2. 修改 `extract_frames` 函数，将 `scale_w`/`scale_h` 透传给 `extract_frames_uniform`
   （`extract_frames_diff` 不支持 scale，传入时打印 warning 并忽略）
3. 在 `argparse` 段新增两个 CLI 参数：
   - `--scale-w`（`type=int`，默认 `None`，help="Output frame width for scale filter"）
   - `--scale-h`（`type=int`，默认 `None`，help="Output frame height for scale filter"）
   - 在 `main()` 中将 `args.scale_w` / `args.scale_h` 传入 `extract_frames`
4. 补充 `tests/test_extract_video_frames.py`：
   - 单元测试：传入 `scale_w=640, scale_h=360`，验证生成的 vf 字符串包含
     `scale=640:360`（mock subprocess.run，检查调用参数）
   - 单元测试：`scale_w=None` 时，vf 字符串不含 `scale`（不退化）
   - 集成测试（打 `@pytest.mark.integration` 标记）：对合成视频（720p，10 帧）以
     `scale_w=320, scale_h=180` 提取，验证输出 JPEG 宽高为 `320×180`
   - 单元测试：diff 策略传入 `scale_w` 时，stdout 包含 warning，不崩溃

**实现依据**：Proposal 方案设计 Phase A'（"`extract_frames_uniform` 的 ffmpeg `-vf`
参数中追加 `,scale=1352:760`"）；现有代码审查确认 vf 字符串在第 40 行组装。

**验收标准**：
- `pytest tests/test_extract_video_frames.py -m "not integration"` 全部通过，新增用例全绿
- `pytest tests/test_extract_video_frames.py -m integration` 通过（scale 后尺寸正确）
- `python scripts/extract_video_frames.py --help` 输出包含 `--scale-w` 和 `--scale-h`
- 现有 test 无回归（uniform/diff 策略原有用例继续通过）

**代码量估计**：≤ 40 行（脚本修改 ~20 行，测试新增 ~20 行）  
**执行环境**：本地（`.venv/bin/python` + `.venv/bin/pytest`）

---

### Stage 5.2：半分辨率帧提取执行

**目标**：以 `every_n=15, scale_w=1352, scale_h=760` 对全部 4 个视频执行帧提取，
产生 `data/frames_half/` 目录（~4147 帧）。

**具体任务**：

1. 执行提取命令（`.venv` 激活后运行）：
   ```bash
   .venv/bin/python scripts/extract_video_frames.py \
     --videos assets/original/computervisie_2024/eastbound_20240319.MP4 \
              assets/original/computervisie_2024/eastbound_20240530.MP4 \
              assets/original/computervisie_2024/westbound_20240319.MP4 \
              assets/original/computervisie_2024/westbound_20240530.MP4 \
     --out-base data/frames_half \
     --every-n 15 \
     --strategy uniform \
     --scale-w 1352 \
     --scale-h 760
   ```
2. 执行完毕后运行质量检查（复用 Phase A 检查脚本逻辑，路径改为 `frames_half`）：
   - 统计 4 个子目录帧数，与预期值对比（`±10%`）：
     - eastbound_20240319：~1249 帧
     - eastbound_20240530：~1118 帧
     - westbound_20240319：~822 帧
     - westbound_20240530：~958 帧
     - 合计：~4147 帧
   - 验证任意抽样帧尺寸为 `1352×760`（`cv2.imread` 检查 `img.shape[:2] == (760, 1352)`）
   - 验证无损坏帧（`cv2.imread` 返回 None 的数量 = 0）
3. **此 Stage 无代码修改**，纯执行步骤；若提取失败，回到 Stage 5.1 排查 vf 字符串

**放行条件**：4 个子目录合计帧数在预期值 ±10% 内，抽样尺寸验证通过，无损坏帧。

**执行环境**：本地（ffmpeg 调用，预计耗时 5–15 分钟）  
**代码量估计**：0 行（纯执行，不计入代码量）

---

### Phase 5 输出质量评价

| 检查项 | 方法 | 合格标准 | 不合格时处置 |
|--------|------|----------|-------------|
| **帧总数** | `ls data/frames_half/*/*.jpg \| wc -l` | 3730–4560（预期 4147 ±10%） | 检查 every_n 参数，重新运行 |
| **帧尺寸** | 随机抽 20 帧，`cv2.imread(p).shape` | 全部为 `(760, 1352, 3)` | 检查 vf 字符串中 scale 参数顺序（w:h） |
| **无损坏帧** | `cv2.imread` 批量读取返回 None 数 | = 0 | 删除损坏帧，对应视频重新提取 |
| **命名格式** | `ls data/frames_half/eastbound_20240319/ \| head -5` | `frame_000001.jpg`（6 位零填充） | 检查 ffmpeg output pattern |
| **目录结构** | `ls data/frames_half/` | 4 个子目录，名称与视频 stem 一致 | 检查 `--out-base` 参数 |

**Phase 5 放行条件**：上表全部通过，方可进入 Phase 6。

---

## Phase 6：伪标签重生成 + 数据集重组

**目标**：对 `data/frames_half/` 重新运行 GDINO 推理，生成 `data/pseudo_labels_half/`；
重新运行 `split_dataset.py` 生成新 `data.yaml`，供 Phase 7 训练使用。  
**执行环境**：本地  
**依赖**：Phase 5（`data/frames_half/` 已存在且通过质量评价）  
**代码量估计**：≤ 15 行（两个脚本均无需修改；仅新增 split_dataset 路径参数行为测试 ~15 行）  
**涉及文件**：
- `scripts/generate_pseudo_labels.py`（无需修改；已有 `--frames-base`/`--out-base`/`--videos`/`--box-threshold` CLI 参数）
- `scripts/split_dataset.py`（无需修改；已有 `--pseudo-labels-base`/`--out-dir` CLI 参数）
- `tests/test_pseudo_labels.py`（补充 split_dataset 路径参数化行为验证测试）

---

### Stage 6.1：GDINO 伪标签重生成（执行步骤）

**目标**：验证 `generate_pseudo_labels.py` 现有 CLI 已支持 `--frames-base` 和
`--out-base` 参数；执行对 `data/frames_half/` 的推理。

**前置验证**（无代码修改，仅确认；实测已全部满足）：
1. 运行 `python scripts/generate_pseudo_labels.py --help`，确认以下参数存在（均已实现）：
   - `--frames-base`（输入帧根目录，默认 `data/frames`）
   - `--out-base`（输出伪标签根目录，默认 `data/pseudo_labels`）
   - `--videos`（可选，用于分批串行处理）
   - `--box-threshold`（GDINO 检测置信度阈值，默认 0.30）
   - `--score-thr`（后处理 NMS 前分数过滤阈值，默认 0.35；调低可提升召回率）
   - `--batch-size`（默认 4）、`--device`（默认 cuda）
   
   > 所有参数已存在，**无需任何代码修改**

**执行命令**（分 4 个视频串行，避免 OOM）：
```bash
for VIDEO in eastbound_20240319 eastbound_20240530 westbound_20240319 westbound_20240530; do
  .venv/bin/python scripts/generate_pseudo_labels.py \
    --frames-base data/frames_half \
    --out-base data/pseudo_labels_half \
    --videos $VIDEO
done
```

> 预计耗时：35 分钟–1.5 小时（~4147 帧，本地 GPU，分 4 轮串行）  
> 执行期间可在另一终端并行进行 Stage 7.1

**执行完毕后质量检查**：
```bash
python -c "
import glob, os, statistics
for video in ['eastbound_20240319','eastbound_20240530','westbound_20240319','westbound_20240530']:
    labels = glob.glob(f'data/pseudo_labels_half/{video}/labels/all/*.txt')
    total = len(labels)
    if total == 0: print(f'{video}: 无标注文件'); continue
    nonempty = sum(1 for p in labels if os.path.getsize(p) > 0)
    counts = [sum(1 for _ in open(p)) for p in labels if os.path.getsize(p) > 0]
    mean_c = statistics.mean(counts) if counts else 0
    print(f'{video}: {total} 帧, 覆盖率={nonempty/total*100:.1f}%, bbox均值={mean_c:.1f}')
"
```

**放行条件**：4 个视频各自标注覆盖率 ≥ 50%，坐标值均在 [0, 1] 内。

**代码量估计**：0 行（脚本无需修改；纯执行步骤）  
**执行环境**：本地（`.venv/bin/python`，GDINO tiny bf16 推理）

---

### Stage 6.2：split_dataset.py 数据集划分执行 + data.yaml 重生成

**目标**：`scripts/split_dataset.py` 已支持 `--pseudo-labels-base` 和 `--out-dir`
CLI 参数（当前代码实测：`argparse` 第 74–75 行），**无需修改**；直接以新路径执行划分，
生成 `data/pseudo_labels_half/all_4videos/data.yaml` 供 Phase 7 训练使用。

**具体任务**：

1. 确认脚本 CLI 参数（一次性验证，无代码修改）：
   ```bash
   .venv/bin/python scripts/split_dataset.py --help
   # 应输出 --pseudo-labels-base 和 --out-dir 两个参数
   ```
2. 补充 `tests/test_pseudo_labels.py::TestSplitPathParam`（验证现有参数行为）：
   - 用 `tmp_path` 构造 `pseudo_labels_half` 目录结构（4 视频各 5 帧）
   - 验证 `--out-dir` 传入自定义路径时，data.yaml 中 `path` 为该路径的绝对值
   - 验证 `--pseudo-labels-base` 传入时，正确读取该目录下的图像和标注
3. 执行划分命令：
   ```bash
   .venv/bin/python scripts/split_dataset.py \
     --pseudo-labels-base data/pseudo_labels_half \
     --out-dir data/pseudo_labels_half/all_4videos
   ```
4. 验证生成的数据集结构：
   ```
   data/pseudo_labels_half/all_4videos/
   ├── images/
   │   ├── train/    # eastbound_20240319 + westbound_20240319 帧（~2071 张）
   │   └── val/      # eastbound_20240530 + westbound_20240530 帧（~2076 张）
   ├── labels/
   │   ├── train/
   │   └── val/
   └── data.yaml     # path 为绝对路径，nc=1，names=['tree']
   ```
5. 运行 ultralytics 兼容性验证：
   ```bash
   .venv/bin/python -c "
   from ultralytics import YOLO
   YOLO('yolo26n.pt').val(data='data/pseudo_labels_half/all_4videos/data.yaml', epochs=0)
   "
   ```

**验收标准**：
- `pytest tests/test_pseudo_labels.py::TestSplitPathParam` 全部通过（若有修改）
- train 目录帧数约 2071（允许 ±5%），val 目录帧数约 2076（允许 ±5%）
- `data.yaml` 中 `path` 以 `/` 开头（绝对路径）
- ultralytics 兼容性验证无 KeyError 或路径错误

**代码量估计**：≤ 15 行（split_dataset.py 无需修改；测试新增 ~15 行验证现有行为）  
**执行环境**：本地（`.venv/bin/python`）

---

### Phase 6 输出质量评价

| 检查项 | 方法 | 合格标准 | 不合格时处置 |
|--------|------|----------|-------------|
| **GDINO scale 提升验证** | 在推理日志中检查输入图像尺寸（若有输出）| 输入到 GDINO 的图像短边 ≈ 760px（vs 当前 749px），长边 ≤ 1333px | 检查 gdino_preprocess 函数参数是否与现有一致 |
| **标注覆盖率** | 4 个视频各自 `nonempty_txts / total_txts` | ≥ 50% | 优先降低 `--score-thr`（0.35 → 0.28，post-NMS 过滤）；若仍不足再降低 `--box-threshold`（0.30 → 0.25，GDINO 检测阈值）重新推理 |
| **train/val 帧数** | `ls data/pseudo_labels_half/all_4videos/images/{train,val}/ \| wc -l` | train ≈ 2071，val ≈ 2076（±5%） | 检查 split_dataset.py 划分逻辑 |
| **data.yaml 路径** | `grep ^path data/pseudo_labels_half/all_4videos/data.yaml` | 以 `/` 开头的绝对路径 | 重新运行 split_dataset.py，确认 `--out-dir` 为绝对路径 |
| **ultralytics 兼容** | 见 Stage 6.2 验证命令 | 无报错 | 对照 YOLO26 数据集文档修正 data.yaml |
| **bbox 视觉质量** | `visualize_pseudo_labels.py`，各视频随机抽 5 帧 | ≥ 70% bbox 覆盖树冠，误检率 < 20% | 提高 score_thr（0.35 → 0.40）或降低 NMS IoU |

**Phase 6 放行条件**：覆盖率 ≥ 50%，train/val 帧数在预期范围内，ultralytics 兼容，
视觉质量抽查通过，方可进入 Phase 7（数据上传 Drive 并在 Colab 执行）。

---

## Phase 7：YOLO26s 重训 notebook（Colab L4 执行）

**目标**：
1. 改造 `scripts/train_yolo26.py`，使训练超参（`mosaic`/`scale`/`translate`）可通过 CLI 传入，支持本地验证新配置。
2. 新建 `notebooks/train_yolo26s_colab_halfres.ipynb`，包含完整的增量恢复逻辑（从 Drive 读取 `last.pt`、resume、训练后写回 Drive）。

**执行环境**：Stage 7.1 本地；Stage 7.2/7.3 Colab L4  
**依赖**：Phase 6（`data/pseudo_labels_half/all_4videos/data.yaml` 及数据集已上传 Drive）  
**代码量估计**：≤ 160 行（跨三个 Stage；见汇总表：7.1 约 40 行，7.2/7.3 约 120 行）  
**涉及文件**：
- `scripts/train_yolo26.py`（修改）
- `tests/test_train_yolo26.py`（补充）
- `notebooks/train_yolo26s_colab_halfres.ipynb`（新建）

---

### Stage 7.1：train_yolo26.py 训练超参 CLI 化

**目标**：将 `train_yolo26.py` 中硬编码的 `mosaic=0.5`、`scale=0.3`、`translate=0.1`
改为 CLI 参数，支持从命令行传入新配置，同时保留原有默认值（向后兼容）。

**具体任务**：

1. 在 `argparse` 段新增三个 CLI 参数：
   - `--mosaic`（`type=float`，默认 `0.5`，help="Mosaic augmentation probability"）
   - `--scale`（`type=float`，默认 `0.3`，help="Image scale augmentation range"）
   - `--translate`（`type=float`，默认 `0.1`，help="Image translate augmentation"）
2. 在 `train(args)` 函数中，将 `model.train(...)` 调用里的三行硬编码值替换为：
   - `mosaic=args.mosaic`
   - `scale=args.scale`
   - `translate=args.translate`
3. 补充 `tests/test_train_yolo26.py::TestTrainArgs`：
   - 验证默认值不变：`parse_args([])` 时 `args.mosaic == 0.5`、`args.scale == 0.3`、
     `args.translate == 0.1`
   - 验证可覆盖：`parse_args(['--mosaic', '0.0', '--scale', '0.5', '--translate', '0.15'])`
     后三个参数均正确解析
   - 用 `unittest.mock.patch` mock `YOLO.train`，验证 `train(args)` 确实将
     `args.mosaic`/`args.scale`/`args.translate` 传入 `model.train(...)` 调用（不漏传）
4. 在改动后手动验证 `--help` 输出包含新参数，原有训练逻辑无回归

**实现依据**：Proposal 现有代码复用表（"`train_yolo26.py`：mosaic/scale/translate 当前
硬编码，需修改"）；Phase C' 配置（`mosaic=0.0, scale=0.5, translate=0.15`）。

**验收标准**：
- `pytest tests/test_train_yolo26.py -m "not integration"` 全部通过（含新增用例）
- `python scripts/train_yolo26.py --help` 输出包含 `--mosaic`、`--scale`、`--translate`
- 原有默认行为（`mosaic=0.5, scale=0.3, translate=0.1`）不变（旧调用方式兼容）

**代码量估计**：≤ 40 行（脚本修改 ~15 行，测试新增 ~25 行）  
**执行环境**：本地（`.venv/bin/python` + `.venv/bin/pytest`）

---

### Stage 7.2：新建 Colab 训练 notebook

**目标**：新建 `notebooks/train_yolo26s_colab_halfres.ipynb`，基于现有
`notebooks/train_yolo26l_colab.ipynb` 改造，将所有参数更新为 Phase C' 配置。

**具体任务**：

1. 以现有 notebook 为基础新建，修改以下参数（每处改动 ≤ 1 行）：
   - `model`：`yolo26l.pt` → `yolo26s.pt`
   - `freeze`：`5` → `10`
   - `mosaic`：`0.5` → `0.0`
   - `scale`：`0.3` → `0.5`
   - `translate`：`0.1` → `0.15`
   - `name`：修改为 `"tree_yolo26s_halfres"`
   - `data`：指向 Drive 上传后的 `data.yaml` 路径
2. notebook 结构（Cell 划分）：
   ```
   Cell 1  [markdown]  标题 + 配置说明（Phase 7 参数 vs 现有 notebook 对比表）
   Cell 2  [code]      环境准备（Drive 挂载、依赖安装、路径常量定义）
   Cell 3  [code]      数据集准备（从 Drive 复制 data/ 到 /content/）
   Cell 4  [code]      增量恢复检测（读 Drive 的 last.pt，决定 resume vs fresh）← Stage 7.3
   Cell 5  [code]      训练主体（model.train(...)，新参数）
   Cell 6  [code]      训练后权重写回 Drive（best.pt + last.pt + results.csv）
   Cell 7  [markdown]  结果说明 + 下一步指引
   ```
3. 在 Cell 2 中定义路径常量（Drive 路径需与实际 Drive 目录一致）：
   ```python
   from pathlib import Path
   DRIVE_DIR = Path("/content/drive/MyDrive/TreeLearn/tree_yolo26s_halfres")
   DATA_DIR  = Path("/content/data/pseudo_labels_half/all_4videos")
   RUN_DIR   = Path("/content/runs/detect/tree_yolo26s_halfres")
   ```
4. Cell 2 末尾加 cell-level 断言（验证 Drive 已挂载）：
   ```python
   assert Path("/content/drive/MyDrive").exists(), "Drive 未挂载，请先执行 drive.mount()"
   ```

**实现依据**：Proposal Phase C' 完整训练配置代码块；Proposal 执行顺序第 7 步。

**验收标准**：
- notebook 可在本地被 `nbformat.read` 正常解析（无 JSON 错误）
- Cell 1 Markdown 包含参数对比表（model/freeze/mosaic/scale/translate 前后值）
- Cell 5 的 `model.train(...)` 调用包含 `mosaic=0.0, scale=0.5, translate=0.15,
  freeze=10`（可通过 `grep` notebook JSON 验证）
- 在 Colab L4 上手动运行 Cell 1–3 无报错（在 Stage 7.3 完成后整体验证）

**代码量估计**：≤ 120 行（notebook cells 总计；含 markdown）  
**执行环境**：notebook 文件在本地创建；Cell 5 在 Colab L4 上执行

---

### Stage 7.3：Drive 增量恢复逻辑

**目标**：在 notebook Cell 4（恢复检测）和 Cell 6（权重写回）中实现完整的 Drive
增量恢复逻辑，支持 Colab session 中断后从 `last.pt` 恢复。

**具体任务**：

1. **Cell 4 — 恢复检测逻辑**（完整实现）：
   ```python
   import shutil
   from pathlib import Path
   from ultralytics import YOLO

   LOCAL_LAST = RUN_DIR / "weights" / "last.pt"
   DRIVE_LAST = DRIVE_DIR / "weights" / "last.pt"

   if DRIVE_LAST.exists():
       print(f"[Resume] 检测到 Drive 中的 last.pt，复制到本地...")
       LOCAL_LAST.parent.mkdir(parents=True, exist_ok=True)
       shutil.copy2(DRIVE_LAST, LOCAL_LAST)
       print(f"[Resume] last.pt → {LOCAL_LAST}，将在 Cell 5 中 resume=True 训练")
       RESUME = True
   else:
       print("[Fresh] Drive 中无 last.pt，从头开始训练")
       RESUME = False
   ```
2. **Cell 5 — 训练主体**（根据 `RESUME` 标志分支）：
   ```python
   if RESUME:
       # resume=True 时只传 last.pt 路径，所有超参从 checkpoint 自动恢复
       # 不得重复传入 data/epochs/imgsz 等（会触发 Ultralytics "cannot override" 错误）
       model = YOLO(str(LOCAL_LAST))
       results = model.train(resume=True)
   else:
       model = YOLO("yolo26s.pt")
       results = model.train(
           data=str(DATA_DIR / "data.yaml"),
           epochs=100,
           imgsz=1280,
           batch=-1,            # autobatch（L4 23GB VRAM，预计 ~24–32）
           freeze=10,
           optimizer="auto",
           lr0=1e-3,
           lrf=0.01,
           cos_lr=True,
           patience=20,
           save=True,
           project="/content/runs/detect",
           name="tree_yolo26s_halfres",
           mosaic=0.0,
           mixup=0.0,
           degrees=10.0,
           translate=0.15,
           scale=0.5,
           flipud=0.0,
           fliplr=0.5,
           device=0,
           workers=4,
           cache=True,
           rect=True,
           deterministic=False,
       )
   ```
3. **Cell 6 — 权重写回 Drive**：
   ```python
   DRIVE_WEIGHTS = DRIVE_DIR / "weights"
   DRIVE_WEIGHTS.mkdir(parents=True, exist_ok=True)

   for fname in ("best.pt", "last.pt"):
       src = RUN_DIR / "weights" / fname
       if src.exists():
           shutil.copy2(src, DRIVE_WEIGHTS / fname)
           print(f"[Saved] {fname} → {DRIVE_WEIGHTS}")

   # results.csv 在 run_dir 根目录（非 weights/ 子目录）
   results_src = RUN_DIR / "results.csv"
   if results_src.exists():
       shutil.copy2(results_src, DRIVE_WEIGHTS / "results.csv")
       print(f"[Saved] results.csv → {DRIVE_WEIGHTS}")
   ```
4. **已知风险处理说明**（Markdown Cell 或注释）：
   - resume 后出现 NaN loss（GitHub #3299/#17282）：确认 `last.pt` 是 epoch 末尾写入的
     完整 checkpoint；若 NaN，删除 Drive 中 `last.pt`，重新从头训练或从 `best.pt`
     fine-tune
   - autobatch 异常（`batch=-1` 探测失败）：手动设 `batch=16`
   - cache=True RAM 不足：改 `cache="disk"`（Colab L4 约 52 GB RAM，通常够用）
5. **cell-level 断言**（Cell 4 末尾）：
   ```python
   assert isinstance(RESUME, bool), "RESUME 必须是 bool"
   if RESUME:
       assert LOCAL_LAST.exists(), f"last.pt 未复制成功: {LOCAL_LAST}"
   ```

**实现依据**：Proposal "增量训练恢复"节完整代码块；Proposal 已知风险说明（NaN loss
缓解措施）；Proposal 勘误（results.csv 路径在 run_dir 根目录）。

**验收标准**：
- 首次运行（无 `last.pt`）：Cell 4 打印 "[Fresh]"，Cell 5 以完整参数从头训练
- 中断后恢复（Drive 中有 `last.pt`）：Cell 4 打印 "[Resume]"，Cell 5 以 `resume=True`
  恢复，不重复传超参
- Cell 6 执行后，Drive 中 `weights/best.pt`、`weights/last.pt`、`weights/results.csv`
  均存在
- Cell-level 断言在两种分支下均通过

**代码量估计**：≤ 120 行（Cell 4/5/6 代码 + 说明注释；含 Cell 5 完整 train 参数列表）  
**执行环境**：Colab L4（notebook 在本地创建，在 Colab 执行）

---

### Phase 7 输出质量评价

| 检查项 | 方法 | 合格标准 | 不合格时处置 |
|--------|------|----------|-------------|
| **训练收敛性** | Drive 中 `results.csv` 后半段 `val/box_loss` 下降趋势 | 无持续震荡 | 降低 `lr0`（1e-3 → 5e-4）后重训 |
| **loss 健康** | Colab 训练日志无 NaN/inf | 全程无 NaN/inf | 删除损坏的 last.pt，重新从头训练；或切换 `optimizer="AdamW"` |
| **mAP@0.5 目标** | Drive 中 `results.csv` 最后一行 `metrics/mAP50` 字段 | > 0.90（基线 0.874） | 检查数据集质量（Phase 6 覆盖率、标注视觉质量）；尝试解冻更多层（`freeze=5`） |
| **mAP@0.5:0.95** | 同上 `metrics/mAP50-95` 字段 | > 0.70 | 增加训练 epochs（100 → 150），放宽 patience（20 → 30） |
| **权重完整性** | `best.pt` 和 `last.pt` 均在 Drive 中 | 两个文件均存在 | Cell 6 重新执行 |
| **resume 功能验证** | 手动中断训练后重新运行 notebook，观察 epoch 计数是否从中断处续接 | epoch 计数从上次中断 epoch + 1 开始 | 检查 Cell 4 是否成功复制 last.pt 到本地路径 |
| **autobatch 结果** | Colab 训练日志中的 batch size 探测输出 | batch ≥ 16（L4 23GB VRAM 应能支持） | 若 < 8，手动设 `batch=16`；若 OOM，设 `batch=8` |

**决策矩阵**：

| 训练结果 | 推荐行动 |
|----------|----------|
| mAP@0.5 > 0.90 | 目标达成；可尝试 YOLO26l 对比实验（以相同配置） |
| 0.874 ≤ mAP@0.5 ≤ 0.90 | 未超越基线；检查数据集质量后重训，或增加 epochs |
| mAP@0.5 < 0.874 | 回退到基线（`runs/detect/tree_yolo26s_pseudo/weights/best.pt`）；排查 Phase 6 数据集问题 |

---

## 汇总：文件清单与代码量

| 文件 | Phase/Stage | 估计修改行数 | 类型 |
|------|-------------|------------|------|
| `scripts/extract_video_frames.py` | 5.1 | ~20 | 修改 |
| `tests/test_extract_video_frames.py` | 5.1 | ~20 | 补充 |
| `scripts/generate_pseudo_labels.py` | 6.1 | 0 | 无需修改（已有完整 CLI 参数） |
| `scripts/split_dataset.py` | 6.2 | 0 | 无需修改（已有 `--pseudo-labels-base`/`--out-dir`） |
| `tests/test_pseudo_labels.py` | 6.2 | ~15 | 补充（验证现有 CLI 参数行为） |
| `scripts/train_yolo26.py` | 7.1 | ~15 | 修改 |
| `tests/test_train_yolo26.py` | 7.1 | ~25 | 补充 |
| `notebooks/train_yolo26s_colab_halfres.ipynb` | 7.2, 7.3 | ~120 | 新建 |

**Phase 5 总计**：~40 行（≤ 500 行限制内）  
**Phase 6 总计**：~15 行（两个脚本均无需修改；仅新增测试；≤ 500 行限制内）  
**Phase 7 总计**：~160 行（≤ 500 行限制内）  
**全局总计**：~215 行代码修改/新增（远低于 3×500 = 1500 行限制）

---

## 关键决策引用索引

| 决策 | 来源（Proposal 章节） | 具体值 |
|------|-----------------------|--------|
| every_n 参数 | Phase A' + 审查修正 | 15（fps=59.9，输出≈4fps，4× 当前密度） |
| 目标帧分辨率 | Phase A' 方案设计 | 1352×760（长宽各 ×0.5） |
| 预期总帧数 | Phase A' 表格（ffprobe 实测） | ~4147（±10%）|
| train/val 划分 | Phase B' 方案设计 | train=20240319，val=20240530（跨日期） |
| GDINO 参数 | Phase B'（无需修改） | SHORTEST_EDGE=800, LONGEST_EDGE=1333 |
| GDINO scale 提升 | Proposal 核心思路 | 0.49 → ≈0.986（几乎无信息损失） |
| model | Phase C' 关键配置变更 | `yolo26s.pt`（已验最优，26l 数据量不足） |
| freeze | Phase C' 关键配置变更 | 10（回到已验证最优） |
| mosaic | Phase C' 关键配置变更 | 0.0（关闭；bbox 高度 47% frameh，mosaic 有害） |
| scale 增强 | Phase C' 关键配置变更 | 0.5（替代 mosaic，大目标场景推荐） |
| translate | Phase C' 关键配置变更 | 0.15（略加强，弥补 mosaic=0） |
| batch | Phase C'（autobatch） | -1（L4 23GB VRAM，预计 ~24–32） |
| cache | Phase C' | True（~12 GB RAM，L4 约 52 GB，可承载） |
| resume 方式 | Proposal 增量训练恢复节 | `YOLO(last.pt).train(resume=True)`，不重传超参 |
| results.csv 路径 | Proposal 勘误 | `run_dir/results.csv`（非 `run_dir/weights/results.csv`） |
| NaN loss 缓解 | Proposal 架构注意 | 确认完整 checkpoint 后再恢复；异常时从 best.pt 重训 |
| 成功指标 | Proposal 成功指标表 | mAP@0.5 > 0.90（基线 0.874） |
