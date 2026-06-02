# Plan: Video-Based Tree Detection via Grounding DINO Pseudo-GT + YOLO26 Fine-Tuning

**状态**: Draft  
**日期**: 2026-05-30  
**对应 Proposal**: `docs/proposals/proposal-tree-detection-gdino-yolo.md`  
**Phase 编号**: 4（接续 `1-2-3-tdus-cutout-svm-classification.md`）

---

## 总体架构

```
Phase A: 视频帧提取脚本（scripts/extract_video_frames.py）
    └── Stage A1: 基础帧提取（等间隔策略，uniform）
    └── Stage A2: 差异采样策略（diff）+ CLI 参数化

Phase B: Grounding DINO 伪标签生成与过滤
    └── Stage B1: GDINO 批量推理骨架（改造现有推理逻辑）
    └── Stage B2: NMS 过滤 + YOLO txt 格式输出
    └── Stage B3: 数据集目录结构 + data.yaml 生成

Phase C: YOLO26 Fine-Tuning
    └── Stage C1: data.yaml 验证 + 训练脚本入口
    └── Stage C2: 训练主逻辑 + 超参配置
    └── Stage C3: 验证与结果输出

Phase D: 迭代自训练（可选）
    └── Stage D1: 自训练循环骨架
    └── Stage D2: bbox 交集过滤 + 置信度递增策略
```

---

## 依赖关系

```
Phase A → Phase B（帧图像集是伪标签生成的输入）
Phase B → Phase C（YOLO 格式数据集是训练的输入）
Phase C → Phase D（fine-tune 权重是自训练迭代的起点）
Phase D 依赖手工标注 val set（20–50 张）——需在 Phase C 完成后手动完成
```

---

## 测试策略

- **倾向 TDD**：每个 Stage 先写测试，再写实现，测试覆盖率目标 ≥ 80%
- **单元测试**：帧提取逻辑（帧数、命名格式）、YOLO txt 格式转换（坐标归一化、值域）、NMS 过滤（重叠框去除）
- **集成测试**：对少量帧（≤ 20 张）跑通完整流水线（提取 → 伪标签生成 → 训练 1 epoch）
- **测试工具**：`pytest`（已在 `.venv` 中可用）；测试文件放 `tests/` 目录
- **不需要单元测试**：YOLO26 训练本身（模型库内部逻辑）——用集成测试验证流程可通跑即可

---

## Phase A：视频帧提取脚本

**目标**：新建 `scripts/extract_video_frames.py`，从全部 4 个行车视频中提取代表性帧。  
**依赖**：无（Phase A 为起点）  
**代码量估计**：≤ 170 行  
**涉及文件**：
- `scripts/extract_video_frames.py`（新建）
- `tests/test_extract_video_frames.py`（新建）

**数据源（4 个视频）**：
```
assets/original/computervisie_2024/eastbound_20240319.MP4   (~2.25 GB)
assets/original/computervisie_2024/eastbound_20240530.MP4   (~2.01 GB)
assets/original/computervisie_2024/westbound_20240319.MP4   (~1.48 GB)
assets/original/computervisie_2024/westbound_20240530.MP4   (~1.73 GB)
```

**前置步骤（Stage A0，非编码任务）**：帧提取前手动运行 ffprobe，确认各视频的实际 fps 和时长：
```bash
for f in assets/original/computervisie_2024/eastbound_20240319.MP4 \
          assets/original/computervisie_2024/eastbound_20240530.MP4 \
          assets/original/computervisie_2024/westbound_20240319.MP4 \
          assets/original/computervisie_2024/westbound_20240530.MP4; do
  ffprobe -v quiet -print_format json -show_streams "$f" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); \
      v=[s for s in d['streams'] if s['codec_type']=='video'][0]; \
      print(f\"{f}: fps={eval(v['r_frame_rate']):.1f}, duration={float(v['duration']):.0f}s\")"
done
```
根据结果确定采样间隔 N（目标 1fps：`N = round(fps)`）。若各视频 fps 不一致，在 CLI 中按视频单独指定。

---

### Stage A1：基础帧提取（等间隔策略）

**目标**：实现 uniform 策略（每 N 帧取一帧），支持批量处理多个视频，作为流水线验证的基础。

**具体任务**：

1. 新建 `scripts/extract_video_frames.py`，包含函数 `extract_frames(video_path, out_dir, every_n, strategy)`
2. 使用 `cv2.VideoCapture` 逐帧读取；`frame_idx % every_n == 0` 时保存
3. 保存格式：JPEG，质量 95（`[cv2.IMWRITE_JPEG_QUALITY, 95]`），文件名 `frame_{idx:06d}.jpg`
4. 输出目录按视频文件名（无后缀）自动命名：`out_base / video_stem /`，例如 `data/frames/eastbound_20240319/`
5. 启动时打印视频元信息：`FPS={fps:.1f}, total_frames={total}, duration={total/fps:.1f}s`
6. 结束时打印：`Saved {saved} frames → {out_dir}`
7. 新建 `tests/test_extract_video_frames.py`，用 `tmp_path` fixture 创建临时目录：
   - 生成 10 帧合成视频（`cv2.VideoWriter`），验证 N=3 时输出 4 张帧（idx=0,3,6,9）
   - 验证文件命名格式符合 `frame_\d{6}\.jpg`
   - 验证输出目录不存在时自动创建
   - 验证输出子目录名与视频文件 stem 一致

**实现依据**：Proposal 阶段一（1A）；JPEG 质量 95 源自 Proposal 阶段零骨架代码；按视频 stem 分目录输出为 Proposal 阶段三跨日期划分的前提。

**验收标准**：
- `pytest tests/test_extract_video_frames.py::TestUniform` 全部通过
- 手动对短视频 `eastbound_20240319_short_20fps_compressed.MP4` 执行脚本做快速验证，N=20，输出帧数符合 `total_frames / 20` 估算（±2 帧）
- 输出 JPEG 文件可被 `cv2.imread` 正常读取，尺寸与原视频一致

**代码量估计**：≤ 80 行（脚本 ~60 行，测试 ~20 行）

---

### Stage A2：差异采样策略 + CLI 参数化

**目标**：新增 diff 策略（基于帧差均值），并将脚本参数化为 CLI 工具。

**具体任务**：

1. 在 `extract_frames` 中新增 `strategy="diff"` 分支：
   - 计算 `cv2.absdiff(gray, prev_gray).mean()`
   - 差值 ≥ `diff_thr` 时保存；默认 `diff_thr=5.0`（Proposal 1B 推荐值）
   - 第一帧无论差值如何，直接保存
2. 用 `argparse` 封装 CLI：`--videos`（支持多个路径，`nargs="+"`)、`--out-base`（输出根目录，默认 `data/frames/`）、`--every-n`、`--strategy`、`--diff-thr` 五个参数；循环处理每个视频，输出到 `out_base/{video_stem}/`
3. 补充 `tests/test_extract_video_frames.py`：
   - 生成含场景变化的合成视频（部分帧灰度值突变），验证 diff 策略保留变化帧、跳过相似帧
   - 验证第一帧无论与后续帧差值大小，必定被保存（边界条件：`prev_gray is None` 分支）
   - 验证 `diff_thr` 参数生效（高阈值下帧数更少）
   - 验证 CLI 接口可被 `subprocess` 正常调用并返回 0

**实现依据**：Proposal 1B；diff 阈值 5.0（"对应约 2% 像素变化"）。

**验收标准**：
- `pytest tests/test_extract_video_frames.py` 全部通过（覆盖 uniform + diff）
- 对真实视频，diff 策略（`diff_thr=5.0`）输出帧数 < uniform（N=1）帧数，且 > uniform（N=20）帧数
- `python scripts/extract_video_frames.py --help` 正常输出（含 `--videos` 多路径参数说明）
- 对 4 个完整视频运行后，`data/frames/` 下产生 4 个独立子目录，命名与视频 stem 一致

**代码量估计**：≤ 70 行（脚本新增 ~40 行，测试新增 ~30 行）

---

### Phase A 输出质量评价

在进入 Phase B 之前，对提取帧集合执行以下检查，确认输出质量达标。

| 检查项 | 方法 | 合格标准 | 不合格时的处置 |
|--------|------|----------|----------------|
| **帧数量** | 统计 4 个子目录文件数，并与 ffprobe 时长估算对比 | 各视频 `duration_s ± 5%`，合计 ~730–1,950 帧（取决于实际 fps） | 检查视频是否完整读取，重新运行 |
| **文件完整性** | `cv2.imread` 批量读取，统计返回 None 的数量 | 损坏帧数 = 0 | 删除损坏帧，检查写入逻辑 |
| **帧尺寸一致性** | 检查所有帧的 `(height, width)` | 与原视频分辨率（2704×1520）完全一致 | 检查 `cv2.VideoCapture` 是否在解码后 resize |
| **时间覆盖率** | 检查首帧和末帧的文件名编号 | 首帧 `frame_000000.jpg`，末帧编号 ≥ `total_frames - every_n` | 检查循环退出条件 |
| **亮度分布** | 统计所有帧灰度均值的分布（min/mean/max） | 均值 > 50（非全暗视频）；若全部帧均值 < 50，说明视频本身问题 | 换用其他视频片段，或降低亮度过滤阈值 |
| **diff 策略多样性**（使用 diff 时） | 目视抽查 10 帧，确认帧间内容有差异 | 无连续 5 帧视觉上完全相同 | 降低 `diff_thr`（如从 5.0 → 3.0） |

**质量评价脚本**：可在终端运行以下命令快速检查（遍历 4 个子目录）：
```bash
python -c "
import cv2, os, glob
for video in ['eastbound_20240319','eastbound_20240530','westbound_20240319','westbound_20240530']:
    frames = sorted(glob.glob(f'data/frames/{video}/*.jpg'))
    sizes, dark = set(), 0
    for p in frames:
        img = cv2.imread(p)
        if img is None: print(f'损坏: {p}'); continue
        sizes.add(img.shape[:2])
        if img.mean() < 50: dark += 1
    print(f'{video}: {len(frames)} 帧, 尺寸={sizes}, 过暗={dark}')
"
```

**Phase A 放行条件**：上表全部检查项合格，方可进入 Phase B。

---

## Phase B：Grounding DINO 伪标签生成与过滤

**目标**：新建 `scripts/generate_pseudo_labels.py`，对帧图像集批量推理生成 YOLO 格式伪标签。  
**依赖**：Phase A（提取后的帧图像集）  
**代码量估计**：≤ 400 行  
**涉及文件**：
- `scripts/generate_pseudo_labels.py`（新建）
- `tests/test_pseudo_labels.py`（新建）
- 参考复用：`gdino_dinov2_svm/03_extract_embeddings.py`（推理骨架）、`validate_gdino_bbox_crop.py`（可视化）

---

### Stage B1：GDINO 批量推理骨架

**目标**：改造现有 `gdino_dinov2_svm/03_extract_embeddings.py` 的推理逻辑，支持对帧目录批量推理并返回原始 bbox + scores。

**具体任务**：

1. 新建 `scripts/generate_pseudo_labels.py`，复用以下现有逻辑（**不修改原文件**）：
   - `gdino_preprocess`：图像 resize + normalize
   - `collate_fn`：批量 pad
   - `post_process_grounded_object_detection`：xyxy 坐标还原到原图尺寸
2. 加载模型：`IDEA-Research/grounding-dino-tiny`，bf16 推理（与现有 notebook 一致，Proposal 背景节）
3. 文本提示固定为 `"tree."`（Proposal 2.1；单词形式已在 TDUS 验证有效）
4. 推理参数：`box_threshold=0.30`（可由 `--box-threshold` CLI 参数覆盖）、`text_threshold=0.25`（Proposal 2.2 三级策略第一、二级）；CLI 应暴露 `--box-threshold` 以支持 Proposal 风险缓解（"伪 GT 漏检远处小树 → 降低至 0.25"）
5. **亮度过滤**（Proposal 风险缓解）：推理前计算帧灰度均值，`mean(gray) < 50` 时跳过该帧（标记为夜间/过暗帧，不生成伪标签），避免黑暗场景产生大量假阳性
6. 输出：每张图对应 `(boxes_xyxy: Tensor[N,4], scores: Tensor[N], img_w: int, img_h: int)` 的字典；被亮度过滤跳过的帧返回空字典（写空 txt）
7. 新建 `tests/test_pseudo_labels.py`：
   - 用 `unittest.mock.patch` mock GDINO 模型返回值，测试 `run_inference` 函数接口（输入图像路径列表，输出字典列表）
   - 验证 OOM 降级逻辑（batch_size 自动减半）
   - 验证亮度过滤：构造 `mean(gray)=30` 的黑色合成帧，确认 `run_inference` 返回空字典（跳过推理）

**实现依据**：Proposal 现有代码复用关系表；`03_extract_embeddings.py`"改造要点"注释；Proposal 风险表（"行车视频场景变化剧烈"和"漏检远处小树"缓解措施）。

**验收标准**：
- `pytest tests/test_pseudo_labels.py::TestInference` 全部通过（含亮度过滤用例）
- 对 10 张真实帧（从 Phase A 输出中取样）执行 `run_inference`，返回字典结构正确，无报错
- bf16 推理不因精度问题导致 scores 全为 0 或 NaN
- `python scripts/generate_pseudo_labels.py --help` 输出包含 `--box-threshold` 参数

**代码量估计**：≤ 120 行（脚本 ~90 行，测试 ~30 行）

---

### Stage B2：NMS 过滤 + YOLO txt 格式输出

**目标**：在 Stage B1 输出基础上，实现分数过滤、NMS 去重，并将结果写入 YOLO txt 格式。

**具体任务**：

1. 实现 `filter_boxes(boxes_xyxy, scores, iou_thr=0.45, score_thr=0.35)` 函数：
   - 先按 `score_thr` 过滤（写入阈值 0.35，Proposal 2.2 三级策略第三级）
   - 再用 `torchvision.ops.nms`（IoU 阈值 0.45，Proposal 2.3）去重
   - 空结果时返回空 Tensor（不报错）
2. 实现 `xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h)` 函数：
   - 返回 `(cx, cy, w, h)`，全部归一化至 `[0, 1]`（Proposal 2.4 公式）
3. 实现 `write_label_file(label_path, boxes_xyxy, img_w, img_h)`：
   - 每行格式：`0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}`（class_id=0）
   - 无有效 bbox 时写空文件（Proposal 2.4："允许为空代表负样本"）
4. 补充 `tests/test_pseudo_labels.py`：
   - 单元测试 `filter_boxes`：两框高度重叠（IoU=0.8）时只保留一框；低分框被过滤
   - 单元测试 `xyxy_to_yolo`：已知坐标验证输出值（精度 1e-6）；验证所有输出值 ∈ [0,1]
   - 单元测试 `write_label_file`：读回 txt 文件验证行数、格式、数值

**实现依据**：Proposal 2.2（三级置信度策略）、2.3（NMS IoU=0.45）、2.4（YOLO 格式公式）。

**验收标准**：
- `pytest tests/test_pseudo_labels.py::TestFilter` 和 `::TestYoloFormat` 全部通过
- 对 20 张样本帧手动运行，输出的 `.txt` 文件可被 `ultralytics` 数据加载器正常读取（无解析错误）
- `xyxy_to_yolo` 单元测试覆盖率 100%（函数短小，须全覆盖）

**代码量估计**：≤ 90 行（脚本新增 ~60 行，测试新增 ~30 行）

---

### Stage B3：数据集目录结构 + data.yaml 生成

**目标**：整合 4 个视频的 Stage B1/B2 输出，按**跨日期划分**（20240319 → train，20240530 → val）组合成符合 YOLO26 要求的完整数据集。

**具体任务**：

1. 在 `generate_pseudo_labels.py` 末尾整合流程（**按视频分目录**，不合并）：
   - 图像复制到 `data/pseudo_labels/{video_name}/images/all/`
   - 标注写入 `data/pseudo_labels/{video_name}/labels/all/`
   - 对 4 个视频分别运行，产生 4 个独立目录
2. 新建 `scripts/split_dataset.py`，实现**跨日期划分**（Proposal 阶段三逻辑）：
   - train：`eastbound_20240319` + `westbound_20240319` 的全部帧
   - val：`eastbound_20240530` + `westbound_20240530` 的全部帧
   - 文件名加视频前缀避免命名冲突：`{video_name}__{frame_name}.jpg`
   - 使用 `shutil.copy`（不用 symlink，避免平台兼容问题）
   - 若标注文件不存在，创建空 `.txt`（负样本帧）
3. 在 `split_dataset.py` 末尾自动生成 `data/pseudo_labels/all_4videos/data.yaml`，`path` 字段写入绝对路径（Proposal 2.6 注意事项）
4. 补充 `tests/test_pseudo_labels.py`：
   - 用 `tmp_path` 模拟 4 个视频目录（各 5 帧），验证 20240319 帧进 train、20240530 帧进 val
   - 验证文件名前缀格式：`{video_name}__frame_{idx:06d}.jpg`
   - 验证跨视频无命名冲突（两个视频的 `frame_000000.jpg` 各自加前缀后共存）
   - 验证 `data.yaml` 中 `path` 为绝对路径、`nc=1`、`names=['tree']`
5. 新建质检脚本（不需要测试）：`scripts/visualize_pseudo_labels.py`，每个视频各随机抽取 5–8 张帧（合计 20–30 张），生成可视化图（Proposal 2.7）

**实现依据**：Proposal 2.5（目录结构）、2.6（data.yaml）、阶段三（跨日期划分）。

**验收标准**：
- `pytest tests/test_pseudo_labels.py::TestSplit` 全部通过
- 生成的目录结构通过 `ultralytics` 内置数据集检查：`YOLO("yolo26n.pt").val(data="data/pseudo_labels/all_4videos/data.yaml")` 无 KeyError 或路径错误
- 验证 train 目录只含 20240319 帧、val 目录只含 20240530 帧（grep 文件名前缀验证）
- 质检可视化：人工确认每个视频抽检帧中，bbox 大体覆盖树冠，无明显路灯/建筑误检
- 整体伪标注覆盖率：4 个视频合计有效 bbox 帧数 ≥ 50%（避免大量空标注导致训练过稀疏）

**代码量估计**：≤ 140 行（`generate_pseudo_labels.py` 整合 ~30 行，`split_dataset.py` ~70 行，`visualize_pseudo_labels.py` ~40 行，测试 ~50 行；质检脚本不计入限制）

---

### Phase B 输出质量评价

在进入 Phase C 之前，对伪标签数据集执行以下检查，确认标注质量和数据集结构达标。

| 检查项 | 方法 | 合格标准 | 不合格时的处置 |
|--------|------|----------|----------------|
| **标注覆盖率** | 统计有效 bbox 帧数（txt 非空）占总帧数比例 | ≥ 50%（过低说明 GDINO 漏检严重） | 降低 `--box-threshold`（0.30 → 0.25）重新推理 |
| **每帧 bbox 数量** | 统计 txt 文件行数的分布（min/mean/max/p95） | 均值 1–10 框；p95 ≤ 30（过多说明误检） | 若 p95 > 30，提高 `score_thr`（0.35 → 0.40）或降低 NMS IoU（0.45 → 0.35） |
| **坐标值域** | 解析全部 txt，检查 cx/cy/w/h 是否均在 [0,1] | 无越界值 | 检查 `xyxy_to_yolo` 的边界 clip 逻辑 |
| **train/val 时间隔离** | 检查 val 中最小帧编号 > train 中最大帧编号 | 成立（无时间泄漏） | 检查 `split_dataset.py` 的排序逻辑 |
| **data.yaml 路径** | `cat data.yaml`，确认 `path` 字段 | 以 `/` 开头的绝对路径 | 重新运行 `split_dataset.py` 并传入 `--out-dir` 绝对路径 |
| **视觉质量抽查** | 运行 `scripts/visualize_pseudo_labels.py`，人工查看 20 张可视化图 | ≥ 70% 的 bbox 覆盖树冠；路灯/建筑误检率 < 20% | 根据误检类型调整 `text_threshold` 或提高 `score_thr` |
| **ultralytics 兼容性** | `python -c "from ultralytics import YOLO; YOLO('yolo26n.pt').val(data='...data.yaml', epochs=0)"` | 无 KeyError、FileNotFoundError、路径错误 | 对照 YOLO26 数据集格式文档修正 `data.yaml` |

**量化报告**：运行以下命令生成各视频标注统计摘要：
```bash
python -c "
import glob, os, statistics
for video in ['eastbound_20240319','eastbound_20240530','westbound_20240319','westbound_20240530']:
    labels = glob.glob(f'data/pseudo_labels/{video}/labels/all/*.txt')
    total = len(labels)
    if total == 0: print(f'{video}: 无标注文件'); continue
    nonempty = sum(1 for p in labels if os.path.getsize(p) > 0)
    counts = [sum(1 for _ in open(p)) for p in labels if os.path.getsize(p) > 0]
    mean_c = statistics.mean(counts) if counts else 0
    print(f'{video}: {total} 帧, 覆盖率={nonempty/total*100:.1f}%, bbox均值={mean_c:.1f}')
"
```

**Phase B 放行条件**：4 个视频各自标注覆盖率 ≥ 50%，视觉质量抽查合格（每视频各抽查 5–8 帧），ultralytics 兼容性验证通过，方可进入 Phase C。

---

## Phase C：YOLO26 Fine-Tuning

**目标**：新建 `scripts/train_yolo26.py`，用 Phase B 生成的伪标签数据集 fine-tune YOLO26s，并输出验证指标。  
**依赖**：Phase B（`data.yaml` + `images/` + `labels/` 完整数据集）  
**代码量估计**：≤ 200 行  
**涉及文件**：
- `scripts/train_yolo26.py`（新建）
- `scripts/validate_yolo26.py`（新建）
- `tests/test_train_yolo26.py`（新建）

---

### Stage C1：data.yaml 验证 + 训练脚本入口

**目标**：确保训练前数据集格式正确；新建带 CLI 的训练脚本入口。

**具体任务**：

1. 新建 `scripts/train_yolo26.py`，包含 `validate_dataset(data_yaml_path)` 函数：
   - 验证 `data.yaml` 存在且包含 `path`、`train`、`val`、`nc`、`names` 字段
   - 验证 `path` 为绝对路径（Proposal 2.6 注意事项）
   - 验证 `images/train/` 和 `images/val/` 目录存在且非空
   - 验证 `labels/train/` 和 `labels/val/` 目录存在
   - 验证标注文件数量与图像文件数量一致（允许差异 ≤ 5，考虑空标注帧）
2. CLI 参数：`--data`（data.yaml 路径）、`--model`（默认 `yolo26s.pt`）、`--epochs`（默认 100）、`--imgsz`（默认 1280）、`--batch`（默认 2）、`--freeze`（默认 10）、`--device`（默认 `auto`）
3. 新建 `tests/test_train_yolo26.py`：
   - 用 `tmp_path` 创建符合规范的最小数据集结构，验证 `validate_dataset` 通过
   - 构造缺少 `path` 字段的非法 `data.yaml`，验证抛出 `ValueError` 并含有描述性错误信息
   - 构造 `images/train/` 为空目录的情况，验证 `validate_dataset` 失败
   - 构造标注文件数量与图像数量差异 > 5（如 20 图 / 10 标注）的情况，验证 `validate_dataset` 抛出 `ValueError`（此边界条件对应"差异 ≤ 5 才允许"的规则，必须有测试覆盖）
   - 构造 `path` 为相对路径的 `data.yaml`，验证 `validate_dataset` 抛出 `ValueError`（Proposal 风险表"path 字段必须为绝对路径"缓解措施）

**实现依据**：Proposal 2.6（data.yaml 格式要求）；Proposal 风险表（"data.yaml 路径错误导致训练失败"缓解措施）。

**验收标准**：
- `pytest tests/test_train_yolo26.py::TestValidateDataset` 全部通过
- 对 Phase B 生成的真实数据集执行 `validate_dataset`，无报错
- `python scripts/train_yolo26.py --help` 正常输出所有参数

**代码量估计**：≤ 80 行（脚本 ~50 行，测试 ~30 行）

---

### Stage C2：训练主逻辑 + 超参配置

**目标**：实现 YOLO26s fine-tuning 主逻辑，含完整超参配置。

**具体任务**：

1. 在 `train_yolo26.py` 中实现 `train(args)` 函数，调用 `ultralytics.YOLO`：
   ```python
   model = YOLO(args.model)
   results = model.train(
       data=args.data,
       epochs=args.epochs,        # 默认 100（Proposal 阶段四）
       imgsz=args.imgsz,          # 默认 1280：与 GDINO 有效分辨率对齐（1333×749）
       batch=args.batch,          # 默认 2：RTX 3060 12GB 实测安全值（rect 1280×736 ≈7GB）
       freeze=args.freeze,        # 默认 10（Proposal："仅训练 neck+head"）
       optimizer="auto",          # Proposal：auto 自动选 AdamW/MuSGD
       lr0=1e-3,
       lrf=0.01,
       cos_lr=True,               # Proposal："余弦退火，防止后期震荡"
       patience=20,               # Proposal：early stopping
       save=True,
       project="runs/detect",
       name=args.run_name,        # 默认 "tree_yolo26s_pseudo"
       mosaic=0.5,                # Proposal："适度降低，避免破坏 bbox 语义"
       mixup=0.0,                 # Proposal："关闭 mixup"
       degrees=10.0,
       translate=0.1,
       scale=0.3,
       flipud=0.0,                # Proposal："行车视频无需上下翻转"
       fliplr=0.5,
       device=args.device,
   )
   ```
2. 训练结束后打印关键指标：`mAP@0.5`、`mAP@0.5:0.95`、最终 epoch 数
3. 新建 `tests/test_train_yolo26.py` 集成测试（需打 `@pytest.mark.integration` 标记，默认不运行）：
   - 用最小数据集（20 张帧 + 标注）运行 1 epoch，验证流程可通跑（不验证 mAP 值）
   - 验证 `runs/detect/{run_name}/weights/best.pt` 存在

**实现依据**：Proposal 阶段四 fine-tuning 配置代码块及关键参数说明。

**验收标准**：
- `pytest tests/test_train_yolo26.py -m "not integration"` 全部通过
- 手动运行集成测试 `pytest tests/test_train_yolo26.py -m integration` 通过（20 帧、1 epoch）
- 完整训练（100 epochs 或 early stopping）正常结束，`best.pt` 权重文件存在
- 训练过程中无 `loss=NaN` 或 `inf`（若出现，Proposal 建议显式指定 `optimizer="AdamW"`）

**代码量估计**：≤ 70 行（脚本新增 ~45 行，测试新增 ~25 行）

---

### Stage C3：验证与结果输出

**目标**：新建验证脚本，对训练好的模型在 val set 上评估并输出指标；提供推理示例。

**具体任务**：

1. 新建 `scripts/validate_yolo26.py`：
   - 加载 `best.pt` 权重，运行 `model.val(data=..., conf=0.25)`
   - 打印 `mAP@0.5`、`mAP@0.5:0.95`、`precision`、`recall`
   - 可选：`--predict-dir` 参数，对指定目录运行推理并保存可视化结果（Proposal 阶段四推理示例）
2. CLI 参数：`--weights`（best.pt 路径）、`--data`（data.yaml 路径）、`--conf`（默认 0.25）、`--predict-dir`（可选）
3. 补充 `tests/test_train_yolo26.py`：
   - 用 mock 验证 `validate_yolo26` 函数接口（输入权重路径和 data.yaml，返回包含 `mAP50` 的字典）
   - 验证 `--conf` 参数正确传入模型

**实现依据**：Proposal 阶段四推理示例（`conf=0.25`）；YOLO26 NMS-free 无需额外后处理（Proposal 阶段四注意事项）。

**验收标准**：
- `pytest tests/test_train_yolo26.py::TestValidate` 全部通过
- 对 Phase C2 训练好的 `best.pt` 运行验证，输出 `mAP@0.5 ≥ 0.30`（伪标签场景基准线；低于此值需检查数据集质量）
- 推理模式（`--predict-dir`）生成含 bbox 的可视化图像，bbox 大体覆盖树冠

**代码量估计**：≤ 60 行（脚本 ~40 行，测试 ~20 行）

---

### Phase C 输出质量评价

在决定是否进入 Phase D（可选）之前，对训练结果执行以下评估，判断模型是否达到可用水平。

| 检查项 | 方法 | 合格标准 | 不合格时的处置 |
|--------|------|----------|----------------|
| **收敛性** | 查看 `runs/detect/.../results.csv` 中 `val/box_loss` 曲线 | 训练后半段 loss 呈下降趋势，无持续震荡 | 减小 `lr0`（1e-3 → 5e-4）重新训练；或减少 `freeze` 层数（10 → 5） |
| **loss 健康** | 检查训练日志中是否出现 NaN/inf | 全程无 NaN/inf | 切换为 `optimizer="AdamW"`（关闭 MuSGD 自动选择） |
| **mAP@0.5（跨日期 val）** | `scripts/validate_yolo26.py --weights best.pt --data data/pseudo_labels/all_4videos/data.yaml` | ≥ 0.25（跨日期 val 比单视频内划分更严格，基准线适当下调） | 检查 Phase B 标注质量；若数据质量无问题，尝试解冻更多层（`freeze=5`） |
| **Precision / Recall 平衡** | 同上输出 | Precision ≥ 0.40 且 Recall ≥ 0.35（避免严重偏科） | 若 Precision 过低：提高推理 conf 阈值；若 Recall 过低：降低 conf 阈值或检查标注漏检 |
| **推理速度** | `model.val(...)` 输出的推理速度 | 单帧推理 ≤ 50ms（640px 输入，GPU） | 若过慢：改用 `yolo26n.pt`（nano 变体）；检查 batch size 设置 |
| **视觉质量抽查** | `--predict-dir` 对 10 张未见过的帧做推理，人工审查 | ≥ 70% 的帧中，明显可见的树有 bbox 覆盖 | 若大量漏检：降低 `conf=0.25` → `0.15` 后再评估；若大量误检：提高 conf 阈值 |
| **early stopping 时机** | 检查实际训练 epoch 数（`results.csv` 行数） | 若 epoch < 30 就停止，说明 patience 过小或学习率过大 | 调整 `patience=30`，`lr0=5e-4` 重新训练 |

**决策矩阵**：

| 评价结果 | 推荐行动 |
|----------|----------|
| mAP ≥ 0.50，视觉质量良好 | 模型可直接部署；Phase D 为锦上添花，可跳过 |
| 0.30 ≤ mAP < 0.50 | 进入 Phase D 迭代自训练以提升质量 |
| mAP < 0.30 | 优先回到 Phase B 改善标注质量，再重训；Phase D 无法弥补根本数据问题 |

**Phase C 放行条件**：mAP@0.5 ≥ 0.30，loss 无 NaN，视觉质量抽查通过，方可决定是否进入 Phase D。

---

## Phase D：迭代自训练（可选）

**目标**：构建可迭代的自训练框架，通过 YOLO 预测与 GDINO 原始输出的交集逐步纯化伪标签。  
**依赖**：Phase C（`best.pt` 权重）+ 手工标注 val set（20–50 张，必须在 Phase C 完成后手动完成）  
**代码量估计**：≤ 300 行  
**涉及文件**：
- `scripts/self_train.py`（新建）
- `tests/test_self_train.py`（新建）

**前置条件（人工操作）**：在 Phase C 完成后，人工标注 20–50 张帧（可使用 Label Studio 或 CVAT），保存为 YOLO 格式，路径 `data/manual_labels/val/`。若跳过此步骤，自训练迭代无客观收敛依据（Proposal 阶段五"前提条件"）。

---

### Stage D1：自训练循环骨架

**目标**：实现自训练循环的框架代码（调度逻辑）；推理和训练逻辑复用 Phase B/C。

**具体任务**：

1. 新建 `scripts/self_train.py`，实现 `run_self_training(config)` 函数：
   - 外层循环 `for k in range(1, max_iter + 1)`（默认 max_iter=5，Proposal 阶段五"超过 5 轮通常无意义"）
   - 每轮：推理 → 交集过滤 → 重新训练 → 在手工 val set 评估 → 判断收敛
   - 收敛条件：连续 2 轮 mAP 无改善（Proposal 阶段五流程第 5 步）
   - 每轮保存模型权重路径：`runs/detect/tree_yolo26s_iter{k}/weights/best.pt`（Proposal 实践建议）
2. CLI 参数：`--weights`（初始 best.pt）、`--frames-dir`（全部帧目录）、`--gdino-labels`（GDINO 原始输出缓存目录）、`--manual-val-data`（手工标注 data.yaml）、`--max-iter`（默认 5）
3. 新建 `tests/test_self_train.py`：
   - mock 推理函数和训练函数，验证循环在 `max_iter` 轮后终止
   - 验证连续 2 轮 mAP 无改善时提前退出

**实现依据**：Proposal 阶段五流程（5 步迭代框架）；`runs/detect/tree_yolo26s_iter{k}/` 命名约定。

**验收标准**：
- `pytest tests/test_self_train.py::TestLoopControl` 全部通过
- `python scripts/self_train.py --help` 正常输出
- 用 mock 数据跑通 3 轮循环（不调用真实 GPU），程序正常退出且日志格式正确

**代码量估计**：≤ 100 行（脚本 ~70 行，测试 ~30 行）

---

### Stage D2：bbox 交集过滤 + 置信度递增策略

**目标**：实现 YOLO 预测与 GDINO 原始输出的 bbox 交集计算（IoU > 0.5）及置信度递增逻辑。

**具体任务**：

1. 实现 `compute_intersection(yolo_boxes, gdino_boxes, iou_thr=0.5)` 函数：
   - 计算两组 bbox 的成对 IoU（可用 `torchvision.ops.box_iou`）
   - 保留 YOLO 预测中，至少有一个 GDINO bbox 与其 IoU > 0.5 的框（Proposal 阶段五步骤 2）
   - 输出：accepted boxes（YOLO 坐标系，xyxy）
2. 实现 `get_score_threshold(k, base_thr=0.35, step=0.05, max_thr=0.55)` 函数：
   - 第 k 轮写入阈值：`min(base_thr + (k-1) * step, max_thr)`（Proposal 实践建议）
   - 第 1 轮 0.35，第 2 轮 0.40，…，上限 0.55
3. 补充 `tests/test_self_train.py`：
   - 单元测试 `compute_intersection`：4 个 YOLO 框中有 2 个与 GDINO 重叠（IoU > 0.5），验证输出恰好 2 框
   - 单元测试 `get_score_threshold`：k=1→0.35, k=2→0.40, k=5→0.55（上限验证）
   - 验证无重叠时返回空 Tensor（不报错）

**实现依据**：Proposal 阶段五步骤 2（`accepted = P_k ∩ G`，IoU > 0.5）；置信度递增策略（"第 1 轮写入阈值 0.35，后续每轮 +0.05"）。

**验收标准**：
- `pytest tests/test_self_train.py` 全部通过，覆盖率 ≥ 80%
- 端到端自训练运行 2 轮（少量帧，不要求收敛），手工 val mAP 第 2 轮 ≥ 第 1 轮（或收敛退出）
- 每轮的置信度阈值在日志中正确打印

**代码量估计**：≤ 90 行（脚本新增 ~60 行，测试新增 ~30 行）

---

### Phase D 输出质量评价

在自训练结束后，评估迭代提升效果，判断是否达到预期收益。

| 检查项 | 方法 | 合格标准 | 不合格时的处置 |
|--------|------|----------|----------------|
| **收敛状态** | 查看 `self_train.py` 输出日志，确认退出原因 | 因"连续 2 轮无改善"退出（正常收敛）；而非因 `max_iter` 强制终止 | 若 5 轮均未收敛，说明自训练无法在此数据上稳定改善；接受当前最优轮 |
| **mAP 提升量** | 对比各轮 `best.pt` 在手工 val set 上的 mAP@0.5 | 最优轮 mAP ≥ Phase C 基线 + 0.05（至少提升 5 个点） | 若提升 < 5 点：检查手工 val set 标注质量；或接受 Phase C 结果直接部署 |
| **标签纯化趋势** | 统计各轮 `labels/all/` 中 bbox 平均数量 | 逐轮略有减少（交集过滤在纯化）；若反而增加说明过滤逻辑有 bug | 检查 `compute_intersection` 的 IoU 计算是否正确 |
| **置信度阈值正常递增** | 查看日志中每轮打印的阈值 | 第 k 轮阈值 = `min(0.35 + (k-1)*0.05, 0.55)` | 若不符，检查 `get_score_threshold` 调用位置 |
| **最优轮视觉质量** | 用最优轮 `best.pt` 对 10 张未见过的帧推理，人工审查 | 相比 Phase C 模型，误检减少或 bbox 更精准 | 若视觉质量无改善：放弃 Phase D 结果，沿用 Phase C 的 `best.pt` |
| **手工 val set 覆盖** | 确认手工标注 val set 包含多种场景（不同天气、光照、树种） | 至少包含 3 种不同场景的帧 | 若 val set 场景单一，mAP 数字不具参考价值 |

**最终模型选取规则**：

```
best_model = argmax_k(mAP@0.5 on manual_val_set, k in 0..max_iter)
# k=0 对应 Phase C 的 best.pt（自训练基线）
```

若 Phase D 最优轮 mAP 与 Phase C 基线差距 < 0.03，则保留 Phase C 模型（自训练收益不显著）。

---

## 汇总：文件清单与代码量

| 文件 | Phase/Stage | 估计行数 | 说明 |
|------|-------------|----------|------|
| `scripts/extract_video_frames.py` | A1, A2 | ~100 | 新建 |
| `tests/test_extract_video_frames.py` | A1, A2 | ~50 | 新建 |
| `scripts/generate_pseudo_labels.py` | B1, B2, B3 | ~180 | 新建 |
| `scripts/split_dataset.py` | B3 | ~60 | 新建 |
| `scripts/visualize_pseudo_labels.py` | B3 | ~40 | 新建（质检用，不计入测试） |
| `tests/test_pseudo_labels.py` | B1, B2, B3 | ~100 | 新建 |
| `scripts/train_yolo26.py` | C1, C2 | ~95 | 新建 |
| `scripts/validate_yolo26.py` | C3 | ~40 | 新建 |
| `tests/test_train_yolo26.py` | C1, C2, C3 | ~75 | 新建 |
| `scripts/self_train.py` | D1, D2 | ~130 | 新建（可选） |
| `tests/test_self_train.py` | D1, D2 | ~60 | 新建（可选） |

**Phase A 总计**：~150 行（≤500 行限制内）  
**Phase B 总计**：~380 行（质检脚本 `visualize_pseudo_labels.py` ~40 行不计入测试覆盖，实际计入 ~340 行；≤500 行限制内）  
**Phase C 总计**：~210 行（`train_yolo26.py` ~95 + `validate_yolo26.py` ~40 + `test_train_yolo26.py` ~75；≤500 行限制内）  
**Phase D 总计**：~190 行（可选；≤500 行限制内）

---

## 关键决策引用索引

| 决策 | 来源（Proposal 章节） | 具体值 |
|------|-----------------------|--------|
| 文本提示 `"tree."` | 阶段二 2.1 | 单词形式，已在 TDUS 验证（recall=100%） |
| box_threshold | 阶段二 2.2 | 0.30（候选框召回） |
| text_threshold | 阶段二 2.2 | 0.25 |
| 写入阈值 | 阶段二 2.2 | 0.35（初始值） |
| NMS IoU | 阶段二 2.3 | 0.45（比标准 0.5 稍严） |
| JPEG 质量 | 阶段零骨架代码 | 95（骨架代码写 95，避免压缩伪影；"≥ 90" 是最低要求，实际取 95） |
| 等间隔采样 N | 阶段一 1A | 默认 20（20 fps 视频，≈1 fps） |
| diff 阈值 | 阶段一 1B | 5.0（灰度帧差均值） |
| 划分策略 | 阶段三 | 跨日期划分：20240319 两视频 → train，20240530 两视频 → val |
| 模型 | 阶段四 | `yolo26s.pt`（精度/速度均衡） |
| imgsz | 阶段四 | 1280（与 GDINO 有效分辨率 1333×749 对齐，避免小树标注丢失） |
| batch | 阶段四 | 2（RTX 3060 12GB 实测：AutoBatch 方形 1280 → batch=1；rect 1280×736 可用 batch=2） |
| freeze | 阶段四 | 10（冻结 backbone 前 10 层） |
| epochs / patience | 阶段四 | 100 / 20（early stopping） |
| optimizer | 阶段四 | `"auto"` → AdamW（lr0=1e-3） |
| mosaic | 阶段四 | 0.5（降低，避免破坏 bbox 语义） |
| 自训练 IoU | 阶段五步骤 2 | > 0.5（交集筛选） |
| 置信度递增步长 | 阶段五实践建议 | +0.05/轮，上限 0.55 |
