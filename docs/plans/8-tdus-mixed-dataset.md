# Plan: TDUS Mixed Dataset — Fine-tune YOLO on Urban Street Trees

**状态**: Draft  
**日期**: 2026-05-31  
**对应 Proposal**: `docs/proposals/proposal-tdus-mixed-dataset.md`  
**Phase 编号**: 8（接续 `5-6-7-higher-res-frames-retrain.md`）

---

## 前置条件

本 Phase 依赖以下已完成的工件：

| 工件 | 路径 | 状态 |
|------|------|------|
| 路面视频统一数据集 | `data/pseudo_labels_half/all_4videos/` | Phase 5–7 已完成，已验证存在 |
| 路面视频最优权重 | `runs/detect/tree_yolo26s_halfres/weights/best.pt` | mAP50=0.886，已下载到本地 |
| TDUS 原始数据集 | `tdus_data/{train,val,test}/` | train=3170, val=395, test=386 |

在开始任何 Stage 之前，确认以上路径均存在。

---

## 总体架构

```
Phase 8.1（本地）: 预缩放 TDUS 图像 → data/tdus_resized/
    └── Stage 8.1: scripts/resize_images.py（新建）

Phase 8.2（本地）: 扩展 generate_pseudo_labels.py 支持平铺目录
    └── Stage 8.2: --flat-dir 模式（TDD）

Phase 8.3（本地）: 为 TDUS train/val 生成 GDino 伪标签
    └── Stage 8.3: 运行 generate_pseudo_labels.py --flat-dir

Phase 8.4（本地）: 构建统一数据集 data/unified_halfres_tdus/
    └── Stage 8.4: scripts/build_unified_dataset.py（新建，TDD）

Phase 8.5（本地）: 扩展 train_yolo26.py CLI 参数
    └── Stage 8.5: --lr0 / --patience / --close-mosaic（TDD）

Phase 8.6（Colab）: 从 best.pt 微调
    └── Stage 8.6: notebooks/finetune_yolo26s_colab.ipynb（新建）
```

---

## Stage 8.1 — 预缩放 TDUS 图像

**目标**：将 `tdus_data/{train,val}/img/` 中的 3024×4032 JPEG 缩放到最长边=1280（输出 960×1280），存入 `data/tdus_resized/`。

**输入**：
- `tdus_data/train/img/`（3170 张）
- `tdus_data/val/img/`（395 张）
- **不处理** `tdus_data/test/img/`（386 张，严格禁止入训练集）

**输出**：
```
data/tdus_resized/
├── train/img/   # 960×1280 JPEG
└── val/img/     # 960×1280 JPEG
```

**脚本**：新建 `scripts/resize_images.py`

```
用法：
.venv/bin/python scripts/resize_images.py \
    --src tdus_data/train/img \
    --dst data/tdus_resized/train/img \
    --max-edge 1280

.venv/bin/python scripts/resize_images.py \
    --src tdus_data/val/img \
    --dst data/tdus_resized/val/img \
    --max-edge 1280
```

**脚本规格**：
- CLI 参数：`--src`、`--dst`、`--max-edge`（默认 1280）、`--quality`（JPEG 质量，默认 90）
- 逐文件处理（无 batch），用 tqdm 显示进度
- 输出文件名与输入相同（保留中文/空格文件名）
- 跳过已存在的输出文件（支持断点续传）
- 磁盘节省估算：3024×4032→960×1280，像素压缩 8.4×，JPEG 下约 **8–10× 文件体积减小**（实测 train 原图 6.4GB，预计缩放后 ~0.6–0.8GB）

**验收**：
- 输出 `data/tdus_resized/train/img/` 含 3170 张，尺寸均为 960×1280
- 输出 `data/tdus_resized/val/img/` 含 395 张，尺寸均为 960×1280

---

## Stage 8.2 — 扩展 generate_pseudo_labels.py 支持平铺目录

**背景**：现有脚本以视频为组织单位（`data/frames_half/{video_name}/`），TDUS 图像是平铺目录（`data/tdus_resized/train/img/`）。

**修改**：在 `scripts/generate_pseudo_labels.py` 中新增 `--flat-dir` 模式。

**新增 CLI 参数**：
```
--flat-dir PATH    以平铺目录模式处理单个图像目录（与 --frames-base/--videos 互斥）
--flat-out PATH    平铺模式的标签输出目录
--flat-conf FLOAT  置信度阈值（默认与现有 --conf 一致）
```

**输出结构**（平铺模式）：
```
<flat-out>/          # 直接包含 YOLO txt 文件
    <stem>.txt       # 与输入图像 stem 同名
```

**TDD**：先写 `tests/test_generate_pseudo_labels_flat.py`，测试：
1. `--flat-dir` 与 `--flat-out` 参数被正确解析
2. 平铺目录模式输出 txt 文件数量与输入图像数量一致（mock GDino）
3. 无检测结果时写空 txt 文件并将该文件路径记录到 `<flat-out>/detection_failures.txt`（不报错，但需留存列表供 Stage 8.4 过滤）

**验收**：现有测试通过；新增测试通过；对 5 张 TDUS 样本图运行并人工检查框质量。

---

## Stage 8.3 — 生成 TDUS GDino 伪标签

**目标**：为 `data/tdus_resized/{train,val}/img/` 生成 YOLO 格式伪标签。

**执行命令**（本地，需 GPU，约 30–60 分钟）：
```bash
# train split
.venv/bin/python scripts/generate_pseudo_labels.py \
    --flat-dir data/tdus_resized/train/img \
    --flat-out data/tdus_resized/train/labels

# val split
.venv/bin/python scripts/generate_pseudo_labels.py \
    --flat-dir data/tdus_resized/val/img \
    --flat-out data/tdus_resized/val/labels
```

**质量检查**（生成后）：
```bash
# 统计有框文件占比（检出率）
python3 -c "
from pathlib import Path
lbls = list(Path('data/tdus_resized/train/labels').glob('*.txt'))
nonempty = sum(1 for p in lbls if p.stat().st_size > 0)
print(f'{nonempty}/{len(lbls)} = {nonempty/len(lbls):.1%} detection rate')
"
```

**目标**：train 检出率 ≥80%（TDUS 图像树木显著，GDino 应有高召回率）。

**检测失败过滤**：
```bash
# 查看失败列表
cat data/tdus_resized/train/labels/detection_failures.txt
# 统计失败率及按物种分布（检测失败是否集中于某几个物种）
python3 -c "
from pathlib import Path
fails = Path('data/tdus_resized/train/labels/detection_failures.txt').read_text().splitlines()
from collections import Counter
species = Counter(Path(p).stem.split('_tree_')[0] for p in fails if p)
for sp, n in species.most_common(10):
    print(f'{n:4d}  {sp}')
"
```

> 若某物种失败率异常高（>50%），需调整 CONF_THRESH 或改用 GT bitmap 转 bbox 作为该物种的标注来源。检测失败的图像**不进入统一数据集**（Stage 8.4 依据 `detection_failures.txt` 排除）。

**人工抽检**：随机抽 20 张有框图像，用 `scripts/visualize_labels.py`（或 matplotlib）叠加框，确认框覆盖树冠主体而非背景。

---

## Stage 8.4 — 构建统一数据集

**目标**：将路面视频帧（`data/pseudo_labels_half/all_4videos/`）和 TDUS 预缩放帧（`data/tdus_resized/`）合并为 `data/unified_halfres_tdus/`。

**脚本**：新建 `scripts/build_unified_dataset.py`

**CLI 参数**：
```
--road-data        PATH   路面视频数据集根目录（含 images/train, images/val, labels/train, labels/val）
--tdus-data        PATH   TDUS 预缩放数据集根目录（含 train/img, train/labels, val/img, val/labels）
--tdus-failures    PATH   Stage 8.3 生成的 detection_failures.txt（过滤无框图像，可选）
--out              PATH   输出目录
--symlink                 使用符号链接而非复制（默认 False；注意 TDUS 文件名含特殊字符，建议先验证 glob 可达性）
```

> **symlink 注意**：TDUS 文件名含空格和括号（如 `Prunus cerasifera f. atropurpurea_tree_1 (64).jpg`），部分 YOLO dataloader 在 glob 时对特殊字符处理有差异。默认使用 copy（--symlink 为可选），确保最大兼容性。

**输出结构**：
```
data/unified_halfres_tdus/
├── images/
│   ├── train/   # 路面 ~2071 帧 + TDUS（3170 减去检测失败数）
│   └── val/     # 路面 ~2076 帧 + TDUS（395 减去检测失败数）
├── labels/
│   ├── train/
│   └── val/
└── data.yaml    # path: <abs>, train: images/train, val: images/val, nc: 1, names: [tree]
```

**文件名冲突处理**：路面视频帧名为 `frame_000001.jpg`，TDUS 文件名含物种名，天然无冲突。

**TDD**：先写 `tests/test_build_unified_dataset.py`，测试：
1. 输出 `data.yaml` 包含正确的 `nc=1`、`names=["tree"]`、绝对路径 `path`
2. train 图像数量 = 路面 train 数 + TDUS train 数（detection_failures 过滤后）
3. 每张图像对应 label 文件存在
4. 不复制 TDUS test 集（验证隔离）
5. detection_failures.txt 中列出的文件不出现在输出目录

**执行命令**：
```bash
.venv/bin/python scripts/build_unified_dataset.py \
    --road-data data/pseudo_labels_half/all_4videos \
    --tdus-data data/tdus_resized \
    --tdus-failures data/tdus_resized/train/labels/detection_failures.txt \
    --out data/unified_halfres_tdus
```

**验收**：`data/unified_halfres_tdus/images/train/` 含 ~5241 张（±检测失败数），`val/` 含 ~2471 张（±检测失败数）；无 TDUS test 集图像。

---

## Stage 8.5 — 扩展 train_yolo26.py CLI 参数

**目标**：新增微调所需的 CLI 参数，避免硬编码。

**新增参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lr0` | float | 1e-3 | 初始学习率（微调时用 2e-4） |
| `--patience` | int | 20 | EarlyStopping patience |
| `--close-mosaic` | int | 10 | 最后 N epoch 关闭 mosaic |

**TDD**：先在 `tests/test_train_yolo26.py` 中追加测试：
1. `test_default_lr0`：默认 lr0=1e-3
2. `test_override_lr0`：`--lr0 2e-4` 正确解析
3. `test_default_patience`：默认 patience=20
4. `test_train_passes_lr0_to_model`：`train(args)` 调用 `mock_model.train()` 时传入正确的 `lr0`

**修改 `scripts/train_yolo26.py`**：
- `add_argument("--lr0", type=float, default=1e-3)`
- `add_argument("--patience", type=int, default=20)`
- `add_argument("--close-mosaic", type=int, default=10)`
- 在 `train()` 中将这三个参数传入 `model.train()`

**验收**：所有测试通过（包含现有测试）。

---

## Stage 8.6 — Colab 微调 Notebook

**目标**：新建 `notebooks/finetune_yolo26s_colab.ipynb`，从 `best.pt` 微调统一数据集。

**Notebook 结构**：

```
Cell 1: Mount Drive + 安装依赖
Cell 2: 从 Drive 下载 best.pt + 统一数据集（或从 Drive 挂载）
Cell 3: Resume 检查（检测 Drive/weights/last.pt）
Cell 4: 训练（或 resume）
Cell 5: 上传 best.pt + last.pt + results.csv 回 Drive
```

**训练参数**（Cell 4 核心配置）：
```python
# best.pt 位于 Google Drive: TreeLearn/tree_yolo26s_halfres/weights/best.pt
model = YOLO(str(local_best_pt))  # 从 tree_yolo26s_halfres best.pt 微调
model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=1280,
    batch=-1,              # autobatch
    freeze=5,              # 微调首选：保留浅层特征，降低路面域遗忘风险
    lr0=2e-4,              # 微调学习率
    lrf=0.01,
    cos_lr=True,
    patience=15,
    rect=False,            # 跨域 mosaic 的必要条件
    mosaic=0.5,
    scale=0.5,
    translate=0.15,
    degrees=10.0,
    fliplr=0.5,
    flipud=0.0,
    mixup=0.0,
    close_mosaic=10,       # 与 --close-mosaic 默认值保持一致
    project="/content/runs/detect",
    name="tree_yolo26s_unified_halfres_tdus",
    device=0,
    workers=4,
    cache=True,
    deterministic=False,
    save=True,
)
```

> **消融实验**：若 TDUS 分类准确率未能显著提升，可将 `freeze` 调为 0 重跑，并对比两次训练的路面 road-only val mAP50（见 Stage 8.6 评估步骤），确认是否发生 catastrophic forgetting。

**Resume 逻辑**（与上一版本相同模式）：
```python
DRIVE_LAST = DRIVE_DIR / "weights" / "last.pt"
if DRIVE_LAST.exists():
    shutil.copy2(DRIVE_LAST, "/content/last.pt")
    model = YOLO("/content/last.pt")
    model.train(resume=True)
else:
    model = YOLO(str(local_best_pt))
    model.train(data=..., ...)
```

**road-only val mAP50 评估**（训练完成后）：
```python
# 在 Colab 中单独对路面 val 子集跑推理，获取 road-only mAP50
road_val_yaml = "/content/data/pseudo_labels_half/all_4videos/data.yaml"
metrics = model.val(data=road_val_yaml, imgsz=1280, batch=16, device=0)
print(f"Road-only mAP50: {metrics.box.map50:.4f}")  # 目标 ≥ 0.85
```

> `results.csv` 中的 mAP50 是混合 val 集（road + TDUS）的综合指标，**不能直接用于验证路面域质量**。必须单独对 road val 子集执行 `model.val()` 才能获得可对比的路面 mAP50。

**验收**：Notebook 在 Colab L4 上无报错启动训练；训练完成后 `results.csv` + road-only mAP50 日志上传 Drive。

---

## 执行顺序

```
前置检查：
  0. 确认 data/pseudo_labels_half/all_4videos/ 存在且含 images/train + val
     确认 runs/detect/tree_yolo26s_halfres/weights/best.pt 存在

本地（准备数据）：
  1. 新建并测试 scripts/resize_images.py
  2. 运行预缩放（tdus_data/train → data/tdus_resized/train/img, 约 5 分钟）
  3. 运行预缩放（tdus_data/val   → data/tdus_resized/val/img,   约 30 秒）
  4. 扩展 generate_pseudo_labels.py --flat-dir（TDD，含 detection_failures.txt 输出）
  5. 运行 GDino 伪标签生成（tdus_resized/train，约 20–40 分钟）
  6. 运行 GDino 伪标签生成（tdus_resized/val，约 2–5 分钟）
  7. 检查 detection_failures.txt：统计失败率及按物种分布，人工抽检 20 张有框图像
  8. 新建并测试 scripts/build_unified_dataset.py（TDD，含 --tdus-failures 过滤）
  9. 运行 build_unified_dataset.py → data/unified_halfres_tdus/
  10. 扩展 train_yolo26.py CLI 参数（TDD）
  11. 上传 data/unified_halfres_tdus/ 到 Google Drive

本地（可选验证）：
  12. 在本地 GPU 以 batch=2 运行 1 个 epoch 验证配置无报错

Colab：
  13. 新建 notebooks/finetune_yolo26s_colab.ipynb
  14. 从 Drive 下载 data/unified_halfres_tdus/ + best.pt（路径：TreeLearn/tree_yolo26s_halfres/weights/best.pt）
  15. 运行微调训练（50 epochs，freeze=5，autobatch）
  16. 训练完成后：对 road-only val 单独执行 model.val() 获取路面 mAP50
  17. 上传 best.pt + last.pt + results.csv + road_only_map50.txt 回 Drive

结果评估：
  18. 下载新 best.pt，替换 YOLO 路径权重
  19. 运行 benchmark_pipeline.py --detector yolo 对比新旧 TDUS 准确率
  20. 若 TDUS 准确率未达 70% 但检测框质量好，考虑在新 crop 上重提取嵌入并重训 SVM
```

---

## 文件清单

| 文件 | 类型 | 状态 |
|------|------|------|
| `scripts/resize_images.py` | 新建 | 待实现 |
| `scripts/generate_pseudo_labels.py` | 修改（+flat-dir） | 待实现 |
| `scripts/build_unified_dataset.py` | 新建 | 待实现 |
| `scripts/train_yolo26.py` | 修改（+lr0/patience/close-mosaic） | 待实现 |
| `notebooks/finetune_yolo26s_colab.ipynb` | 新建 | 待实现 |
| `tests/test_generate_pseudo_labels_flat.py` | 新建 | 待实现 |
| `tests/test_build_unified_dataset.py` | 新建 | 待实现 |
| `tests/test_train_yolo26.py` | 修改（追加测试） | 待实现 |
| `data/tdus_resized/` | 运行时产物 | 不提交 git |
| `data/unified_halfres_tdus/` | 运行时产物 | 不提交 git |

---

## 成功指标

| 指标 | 目标值 | 当前基线 |
|------|--------|---------|
| TDUS top-1 分类准确率（YOLO 路径，benchmark_pipeline） | ≥70% | 33.2% |
| 路面视频 mAP50（road-only val，Colab model.val() 单独测量） | ≥0.85 | 0.886 |
| TDUS 伪标签检出率（train） | ≥80% | 未测 |
| `data/unified_halfres_tdus/images/train/` 总数 | ~5241（±检测失败数） | — |
