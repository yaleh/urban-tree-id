# Plan: YOLOE-26N 替换 GDino 伪标签生成 — 消融验证

**状态**: Draft  
**日期**: 2026-05-31  
**对应 Proposal**: `docs/proposals/proposal-yoloe-pseudo-labels.md`  
**Phase 编号**: 9（接续 `8-tdus-mixed-dataset.md`）

---

## 前置条件

| 工件 | 路径 | 状态 |
|------|------|------|
| TDUS GDino 伪标签（val） | `data/tdus_resized/val/labels/` | Phase 8.3 已完成，作为对比基准 |
| TDUS 预缩放图像（val） | `data/tdus_resized/val/img/` | Phase 8.1 已完成 |
| TDUS GDino 伪标签（train） | `data/tdus_resized/train/labels/` | Phase 8.3 已完成，替换目标 |
| 统一数据集（GDino 版） | `data/unified_halfres_tdus/` | Phase 8.4 已完成，作为对比基线 |
| 微调权重（GDino 版） | Phase 8 Colab 训练产物 | Phase 8.6 完成后方可进行完整对比 |

**开始前必须确认**：
1. `ultralytics` 版本支持 `YOLOE` 类及文本提示推理（`>=8.3`，以实际安装版本为准）
2. `data/tdus_resized/val/labels/` 非空（Phase 8.3 已生成 GDino 参考标签）

---

## 总体架构

```
Phase 9.1（本地）: 验证 YOLOE-26N 推理接口可用
    └── Stage 9.1: 安装/确认版本，5 张样本图冒烟测试

Phase 9.2（本地）: 100 张 val 子集消融对比
    └── Stage 9.2: 对比检出率、mean IoU、框数分布（vs GDino 参考标签）

Phase 9.3（本地，条件执行）: 扩展 generate_pseudo_labels.py 支持 YOLOE 后端
    └── 仅当 9.2 通过时执行

Phase 9.4（本地，条件执行）: 重生成全套 TDUS 伪标签（YOLOE 版）
Phase 9.5（本地，条件执行）: 重建统一数据集（YOLOE 版）
Phase 9.6（Colab，条件执行）: 从 best.pt 微调（YOLOE 伪标签）
Phase 9.7（本地，条件执行）: 对比两套伪标签的最终分类准确率
```

> **决策门**：Stage 9.2 是唯一决策点。若检出率 <98% 或 mean IoU <0.65，整个 Phase 9 终止，结论记录在本文档末尾，GDino 保持现状。

---

## Stage 9.1 — 验证 YOLOE-26N 推理接口

**目标**：确认 ultralytics 当前版本支持 YOLOE，并在 5 张 TDUS 样本图上运行冒烟测试。

**执行**：

```bash
# 确认版本
.venv/bin/python -c "import ultralytics; print(ultralytics.__version__)"

# 下载模型（首次运行自动下载）
.venv/bin/python - <<'EOF'
from ultralytics import YOLOE
model = YOLOE("yoloe-26n.pt")

import glob, pathlib
samples = sorted(pathlib.Path("data/tdus_resized/val/img").glob("*.jpg"))[:5]
results = model.predict(
    source=[str(p) for p in samples],
    text=["tree"],
    conf=0.25,
    iou=0.5,
    imgsz=1280,
    save=False,
    verbose=True,
)
for r, p in zip(results, samples):
    print(f"{p.name}: {len(r.boxes)} boxes")
EOF
```

**验收**：5 张图无报错，至少 4/5 检出至少 1 个框（冒烟级别，不要求 bbox 质量）。

**若 API 不兼容**：记录 ultralytics 版本和报错信息，Phase 9 终止。不升级 ultralytics（可能影响主训练脚本兼容性）。

---

## Stage 9.2 — 100 张 val 子集消融对比

**目标**：量化 YOLOE-26N 与 GDino 在 TDUS val 子集上的框质量差异。

### 新建 `scripts/compare_pseudo_labels.py`

**功能**：
1. 从 `--yoloe-dir` 和 `--gdino-dir` 分别读取 YOLO txt 标签
2. 对每张共有图像，计算 YOLOE bbox 与 GDino bbox 的最大 IoU（Hungarian matching）
3. 输出：检出率对比、mean IoU、框数分布（0框/1框/多框）

**CLI**：
```
.venv/bin/python scripts/compare_pseudo_labels.py \
    --img-dir    data/tdus_resized/val/img \
    --gdino-dir  data/tdus_resized/val/labels \
    --yoloe-dir  /tmp/yoloe_val_labels \
    --n-sample   100 \
    --conf       0.25 \
    --output     /tmp/yoloe_comparison.json
```

**输出示例**：
```
=== YOLOE-26N vs GDino (n=100) ===
检出率: YOLOE 97/100 (97.0%)  GDino 100/100 (100.0%)
mean IoU (matched pairs): 0.71
框数分布:
  YOLOE 0框: 3   1框: 72   多框: 25
  GDino 0框: 0   1框: 68   多框: 32
```

**通过阈值**（两项必须同时满足）：
- 检出率 ≥ 98%
- mean IoU ≥ 0.65

**置信度消融**（若 conf=0.25 未达阈值）：
```bash
# 尝试更低置信度
.venv/bin/python scripts/compare_pseudo_labels.py \
    --conf 0.15 --output /tmp/yoloe_comparison_conf015.json ...
```

若任意置信度均不达标，Phase 9 在此终止。

**验收**：`/tmp/yoloe_comparison.json` 生成，两项指标均达标，决定最优 `conf` 值。

---

## Stage 9.3 — 扩展 generate_pseudo_labels.py 支持 YOLOE 后端

**仅当 Stage 9.2 通过时执行。**

**修改 `scripts/generate_pseudo_labels.py`**：

新增 CLI 参数：
```
--backend {gdino,yoloe}   推理后端（默认 gdino，向后兼容）
--yoloe-model PATH        YOLOE 模型路径（默认 yoloe-26n.pt）
```

实现细节：
- `--backend yoloe` 时，用 `YOLOE(model).predict(source, text=["tree"], conf=args.conf, ...)` 替换 GDino 推理
- 其余逻辑（YOLO txt 写出、detection_failures.txt 记录、flat/video 模式）完全复用
- `--backend gdino` 路径不改动（保持现有行为）

**TDD**：在 `tests/test_generate_pseudo_labels_flat.py` 中追加：
- `test_yoloe_backend_writes_labels`：mock YOLOE，验证输出格式与 gdino 后端相同
- `test_yoloe_empty_detection_writes_failure_log`：空检测时写 detection_failures.txt

**验收**：所有现有测试通过；新增测试通过；`--backend gdino` 路径无变化。

---

## Stage 9.4 — 重生成全套 TDUS 伪标签（YOLOE 版）

**仅当 Stage 9.3 完成时执行。**

输出目录使用新路径，不覆盖 GDino 版本：

```bash
# train（约 5–10 分钟，vs GDino 20–40 分钟）
.venv/bin/python scripts/generate_pseudo_labels.py \
    --backend yoloe \
    --flat-dir data/tdus_resized/train/img \
    --flat-out data/tdus_resized_yoloe/train/labels \
    --conf <Stage_9.2_最优conf>

# val
.venv/bin/python scripts/generate_pseudo_labels.py \
    --backend yoloe \
    --flat-dir data/tdus_resized/val/img \
    --flat-out data/tdus_resized_yoloe/val/labels \
    --conf <Stage_9.2_最优conf>
```

**输出路径**：`data/tdus_resized_yoloe/`（与 `data/tdus_resized/` 并存）

**质量检查**：
```bash
python3 -c "
from pathlib import Path
lbls = list(Path('data/tdus_resized_yoloe/train/labels').glob('*.txt'))
nonempty = sum(1 for p in lbls if p.stat().st_size > 0)
print(f'{nonempty}/{len(lbls)} = {nonempty/len(lbls):.1%}')
"
```

目标：train 检出率 ≥98%。

---

## Stage 9.5 — 重建统一数据集（YOLOE 版）

**仅当 Stage 9.4 完成时执行。**

```bash
.venv/bin/python scripts/build_unified_dataset.py \
    --road-data   data/pseudo_labels_half/all_4videos \
    --tdus-data   data/tdus_resized_yoloe \
    --tdus-failures data/tdus_resized_yoloe/train/labels/detection_failures.txt \
    --out         data/unified_halfres_tdus_yoloe
```

**输出**：`data/unified_halfres_tdus_yoloe/`（与 GDino 版 `data/unified_halfres_tdus/` 并存）

---

## Stage 9.6 — Colab 微调（YOLOE 伪标签版）

**仅当 Stage 9.5 完成时执行。**

复用 `notebooks/finetune_yolo26s_colab.ipynb`，修改以下变量：

```python
DRIVE_DATA_DIR = DRIVE_DIR / "unified_halfres_tdus_yoloe"   # 改为 YOLOE 版数据集
RUN_NAME = "tree_yolo26s_unified_halfres_tdus_yoloe"        # 区分两次实验
```

所有训练超参数与 Phase 8.6 完全相同（freeze=5, lr0=2e-4, rect=False, mosaic=0.5, epochs=50）。

**产物**：
```
gdrive:TreeLearn/tree_yolo26s_unified_halfres_tdus_yoloe/
├── weights/best.pt
├── results.csv
└── road_only_map50.txt
```

---

## Stage 9.7 — 对比两套方案

**仅当 Stage 9.6 完成时执行。**

| 指标 | Phase 8（GDino） | Phase 9（YOLOE） | 目标 |
|------|-----------------|-----------------|------|
| TDUS top-1 分类准确率 | 待测 | 待测 | ≥70% |
| 路面 mAP50（road-only val） | 待测 | 待测 | ≥0.85 |
| 伪标签生成时间（3565 张） | ~30–60 分钟 | ~5–10 分钟 | 速度验证 |
| 依赖复杂度 | groundingdino + transformers | ultralytics only | 降低 |

**决策规则**：
- 若 YOLOE 版两项准确率指标均 ≥ GDino 版 ×0.95（即不超过 5% 相对退化），则采用 YOLOE 替换 GDino，删除 GDino 依赖。
- 否则保留 GDino，YOLOE 方案归档。

---

## 执行顺序

```
前置检查：
  0. 确认 Phase 8.6（Colab 微调）已完成，GDino 版基线指标已记录

本地（验证阶段）：
  1. Stage 9.1: 确认 YOLOE-26N API 可用（5 张冒烟测试）
  2. Stage 9.2: 100 张 val 子集消融对比（新建 compare_pseudo_labels.py）
     → 决策门：两项指标均达标？
       YES → 继续 Stage 9.3
       NO  → Phase 9 终止，记录结论

本地（替换阶段，条件执行）：
  3. Stage 9.3: 扩展 generate_pseudo_labels.py（TDD）
  4. Stage 9.4: 重生成全套标签 → data/tdus_resized_yoloe/
  5. Stage 9.5: 重建统一数据集 → data/unified_halfres_tdus_yoloe/
  6. 上传 data/unified_halfres_tdus_yoloe/ 到 Google Drive

Colab（条件执行）：
  7. Stage 9.6: 微调（YOLOE 伪标签版，复用 finetune_yolo26s_colab.ipynb）

本地（评估）：
  8. Stage 9.7: 对比两套方案，执行 benchmark_pipeline.py
  9. 根据对比结果决定是否将 YOLOE 设为默认后端
```

---

## 文件清单

| 文件 | 类型 | 状态 |
|------|------|------|
| `scripts/compare_pseudo_labels.py` | 新建 | 待实现（Stage 9.2） |
| `scripts/generate_pseudo_labels.py` | 修改（+`--backend yoloe`） | 待实现（Stage 9.3，条件） |
| `tests/test_generate_pseudo_labels_flat.py` | 修改（追加 YOLOE 测试） | 待实现（Stage 9.3，条件） |
| `data/tdus_resized_yoloe/` | 运行时产物 | 不提交 git（条件） |
| `data/unified_halfres_tdus_yoloe/` | 运行时产物 | 不提交 git（条件） |

---

## 成功指标

| 指标 | 目标值 | 备注 |
|------|--------|------|
| Stage 9.2 检出率 | ≥98%（100 张样本） | 决策门 |
| Stage 9.2 mean IoU | ≥0.65 | 决策门 |
| 全集检出率（Stage 9.4） | ≥98% | 续行条件 |
| TDUS top-1 分类准确率（YOLOE 版） | ≥70% 且 ≥ GDino 版 ×0.95 | 最终替换判据 |
| 路面 mAP50（YOLOE 版） | ≥0.85 且 ≥ GDino 版 ×0.95 | 最终替换判据 |

---

## 结论记录（待填写）

> 执行 Stage 9.2 后在此记录量化结果和决策。
