# Proposal: YOLOE-26N 替换 GDino 用于伪标签生成

**Status**: Draft  
**Date**: 2026-05-31  
**Builds on**: `docs/proposals/proposal-tdus-mixed-dataset.md`

---

## 背景

当前伪标签生成（Phase 8.3）依赖 GroundingDINO-tiny（GDino），它以零样本方式在 TDUS 预缩放图像上生成 YOLO 格式 bbox。GDino 在 TDUS 上的检出率达到 3169/3170（≈100%），质量有保证，但有以下代价：

- **依赖链重**：`groundingdino`、`transformers`、`timm`，与主训练栈（ultralytics）完全独立。
- **推理慢**：~2–4 fps（单 GPU），3565 张 TDUS 图像花费约 30–60 分钟。
- **维护风险**：GDino 上游不活跃，ultralytics 生态升级时可能 break。

YOLOE-26N 是 Ultralytics YOLO26 系列的开放词汇检测变体（nano 规模），以文本提示驱动零样本检测。它与训练模型（YOLO26s）共享完全相同的依赖栈和 API，推理速度约快 5–10×。

### 关键数据对比

| 维度 | GDino-tiny（现状） | YOLOE-26N（候选） |
|------|-------------------|-----------------|
| 参数量 | ~172M | ~3M（nano） |
| 推理速度（单 GPU） | ~2–4 fps | ~15–30 fps |
| 依赖栈 | groundingdino + transformers | ultralytics 仅此一个 |
| 零样本能力来源 | 双编码器 GLIP 预训练 | RE-CLIP / 文本对齐头 |
| TDUS 检出率 | 3169/3170（≈100%） | 未知，需实测 |
| bbox 质量 | 经大规模 grounding 预训练，框较紧 | 依赖 YOLO 特征空间，未知 |
| 与训练域一致性 | 异构模型 | 同族模型，特征空间更接近 |

---

## 目标

1. 验证 YOLOE-26N 在 TDUS（行人视角、大目标、竖拍）上的零样本检测质量是否足以替代 GDino。
2. 若验证通过，以 YOLOE-26N 重新生成全套 TDUS 伪标签，消除 GDino 依赖。
3. 对比两套伪标签训练出的 YOLO26s 的 TDUS 分类准确率，确认替换不引入退化。

---

## 方案设计

### 核心思路：消融式验证，通过后再替换

不直接替换，而是在验证集的 100 张子样本上对比 YOLOE-26N 与 GDino 的框质量，以量化指标决定是否替换。

**关键决策**：

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 验证规模 | 100 张 TDUS val 图像 | 足以统计检出率；val 集有 GDino 参考标签可直接对比 |
| 对比指标 | 检出率 + mean IoU（vs GDino bbox）+ 框数分布 | 三项共同衡量质量 |
| 通过阈值 | 检出率 ≥98%，mean IoU ≥0.65 | GDino 基线约 100%；IoU ≥0.65 说明主要树体框重叠 |
| 失败处理 | 保留 GDino，YOLOE-26N 作为备选记录在案 | 不破坏现有 Phase 8 结果 |
| 置信度阈值 | 初始 0.25，消融 0.15 / 0.35 | YOLOE 置信度分布与 GDino 不同，需单独调参 |

### YOLOE-26N 推理接口

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26n.pt")
results = model.predict(
    source="data/tdus_resized/val/img/",
    text=["tree"],        # 文本提示
    conf=0.25,
    iou=0.5,
    imgsz=1280,
    save=False,
)
```

输出 YOLO txt 格式逻辑与现有 `generate_pseudo_labels.py` 中 `run_inference_flat()` 相同，只需替换推理后端。

### 与 GDino 的框质量对比

```python
# 对 100 张 val 图像，计算 YOLOE bbox 与 GDino bbox 的 mean IoU
# GDino 标签已在 data/tdus_resized/val/labels/ 中
# 匹配策略：Hungarian matching，每张图最高 IoU 配对
```

若一张图 YOLOE 检出多框而 GDino 只有一框，则取与 GDino 框 IoU 最高的 YOLOE 框参与统计（其余框视为多余检测，不计入 IoU 但记录框数分布）。

### 脚本改动

| 文件 | 改动 | 幅度 |
|------|------|------|
| `scripts/generate_pseudo_labels.py` | 新增 `--backend {gdino,yoloe}` 参数；YOLOE 路径复用 `run_inference_flat()` 接口 | ~30 行 |
| `scripts/compare_pseudo_labels.py` | **新建**：计算两套标签的检出率差异、mean IoU、框数分布 | ~60 行 |

---

## 权衡分析

### YOLOE-26N（nano）是否足够大？

YOLOE-26N 是同系列最小模型，参数量约为 GDino-tiny 的 1/50。对于"tree"这种单一粗粒度类别，零样本检测难度低；但对于树冠紧贴、多棵树重叠的场景（TDUS 中较常见），nano 规模的特征容量可能不足以区分树与背景。若 nano 检出率不达标，可尝试 YOLOE-26S（参数量约 11M）。

### 特征空间一致性的潜在收益

YOLOE-26N 与训练模型 YOLO26s 共享骨干架构，伪标签框由类似特征空间产生，理论上更符合 YOLO26s 的"看法"。但这一优势尚未有实验支撑，不能作为替换的主要理由。

### 置信度阈值的不可迁移性

GDino 的默认 CONF=0.35 在 TDUS 上产生接近 100% 检出率；YOLOE 的置信度分布由不同训练目标决定，**不能直接沿用同一阈值**，需在验证集上独立校准。

---

## 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| YOLOE-26N 检出率低于 GDino（<98%） | 中 | 高 | 回退到 GDino；可尝试 YOLOE-26S |
| bbox 系统性偏小或偏大 | 中 | 中 | 调整置信度阈值；若无法改善则放弃替换 |
| YOLOE-26N 在 ultralytics 当前版本中不支持纯文本提示 | 低 | 高 | 确认版本后再安装；若 API 不兼容则等待稳定版本 |
| 替换后 SVM 分类准确率下降 | 低 | 高 | 必须重跑 benchmark_pipeline.py 对比两套伪标签训练出的模型 |
| 验证通过但全集替换后质量不稳定 | 低 | 中 | 全集替换后检查 detection_failures 分布；与 GDino 版本的失败图像对比 |

---

## 成功指标

| 指标 | 目标值 | GDino 基线 |
|------|--------|-----------|
| 100 张 val 子集检出率 | ≥98% | 100%（395/395） |
| mean IoU（YOLOE vs GDino bbox） | ≥0.65 | — |
| 框数分布（多框率 / 漏框率） | 与 GDino ±20% 以内 | — |
| 全集替换后 TDUS top-1 分类准确率 | ≥70% | 33.2%（Phase 8 前基线） |
| 全集替换后路面 mAP50 | ≥0.85 | 0.886 |

---

## 参考

- 当前 GDino 伪标签：`data/tdus_resized/{train,val}/labels/`（Phase 8.3 产物）
- 对应 Proposal：`docs/proposals/proposal-tdus-mixed-dataset.md`
- YOLOE 文档：Ultralytics YOLOE（ultralytics >= 8.3）
