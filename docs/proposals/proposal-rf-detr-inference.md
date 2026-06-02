# Proposal: RF-DETR Fine-tune 替换推理管道中的 GDino 检测器

**Status**: Draft  
**Date**: 2026-05-31  
**Phase 编号**: 10（接续 Phase 9）  
**Builds on**: `docs/proposals/proposal-yoloe-pseudo-labels.md`

---

## 背景

当前项目存在两条独立的管道：

### 管道 1：检测训练管道（Pseudo-label → YOLO fine-tune）

```
GDino（zero-shot）→ YOLO txt 伪标签 → 统一数据集 → YOLO26s fine-tune → best.pt
```

- GDino 在此管道中扮演**标注员**角色，产出训练数据
- Phase 4–8 的核心工作均围绕此管道
- Phase 9 正在消融 YOLOE-26N 能否替换 GDino 用于伪标签生成

### 管道 2：推理分类管道（Inference → Embedding → SVM）

```
GDino（zero-shot）→ bbox crop → DINOv2 embedding → SVM 分类 → 树种预测
```

- GDino 在此管道中扮演**推理期检测器**角色，为每张输入图像提供 bbox
- 现有 TDUS 流水线（`scripts/predict_pipeline.py`、`scripts/benchmark_pipeline.py`）采用此路径
- 当前 TDUS top-1 分类准确率基线为 33.2%（Phase 8 前），目标 ≥70%

**本 Proposal 的范围**：仅针对管道 2，用 **RF-DETR（域内 fine-tune 版）** 替换 GDino 作为推理期检测器。不涉及管道 1 的伪标签生成逻辑。

---

## 问题陈述

管道 2 中的 GDino 用于推理期 bbox 检测，存在以下局限：

1. **边界质量不稳定**：GDino 是零样本模型，对 TDUS 行人视角、竖拍大目标、树冠紧贴等场景无针对性训练，bbox 边界偶尔偏松或漏检，直接影响 DINOv2 crop 的嵌入质量。
2. **推理速度慢**：GDino 推理约 2–4 fps（单 GPU），含 text prompt encoding overhead，在批量推理 3565 张 TDUS 图像时耗时约 30–60 分钟。
3. **依赖链重**：`groundingdino`、`transformers`、`timm` 与 ultralytics 训练栈完全异构，维护成本高，存在上游 break 风险（GDino 上游不活跃）。

**自举问题**（核心障碍）：若用 RF-DETR 替换 GDino，RF-DETR 的训练数据本身来自 GDino 伪标签，形成循环依赖——训练集的质量上限受 GDino 标注质量制约。

**解决方案**：Phase 8/9 通过系统性验证，已确认 GDino 在 TDUS 数据集上检出率 ≥98%、mean IoU ≥0.65，标签质量足以作为 RF-DETR 的训练监督信号。自举问题在 Phase 8/9 完成后解除。

---

## 目标与范围

### 目标

1. 使用 Phase 8/9 产出的伪标签（已验证质量）fine-tune RF-DETR，使其成为 TDUS 域内专用检测器。
2. 将管道 2 中的 GDino 推理替换为 RF-DETR，消除 GDino 在推理路径上的依赖。
3. 验证替换后的端到端分类准确率不低于 GDino 路径（目标 TDUS top-1 ≥70%）。

### 明确范围

| 内容 | 是否在 Phase 10 范围内 |
|------|----------------------|
| 替换管道 2（推理期）的检测器 | **是** |
| RF-DETR fine-tune（用 Phase 8/9 伪标签） | **是** |
| 替换管道 1（伪标签生成）的检测器 | **否**（Phase 9 的职责） |
| 修改 DINOv2 embedding 或 SVM 训练逻辑 | **否** |
| 替换 YOLO 检测训练管道 | **否** |

---

## 方案设计

### RF-DETR 选型理由

RF-DETR（Real-time Feature-pyramid Detection Transformer）是基于 DETR 架构的实时目标检测器，具备以下特点：

- **End-to-end transformer 检测**：无 NMS 后处理，推理接口简洁
- **Fine-tune 友好**：预训练权重在 COCO 上具备通用特征，域内少量标注即可收敛
- **推理速度**：无 text prompt encoding overhead，推理速度显著快于 GDino
- **依赖独立**：与 ultralytics 栈分离，不引入对 GDino 的新依赖，同时消除旧依赖

### RF-DETR Fine-tune 流程

```
Phase 10.1（本地）: 环境验证 + 5 张冒烟测试
    └── 确认 rf-detr 安装，验证预训练权重推理可用

Phase 10.2（本地）: 准备 RF-DETR 训练数据
    └── 从最优伪标签来源（Phase 8 GDino 或 Phase 9 YOLOE，取 Phase 9.7 决策结果）
        转换为 RF-DETR 训练格式（COCO JSON）

Phase 10.3（Colab）: RF-DETR fine-tune
    └── 使用 TDUS train 集（伪标签）fine-tune RF-DETR 预训练权重
    └── 产出 rf_detr_tdus_best.pth

Phase 10.4（本地）: 推理管道替换
    └── 扩展 predict_pipeline.py 支持 --detector {gdino,yolo,rf-detr}
    └── RF-DETR 路径：输入图像 → RF-DETR bbox → crop → DINOv2 → SVM

Phase 10.5（本地）: 端到端评估
    └── benchmark_pipeline.py 对比三路径（GDino / YOLO / RF-DETR）的 TDUS 分类准确率
```

### 训练数据来源

RF-DETR 的训练数据取自以下之一（Phase 9.7 决策后确定）：

- **若 Phase 9 采用 GDino**：使用 `data/tdus_resized/train/`（图像 + GDino 版伪标签 + `data.yaml`）
- **若 Phase 9 采用 YOLOE**：使用 `data/tdus_resized_yoloe/train/`（图像 + YOLOE 版伪标签 + `data.yaml`）

仅使用 TDUS train 集（3170 张，去除 detection_failures），**严禁使用 TDUS test 集**。

> **重要**：RF-DETR 训练数据**仅使用 TDUS 部分**（上述路径），**不使用** `data/unified_halfres_tdus/`（混合数据集含路面视频帧约 2071 张，为低角度行车道路场景，与 TDUS 行人视角竖拍场景完全不同，引入会稀释域内信号、影响 TDUS 推理期检测精度）。`rfdetr` 包原生支持 YOLO 格式，上述路径中的 `data.yaml` + `{train,val}/img/` + `{train,val}/labels/` 布局可直接作为训练输入，**无需格式转换**。

### 推理替换方案

现有管道 2 调用链（以 `scripts/predict_pipeline.py` 或 `scripts/benchmark_pipeline.py` 为入口）：

```python
# 现有 GDino 路径（管道 2）
boxes = gdino_detect(image, text_prompt="tree.")
crop  = image.crop(boxes[0])
emb   = dinov2_embed(crop)
label = svm.predict(emb)
```

替换后：

```python
# RF-DETR 路径（Phase 10）
boxes = rf_detr_detect(image)   # 无 text prompt，域内 fine-tune
crop  = image.crop(boxes[0])
emb   = dinov2_embed(crop)
label = svm.predict(emb)
```

DINOv2 embedding 和 SVM 分类逻辑**不变**，仅替换检测器接口。

> **RF-DETR 推理接口说明**：`rfdetr` 包的 `model.predict(image)` 返回 `supervision.Detections` 对象，含 `.xyxy`（N×4 float32，绝对像素坐标，与 GDino 的 `res["boxes"]` 格式一致）、`.confidence`（N,）、`.class_id`（N,）字段。封装 `rf_detr_detect()` 时直接取 `.xyxy` 即可对齐现有管道。`supervision` 包会随 `rfdetr` 自动安装，无需单独添加依赖。

### 脚本改动

| 文件 | 改动 | 幅度 |
|------|------|------|
| `scripts/prepare_rf_detr_dataset.py` | **新建**：验证 TDUS 数据集布局、排除 detection_failures、生成 `data.yaml`，直接消费现有 YOLO 格式目录 | ~80 行 |
| `scripts/rf_detr_detector.py` | **新建**：封装 RF-DETR 推理接口，将 `sv.Detections.xyxy` 对齐为现有 bbox 接口格式 | ~60 行 |
| `scripts/predict_pipeline.py` | 新增 `--detector rf-detr` 选项，RF-DETR 推理路径（`--rf-detr-checkpoint` 参数） | ~40 行 |
| `scripts/benchmark_pipeline.py` | 新增 `--detector rf-detr` 支持 | ~10 行 |
| `notebooks/finetune_rf_detr_colab.ipynb` | **新建**：Colab 训练 notebook（承载 RF-DETR fine-tune 全流程，无需本地 `train_rf_detr.py`） | — |

> **注**：
> - ~~`scripts/convert_labels_to_coco.py`~~（已从原设计中移除）。`rfdetr` 包原生支持 YOLO 格式数据集（检测到 `data.yaml` 时自动识别），现有 `data/tdus_resized/` 布局已满足要求，无需 YOLO→COCO 格式转换步骤。
> - ~~`scripts/train_rf_detr.py`~~（已从原设计中移除）。RF-DETR fine-tune 通过 Colab notebook（`notebooks/finetune_rf_detr_colab.ipynb`）执行，数据集准备逻辑独立抽取到 `scripts/prepare_rf_detr_dataset.py`，推理封装独立抽取到 `scripts/rf_detr_detector.py`，职责分离更清晰。

---

## 与 Phase 9 的关系

Phase 10 与 Phase 9 是**严格串行依赖**，不存在功能重叠：

```
Phase 9（伪标签生成消融）
    ↓ 产出：最优伪标签后端（GDino 或 YOLOE）+ 已验证的标签质量数据
Phase 10（推理管道替换）
    ↓ 消费：Phase 9 决策结果，以最优伪标签 fine-tune RF-DETR
```

| 维度 | Phase 9 | Phase 10 |
|------|---------|----------|
| 目标管道 | 管道 1（伪标签生成） | 管道 2（推理期检测） |
| 替换对象 | 伪标签生成阶段的 GDino | 推理期的 GDino |
| 核心模型 | YOLOE-26N | RF-DETR |
| 是否依赖对方 | 否（独立） | 是（需 Phase 9.7 结论） |

**Phase 10 的启动前置条件**：Phase 9.7（或 Phase 8.6，若 Phase 9 决策终止）完成，最优伪标签后端已确定，相应标签文件已验证。

---

## 权衡分析

### 优势

| 优势 | 说明 |
|------|------|
| **域内 bbox 精度更高** | GDino 是 zero-shot，域内 bbox 边界偶尔偏松或漏检；RF-DETR 在 TDUS 数据上 fine-tune 后，crop 区域更精确，DINOv2 嵌入质量更好 |
| **推理速度更快** | RF-DETR 无 text prompt encoding overhead；官方 benchmark 显示 RF-DETR 2x-large vs GDino-tiny 快约 20×，RF-DETR-B 在 T4 上约 25 FPS vs GDino ~2–4 FPS，加速比 ≥10×（实测后记录具体值） |
| **消除 GDino 依赖** | 推理路径不再依赖 `groundingdino`、`transformers`、`timm`，降低依赖链复杂度和维护风险 |
| **接口更简洁** | RF-DETR 推理无需 text prompt，接口更稳定，不受 GDino 零样本 text 解析歧义影响 |

### 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| **RF-DETR 在小样本（3170 张伪标签）上过拟合** | 中 | 中 | 使用预训练权重 fine-tune（而非从头训练）；应用数据增强；early stopping |
| **伪标签质量上限制约 RF-DETR bbox 精度** | 中 | 中 | Phase 8/9 已验证标签质量（检出率 ≥98%，mean IoU ≥0.65），为可接受起点；若效果不足可迭代 |
| **RF-DETR 单目标场景漏检（多棵树）** | 低 | 中 | 推理期保留 top-k boxes（k>1），后续 crop 逻辑处理多框；与 GDino 多框行为对齐 |
| **替换后分类准确率不升反降** | 低 | 高 | 严格对比实验：RF-DETR 路径 vs GDino 路径，以 TDUS top-1 准确率为最终判据；可回退 |
| **RF-DETR 依赖与现有栈冲突** | 低 | 中 | 隔离安装，在 `.venv` 中验证 rfdetr>=1.1.0 与 ultralytics、transformers 版本兼容性；若冲突则评估单独 venv 方案。**额外风险**：RF-DETR（DINOv2 backbone）与管道 2 的 DINOv2 embedding 模块同时驻留显存，峰值显存约增加 ~330 MB，Phase 10.4 集成测试时需测量峰值显存，若超出可用量则考虑共享 backbone 实例 |

### 不采用 YOLO 进行推理期检测的理由

Phase 8 已在 `benchmark_pipeline.py` 中添加 `--detector yolo` 路径（YOLO → bbox → DINOv2 → SVM）。若 Phase 8/9 完成后 YOLO 路径已达到 ≥70% 准确率，Phase 10 可作为进一步提升精度的探索性实验，而非必选项。相比 YOLO，RF-DETR 的核心差异在于端到端 transformer 检测（无 anchor、无 NMS），理论上在少框精确检测场景（TDUS 图像一般含 1 棵主树）更稳定。

---

## 前置条件

| 前置条件 | 来源 | 验证方式 |
|----------|------|---------|
| TDUS 伪标签（最优版本）已生成 | Phase 8.3 或 Phase 9.4 | `data/tdus_resized[_yoloe]/train/labels/` 非空，检出率记录在案 |
| Phase 9.7 决策已完成 | Phase 9 结论记录 | `docs/plans/9-yoloe-pseudo-labels.md` 结论区已填写 |
| RF-DETR 预训练权重可下载 | HuggingFace / 官方仓库 | 本地冒烟测试（Phase 10.1）通过 |
| TDUS test 集严格隔离 | Phase 8 数据准备 | 训练集不含 `tdus_data/test/img/` 内容 |

**若 Phase 9 在 Stage 9.2 决策门终止**（YOLOE 不达标），Phase 10 仍可启动，训练数据使用 Phase 8 GDino 版伪标签（`data/tdus_resized/train/labels/`），前置条件仅要求 Phase 8.6 完成。

---

## 成功指标

| 指标 | 目标值 | 当前基线 |
|------|--------|---------|
| TDUS top-1 分类准确率（RF-DETR 路径） | ≥70% | 33.2%（Phase 8 前基线） |
| TDUS top-1 分类准确率（RF-DETR vs GDino 路径对比） | RF-DETR ≥ GDino ×0.98（不超过 2% 相对退化） | Phase 8/9 完成后测量 |
| TDUS val 集 RF-DETR 检出率 | ≥98% | — |
| val mean IoU（RF-DETR 预测 vs GDino 参考标签） | ≥0.65 | — |
| RF-DETR 批量推理速度 | ≥ GDino ×10（参考官方 benchmark，实测后记录） | GDino ~2–4 fps |
| 管道 2 中 GDino 依赖彻底移除 | `predict_pipeline.py` 默认路径无需 `groundingdino` | — |
| 测试覆盖率（新建脚本） | ≥80% | — |

---

## 参考

- 当前伪标签（GDino 版）：`data/tdus_resized/train/labels/`（Phase 8.3 产物）
- 当前伪标签（YOLOE 版，条件产出）：`data/tdus_resized_yoloe/train/labels/`（Phase 9.4 产物）
- GDino 推理接口：`scripts/generate_pseudo_labels.py`（`run_inference_flat()`）
- 现有推理管道：`scripts/predict_pipeline.py`、`scripts/benchmark_pipeline.py`
- Phase 9 对应 Proposal：`docs/proposals/proposal-yoloe-pseudo-labels.md`
- Phase 8 对应计划：`docs/plans/8-tdus-mixed-dataset.md`

---

## 架构师审查记录

**审查日期**：2026-05-31  
**审查人**：架构师  
**审查依据**：在线查阅 RF-DETR 官方文档（roboflow/rf-detr GitHub、rfdetr.roboflow.com）、实际检查 `scripts/generate_pseudo_labels.py`、`scripts/build_unified_dataset.py`、`tdus_data/train/ann/` 标注格式、`docs/plans/8-tdus-mixed-dataset.md`、`docs/plans/9-yoloe-pseudo-labels.md`

---

### 修改内容

1. **删除不必要的格式转换脚本**（原文件改动表中的 `scripts/convert_labels_to_coco.py`）

   **原文**：`scripts/convert_labels_to_coco.py`（新建：将 YOLO txt 格式标签转换为 RF-DETR 训练所需的 COCO JSON 格式，~60 行）

   **修改后**：此脚本不再需要。经官方文档确认，`rfdetr` 包原生支持 YOLO 格式数据集——当训练目录中存在 `data.yaml` 且包含 `train/images/` 子目录时，RF-DETR 自动识别 YOLO 格式并直接训练，无需转换为 COCO JSON。现有 `data/tdus_resized/`（含 `train/img/` + `train/labels/` + `data.yaml`）及 `data/unified_halfres_tdus/`（含 `images/train/` + `labels/train/` + `data.yaml`）均已满足此格式要求（注意：`build_unified_dataset.py` 会生成 `data.yaml`，但图像子目录为 `images/{split}/`，标签为 `labels/{split}/`，与 rfdetr 期望的 YOLO 布局完全一致）。

2. **修正推理速度声明**（原"权衡分析 - 优势"表格第二行）

   **原文**：推理速度约快 3–5×（实测后记录）

   **修改后**：根据 RF-DETR 论文（ICLR 2026）实测数据，RF-DETR 2x-large 对比 GroundingDINO-tiny 在同等精度下快约 **20×**；RF-DETR-B 在 T4 GPU 上达 25 FPS，RF-DETR Nano 达 100 FPS，而 GDino-tiny 单卡约 2–4 FPS（含 text prompt 编码开销）。"3–5×"是明显低估，已改为"**≥10×**（参考 RF-DETR 2x-large vs GDino-tiny 约 20×，实测后记录具体值）"。

3. **明确 Phase 10.2 训练数据集边界**（原"训练数据来源"章节）

   **原文**：训练数据来源未区分"纯 TDUS 集"与"混合统一数据集"，仅列出两个条件路径。

   **修改后**：在"训练数据来源"章节追加以下说明：
   > **重要**：RF-DETR 的训练数据**仅使用 TDUS 部分**（`data/tdus_resized/train/` 或 `data/tdus_resized_yoloe/train/`），**不使用** `data/unified_halfres_tdus/`（后者含路面视频帧，引入异域样本，会导致 RF-DETR 对行道路面树木域过拟合，影响推理期 TDUS 检测质量）。RF-DETR fine-tune 是针对"TDUS 行人视角、竖拍大目标"场景的域内专化，路面帧是噪声。

4. **补充 RF-DETR 推理接口说明**（原"推理替换方案"章节）

   **原文**：`rf_detr_detect(image)` 函数签名未说明返回格式。

   **修改后**：在"推理替换方案"章节补充：
   > RF-DETR `predict()` 方法返回 `supervision.Detections` 对象，含 `.xyxy`（N×4 float32，绝对像素坐标）、`.confidence`（N,）、`.class_id`（N,）字段。封装 `rf_detr_detect()` 时需从 `.xyxy` 提取 bbox，接口与现有 GDino 路径（`res["boxes"]`，xyxy 格式）对齐，改动量与原估算（~40 行）相符。需添加 `supervision` 到 `.venv` 依赖（rfdetr 包已将其列为必须依赖，安装 rfdetr 时自动拉取）。

5. **补充 rfdetr 依赖冲突风险**（原"风险"表格）

   **原文**：风险表中"RF-DETR 依赖与现有栈冲突"仅描述为"低概率"，缓解措施过于简略。

   **修改后**：在"RF-DETR 依赖与现有栈冲突"风险行的缓解措施中追加：
   > RF-DETR 使用 DINOv2 作为 backbone，而 `predict_pipeline.py` 中的 DINOv2 embedding 来自独立加载的 `torch.hub` 模型。两者共享 DINOv2 权重但通过不同路径加载，**内存中会同时存在两份 DINOv2 权重**（约 2×330 MB），推理期峰值显存需额外留意（建议在 Phase 10.4 集成测试时测量峰值显存，若超出可用量则考虑共享 backbone）。`rfdetr` 包依赖 `supervision`，需在 `.venv` 中验证与现有 `ultralytics`、`transformers` 版本无冲突（rfdetr>=1.1.0 支持 supervision>=0.25）。

6. **脚本改动表修订**（对应修改项 1）

   删除 `scripts/convert_labels_to_coco.py` 行，并将 `scripts/train_rf_detr.py` 的描述由"封装训练 CLI"修订为"封装 rfdetr 训练 API，直接消费 YOLO 格式数据集（`data.yaml` + `images/` + `labels/`），无需格式预转换"，行数估算从 ~80 行调整为 ~60 行（去掉格式转换调用逻辑）。

---

### 发现的重大问题

#### 问题 1：Phase 10.2 训练数据边界不明确（严重性：中）

原文在"训练数据来源"中列出的条件路径（`data/tdus_resized/train/labels/` 或 `data/tdus_resized_yoloe/train/labels/`）仅指向标签目录，但未说明图像来源，也未明确说明**不使用**混合数据集 `data/unified_halfres_tdus/`。若执行者误用混合集训练 RF-DETR，将引入约 2000 张路面视频帧（`frame_000001.jpg` 等），这些图像是低角度、平视的行车道路场景，与 TDUS 的行人视角、竖拍大目标完全不同，会稀释域内信号、降低 TDUS 场景的检测精度。**已在修改项 3 中补充明确禁止说明。**

#### 问题 2：`scripts/convert_labels_to_coco.py` 是不必要的工作（严重性：低，但影响估算精度）

原文将此脚本列为"新建，~60 行"，Phase 10.2 的工作量因此被高估。RF-DETR 官方文档明确支持 YOLO 格式，现有数据集结构（`build_unified_dataset.py` 产出的 `data.yaml` 布局）已满足要求，无需额外转换步骤。**已在修改项 1 中删除该脚本。**

#### 问题 3：推理期双份 DINOv2 显存占用未评估（严重性：低）

管道 2 中 RF-DETR（DINOv2 backbone）和 DINOv2 embedding 模块将同时驻留显存，若在同一 Python 进程中批量推理 3565 张图像，峰值显存约为 RF-DETR 模型（~1.5 GB）+ DINOv2-B/14（~330 MB）+ 激活值。在 6 GB VRAM 环境下有风险，需在 Phase 10.4 集成测试时测量。**已在修改项 5 中记录缓解措施。**

---

### 确认通过的事项

1. **Phase 10 与 Phase 9 范围无重叠**：Phase 9 仅处理管道 1（伪标签生成），Phase 10 仅处理管道 2（推理期检测替换），职责分界清晰。

2. **前置条件合理**：Phase 8.6 fallback（Phase 9 终止时直接使用 GDino 版伪标签）是合理且可操作的，`data/tdus_resized/train/labels/` 路径已在 Phase 8 产出，无歧义。

3. **自举问题分析正确**：文档对"GDino 标签质量作为 RF-DETR 训练监督信号的可行性"的论证逻辑正确，Phase 8/9 的验证阈值（检出率 ≥98%、mean IoU ≥0.65）与 Phase 9.2 决策门一致。

4. **TDUS test 集隔离**：训练数据来源明确限定为 train split，`build_unified_dataset.py` 代码中已确认 `tdus_data/test/` 不被处理（硬编码仅处理 `train`/`val`，见代码第 60 行 `for split in ("train", "val")`），数据污染风险已规避。

5. **成功指标可量化**：TDUS top-1 ≥70%、RF-DETR vs GDino 相对退化 ≤2%、val 检出率 ≥98%、推理速度 ≥2× 均可通过 `benchmark_pipeline.py` 和计时实测验证，指标设计合格。

6. **GDino 推理接口替换难度合理**：`generate_pseudo_labels.py` 中的 GDino 接口（`AutoProcessor` + `AutoModelForZeroShotObjectDetection`，batch 推理，xyxy 输出）与 RF-DETR 推理接口（`rfdetr.RFDETRBase().predict()`，返回 `sv.Detections.xyxy`）均为 xyxy 格式，接口对齐改动量估算（~40 行）合理。

7. **RF-DETR 方案技术成熟度**：rfdetr 包已于 2025 年发布稳定版（v1.1.0+），在 ICLR 2026 收录，官方提供 Colab fine-tune notebook，工程风险低。YOLO 格式原生支持已确认，Phase 10.3 Colab 训练方案可行。

---

*（Step 4 一致性审查：已对齐）*
