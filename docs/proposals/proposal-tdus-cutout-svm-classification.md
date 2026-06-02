# Proposal: TDUS Cutout-SVM Tree Species Classification

**Status**: Draft  
**Date**: 2026-05-28  
**Dataset**: TDUS (Tree Dataset of Urban Street) — train 3168 / val 395 / test 386 images, 22 species

---

## 背景

当前 CV 验证流程已在两类任务上积累了可复用的基础设施：

- **图像分类**（`validate_plant_seedlings.py`）：DINOv2 vits14 CLS token + SVM，已在 plant seedlings 及 erickendric urban street tree 数据集上验证，pipeline 完整可运行。
- **实例分割**（`validate_segmentation_tdus.py`、`validate_tree_extraction_sam.py`）：GT mask 解码、白背景抠图、SAM 分割，已在 TDUS 上取得 IoU ≈ 0.62。

TDUS 数据集的独特价值在于它同时提供：
1. **树种标签**（annotation JSON 中的 `classTitle` 字段）
2. **逐像素 GT 实例 mask**（Supervisely bitmap 格式，base64+zlib+PNG）

这为一个关键科学问题提供了天然实验条件：**在分类之前先用 GT mask 抠去背景，是否能提升 DINOv2 特征的判别力？**

---

## 目标

1. 构建"GT 抠图 → DINOv2 CLS token → SVM"的完整分类 pipeline（称为 **Cutout 方案**）。
2. 在完全相同条件下训练对照组（原图直接分类，称为 **Original 方案**）。
3. 在 TDUS val/test split 上定量比较两方案，评估背景去除对 22 类树种分类的影响。

---

## 方案设计

### 阶段一：预处理——抠图与尺寸归一化

**输入**：`tdus_data/{train,val,test}/` 中的原始图像及其 annotation JSON。

**步骤**：

1. **GT mask 解码**  
   复用 `validate_segmentation_tdus.py` 中已实现的 `decode_gt_mask(ann_path, img_w, img_h)` 函数，该函数封装了完整的 base64 → zlib 解压 → PIL PNG → boolean ndarray 流程，直接输出与原图等大的布尔掩膜。

2. **背景填白**  
   复用 `validate_tree_extraction_sam.py` 中的 `apply_mask_white_bg(img_np, mask)` 函数，将 mask 外所有像素设为 RGB (255, 255, 255)。  
   **注意**：该函数要求 `img_np` 为 `float32 [0, 1]` 格式（`np.array(pil_img) / 255.0`），输出同样为 `[0, 1]` float。保存为 JPEG 前须转换回 `(out * 255).astype(np.uint8)` 并通过 `Image.fromarray()` 重建 PIL 图像。

3. **Tight crop**  
   按 mask bounding box 裁切图像，仅保留树木区域。设最小尺寸阈值为 32px（即 crop_w < 32 或 crop_h < 32 时跳过该样本并记录到跳过日志）。

4. **等比缩放 + 白边填充（448×448）**  
   - `scale = 448 / max(crop_w, crop_h)`，等比缩放使长边恰好为 448px
   - 新建 448×448 白色 PIL 画布，将缩放后图像居中粘贴
   - 对照组：原图直接 `Resize(448)` + `CenterCrop(448)`  
   **注意**：`validate_plant_seedlings.py` 实际使用 `Resize(244)` + `CenterCrop(224)`（DINOv2 默认分辨率），不涉及 448。原图对照组与抠图组应使用**相同的目标尺寸**，此处统一选用 448×448（合法的 DINOv2 输入尺寸，448 = 14 × 32），以充分保留树冠细节；DINOv2 内部 Resize transform 统一调整为 `T.Resize(448)` + `T.CenterCrop(448)`，代替现有脚本中的 224。

5. **多实例处理**  
   实测发现训练集中存在 72 张（约 2.3%）含多个 `objects` 的 annotation（同一图像标注了两棵树）。`decode_gt_mask()` 当前只读取 `ann["objects"][0]`，仅处理第一个实例。实现时应保持此行为（取首个对象），并在日志中记录多实例跳过数量，不做多实例展开，避免同图多样本引入标签泄漏。

6. **落盘**  
   按 `tdus_cutouts/{train,val,test}/{species}/` 目录结构保存，species 取自 annotation 的 `classTitle`（如 `acer_palmatum`，已由数据集统一使用下划线小写格式，可直接用作目录名）。

**输出目录**：
```
tdus_cutouts/
  train/{species}/
  val/{species}/
  test/{species}/
```

---

### 阶段二：DINOv2 特征提取

**模型**：`dinov2_vits14`（384 维 CLS token），通过 `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` 加载，与 `validate_plant_seedlings.py` 和 `validate_tree_extraction_sam.py` 中的加载方式一致。

**预处理**：ImageNet 归一化（`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`），参考 `validate_tree_extraction_sam.py` 中的 `dino_transform`，但 `Resize` 和 `CenterCrop` 目标尺寸改为 448（见阶段一步骤4）。  
**注意**：`validate_plant_seedlings.py` 中的 `transform_image` 使用 `T.Normalize([0.5], [0.5])`（非 ImageNet 标准值），且 `Resize(244)` 存在笔误（244 非 14 的整倍数）。本方案应以 `validate_tree_extraction_sam.py` 的 `dino_transform`（ImageNet mean/std，尺寸改为 448）为准，不复用 `validate_plant_seedlings.py` 的 transform。

**提取方式**：调用 `dinov2.forward_features(x)["x_norm_clstoken"]` 获取 CLS token，参考 `validate_tree_extraction_sam.py` 中的 `cls_similarity_map()` 函数实现。  
**注意**：`validate_plant_seedlings.py` 的 `compute_embeddings()` 使用 `dinov2(x)`（直接 forward，返回 logits 而非特征向量），不能用于此处。本方案须通过 `forward_features()` 显式提取 `x_norm_clstoken`。

**缓存策略**：提取结果缓存为 `.npz` 文件（`embeddings` 数组 + `labels` 数组），避免重复推理：
```
tdus_embeddings/
  cutout_train.npz  cutout_val.npz  cutout_test.npz
  original_train.npz  original_val.npz  original_test.npz
```

---

### 阶段三：SVM 分类

**分类器**：`sklearn.svm.SVC(gamma='scale', class_weight='balanced')`

- `class_weight='balanced'` 处理 22 个类之间可能存在的样本不均衡
- 在 train split 上拟合，在 val 和 test split 上评估
- Cutout 方案与 Original 方案使用完全相同的超参数，保证对比公平

**参考**：`validate_plant_seedlings.py` 中已有 `svm.SVC(gamma="scale")` 的完整训练—评估流程。

---

### 评估指标

| 指标 | 说明 |
|------|------|
| Accuracy | 整体准确率 |
| Weighted F1 | 按类别样本量加权，反映实际分布下的性能 |
| Macro F1 | 各类等权平均，对稀有类更敏感 |
| Confusion Matrix | 22×22 热图，可视化易混淆的树种对 |

---

### 输出结构

```
tdus_cutouts/{train,val,test}/{species}/    # 预处理后的抠图图像
skip_log.txt                               # 跳过记录（too_small / no_objects / 多实例统计）
tdus_embeddings/
  cutout_{train,val,test}.npz              # Cutout 方案特征缓存
  original_{train,val,test}.npz            # Original 方案特征缓存
tdus_clf_results/
  cutout_report.txt                        # Cutout 方案分类报告
  cutout_confusion.png                     # Cutout 方案混淆矩阵
  original_report.txt                      # Original 方案分类报告
  original_confusion.png                   # Original 方案混淆矩阵
  comparison_summary.txt                   # 两方案指标对比及自动解读
```

---

## 权衡分析

### 为什么选择 GT mask 而非预测 mask

当前实验目标是验证"背景去除本身是否有益"，而非端到端系统性能。使用 GT mask 可以排除分割误差对分类结果的干扰，得到理论上界估计。

若最终结论是 Cutout 方案显著优于 Original 方案，则下一步自然延伸是将 GT mask 替换为 SAM 预测 mask（`validate_segmentation_tdus.py` 已实现），验证端到端方案是否也能保持优势。

### 等比缩放 + 白边填充 vs. 直接拉伸

直接将 crop 拉伸至 448×448 会引入纵横比形变，对树干/树冠等依赖形状的特征有损。等比缩放保持真实比例，代价是引入白边区域。由于 DINOv2 的 patch token 计算对均匀白色区域的响应接近零（与抠图背景一致），白边对 CLS token 的影响预期较小。

### CLS token vs. patch token 平均

CLS token 是 DINOv2 训练中聚合全局语义信息的单一向量，已在 `validate_plant_seedlings.py` 中验证其对图像级分类任务的有效性，SVM 训练开销也远低于对 patch token 降维后的方案。若 CLS token 方案出现明显瓶颈，可考虑引入 patch token mean pooling 作为补充。

### SVM vs. 线性探针 / kNN

SVM 在中等规模（~3000 训练样本）、中等维度（384 维）设置下通常具备良好泛化性，且与 `validate_plant_seedlings.py` 的已有验证保持一致，便于横向比较。若 22 类分类存在明显的非线性决策边界，可追加 kNN（k=5, 10）作为参考基线。

---

## 风险

### 1. 样本跳过率

部分 TDUS 图像的 GT mask 面积极小（如远景树木），tight crop 后可能低于 32px 阈值被跳过。若某类树种的跳过率显著偏高，会导致该类在 train/val/test 分布不一致，影响评估公平性。

**缓解措施**：预处理阶段统计各类别跳过数量，若某类跳过率 > 20% 则在报告中标注，并在 confusion matrix 中对应列/行加注释。

### 2. GT mask 边界质量

Supervisely bitmap 格式的 GT mask 由人工标注，边界精度依赖标注者。对于叶片密集的树种，mask 边缘可能包含部分背景天空或道路，这会在 Cutout 方案中引入轻微噪声。

**缓解措施**：此风险在对照实验框架内是系统性的（所有样本同等受影响），不影响方案间的相对比较，但需在报告中说明。

### 3. 类别不均衡

TDUS 22 个树种的样本分布未必均匀。`class_weight='balanced'` 可在一定程度上缓解 SVM 对多数类的偏向，但若某类样本极少（< 10 张），分类器仍可能不稳定。

**缓解措施**：使用 macro F1 作为主要报告指标，同时在报告中附上各类别样本数统计。

### 4. 磁盘空间

全量 train/val/test 抠图（3949 张 × 448×448 RGB JPEG）估计约 2-4 GB，加上 `.npz` 缓存约 0.1 GB。需确认工作目录有足够空间。

**缓解措施**：预处理时以 JPEG quality=90 保存，可显著减小磁盘占用，且对 DINOv2 特征提取影响可忽略。

---

## 现有代码可复用清单

| 功能 | 来源文件 | 函数/代码段 |
|------|----------|-------------|
| GT mask 解码 | `validate_segmentation_tdus.py` | `decode_gt_mask()` |
| 白背景填充 | `validate_tree_extraction_sam.py` | `apply_mask_white_bg()` |
| DINOv2 加载 | `validate_tree_extraction_sam.py` | `torch.hub.load(..., "dinov2_vits14")` |
| DINOv2 forward_features | `validate_tree_extraction_sam.py` | `cls_similarity_map()` 内部实现 |
| ImageNet 归一化 transform | `validate_tree_extraction_sam.py` | `dino_transform` |
| SVM 训练—评估流程 | `validate_plant_seedlings.py` | `svm.SVC(gamma="scale")` 段落 |
| TDUS 数据集目录收集 | `validate_segmentation_tdus.py` | `collect_tdus()` |

---

## 下一步

1. 实现 `validate_tdus_cutout_classification.py`，按本 proposal 的三阶段设计构建完整脚本。
2. 先在小规模子集（每类 5 张）上验证预处理输出的视觉正确性。
3. 全量运行后对比 Cutout 与 Original 方案的 val/test 指标，记录结论。
4. 若 Cutout 方案有显著提升（Δ macro F1 > 3pp），考虑以 SAM 预测 mask 替换 GT mask 进行端到端验证。

---

## 架构师审查备注

**审查日期**：2026-05-28  
**审查范围**：完整性、一致性、可行性、最佳实践、风险

---

### 发现的问题及修改内容

#### 问题 1：`apply_mask_white_bg` 输入格式未说明（严重）

原文未交代该函数对输入格式的要求。经代码确认（`validate_tree_extraction_sam.py` 第 184 行注释）：`img_np` 必须是 `float32 [0, 1]`，而不是 uint8。若直接传入 `np.array(pil_img)`（uint8），mask 外像素会被设为 `1.0`（几乎为黑色）而非白色，导致抠图结果完全错误。

**修改**：在阶段一步骤 2 补充了 `/ 255.0` 转换要求及保存前的 `* 255` 反转换说明。

#### 问题 2：对照组 transform 尺寸描述错误（严重）

原文称"对照组原图直接 `Resize(448)` + `CenterCrop(448)`，与 `validate_plant_seedlings.py` 保持一致"。经核查，`validate_plant_seedlings.py` 实际使用 `Resize(244)` + `CenterCrop(224)`（且 244 是笔误，非 14 的整倍数），与 448 完全无关。描述来源不一致会使实现者困惑并可能复制错误的 transform。

**修改**：纠正引用来源，明确 448 是本方案的主动选择（448 = 14 × 32，合法的 DINOv2 输入），不依赖 `validate_plant_seedlings.py`。

#### 问题 3：`validate_plant_seedlings.py` 的 transform 使用非标准归一化（中等）

原文将 `validate_plant_seedlings.py` 作为 ImageNet 归一化的参考来源之一。实际上该脚本使用 `T.Normalize([0.5], [0.5])`，与 ImageNet mean/std 不同，会导致特征分布偏移。

**修改**：在阶段二明确说明 transform 应以 `validate_tree_extraction_sam.py` 的 `dino_transform` 为准，并标注 `validate_plant_seedlings.py` transform 不可复用。

#### 问题 4：`validate_plant_seedlings.py` 使用 `dinov2(x)` 而非 `forward_features()`（严重）

原文在"现有代码可复用清单"中将 `validate_plant_seedlings.py` 列为 SVM 训练参考，同时将 `validate_tree_extraction_sam.py` 列为 `forward_features` 参考，暗示两者可组合。实际上 `validate_plant_seedlings.py` 的 `compute_embeddings()` 调用 `dinov2(x)`——对于 `dinov2_vits14`（hub 模型），直接 forward 返回的是分类 logit 向量（1000 维），不是 384 维 CLS token 特征，两者语义完全不同。若实现者混用，会得到错误的特征维度。

**修改**：在阶段二补充警告，明确 `forward_features()` 为唯一正确的特征提取方式。

#### 问题 5：多实例 annotation 未处理（中等）

原文完全未提及多实例情况。实测发现训练集中有 72 张图（2.3%）包含多个 `objects`，`decode_gt_mask()` 只处理 `objects[0]`，静默忽略其余实例。此行为可接受，但需显式决策并记录。

**修改**：在阶段一新增步骤 5（多实例处理）说明此边界条件及选取首个对象的理由。

#### 问题 6："数据集目录收集"复用来源错误（轻微）

"现有代码可复用清单"将 `collect_images()` 的来源标注为 `validate_plant_seedlings.py`，但该函数假设目录结构为 `root/{class}/img.jpg`（适用于 plant seedlings 的平铺布局）。TDUS 的布局是 `split/img/` + `split/ann/`，对应函数是 `validate_segmentation_tdus.py` 中的 `collect_tdus()`。

**修改**：将复用清单中的来源改为 `collect_tdus()`（`validate_segmentation_tdus.py`）。

---

### 验证通过的设计决策

1. **`decode_gt_mask()` 函数接口**：签名 `decode_gt_mask(ann_path, img_w, img_h)` 与 `validate_segmentation_tdus.py` 第 51 行完全一致，base64 → zlib → PIL PNG → boolean ndarray 流程准确。

2. **`forward_features(x)["x_norm_clstoken"]` API**：在 `validate_tree_extraction_sam.py` 的 `cls_similarity_map()` 中已验证可正常调用，返回 384 维归一化 CLS token，适用于 SVM 输入。

3. **448×448 输入尺寸**：448 = 14 × 32，是 DINOv2 patch size 14 的整数倍，符合模型要求。相比 224，448 提供更多 patch token（32×32 = 1024 vs 16×16 = 256），对树冠细节保留更有利，代价是推理时间增加约 4×。

4. **`SVC(gamma='scale', class_weight='balanced')`**：在 22 类、~3000 样本、384 维特征空间下，RBF kernel + scale gamma 是合理默认值；`class_weight='balanced'` 对不均衡类分布的缓解有据可查。

5. **NPZ 缓存策略**：将 embeddings 和 labels 分开存储为 `.npz` 是标准做法，避免每次重新推理，在实验迭代中节省时间成本合理。

6. **classTitle 作为标签来源**：实测 TDUS annotation 中 `classTitle` 字段为下划线小写格式（如 `acer_palmatum`），在全数据集中唯一对应树种，适合直接用作目录名和分类标签。

7. **Macro F1 作为主要指标**：在类别不均衡场景下优于 Accuracy，与学术惯例一致。

8. **等比缩放 + 白边填充**：保持树木纵横比，避免形变对形状特征的干扰。DINOv2 对均匀白色 patch 的响应接近零的推断来自 PCA foreground extraction 的已有实验，理论支撑充分。

---

### 重大风险补充

#### 新增风险：DINOv2 对白色背景的处理并非完全中立

DINOv2 在 ImageNet 上预训练，训练图像极少含大面积纯白背景。448×448 画布上若树冠面积较小（如细高树种），白边可能占总面积 70% 以上，此时 CLS token 会吸收大量白色区域的 attention，稀释树种判别信息。  
**缓解**：在报告中按"mask 面积占图像面积比"分析 Cutout 方案的精度分布，若小 mask 比例类的准确率显著低于 Original 方案，应在结论中注明。

#### 新增风险：两方案 DINOv2 transform 不一致导致结果不可比

若 Cutout 方案使用 448×448 DINOv2 transform，而 Original 方案仍沿用 `validate_plant_seedlings.py` 的 `Resize(224)` + `Normalize([0.5])`，则特征空间不同，比较无效。  
**缓解**：两方案必须使用完全相同的 transform（均用 `T.Resize(448), T.CenterCrop(448), T.ToTensor(), T.Normalize(ImageNet_mean, ImageNet_std)`），此点在实现阶段需严格校验。

#### 已有风险 3（类别不均衡）的补充

实测 22 个类的样本数来自文件名前缀计数，train 3168 张均匀分配则约 144 张/类，但分布差异可能更大。建议在步骤 2 子集验证前先运行一次类别计数统计（直接遍历 `tdus_data/train/ann/`），对 < 30 张训练样本的类提前预警。
