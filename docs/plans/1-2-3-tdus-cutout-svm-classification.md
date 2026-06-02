# Plan: TDUS Cutout-SVM Tree Species Classification

**Status**: Draft  
**Date**: 2026-05-28  
**Based on**: `docs/proposals/proposal-tdus-cutout-svm-classification.md`（审查后版本）  
**Phases**: 1–3

---

## Phase 总览

| Phase | 脚本 | 依赖 | 预估行数 |
|-------|------|------|---------|
| Phase 1 | `preprocess_tdus_cutouts.py` | 无 | ~220 行 |
| Phase 2 | `extract_embeddings_tdus.py` | Phase 1 输出（`tdus_cutouts/`） | ~180 行 |
| Phase 3 | `classify_tdus_svm.py` | Phase 2 输出（`tdus_embeddings/`） | ~200 行 |

**总依赖链**：Phase 1 → Phase 2 → Phase 3（顺序执行）

---

## Phase 1：预处理脚本（`preprocess_tdus_cutouts.py`）

**目标**：将 TDUS 原始图像 + annotation JSON 转化为等比缩放、白边填充后的 448×448 抠图，按树种分类保存，供后续特征提取使用。

**输入**：
- `tdus_data/{train,val,test}/img/*.jpg`
- `tdus_data/{train,val,test}/ann/*.json`

**输出**：
```
tdus_cutouts/
  train/{species}/*.jpg
  val/{species}/*.jpg
  test/{species}/*.jpg
skip_log.txt   # 记录被跳过的样本及原因
```

**依赖**：无（Phase 1 为起点）

---

### Stage 1.1：decode_gt_mask 复用 + tight crop + 最小尺寸过滤

**输入**：单张图像路径 + 对应 annotation JSON 路径  
**输出**：`(pil_img_cropped, mask_cropped, species_name)` 或跳过信号

**关键实现要点**：

1. 直接 import 并调用 `validate_segmentation_tdus.py` 中的 `decode_gt_mask(ann_path, img_w, img_h)`，不重复实现。该函数返回与原图等大的 boolean ndarray mask。
2. 读取 annotation JSON，取 `ann["objects"][0]` 的 `classTitle` 作为 `species_name`；若 `ann["objects"]` 为空列表，则跳过并记录原因 `"no_objects"`。
3. 多实例处理：若 `len(ann["objects"]) > 1`，仅使用 `objects[0]`，在 `skip_log.txt` 中用独立计数器统计多实例样本数（不跳过，但需记录）。
4. 计算 mask bounding box：`np.where(mask)` 得到行/列范围，推导 `(x_min, y_min, x_max, y_max)`。
5. 对原图和 mask 同步裁切到 bounding box 区域。
6. 最小尺寸过滤：若 `crop_w < 32` 或 `crop_h < 32`，跳过并记录原因 `"too_small"`及尺寸。

**代码复用来源**：
- `decode_gt_mask()`：`validate_segmentation_tdus.py`
- `collect_tdus()`：`validate_segmentation_tdus.py`（用于遍历 split 目录）

**验收标准**：
- [ ] 对 train/val/test 中任意一张样本，`decode_gt_mask()` 可正常返回 boolean ndarray，形状与原图相同
- [ ] bounding box 裁切后的图像尺寸与 mask 紧密贴合（无多余背景行/列）
- [ ] `crop_w < 32` 或 `crop_h < 32` 的样本被跳过，skip_log 有对应记录
- [ ] 多实例样本（`len(objects) > 1`）只取 `objects[0]`，不额外生成样本
- [ ] 本 Stage 无落盘操作，仅返回中间结果供 Stage 1.2 处理

**估计行数**：~80 行（含 import 和辅助函数）

---

### Stage 1.2：等比缩放 + 白边填充 → 448×448

**输入**：Stage 1.1 输出的 `(pil_img_cropped, mask_cropped, species_name)`  
**输出**：448×448 PIL Image（白背景抠图）

**关键实现要点**：

1. **背景填白**：
   - 将 `pil_img_cropped` 转为 float32：`img_np = np.array(pil_img_cropped) / 255.0`
   - 调用 `apply_mask_white_bg(img_np, mask_cropped)`，函数要求 `img_np` 为 `float32 [0, 1]`，输出同为 `[0, 1]`
   - 反转换：`out_uint8 = (out * 255).astype(np.uint8)`，再用 `Image.fromarray(out_uint8)` 重建 PIL 图像

2. **等比缩放**：
   - 计算 `scale = 448 / max(crop_w, crop_h)`
   - 新宽高：`new_w = int(crop_w * scale)`，`new_h = int(crop_h * scale)`
   - 用 `PIL.Image.LANCZOS` 进行高质量缩放

3. **白边填充**：
   - 新建 `Image.new("RGB", (448, 448), (255, 255, 255))`
   - 计算居中偏移：`offset_x = (448 - new_w) // 2`，`offset_y = (448 - new_h) // 2`
   - `canvas.paste(scaled_img, (offset_x, offset_y))`

**代码复用来源**：
- `apply_mask_white_bg()`：`validate_tree_extraction_sam.py`

**验收标准**：
- [ ] 输出图像尺寸恰好为 (448, 448)
- [ ] 树木区域（原 mask 内）像素与原图保持一致（颜色未损失）
- [ ] mask 外区域（背景）为纯白 RGB(255, 255, 255)
- [ ] 等比缩放后长边为 448px，短边 ≤ 448px（白边居中填充）
- [ ] 对纵横比极端样本（如高宽比 5:1 的树干）输出仍为 448×448

**估计行数**：~50 行

---

### Stage 1.3：多实例 annotation 统计记录（仅取首个对象）

> **说明**：根据 proposal 决策，多实例样本仅取 `objects[0]` 的 mask，不做并集处理，保持与 `decode_gt_mask()` 当前行为一致。本 Stage 转为记录与报告模块。

**输入**：annotation JSON 文件路径  
**输出**：`multi_instance_count`（整数，累计到 skip_log）

**关键实现要点**：

1. 在遍历 annotation 时，统计 `len(ann["objects"]) > 1` 的样本数量。
2. 在 `skip_log.txt` 末尾附加汇总行：`"Multi-instance samples (objects[0] used): N"`。
3. 无需额外代码逻辑，确保 `decode_gt_mask()` 只看 `objects[0]` 即可（其内部已实现）；实现者无需修改该函数。

**验收标准**：
- [ ] skip_log.txt 包含多实例汇总行，数量与手动统计一致
- [ ] 多实例图像在 `tdus_cutouts/` 中只产生一张输出图像
- [ ] 无新增函数，Stage 1.3 的逻辑嵌入在主批处理循环中

**估计行数**：~10 行（嵌入循环内）

---

### Stage 1.4：完整 train/val/test 批处理 + 跳过记录

**输入**：`tdus_data/{train,val,test}/`  
**输出**：`tdus_cutouts/{train,val,test}/{species}/*.jpg` + `skip_log.txt`

**关键实现要点**：

1. 使用 `collect_tdus(split_dir)` 遍历每个 split，返回 `(img_path, ann_path)` 对列表。
2. 外层循环遍历 `["train", "val", "test"]`，内层循环遍历该 split 的所有样本对。
3. 创建输出目录：`tdus_cutouts/{split}/{species}/`，使用 `os.makedirs(..., exist_ok=True)`。
4. 落盘：调用 `cutout_img.save(out_path, "JPEG", quality=90)`，文件名保留原图基名。
5. skip_log 追加记录格式：`{split}/{img_stem}: {reason}({detail})`。
6. 处理完成后打印统计摘要：每个 split 的处理总数、跳过数、各 species 样本数。
7. **类别计数预警**：对 train split 中样本数 < 30 的 species，打印警告信息。

**验收标准**：
- [ ] `tdus_cutouts/` 三个 split 目录均已生成，子目录按 species 分类
- [ ] 对 train 3168 张（预估跳过 < 5%），val 395 张，test 386 张，全量处理无崩溃
- [ ] skip_log.txt 逐条记录跳过原因，总跳过数与摘要统计一致
- [ ] JPEG quality=90 输出文件，单张约 30-80 KB，train 全量约 ≤ 500 MB
- [ ] 样本数 < 30 的 train species 有控制台警告

**估计行数**：~80 行（含 main 函数、参数解析、循环）

**Phase 1 总估计行数**：~220 行（所有 Stage 合计，含 import、logging、argparse）

---

## Phase 2：特征提取脚本（`extract_embeddings_tdus.py`）

**目标**：用 DINOv2 vits14 对 Cutout 抠图和 Original 原图分别提取 CLS token 特征，缓存为 .npz 文件。

**输入**：
- Cutout 组：`tdus_cutouts/{train,val,test}/{species}/*.jpg`
- Original 组：`tdus_data/{train,val,test}/img/*.jpg` + 对应 annotation（用于获取 species label）

**输出**：
```
tdus_embeddings/
  cutout_train.npz   cutout_val.npz   cutout_test.npz
  original_train.npz original_val.npz original_test.npz
```
每个 .npz 包含：`embeddings`（shape: N×384, float32）和 `labels`（shape: N, str）

**依赖**：Phase 1 输出（`tdus_cutouts/`）

---

### Stage 2.1：DINOv2 forward_features CLS token 提取（抠图组）

**输入**：`tdus_cutouts/{split}/{species}/*.jpg`  
**输出**：`cutout_{split}.npz`（embeddings + labels）

**关键实现要点**：

1. **模型加载**：`torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`，`.eval().cuda()`（或 `.cpu()` fallback），与 `validate_tree_extraction_sam.py` 一致。

2. **Transform**（严格遵循 `validate_tree_extraction_sam.py` 的 `dino_transform`，目标尺寸改为 448）：
   ```
   T.Resize(448), T.CenterCrop(448),
   T.ToTensor(),
   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ```
   **禁止**使用 `validate_plant_seedlings.py` 的 `T.Normalize([0.5], [0.5])` 或 `Resize(244)`。

3. **特征提取**：
   - `with torch.no_grad(): feats = dinov2.forward_features(x_batch)["x_norm_clstoken"]`
   - **禁止**使用 `dinov2(x)`（返回 logits，非 CLS token）
   - 参考 `validate_tree_extraction_sam.py` 中 `cls_similarity_map()` 的实现方式

4. **数据集遍历**：按目录结构读取 `cutouts/{split}/{species}/`，从 species 目录名直接获取标签，无需读取 annotation JSON。

5. **Batch 推理**：batch size=32（可通过参数调整），使用 `torch.utils.data.DataLoader`。

**验收标准**：
- [ ] 输出 .npz 中 `embeddings.shape == (N, 384)`，dtype=float32
- [ ] `labels.shape == (N,)`，值与 `tdus_cutouts/` 中的 species 目录名一致
- [ ] 不同 split 的 N 与 Phase 1 实际输出数量（扣除跳过后）一致
- [ ] 使用 `forward_features()["x_norm_clstoken"]`，可通过 `assert feats.shape[-1] == 384` 验证

**估计行数**：~80 行

---

### Stage 2.2：原图对照组特征提取（相同 transform）

**输入**：`tdus_data/{split}/img/*.jpg` + annotation JSON（获取 species label）  
**输出**：`original_{split}.npz`（embeddings + labels）

**关键实现要点**：

1. 使用与 Stage 2.1 **完全相同**的 transform（`Resize(448)`, `CenterCrop(448)`, `ToTensor()`, `Normalize(ImageNet)`），保证两组特征空间可比。

2. 通过 annotation JSON 获取 species label：读取 `ann["objects"][0]["classTitle"]`，与 Phase 1 跳过逻辑保持一致——若该图像在 Phase 1 被跳过（如 `too_small`），则 Original 对照组也**跳过同一张图**，确保 Cutout 与 Original 两组样本集合完全对齐。

3. 可选：读取 Phase 1 生成的 `skip_log.txt`，构建跳过图像集合，在 Original 组遍历时排除这些图像。

4. 模型实例复用：在同一脚本运行中，Cutout 和 Original 共享同一个 DINOv2 模型实例，避免重复加载。

**验收标准**：
- [ ] `original_{split}.npz` 的 N 与 `cutout_{split}.npz` 的 N 完全相等（样本对齐）
- [ ] 两组 `labels` 数组的 species 分布完全一致（可通过 `np.array_equal(sorted_labels_cutout, sorted_labels_original)` 验证）
- [ ] Transform 参数与 Stage 2.1 一致，可通过代码复用（同一 transform 对象）保证

**估计行数**：~60 行（大量复用 Stage 2.1 的辅助函数）

---

### Stage 2.3：.npz 缓存写入与验证

**输入**：Stage 2.1/2.2 的 embeddings 列表和 labels 列表  
**输出**：6 个 .npz 文件 + 控制台验证报告

**关键实现要点**：

1. 写入：`np.savez(out_path, embeddings=emb_array, labels=label_array)`，其中 `emb_array` 为 `np.vstack(batch_feats).astype(np.float32)`，`label_array` 为 `np.array(labels)`（dtype=str）。

2. 写入前创建目录：`os.makedirs("tdus_embeddings", exist_ok=True)`。

3. **验证**（写入后立即执行）：
   - 重新加载 .npz：`data = np.load(path, allow_pickle=True)`
   - 打印：文件名、N（样本数）、维度（应为 384）、species 分布（unique labels + counts）
   - 断言：`assert data["embeddings"].shape[1] == 384`

4. 缓存检查：脚本开头检查 .npz 是否已存在，若存在则跳过重新提取（添加 `--force` 标志可强制重提取）。

**验收标准**：
- [ ] 6 个 .npz 文件均可被 `np.load()` 正常读取
- [ ] 每个文件的 `embeddings` shape 第二维为 384
- [ ] 验证报告中 species 分布与 `tdus_cutouts/` 目录统计一致
- [ ] 重复运行脚本时，已有 .npz 被跳过（无重复推理）

**估计行数**：~40 行

**Phase 2 总估计行数**：~180 行（含 import、argparse、DataLoader 封装）

---

## Phase 3：SVM 分类与评估脚本（`classify_tdus_svm.py`）

**目标**：加载 Phase 2 缓存的 embeddings，训练 SVM 分类器，在 val/test 上评估，生成分类报告和混淆矩阵热图，对比 Cutout 与 Original 两组结果。

**输入**：`tdus_embeddings/*.npz`（6 个文件）  
**输出**：
```
tdus_clf_results/
  cutout_report.txt
  cutout_confusion.png
  original_report.txt
  original_confusion.png
  comparison_summary.txt
```

**依赖**：Phase 2 输出（`tdus_embeddings/`）

---

### Stage 3.1：SVM 训练（train split）

**输入**：`cutout_train.npz` 和 `original_train.npz`  
**输出**：两个训练好的 SVM 模型对象（内存中）

**关键实现要点**：

1. 加载训练数据：`X_train = data["embeddings"]`，`y_train = data["labels"]`。

2. 分类器：`SVC(kernel="rbf", gamma="scale", class_weight="balanced")`，参考 `validate_plant_seedlings.py` 中的 `svm.SVC(gamma="scale")` 用法，补充 `class_weight="balanced"`。

3. Cutout 和 Original 使用完全相同的超参数，保证对比公平。

4. 训练前打印各 species 样本数（`np.unique(y_train, return_counts=True)`），对样本数 < 10 的类发出警告。

5. 计时：记录 SVM 训练耗时，打印到控制台（3168 样本、384 维约需 10-60 秒）。

**代码复用来源**：`validate_plant_seedlings.py` 中的 SVM 训练段落

**验收标准**：
- [ ] 两个 SVM 模型均成功 `.fit()`，无异常
- [ ] 训练集 accuracy 可通过 `svm.score(X_train, y_train)` 获取并打印（用于 sanity check，预期接近 100%）
- [ ] 样本数 < 10 的 species 有控制台警告
- [ ] 训练耗时被记录

**估计行数**：~50 行

---

### Stage 3.2：val/test 评估 + classification report

**输入**：Stage 3.1 的两个 SVM 模型 + `{cutout,original}_{val,test}.npz`  
**输出**：`{cutout,original}_report.txt`

**关键实现要点**：

1. 对 val 和 test 分别评估，共 4 次 `svm.predict(X)`。

2. 计算指标：
   - `accuracy_score(y_true, y_pred)`
   - `f1_score(y_true, y_pred, average="weighted")`
   - `f1_score(y_true, y_pred, average="macro")`
   - `classification_report(y_true, y_pred)`（包含每类 precision/recall/f1/support）

3. 报告格式（写入 txt 文件）：
   ```
   === Cutout / Original Report ===
   Val  Accuracy: X.XXX  Weighted-F1: X.XXX  Macro-F1: X.XXX
   Test Accuracy: X.XXX  Weighted-F1: X.XXX  Macro-F1: X.XXX

   --- Val Classification Report ---
   <sklearn classification_report 输出>

   --- Test Classification Report ---
   <sklearn classification_report 输出>
   ```

4. 对 Phase 1 标注跳过率 > 20% 的 species，在报告中对应行加注释 `# high skip rate`。

**验收标准**：
- [ ] `{cutout,original}_report.txt` 均成功写入
- [ ] 报告中包含 val 和 test 两部分的三项指标（accuracy、weighted-F1、macro-F1）
- [ ] `classification_report` 输出覆盖全部 22 个 species
- [ ] 高跳过率 species 有标注（若有）

**估计行数**：~60 行

---

### Stage 3.3：Confusion matrix 热图生成

**输入**：Stage 3.2 中 test split 的 `y_true` 和 `y_pred`  
**输出**：`{cutout,original}_confusion.png`

**关键实现要点**：

1. 使用 `sklearn.metrics.confusion_matrix(y_true, y_pred, labels=sorted_species)` 生成 22×22 矩阵。

2. 可视化：
   - `matplotlib` + `seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues")`
   - x 轴：Predicted，y 轴：True
   - 标签使用 species 名（22 个）
   - 图尺寸：`figsize=(16, 14)`，标签字号 6-8pt（避免重叠）
   - title：`"Cutout (Test) Confusion Matrix"` / `"Original (Test) Confusion Matrix"`

3. 保存：`plt.savefig(out_path, dpi=150, bbox_inches="tight")`

4. 若某 species 在 test 中无样本（因跳过），该行/列全为 0，不影响矩阵生成但需在 report.txt 中注明。

**验收标准**：
- [ ] 两张 .png 均可正常打开，热图尺寸为 22×22
- [ ] 主对角线为各类正确预测数，矩阵行和等于各类 test 样本数
- [ ] axes 标签清晰可读（字体不重叠）
- [ ] 图文件大小合理（< 2 MB）

**估计行数**：~50 行

---

### Stage 3.4：两组结果对比汇总

**输入**：Stage 3.2 计算的两组指标  
**输出**：`comparison_summary.txt`

**关键实现要点**：

1. 对比表格格式：

   ```
   === Comparison Summary (Test Split) ===

   Metric           Cutout     Original   Delta
   ─────────────────────────────────────────────
   Accuracy         X.XXX      X.XXX      +X.XXX
   Weighted F1      X.XXX      X.XXX      +X.XXX
   Macro F1         X.XXX      X.XXX      +X.XXX

   === Interpretation ===
   - 若 Delta Macro F1 > +0.03: Cutout 方案显著优于 Original，建议后续以 SAM 预测 mask 替换 GT mask 验证端到端效果
   - 若 |Delta Macro F1| ≤ 0.03: 两方案性能相当，背景去除对 DINOv2+SVM 的增益有限
   - 若 Delta Macro F1 < -0.03: Cutout 方案劣于 Original，白边效应或裁切信息损失超过背景噪声去除的收益
   ```

2. Delta = Cutout - Original（正值代表 Cutout 方案更优）。

3. 自动填充 Interpretation（根据 delta 阈值选择对应文本）。

4. val split 对比数据也一并写入（格式相同，标注 "Val Split"）。

**验收标准**：
- [ ] `comparison_summary.txt` 包含 val 和 test 两部分的三项指标对比
- [ ] Delta 列计算正确（Cutout - Original）
- [ ] Interpretation 文本根据实际 delta 值自动填写（不硬编码结论）
- [ ] 文件可直接作为实验结论摘录，无需人工编辑

**估计行数**：~40 行（含格式化输出）

**Phase 3 总估计行数**：~200 行（含 import、argparse、main 函数）

---

## 测试策略

### 单元测试（Stage 级别）

| Stage | 测试方法 |
|-------|----------|
| Stage 1.1 | 取 1 张已知 annotation 的图像，验证 `decode_gt_mask()` 返回 shape 与原图一致；模拟 `crop_w=10`（< 32）触发跳过路径 |
| Stage 1.2 | 对单张裁切图像验证输出为 (448, 448)；检查 mask 外区域像素值 == 255 |
| Stage 2.1 | 取 1 张 cutout，确认 `forward_features()["x_norm_clstoken"].shape == (1, 384)` |
| Stage 2.3 | 写入后 `np.load()` 重新读取，断言 shape 和 dtype 正确 |
| Stage 3.1 | 以 10 张样本（2类）训练 SVM，确认 `.predict()` 返回长度正确 |
| Stage 3.3 | 以 dummy y_true/y_pred 生成混淆矩阵，确认图文件可写入 |

### 端到端小规模验证

在全量运行前，先以每个 species 5 张（共约 110 张）运行完整三阶段 pipeline：
1. 目视检查 `tdus_cutouts/train/` 中几张抠图的视觉正确性（背景为白色，树木区域保留）
2. 确认 .npz 文件生成正常，embeddings 维度为 384
3. 确认 SVM 可完成训练和预测（小规模 accuracy 不要求高，仅验证 pipeline 不崩溃）

### 回归检查（Phase 间）

- Phase 2 开始前：确认 `tdus_cutouts/` 三个 split 的文件总数与 Phase 1 摘要统计一致
- Phase 3 开始前：确认 6 个 .npz 文件均存在且可正常加载，`cutout_{split}` 与 `original_{split}` 的 N 相等

---

## 风险与缓解措施（来自 proposal 审查）

| 风险 | 缓解 |
|------|------|
| 样本跳过率过高（某类 > 20%）| Phase 1 输出 skip_log，Phase 3 报告中对高跳过率类标注 |
| 两组 transform 不一致导致结果不可比 | Stage 2.2 明确复用 Stage 2.1 的同一 transform 对象 |
| DINOv2 白色背景处理非中立（白边占比大时稀释 CLS token）| Phase 3 报告中按 mask 面积占比分析精度分布（可选后处理） |
| 类别不均衡（某类 < 30 张训练样本）| Stage 1.4 和 Stage 3.1 均有预警；SVM 使用 `class_weight="balanced"` |
| 磁盘空间（估计 2-4 GB cutouts + 0.1 GB npz）| JPEG quality=90 保存，预处理前确认可用空间 |
| `apply_mask_white_bg` 输入格式错误（uint8 而非 float32）| Stage 1.2 验收标准要求检查 mask 外像素值 == 255（uint8 错误时会为 1） |

---

## 文件产出清单

| 文件 | Phase | 说明 |
|------|-------|------|
| `preprocess_tdus_cutouts.py` | Phase 1 | 预处理脚本 |
| `extract_embeddings_tdus.py` | Phase 2 | 特征提取脚本 |
| `classify_tdus_svm.py` | Phase 3 | 分类与评估脚本 |
| `tdus_cutouts/` | Phase 1 输出 | 抠图图像目录 |
| `skip_log.txt` | Phase 1 输出 | 跳过记录 |
| `tdus_embeddings/*.npz` | Phase 2 输出 | 特征缓存（6 个文件） |
| `tdus_clf_results/cutout_report.txt` | Phase 3 输出 | Cutout 方案分类报告 |
| `tdus_clf_results/original_report.txt` | Phase 3 输出 | Original 方案分类报告 |
| `tdus_clf_results/comparison_summary.txt` | Phase 3 输出 | 两方案指标对比及自动解读 |
| `tdus_clf_results/*.png` | Phase 3 输出 | 混淆矩阵热图（2 张） |

---

## 审查备注

**审查日期**：2026-05-28
**审查范围**：Proposal ↔ Plan 一致性、Phase/Stage 粒度、依赖顺序、遗漏检查、可执行性

---

### 发现的不一致项及处理方式

#### 不一致 1：Phase 总览表行数与各 Phase 末尾统计矛盾（严重）

**发现**：总览表中 Phase 1 写 ~400 行、Phase 2 写 ~250 行、Phase 3 写 ~300 行，与各 Phase 末尾统计（~220、~180、~200）相差 80-200 行。各 Stage 逐项加总与末尾统计吻合，总览表数字是草稿遗留值。

**处理**：将总览表修正为 Phase 1 ~220 行、Phase 2 ~180 行、Phase 3 ~200 行，与末尾统计对齐。三个 Phase 总计 ~600 行，任何单 Phase ≤ 220 行，符合"Phase ≤ 500 行、单 Stage ≤ 200 行"的粒度要求。

#### 不一致 2：Stage 1.3 标题与正文行为相悖（中等）

**发现**：Stage 1.3 标题为"多实例 annotation 处理（取并集 mask）"，但正文第一句明确说明"本 Stage 转为记录与报告模块"，实际不做并集处理，标题具有误导性。

**处理**：标题改为"多实例 annotation 统计记录（仅取首个对象）"，与 Proposal 阶段一步骤 5 的决策一致。

#### 不一致 3：Proposal 输出结构缺少 Plan 新增的两个产出物（中等）

**发现**：Plan Stage 3.4 新增了 `comparison_summary.txt`，但 Proposal 的"输出结构"章节未列出该文件；Proposal 也未列出 `skip_log.txt`，而 Plan Phase 1 已在输出中明确定义。

**处理**：在 Proposal 输出结构中补充 `skip_log.txt` 和 `comparison_summary.txt`，两个文档输出清单现已对齐。Plan 文件产出清单中将 `tdus_clf_results/*.txt` 拆分为三条具体条目，避免歧义。

---

### 验证通过的对齐项

1. **数据流连贯性**：Phase 1 → 2 → 3 的输入/输出格式完全衔接；Phase 2 Stage 2.2 正确引用 `skip_log.txt` 实现样本对齐，保证 Cutout 与 Original 两组 N 相等。

2. **Proposal 三大目标全覆盖**：目标 1（Cutout pipeline）→ Phase 1+2+3.1~3.3；目标 2（Original 对照组）→ Stage 2.2+3.1；目标 3（定量比较）→ Stage 3.2+3.4。

3. **评估指标全覆盖**：Proposal 规定的 Accuracy、Weighted F1、Macro F1、Confusion Matrix 四项指标分别在 Stage 3.2 和 Stage 3.3 中实现，Stage 3.4 的对比摘要包含全部三项数值指标。

4. **Stage 粒度合理**：最大单 Stage 估计行数为 Stage 1.1/1.4 的各 ~80 行，远低于 200 行阈值；Stage 1.3 仅 ~10 行，可接受（嵌入主循环内，无需独立函数）。

5. **技术细节对齐**：transform（ImageNet mean/std，Resize(448)）、forward_features API、SVC 超参数、JPEG quality=90、等比缩放逻辑等关键技术决策在 Plan 各 Stage 中均与 Proposal 一致，且保留了 Proposal 审查备注中的警告说明。

6. **风险缓解完整映射**：Proposal 中 4 条原始风险和 3 条审查补充风险，全部在 Plan 的"风险与缓解措施"表中有对应条目，无遗漏。

---

### 执行风险提示

1. **Stage 1.3 实际代码量极低（~10 行）**，建议直接嵌入 Stage 1.4 的主循环，不单独成函数，避免过度分层。如果实现者将其独立为函数，需注意计数器状态需跨循环维护。

2. **Phase 2 Stage 2.2 的样本对齐逻辑是隐性风险**：读取 `skip_log.txt` 解析跳过图像集合的方案依赖日志格式的稳定性；建议改为直接遍历 `tdus_cutouts/{split}/{species}/` 目录，用已生成文件的文件名集合反推 Original 组应包含的样本，避免日志解析出错。此替代方案更健壮，但 Plan 中未提及，实现时注意选择。

3. **混淆矩阵 22 类标签在 16×14 英寸图中字号 6-8pt 可能仍然拥挤**，建议实现时先用 `plt.tight_layout()` 测试，必要时将标签旋转 45° 或缩减 `annot=True` 改为仅高亮对角线。
