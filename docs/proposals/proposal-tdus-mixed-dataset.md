# Proposal: TDUS Mixed Dataset — Fine-tune YOLO on Urban Street Trees

**Status**: Draft  
**Date**: 2026-05-31  
**Builds on**: `docs/proposals/proposal-higher-res-frames-retrain.md`

---

## 背景

当前 YOLO 模型（`tree_yolo26s_halfres/weights/best.pt`，mAP50=0.886）在路面行车视频上表现优秀，但在 TDUS 数据集（城市街道行人视角树木照片）上仅达到 **33.2% top-1 分类准确率**，而同条件下 GDino 路径达到 **93.5%**。

根本原因：YOLO 训练数据（行车记录仪视角，1352×760，AR=1.78）与 TDUS 图像（行人视角竖拍，3024×4032，AR=0.75）存在严重域迁移问题。YOLO 在 TDUS 图像上选择碎片化区域（树干角落、细条状切片）而非完整树冠，导致 DINOv2 嵌入质量低劣。

### 数据集对比

| 维度 | 路面视频帧 | TDUS |
|------|-----------|------|
| 分辨率 | 1352×760 | 3024×4032 |
| 宽高比 | 1.78（横） | 0.75（竖） |
| bbox 面积均值 | ~0.10（实测，2021 帧标签） | ~0.25（推定，GDino 尚未在 TDUS 上运行） |
| 拍摄视角 | 行车记录仪（车载） | 行人（地面） |
| 数量（train） | ~2071 帧 | 3170 张 |
| 数量（val） | ~2076 帧 | 395 张 |
| 数量（test） | — | 386 张（**禁止入训练集**） |

---

## 目标

1. 通过加入 TDUS 训练图像，使 YOLO 能正确检测行人视角的城市树木。
2. 维持路面视频域的检测质量（mAP50 不低于 0.85）。
3. 使 YOLO+DINOv2+SVM 路径在 TDUS 测试集上的分类准确率达到 **≥70%**（当前 33.2%）。
4. 整个 pipeline 保持自洽：TDUS 训练标注使用 GDino 伪标签（而非 GT bitmap mask），与现有 SVM 分类器的训练数据一致。

---

## 方案设计

### 核心思路：跨域混合数据集 + 微调

从已收敛的 `best.pt` 微调（而非从头训练），加入 TDUS 数据引入行人视角域知识，同时保留路面视频域的特征。

**关键决策**：

| 决策点 | 选择 | 理由 |
|--------|------|------|
| TDUS 标注来源 | GDino 伪标签 | 与 SVM 训练数据一致；GT bitmap 仅标注类别轮廓，不适合直接用作检测框 |
| 预缩放 | 最长边=1280（960×1280） | 3024×4032 原图 train 集实测 6.4GB，缩放后预计 ~0.6–0.8GB（约 8–10×）；避免训练时实时缩放的内存压力；GDino 伪标签生成时也以相同分辨率输入 |
| rect 模式 | `rect=False` | `rect=True` 按宽高比分桶（road AR=1.78 vs TDUS AR=0.75），两类图像进不同 batch，无法 mosaic 混合；`rect=False` 才能实现跨域 mosaic |
| 基础模型 | `best.pt`（非从头训练） | 路面视频特征已收敛，微调成本低；从头训练需要更大 lr + 更多 epoch |
| 学习率 | lr0=2e-4（vs 训练时 1e-3） | 微调标准做法：低学习率保留已习得特征 |
| TDUS test 集处理 | 完全排除 | 386 张测试图不进入训练/val 集，防止数据污染 |

### 数据预处理流程

```
TDUS train/img/（3170 张，3024×4032）
    └── 预缩放 960×1280（最长边=1280）
            → data/tdus_resized/train/img/
    └── GDino 伪标签推理
            → data/tdus_resized/train/labels/   # YOLO txt 格式

TDUS val/img/（395 张，3024×4032）
    └── 预缩放 960×1280
            → data/tdus_resized/val/img/
    └── GDino 伪标签推理
            → data/tdus_resized/val/labels/
```

### 统一数据集结构

```
data/unified_halfres_tdus/
├── images/
│   ├── train/   # 路面视频 ~2071 帧 + TDUS 3170 张（预缩放）= ~5241 张
│   └── val/     # 路面视频 ~2076 帧 + TDUS 395 张 = ~2471 张
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

> 路面视频帧（1352×760）与 TDUS 缩放帧（960×1280）直接混合，YOLO 在 `rect=False` 模式下以 `imgsz=1280` 统一处理，两种尺寸都会被 letterbox 填充到 1280×1280 正方形 batch。

### 训练配置

| 参数 | 值 | 对比上一版本 |
|------|-----|-----------|
| 起点模型 | `tree_yolo26s_halfres/weights/best.pt` | 从 best.pt 微调 |
| lr0 | **2e-4** | 从 1e-3 降低（微调） |
| lrf | 0.01 | 不变 |
| epochs | 50 | 从 100 减少（微调收敛快） |
| patience | 15 | 从 20 减少 |
| freeze | **5** | 微调首选：保留浅层特征，降低路面域遗忘风险；可消融 freeze=0，但须对比路面 mAP50 是否下降 |
| rect | **False** | 从 True 改变（跨域 mosaic 的必要条件） |
| mosaic | 0.5 | 开启（跨域混合效果） |
| scale | 0.5 | 保持加强版 |
| translate | 0.15 | 保持加强版 |
| degrees | 10.0 | 不变 |
| batch | autobatch | 不变 |
| imgsz | 1280 | 不变 |
| cache | True | 不变（约 15 GB RAM） |

### 脚本改动

| 文件 | 改动 | 幅度 |
|------|------|------|
| `scripts/resize_images.py` | **新建**：将 TDUS img/ 缩放至最长边=1280，输出 960×1280 JPEG | ~50 行 |
| `scripts/generate_pseudo_labels.py` | **扩展**：新增 `--flat-dir` 模式，支持平铺目录（非按视频组织） | ~30 行 |
| `scripts/build_unified_dataset.py` | **新建**：合并路面视频帧 + TDUS 预缩放帧，生成 `data/unified_halfres_tdus/data.yaml` | ~60 行 |
| `scripts/train_yolo26.py` | **扩展**：新增 `--lr0`、`--patience`、`--close-mosaic` CLI 参数 | ~10 行 |
| `notebooks/finetune_yolo26s_colab.ipynb` | **新建**：Colab 微调 notebook（从 best.pt，rect=False） | notebook |

---

## 权衡分析

### 使用 GDino 伪标签 vs GT Bitmap Mask

**GDino 伪标签**：
- 与 SVM 训练数据一致（SVM 在 GDino 裁剪的嵌入上训练）
- 可能漏检或框不准（GDino 在 960×1280 上有效 scale=0.833，即 gdino_preprocess 将 960→800px；尚可接受，但弱于路面帧的 0.986）
- **选择此方案**

**GT Bitmap Mask**：
- 更精确，但需要从 Supervisely bitmap 转换为 bbox（需额外脚本）
- 引入 GT/伪标签不一致问题：YOLO 学 GT bbox，SVM 在 GDino bbox 上运行
- 两套标注体系增加维护复杂度

### rect=False 的代价

`rect=False` 要求在 `imgsz×imgsz` 正方形 batch 内 letterbox 所有图像，会引入 padding 计算量。路面视频帧（AR=1.78）会在上下各填充约 280px，TDUS 帧（AR=0.75）会在左右各填充约 160px。额外计算量约 20–30%，但这是实现跨域 mosaic 的必要代价。

### TDUS 标注噪声

GDino 在行人视角树木上的伪标签质量未经验证（只在行车视频上验证过）。TDUS 图像中树木通常占据图像的大部分（bbox 面积均值约 0.25，**推定值**，GDino 尚未在 TDUS 上运行），GDino 应能检出，但可能产生多框或边框偏松。可在生成后人工抽样 20–30 张检查框质量，必要时调整 GDino 推理的置信度阈值。

---

## 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| TDUS GDino 伪标签质量差 | 中 | 高 | 生成后人工抽检 20–30 张；若框质量差则降低 CONF_THRESH |
| SVM crop 分布偏移：SVM 在原始 3024×4032 GDino 裁剪嵌入上训练，推理时 YOLO 从 960×1280 图像产生不同 crop | 中 | 高 | 若 TDUS 准确率未达 70% 但检测框质量好，优先重新在新 crop 上提取嵌入并重训 SVM |
| freeze=5 导致 TDUS 域适应不足 | 中 | 中 | 可消融 freeze=0；对比两次运行的路面 mAP50（road-only val 子集）确认 forgetting 程度 |
| rect=False 导致路面视频 mAP 下降 | 中 | 中 | 可消融：先只用 TDUS+rect=False 训一个 toy run，观察 val 曲线 |
| TDUS 文件名含特殊字符（空格、括号）与 symlink 结合导致 YOLO dataloader glob 失败 | 中 | 中 | build_unified_dataset.py 以 copy 为默认（而非 symlink）；symlink 作为可选项，使用前在目标平台验证 glob 可达性 |
| lr0=2e-4 微调不收敛 | 低 | 中 | 可消融 lr0=5e-4；若 loss 不下降则增大 lr |
| TDUS 3170 张全部进训练集，数据量比路面视频多 | 低 | 低 | TDUS 3170 vs 路面 2071，比例约 3:2，可接受；若过拟合 TDUS 可降采样 |
| Colab session 中断 | 高 | 中 | resume=True + last.pt 上传 Drive，与上一版本相同策略 |

---

## 成功指标

| 指标 | 目标值 | 当前基线 |
|------|--------|---------|
| TDUS top-1 分类准确率（YOLO 路径） | ≥70% | 33.2% |
| 路面视频 mAP50（road-only val 子集，需单独推理） | ≥0.85 | 0.886 |
| TDUS val GDino 伪标签检出率（有框占比） | ≥80% | 未测 |

---

## 参考

- 当前最优 YOLO 模型：`runs/detect/tree_yolo26s_halfres/weights/best.pt`（mAP50=0.886）
- 上一版 proposal：`docs/proposals/proposal-higher-res-frames-retrain.md`
- TDUS 数据集：`tdus_data/`（train=3170, val=395, test=386，Supervisely bitmap 标注）
- GDino 伪标签脚本：`scripts/generate_pseudo_labels.py`
- TDUS 分类结果对比：YOLO 33.2% vs GDino 93.5%（TDUS test，22 species）
