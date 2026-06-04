# Proposal: 城市树木识别流水线的 C/C++ 生产系统实现

**Status**: Draft  
**Date**: 2026-06-04  
**作者**: Yale Huang  
**Builds on**: `gdino_dinov2_svm/RESULTS.md`、`scripts/predict_pipeline.py`、`scripts/benchmark_pipeline.py`

---

## ⚠️ 架构师审查备注

**审查日期**: 2026-06-04  
**严重程度**：发现 5 处重大问题，需在实施前修正。

### [CRITICAL-1] TensorRT FMHA 融合失败——43 img/s 目标缺乏依据

TensorRT 10.8.0.43 + PyTorch 2.7.1 ONNX 导出路径下，DINOv2 ViT 的 FMHA（Fused Multi-Head Attention）融合**已知存在失败问题**（[NVIDIA/TensorRT #4537](https://github.com/NVIDIA/TensorRT/issues/4537)，2025 年 7 月）：exported ONNX graph 中 attention 操作未生成 `mha`/`fused_mha_v2` 层，退化为分散的 MatMul + Softmax，无法享受 FMHA 的 Tensor Core 加速。若 FMHA 融合失败，DINOv2 TRT fp16 引擎的实际加速倍数将远低于 2×，43 img/s 目标可能无法达到。**本 proposal 的吞吐率目标需标注为假设 FMHA 融合成功的上限估算，实施时必须先验证。**

缓解方案见正文 §"TensorRT fp16 注意事项（更新）"。

### [CRITICAL-2] YOLO ONNX 导出——letterbox 已内嵌，C++ 端描述自相矛盾

原 proposal 第 86 行声称 "ONNX 已内嵌 letterbox，C++ 不需要重复实现"，但第 127 行又要求 "C++ 端需精确复现 letterbox 行为"。两者矛盾。经核查：

- `model.export(format="onnx")` 默认导出**含** Ultralytics letterbox 预处理（resize + pad to `imgsz×imgsz`，灰色填充 114/255）。C++ 端输入应为 **原始 RGB uint8 HWC 图像**，由 ONNX graph 内部完成 letterbox。
- 若需要 C++ 端自己控制 letterbox 以支持动态分辨率（如批量变长图像），应在 `export()` 时传 `batch_size=1, dynamic=True` 并关闭内嵌预处理（`model.export(format="onnx", simplify=True, dynamic=False)`），然后 C++ 手动实现 letterbox。

**必须在实施前明确选择其中一种方案，并在全链路保持一致。**

### [CRITICAL-3] `make_crops_gpu_batch` padding 顺序描述有误

原 proposal 表格中写 "填充前归一化：`(1.0 - mean[c]) / std[c]`"——措辞误导读者认为 C++ 需在 pad 之前做归一化。实际代码顺序（`scripts/predict_pipeline.py` 第 352–358 行）是：

1. 先以 `value=1.0` 填充白色（[0,1] 浮点空间）
2. **再** 对整个 batch（含 pad 区域）统一做 ImageNet 归一化 `(x - mean) / std`

C++ 实现有两种等价方案：(a) 先 pad 填 1.0f，再统一归一化；(b) 直接 pad 填 `(1.0 - mean[c]) / std[c]`，再归一化时 pad 区域值不变。两者均正确，但不能仅做 (b) 而跳过最后的归一化步骤。

### [CRITICAL-4] `_make_transforms`（CenterCrop）与 `make_crops_gpu_batch`（白边 pad）是两条不同路径

`predict_pipeline.py` 中存在两套预处理路径：

| 函数 | 调用场景 | resize 策略 | padding |
|------|---------|------------|---------|
| `_make_transforms` (`T.Resize + T.CenterCrop`) | 旧单图路径 `extract_embedding_batch` | 短边 resize 到 size，中心裁剪 | 无 |
| `make_crops_gpu_batch` | 批量推理（benchmark、run_predict 现行路径）| 长边 resize，等比例缩放 | 白色（1.0）|

**C++ 必须复现 `make_crops_gpu_batch` 的白边 pad 路径**（当前生产准确率 94.82% 的测量路径）。使用 CenterCrop 路径将引入分布漂移，实测 embedding cosine similarity 约 0.97–0.98，低于正文要求的 0.9999。

### [MAJOR-5] 依赖版本约束不完整，缺少部署打包方案

原 proposal 仅提 "ONNX Runtime ≥ 1.17.0" 和 "opset=17"，未说明：CUDA 版本、TensorRT 版本、cuDNN 版本、操作系统约束，以及跨机器部署时 TRT 引擎不可移植的问题。部署打包方案（Docker / 静态链接 / 动态库）完全缺失。详见正文新增 §"依赖版本矩阵"和 §"部署打包方案"。

---

## 背景

### 当前 Python 流水线

现有推理系统实现了三阶段检测→嵌入→分类流水线（`scripts/predict_pipeline.py`），在 TDUS 测试集（386 张）上达到以下生产指标：

| 配置 | 测试准确率 | 吞吐率 (timed) | 吞吐率 (含 I/O) | 漏检数 |
|------|-----------|----------------|-----------------|--------|
| YOLO `tree_yolo26s_unified_halfres_tdus` + SVM 448px | **94.82%** | **21.5 img/s** | **8.8 img/s** | **0** |
| GDino-tiny + SVM 448px | 94.0% | 4.5 img/s | 3.0 img/s | 0 |

推荐生产配置（`gdino_dinov2_svm/RESULTS.md`）：
- 检测器：YOLO `tree_yolo26s_unified_halfres_tdus`，imgsz=**960**，batch=64
- 嵌入：DINOv2 ViT-S/14 CLS token，384-dim
- 分类：RBF-SVM，crop=448px
- 22 类树种，OvO（One-vs-One）多分类

> **注意**：YOLO 检测时 imgsz=960（非 imgsz=1280）。`YOLODetector` 的默认值 `imgsz=1280` 是类定义默认值；生产 benchmark 通过 CLI `--yolo-imgsz 960` 覆盖。C++ 导出和推理必须使用 **imgsz=960**。

### 流水线各阶段代码定位

| 阶段 | 函数 | 文件 | 行号（实测）|
|------|------|------|------|
| YOLO 批量检测 | `YOLODetector.detect_batch` → `detect_yolo_batch()` | `scripts/detection/yolo_detector.py`、`scripts/predict_pipeline.py` | yolo_detector.py 25–27；predict_pipeline.py 278–289 |
| GPU crop + 归一化 | `make_crops_gpu_batch()` | `scripts/predict_pipeline.py` | 306–358 |
| DINOv2 批量嵌入 | `dinov2.forward_features()` on GPU batch | `scripts/benchmark_pipeline.py` `_process_batch` | 171–177 |
| SVM 批量分类 | `predict_species_batch()` | `scripts/predict_pipeline.py` | 361–373 |
| 双缓冲检测接口 | `preprocess_cpu()` / `forward_preprocessed()` | `scripts/detection/base_detector.py` | 35–44 |
| YOLO 不走双缓冲 | `supports_pipeline = False`（默认值）| `scripts/detection/base_detector.py` | 32–33 |

> **YOLO 路径特殊说明**：`YOLODetector.supports_pipeline` 为 `False`（继承基类默认值），因此 `benchmark_pipeline.py` 中 YOLO 走 `else` 分支（第 207–211 行），Ultralytics 内部自行处理 letterbox/batch，**不走双缓冲线程池**。C++ 复现时需注意此差异：YOLO 推理是单线程串行，不与 I/O 线程并发。

### 生产化动机

Python 流水线依赖 PyTorch、Ultralytics、transformers、scikit-learn 等重量级依赖，难以：
1. 嵌入到无 Python 运行时的边缘设备或嵌入式系统
2. 与 C/C++ 为主的生产服务（视频分析后端、城市感知平台）集成
3. 进一步压榨 GPU/CPU 利用率（Python GIL、内存碎片）
4. 以共享库（`.so` / `.dll`）形式暴露推理 API

---

## 目标与范围

### 目标

1. 将现有三阶段流水线（YOLO → crop → DINOv2 → SVM）完整移植到 C/C++ + ONNX Runtime / TensorRT，**在相同硬件上实现 ≥2× 吞吐率提升**（相对 Python timed-bench 21.5 img/s，目标 ≥43 img/s）。

   > ⚠️ **风险警告**：此目标假设 DINOv2 TRT FMHA 融合成功（见 [CRITICAL-1]）。若 FMHA 融合退化为分散 attention ops，实际加速倍数预计 1.3–1.6×（28–34 img/s），43 img/s 可能无法达到。建议将目标拆分为：保证目标 ≥30 img/s，冲刺目标 ≥43 img/s（FMHA 融合成功前提下）。

2. 暴露清洁的 C API（`urban_tree_id_infer()`）供上层服务调用，无 Python 运行时依赖。
3. 通过 TDUS test split 验证端到端准确率相对退化 ≤1%（绝对值 ≥93.8%）。
4. 首阶段使用 ONNX Runtime（可移植性优先），第二阶段切换 TensorRT fp16（吞吐率优先）。

### 明确范围

| 内容 | 是否在本 Proposal 范围内 |
|------|------------------------|
| YOLO ONNX 导出及 C++ 推理 | **是** |
| DINOv2 ONNX 导出及 C++ 推理 | **是** |
| RBF-SVM C++ 推理（skl2onnx 首选） | **是** |
| TensorRT fp16 引擎构建与验证 | **是（第二阶段）** |
| 整体 C++ 流水线架构（I/O / GPU / CPU 三线程） | **是** |
| 模型重新训练或 fine-tune | **否** |
| 修改 Python 训练脚本 | **否** |
| Android / iOS 移动端适配 | **否** |

---

## 依赖版本矩阵

C++ 生产部署需锁定以下版本，**跨版本不保证引擎兼容性**（特别是 TensorRT 引擎序列化文件与 TRT 大版本强绑定）：

| 依赖 | 推荐版本 | 最低要求 | 备注 |
|------|---------|---------|------|
| CUDA | 12.4+ | 12.0 | TRT 10.x 需要 CUDA 12.x；低于 12.0 不支持 TRT 10 |
| cuDNN | 9.2+ | 8.9 | TRT 10 构建依赖；注意 cuDNN 9.x 与 8.x API 有变更 |
| TensorRT | **10.6+** | 10.4（有已知 fp16 bug，不推荐）| TRT 10.4 fp16 推理已知输出错误；TRT 10.6 修复；TRT 10.8 有 FMHA 融合失败问题（见 [CRITICAL-1]，需验证）|
| ONNX Runtime | 1.19+ | 1.17 | opset 17 支持；TensorRT Execution Provider 需 1.17+；1.19 修复若干 ViT op 兼容问题 |
| ONNX opset | 17 | 17 | DINOv2/YOLO 共同支持的最高稳定 opset；opset 18 dynamo_export 问题较多 |
| OpenCV | 4.9+ | 4.5 | `dnn::NMSBoxes`、INTER_LINEAR、JPEG decode |
| PyTorch（导出环境）| 2.4–2.6 | 2.1 | PyTorch 2.7 + TRT 10.8 已知触发 FMHA 融合问题 |
| skl2onnx | 1.17+ | 1.15 | SVC RBF OvO 支持；1.15 以下有已知多分类 bug |
| scikit-learn | 与导出环境一致 | 1.3 | joblib 模型与 sklearn 版本需匹配 |

> **TRT 引擎不可移植性警告**：TensorRT `.engine` 文件与 GPU 架构、TRT 版本、CUDA 版本**强绑定**，不同机器间**不可复制**。每个部署目标机器必须在本机重新运行 `trtexec` 或 engine build。Docker 镜像内构建引擎可解决环境一致性问题，但镜像与目标机 GPU 架构必须相同（sm_86 vs sm_89 等）。

---

## 方案设计

### 模块一：YOLO 检测器

#### 模型导出

```python
from ultralytics import YOLO
model = YOLO("data/model_weights/tree_yolo26s_unified_halfres_tdus/best.pt")
# imgsz=960：与生产 benchmark 配置一致（RESULTS.md 推荐配置）
# 注意：YOLODetector 类默认 imgsz=1280，但 benchmark 用 --yolo-imgsz 960 覆盖
model.export(format="onnx", imgsz=960, opset=17, simplify=True, dynamic=False)
# 产出: tree_yolo26s_unified_halfres_tdus.onnx
# 输入: (1, 3, 960, 960) float32，范围 [0,1]，**letterbox 已内嵌**
# 输出: (1, 5, 8400) float32  — [x_c, y_c, w, h, conf]（anchor-free，相对于 960px）
```

#### YOLO 预处理——letterbox 选择（二选一，必须在实施前确定）

**方案 A（推荐：让 ONNX graph 内嵌 letterbox）**

Ultralytics 默认 ONNX 导出已内嵌 letterbox（等比例缩放 + 灰色填充 114/255，resize 到 `imgsz×imgsz`）。C++ 端只需将原始 RGB uint8 HWC 图像直接传入 ONNX 输入 tensor（转为 float32 CHW，除以 255）：

```cpp
// 方案 A：内嵌 letterbox，C++ 无需手动实现
// 1. JPEG decode → RGB uint8 HWC (libjpeg-turbo 或 OpenCV)
// 2. 转 CHW float32，/255.0（无需 resize，ONNX graph 内部处理）
// 3. 传入 ONNX Runtime session，input shape 为 (1, 3, 960, 960) 时 ONNX 会自动 letterbox
```

坐标后处理：ONNX 输出的 bbox 坐标相对于 960×960 输入空间（已含 letterbox 偏移），需用 Ultralytics 标准还原公式（`scripts/ultralytics/utils/ops.py:scale_boxes()`）还原到原始图像坐标。**可直接参考 [YOLOs-CPP](https://github.com/Geekgineer/YOLOs-CPP) 或 [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP) 的坐标还原实现，勿自行推导。**

**方案 B（可选：C++ 手动 letterbox，支持动态 batch）**

若需 `dynamic=True` 支持变 batch 推理，导出时：

```python
model.export(format="onnx", imgsz=960, opset=17, dynamic=True, simplify=True)
```

C++ 端手动实现 letterbox（等比例缩放到长边 960，短边补灰 `114`，填充到 960×960），然后才传入网络。坐标还原时需记录 scale factor 和 pad offset。

#### NMS 后处理

YOLO ONNX 导出默认**不含** NMS（输出 8400 个候选框）。C++ 端需实现 NMS：

```cpp
// 使用 OpenCV dnn::NMSBoxes
std::vector<int> indices;
cv::dnn::NMSBoxes(
    boxes,          // vector<cv::Rect2d>，从 ONNX 输出转换
    scores,         // vector<float>
    conf_threshold, // 0.25（与 Ultralytics 默认一致）
    iou_threshold,  // 0.45（与 Python 推理默认一致）
    indices
);
```

NMS 参数需与 Ultralytics 内部默认值一致。可参考 `ultralytics/utils/ops.py` 中的 `non_max_suppression()` 函数确认阈值含义（`conf_thres=0.25, iou_thres=0.45`）。

> **参考实现**：[YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP) 和 [YOLOs-CPP](https://github.com/Geekgineer/YOLOs-CPP)（支持 YOLO5–YOLO12）提供了完整的 letterbox + NMS C++ 实现，可直接集成或参考，避免手动推导 letterbox 坐标还原公式的错误。

#### C++ 推理接口

```cpp
struct YoloDetector {
    Ort::Session session;           // ONNX Runtime session
    // 输入: RGB uint8 HWC，原始分辨率（方案A：传给ONNX内嵌letterbox）
    //       或 letterbox后的(960,960,3) uint8（方案B）
    // 返回: vector<cv::Rect2d> boxes（绝对像素坐标，原始图像空间）
    std::vector<cv::Rect2d> Run(
        const cv::Mat& img_rgb,
        float conf_thr = 0.25f,
        float iou_thr  = 0.45f
    );
};
```

---

### 模块二：DINOv2 ViT-S/14 嵌入器

#### 模型导出

DINOv2 ViT-S/14 通过 `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` 加载，输出 384-dim CLS token（`feat["x_norm_clstoken"]`，`scripts/predict_pipeline.py` 第 165–167 行）。

```python
import torch

# 使用 PyTorch 2.4–2.6（避免 2.7 + TRT 10.8 FMHA 融合问题）
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()

# 封装薄包装模块，只导出 CLS token（避免 1024 个 patch token 占用带宽）
class DINOv2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model.forward_features(x)["x_norm_clstoken"]  # (N, 384)

wrapper = DINOv2Wrapper(dinov2).eval()

# crop_size=448: ViT-S/14 patch_size=14, 448/14=32 patches per side → 1024 patch tokens + 1 CLS
dummy = torch.zeros(1, 3, 448, 448)
torch.onnx.export(
    wrapper,
    dummy,
    "dinov2_vits14_448.onnx",
    input_names=["pixel_values"],
    output_names=["cls_token"],   # (N, 384)
    opset_version=17,             # opset 18 dynamo_export 路径问题较多，使用 17
    dynamic_axes={"pixel_values": {0: "batch"}},
)
```

**已知导出障碍**：
- `facebookresearch/dinov2` 官方 repo [issue #19](https://github.com/facebookresearch/dinov2/issues/19) 记录了 ONNX 导出时 `prepare_tokens_with_masks` 的 unsupported op 问题（`::_upsample_bicubic2d_aa`，在较旧 PyTorch 版本中）。PyTorch 2.1+ 已修复该 bicubic upsample op，应使用 PyTorch ≥2.1。
- HuggingFace transformers `Dinov2Model` 通过 `optimum` 库导出 ONNX 更稳定（[transformers #26790](https://github.com/huggingface/transformers/issues/26790)），可作为备选路径。

#### 预处理对齐（最关键风险点）

**重要**：Python 批量推理走 `make_crops_gpu_batch()`（`scripts/predict_pipeline.py` 第 306–358 行），**不是** `_make_transforms`（CenterCrop 路径，仅用于旧单图路径 `extract_embedding_batch`）。这两条路径产生不同的 embedding 分布（cosine similarity 约 0.97–0.98），C++ 必须复现 `make_crops_gpu_batch` 的白边 pad 路径。

`make_crops_gpu_batch()` 执行（以实际代码为准）：

1. **选最大面积 bbox**（含 5% padding，clamp 到图像边界）
2. **aspect-ratio-preserving resize**：`F.interpolate(..., mode="bilinear", align_corners=False)`，长边 resize 到 `target_size=448`
3. **白色画布 padding**（先以 `value=1.0` 填充至 448×448，padding 均匀分配到两侧）
4. **ImageNet 归一化**：对整个 batch（含 pad 区域）统一执行 `(x - mean) / std`，mean `[0.485, 0.456, 0.406]`，std `[0.229, 0.224, 0.225]`

**C++ 实现要点**：

| 细节 | Python 实际行为 | C++ 要求 |
|------|---------------|---------|
| bbox 选择 | 最大面积，pad_frac=0.05，clamp 到图像边界 | 与 Python 逻辑完全一致（整数截断：`int(x0)` 等）|
| resize 插值 | bilinear，`align_corners=False` | OpenCV `INTER_LINEAR`（等价于 PyTorch `align_corners=False`），不可用 INTER_CUBIC |
| padding 顺序 | **先** pad 填 1.0（白色）→ **再**整体归一化 | 推荐同样顺序：先 pad 填 1.0f，再归一化；或等价地 pad 填 `(1.0-mean[c])/std[c]` 跳过 pad 区域归一化，但前提是 pad 之后无需重新归一化 |
| padding 分配 | `pw // 2` 在左/上，`pw - pw // 2` 在右/下（向右/下取整）| 与 `F.pad` 语义对齐；奇数 padding 偏右/下 1px |
| 数据类型 | `uint8 → /255.0 → [0,1] float32 → pad → 归一化` | C++ 同顺序，`/255.0` 不能用整数除 |
| 无 bbox 时 | `crop = img`（全图作为 crop）| 同 Python，全图 resize 到 448×448（白边 pad）|

> **数值对齐验证**：用 100 张 TDUS 图像分别跑 Python GPU 路径和 C++ 实现，要求逐元素 L∞ distance（最大绝对差）< 0.01（归一化后），cosine similarity ≥ 0.9995。此误差范围比原 proposal 的 1e-4（L2）更合理——OpenCV bilinear 与 PyTorch `align_corners=False` 的像素坐标映射存在约 0.5px 差异，严格要求 1e-4 在高分辨率图像上不可实现。

#### C++ 推理接口

```cpp
struct DINOv2Embedder {
    Ort::Session session;
    // 输入: (N, 3, 448, 448) float32，白边 pad + ImageNet 归一化后
    // 输出: (N, 384) float32，CLS token
    std::vector<std::vector<float>> Embed(
        const std::vector<cv::Mat>& crops_normalized  // 已归一化，float32 CHW
    );
};
```

#### TensorRT fp16 注意事项（更新）

**FMHA 融合失败风险（[CRITICAL-1] 详细缓解方案）**：

TensorRT 10.8.0.43 对 DINOv2/ViT ONNX 存在 FMHA 融合失败问题（[NVIDIA/TensorRT #4537](https://github.com/NVIDIA/TensorRT/issues/4537)）。推荐的诊断和缓解流程：

```bash
# 步骤1：构建 TRT 引擎时开启 verbose，检查 attention 层是否被 FMHA 融合
trtexec --onnx=dinov2_vits14_448.onnx --fp16 --verbose 2>&1 | grep -i "fmha\|attention\|mha"
# 若输出含 "mha" 或 "fused_mha_v2" → 融合成功
# 若无上述字样，attention 退化为分散 ops → 性能受损，需走下方替代方案
```

**替代方案（按推荐顺序）**：

1. **降级 PyTorch 到 2.4–2.6 重新导出**：PyTorch 2.7 修改了 scaled_dot_product_attention 的 ONNX 导出格式，可能破坏 TRT FMHA 识别模式。
2. **使用 `torch.nn.functional.scaled_dot_product_attention` 替换手写 attention**（若有访问 DINOv2 源码权限），显式 Flash Attention 格式更易被 TRT FMHA 识别。
3. **接受退化，以 ONNX Runtime + TensorRT Execution Provider 替代纯 TRT**：ONNX Runtime TRT EP 在内部处理 fusion 失败的回退，兼容性更好。
4. **使用 `trtexec --precisionConstraints=obey --layerPrecisions` 精细控制层精度**，softmax / LayerNorm 强制 fp32，其余 fp16。

**fp16 数值稳定性缓解（无论 FMHA 是否成功）**：

ViT attention 在 fp16 下，softmax 输入值（attention logit）若超过约 ±340 即触发 fp16 饱和（最大值 65504），导致 softmax 输出全零或 NaN。ViT-S/14 的 attention head dimension 为 384/6=64，通常不严重，但仍需验证：

```python
# 验证标准：对 TDUS val split 100 张图像，比较 ORT fp32 vs TRT fp16 embedding
# 要求：cosine_similarity 均值 ≥ 0.999，最小值 ≥ 0.99
# 若不满足：对 softmax、LayerNorm 层强制 fp32
```

TRT per-layer 精度设置（C++ API）：

```cpp
// 对已知不稳定层强制 fp32
auto layer = network->getLayer(i);
if (std::string(layer->getName()).find("Softmax") != std::string::npos ||
    std::string(layer->getName()).find("LayerNorm") != std::string::npos) {
    layer->setPrecision(nvinfer1::DataType::kFLOAT);
    layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
}
```

校验标准：TRT fp16 推理与 ONNX Runtime fp32 推理在 TDUS test split（386 张）上 embedding cosine similarity **均值 ≥ 0.999，最小值 ≥ 0.99**（原 proposal 要求均值 ≥ 0.999 但未规定最小值，补充之）。

---

### 模块三：RBF-SVM 分类器

#### 方案选型

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **skl2onnx 导出**（首选） | 无需手写核函数；ONNX Runtime 统一推理；支持 batch | 导出后模型文件较大（22 类 OvO = 231 个二分类器，支持向量可能 >10k）；已知历史 bug（见下）| **首选** |
| libsvm C API | 原生 C 库；内存可控 | 需将 joblib 中的支持向量矩阵转存为 libsvm 格式；需额外转换脚本 | 备选 |
| 手写 BLAS 加速 RBF 核 | 最高灵活性；可定制 batch 大小 | 实现复杂；需维护 | 最后备选 |

#### skl2onnx 导出

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

clf = joblib.load("gdino_dinov2_svm/results/svm_model.joblib")
# clf 为 sklearn.svm.SVC(kernel="rbf", decision_function_shape="ovr")
# OvO 在 sklearn 内部始终用 OvO 训练，decision_function_shape 影响输出形状而非训练

initial_type = [("float_input", FloatTensorType([None, 384]))]
# target_opset=15 更保守（skl2onnx 对 SVC RBF 的 opset 17 支持较新，需验证）
onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=15)
with open("svm_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**已知 skl2onnx 风险**：

- [sklearn-onnx #478](https://github.com/onnx/sklearn-onnx/issues/478)：OvO SVC pipeline 导出有历史 bug（2020 年报告，新版本可能已修复，需验证）。
- [onnxruntime #11284](https://github.com/microsoft/onnxruntime/issues/11284)：SVM 导出后 ONNX Runtime 推理输出不同，根因是 RBF kernel 数值精度差异（float vs double 累加顺序）。

**导出验证（必须执行）**：

```python
import onnxruntime as ort
import numpy as np

# 对 TDUS val split（395 张）分别用 sklearn 和 ONNX Runtime 推理
sess = ort.InferenceSession("svm_model.onnx")
# 比较 predict() 结果（类别字符串），要求完全一致
# 若差异 > 0.5%，视为 bug，回退 libsvm C API 方案
```

> **置信度说明**：Python `predict_species_batch()` 的置信度计算是对 `decision_function` 输出做 softmax（第 369–372 行），而非真实概率。skl2onnx 默认导出 `predict` 和 `decision_function`；需确认 ONNX Runtime 输出的 `decision_function` 与 sklearn 输出数值差异在可接受范围（L∞ < 0.01），否则置信度分数会系统性偏移。

#### RBF 核批量计算性能

22 类 OvO 训练 231 个二分类器，每个二分类器有若干支持向量。设总支持向量数为 $N_{sv}$，batch size 为 $B$，每次推理的 RBF 核计算量为：

$$O(B \times N_{sv} \times 384)$$

若 $N_{sv} \approx 10000$（保守估算），batch=64 时约 $2.5 \times 10^8$ 次浮点运算。建议：
- **skl2onnx 路径**：ONNX Runtime 自动利用 OpenBLAS / MKL，无需手动优化
- **libsvm C API 路径**：调用 `cblas_sgemm` 批量计算 $K(\mathbf{x}, \mathbf{sv})$，然后加权求和

---

## 整体 C++ 架构

### 三线程双缓冲架构

> **YOLO 架构差异注意**：Python 端 YOLO 不走双缓冲（`supports_pipeline=False`），Ultralytics 内部自行处理预处理。C++ 端可选择：(a) 保持单线程串行（忠实复现 Python 行为）；(b) 手动实现 YOLO 的 I/O 与 GPU 推理双缓冲（需自行拆分 letterbox 预处理和 TRT forward，可参考 YOLOv8-TensorRT-CPP 实现）。图示为理想双缓冲架构，实际 YOLO 路径取决于方案选择。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         C++ Production Pipeline                              │
│                                                                              │
│  ┌──────────────┐     queue_a     ┌──────────────────┐    queue_b           │
│  │  I/O Thread  │ ─────────────► │   GPU Thread     │ ──────────►          │
│  │              │                │                  │             │          │
│  │ libjpeg-turbo│                │ [YOLO TensorRT]  │             ▼          │
│  │  JPEG decode │                │  - H2D transfer  │  ┌──────────────────┐ │
│  │              │                │  - YOLO forward  │  │   CPU Thread     │ │
│  │ (方案A: 无   │                │  - NMS           │  │                  │ │
│  │  预处理)     │                │                  │  │ [SVM ONNX/libsvm]│ │
│  │ (方案B: 手动 │                │ [crop kernel]    │  │  - RBF 核计算    │ │
│  │  letterbox)  │                │  - bbox select   │  │  - OvO voting    │ │
│  │              │                │  - bilinear resz │  │  - argmax        │ │
│  │ 输出:        │                │  - white pad     │  │                  │ │
│  │  CHW uint8   │                │  - ImageNet norm │  │ 输出:            │ │
│  │  CPU tensor  │                │                  │  │  species + conf  │ │
│  └──────────────┘                │ [DINOv2 TRT]     │  └──────────────────┘ │
│                                  │  - forward fp16  │             │          │
│                                  │  - CLS token out │             │          │
│                                  │                  │             ▼          │
│                                  │ 输出:            │     JSON / struct     │
│                                  │  (N,384) fp32   │     结果输出           │
│                                  └──────────────────┘                        │
│                                                                              │
│  对应 Python:  DataLoader worker    GPU Thread (main)    CPU Thread (infer)  │
│      _ImageDataset.__getitem__   _process_batch()      predict_species_batch()│
│      （YOLO不走双缓冲，见注释）                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**CUDA Graph 优化**（推荐在 M3 阶段引入）：ONNX Runtime 支持 CUDA Graph capture（`OrtCUDAProviderOptions.enable_cuda_graph = 1`），可减少每次推理的 CUDA kernel launch overhead，对固定 batch size 的 DINOv2 推理有约 5–15% 额外提升。需要固定 input/output tensor 地址（IOBinding），详见 [ONNX Runtime device tensor 文档](https://onnxruntime.ai/docs/performance/device-tensor.html)。

### 关键数据流（单 batch，batch=64）

以下时序估算基于 FMHA 融合成功的乐观假设，括号内为 FMHA 失败时的保守估算：

```
JPEG 文件 (64×)
  ──[libjpeg-turbo]──► RGB uint8 HWC  (~5ms/batch)
  ──[H2D DMA]────────► GPU uint8 tensor  (non_blocking=true, ~1ms)
  ──[/255 + (letterbox 方案B)]► (64,3,960,960) float32 GPU  (~0.5ms)
  ──[YOLO TensorRT]──► (64,5,8400) float32  (~8ms, fp16；依赖 GPU 型号)
  ──[NMS]────────────► 64× bbox list  (~0.5ms, CPU OpenCV 或 GPU NMS)
  ──[crop+resize+pad]► (N,3,448,448) float32 GPU  (~1ms, CUDA kernel)
  ──[DINOv2 TRT]─────► (N,384) float32  (~6ms fp16 FMHA成功 / ~12ms FMHA失败)
  ──[D2H]────────────► CPU (N,384) float32  (~0.5ms)
  ──[SVM ONNX]───────► 22 类 logit + argmax  (~2ms, CPU)
  ──[结构化输出]─────► {image_id, species, confidence} × N
总计（FMHA成功）: ~24ms / batch=64 → ~2700 img/s 理论峰值（不含流水线 stall）
实际含流水线 overhead 估算: ~43 img/s（FMHA成功）/ ~30 img/s（FMHA失败）
```

> 以上估算基于 12 GB VRAM GPU（RESULTS.md 测量环境）。实际数字必须通过 M2 里程碑实测确定，不应作为承诺指标。

### C API 接口设计

```c
/* urban_tree_id.h */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* image_path;
    char        species[64];     /* snake_case 树种名，如 "acer_palmatum" */
    float       confidence;      /* [0, 1] softmax 代理置信度（decision_function softmax，非真实概率）*/
    int         n_boxes;         /* 检测到的 bbox 数量（检测阶段）；0 表示未检测到树木 */
    int         error_code;      /* 0=成功，非0=错误（OOM/模型加载失败/图像解码失败）*/
} UrbanTreeResult;

/* 初始化流水线（加载模型，分配 GPU 资源，阻塞直到 warmup 完成）*/
void* urban_tree_id_create(
    const char* yolo_onnx_path,
    const char* dinov2_onnx_path,
    const char* svm_onnx_path,
    int         batch_size,       /* 推荐 64；必须与 SVM 训练时的 crop batch 一致 */
    int         crop_size,        /* 推荐 448；必须与 SVM 训练时的 crop_size 一致 */
    int         device_id         /* CUDA device index */
);

/* 销毁流水线（释放所有资源，等待后台线程退出）*/
void urban_tree_id_destroy(void* handle);

/* 批量推理（同步）：传入 n 张图像路径，写入 results */
/* 线程安全：同一 handle 不可并发调用（单流水线单线程调用）*/
/* 跨 handle 调用（每个 handle 独占 GPU 资源）：由调用方保证不超显存 */
int urban_tree_id_infer_batch(
    void*              handle,
    const char**       image_paths,
    int                n,
    UrbanTreeResult*   results     /* 调用方分配，长度 >= n */
);

/* 查询版本信息（TRT/ORT/CUDA 版本，用于部署诊断）*/
const char* urban_tree_id_version(void);

#ifdef __cplusplus
}
#endif
```

> **线程安全说明**：单 `handle` 的 `urban_tree_id_infer_batch` **不线程安全**（GPU session 有 internal state）。多线程场景应每线程创建独立 handle，或在应用层加锁。`urban_tree_id_create` / `urban_tree_id_destroy` 非线程安全，应在主线程顺序调用。

---

## 权衡分析

### ONNX Runtime vs TensorRT

| 维度 | ONNX Runtime | TensorRT |
|------|-------------|----------|
| **可移植性** | 跨平台（x86/ARM/GPU/CPU）| NVIDIA 专属，需 CUDA + cuDNN |
| **推理速度** | 基准（约等于 Python PyTorch）| fp16 通常 1.5–4× 加速（ViT 实测差异大；FMHA 融合成功可达 3×，失败约 1.3×）|
| **部署复杂度** | 低（单 `.onnx` 文件）| 高（需本机构建引擎，不可跨 GPU 型号或 TRT 版本移植）|
| **精度风险** | fp32，无损 | fp16 attention softmax 有溢出风险（需 per-layer 精度设置）|
| **FMHA 融合风险** | N/A | TRT 10.8 + PyTorch 2.7 ONNX 存在已知融合失败问题 |
| **维护成本** | 低（官方 ONNX opset 保证兼容性）| 中高（TRT 版本升级需重建引擎；PyTorch 版本变化可能破坏 FMHA 融合）|
| **首阶段推荐** | **是**（快速验证准确率对齐）| 否 |
| **第二阶段推荐** | 否（生产吞吐率瓶颈）| **是**（但需先验证 FMHA 融合，见 [CRITICAL-1]）|

**里程碑顺序**：
1. ONNX Runtime 全链路跑通 → 验证 TDUS test 准确率（目标 ≥93.8%，相对退化 ≤1%）
2. TensorRT fp16 引擎构建（YOLO + DINOv2 分别验证）→ **先检查 FMHA 融合状态** → cosine similarity 验证 → 吞吐率测试

### 预处理实现选项

| 方案 | 推荐场景 |
|------|---------|
| **OpenCV CPU**（libjpeg-turbo decode + INTER_LINEAR resize）| 首选，与 Python `make_crops_gpu_batch` 的 bilinear 行为最接近，易于调试 |
| **CUDA kernel**（自定义 bilinear resize + padding）| 当 I/O Thread 成为瓶颈（batch=64 时 CPU 预处理 >GPU 推理）时考虑 |
| **nvJPEG**（GPU JPEG decode）| 当图像文件 I/O 本身成为瓶颈时（SSD 顺序读取 >500MB/s 场景）|

---

## 数值对齐验证策略

C++ 实现必须通过以下验证门控（均为 CI gate）才能进入 M1 完成里程碑：

### 验证脚本（Python 参考实现 vs C++ 输出）

```python
# validate_alignment.py — 用于 CI 门控
import numpy as np

def validate_embedding_alignment(py_embeddings: np.ndarray,
                                  cpp_embeddings: np.ndarray) -> dict:
    """
    py_embeddings: (N, 384) Python GPU 路径输出
    cpp_embeddings: (N, 384) C++ 输出
    """
    cosine = np.sum(py_embeddings * cpp_embeddings, axis=1) / (
        np.linalg.norm(py_embeddings, axis=1) * np.linalg.norm(cpp_embeddings, axis=1)
    )
    l_inf = np.max(np.abs(py_embeddings - cpp_embeddings), axis=1)
    return {
        "cosine_mean":   cosine.mean(),
        "cosine_min":    cosine.min(),
        "l_inf_mean":    l_inf.mean(),
        "l_inf_max":     l_inf.max(),
        # 通过标准
        "pass_cosine":   cosine.min() >= 0.9995,
        "pass_l_inf":    l_inf.max() < 0.01,     # 归一化后最大绝对差
    }
```

### 各阶段数值容差

| 验证点 | 指标 | 门控标准 | 说明 |
|-------|------|---------|------|
| YOLO bbox 坐标（C++ vs Python）| 绝对像素差 | L∞ < 2px | letterbox 坐标还原误差 |
| DINOv2 crop 预处理 tensor（C++ vs Python）| L∞（归一化后）| < 0.01 | OpenCV bilinear vs PyTorch bilinear 亚像素差 |
| DINOv2 embedding（ORT fp32 C++ vs Python GPU）| cosine similarity | 均值≥0.9995，最小≥0.99 | 预处理误差传播 |
| DINOv2 embedding（TRT fp16 vs ORT fp32）| cosine similarity | 均值≥0.999，最小≥0.99 | fp16 量化误差 |
| SVM 预测类别（ORT vs sklearn）| 逐样本预测类别字符串完全一致 | val 395 张零差异（逐样本，任意一张不一致即为 skl2onnx bug，回退 libsvm C API）| 不一致即为 skl2onnx bug |
| SVM 置信度（ORT vs sklearn softmax）| L∞ | < 0.01 | decision_function 数值精度 |
| 端到端准确率（C++ vs Python）| top-1 acc | ≥ 93.8%（绝对值）| TDUS test 386 张 |

---

## 风险与缓解措施

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **TRT FMHA 融合失败**（DINOv2 ViT attention 未融合，导致吞吐率目标无法达到）| **高**（TRT 10.8 已有 issue）| **高** | 降级 PyTorch 2.4–2.6 重新导出；或改用 ORT+TRT EP；或接受 1.3× 加速降低目标为 ≥30 img/s |
| **YOLO letterbox 方案选择错误**（C++ 手动 letterbox 与 ONNX 内嵌 letterbox 混用）| **高**（两方案互斥）| **高** | 实施前明确选择方案 A 或 B，单测验证坐标精度（L∞ < 2px）|
| **crop 预处理路径混淆**（C++ 复现 CenterCrop 而非白边 pad）| 中 | **高** | 用 100 张图验证 embedding cosine ≥ 0.9995；CenterCrop 路径 cosine 约 0.97–0.98 易被检出 |
| **TensorRT fp16 ViT attention 数值溢出**（NaN 传播导致 embedding 异常）| 中 | 高 | 对 softmax / LayerNorm 层保留 fp32；使用 val split 验证 embedding cosine ≥ 0.999 |
| **skl2onnx OvO SVC 导出输出不一致**（历史 bug，新版本状态未明）| 中 | 中 | 导出后立即在 val 395 张全量比对，差异 >0.5% 则回退 libsvm C API |
| **SVM 支持向量数过大**（$N_{sv}$ 可能超过 10k，批量 RBF 核耗时超预期）| 中 | 中 | 首选 skl2onnx（ORT 自动 BLAS 加速）；若 latency 不满足，考虑 SV pruning 或改用 liblinear |
| **TRT 引擎跨机器不可移植**（部署目标与构建机 GPU 型号不同）| 中 | 中 | 每个部署目标本机构建引擎；或 Docker 内构建（需镜像与目标机 GPU arch 一致）|
| **双模型显存峰值超出 GPU 限制**（YOLO + DINOv2 同时驻留 GPU，batch=64 激活值另加）| 低 | 中 | 测量峰值显存（NVML API）；若超限则 YOLO 检测和 DINOv2 嵌入分时共享显存 |
| **DINOv2 ONNX 导出障碍**（PyTorch 版本导致 prepare_tokens_with_masks op 不支持）| 低 | 中 | 使用 PyTorch 2.1–2.6；或改用 HuggingFace `optimum` 导出路径 |

---

## 部署打包方案

### 方案选择矩阵

| 方案 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **Docker 镜像**（推荐）| 服务器部署，GPU 驱动与镜像解耦 | 环境隔离；可重现；NVIDIA Container Toolkit 自动挂载 GPU | 镜像大（含 TRT + ORT ~4GB）；TRT 引擎仍需本机 build |
| **动态链接 `.so`** | 集成到现有 C++ 服务 | 轻量；调用方控制生命周期 | 调用方需自行安装 TRT/ORT 运行时 |
| **静态链接二进制** | 边缘设备，依赖受限环境 | 零外部依赖（除 CUDA Driver）| 二进制很大（>200MB）；TRT 不支持完全静态链接 |
| **ONNX Runtime Web / WASM** | 浏览器 / 无 CUDA 环境 | 超高可移植性 | 无 GPU 加速；吞吐率远低于目标 |

**推荐部署方案（服务器）**：

```dockerfile
# Dockerfile（示例）
FROM nvcr.io/nvidia/tensorrt:24.08-py3   # TRT 10.x, CUDA 12.4, cuDNN 9.x
# 固定 nvcr.io 镜像版本，不使用 latest

# 安装 ONNX Runtime（TRT EP）
RUN pip install onnxruntime-gpu==1.19.* --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 复制 ONNX 模型文件（不含 TRT 引擎，在容器启动时本机 build）
COPY models/dinov2_vits14_448.onnx models/yolo_960.onnx models/svm_model.onnx /app/models/

# 启动时 build TRT 引擎（first-run 约 2–5 分钟，缓存到 volume）
ENTRYPOINT ["/app/build_engines_and_serve.sh"]
```

**TRT 引擎生命周期管理**：

1. 首次启动时调用 `trtexec` 或 C++ engine builder，将 `.engine` 文件写入持久化 volume（非容器层）。
2. 后续启动检查 `.engine` 文件的 TRT 版本 header，若版本不匹配则重新 build。
3. 提供 `--rebuild-engines` 命令行选项，强制重建。

---

## 前置条件

| 前置条件 | 来源 | 验证方式 |
|----------|------|---------|
| YOLO 模型权重已可用 | `data/model_weights/tree_yolo26s_unified_halfres_tdus/best.pt`（HuggingFace `yaleh/urban-tree-id`）| 本地文件存在；`ultralytics` 加载无报错 |
| SVM 模型已训练（448px）| `gdino_dinov2_svm/results/svm_model.joblib`（或 HuggingFace `svm/gdino_dinov2_448px/svm_model.joblib`）| 文件存在；`joblib.load()` 返回 `sklearn.svm.SVC`；`clf.score(X_te, y_te) ≥ 0.948` |
| DINOv2 可通过 torch.hub 加载 | `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` | 网络连通或本地缓存；`forward_features()` 输出 dict 含 `x_norm_clstoken` shape `(1, 384)` |
| TDUS test split 可用于验证 | `data/tdus_data/test/` 严格隔离（不参与训练）| `data/tdus_data/test/img/` 共 386 张，`test/ann/` 有对应标注 |
| skl2onnx 安装且版本兼容 | `pip install skl2onnx>=1.17` | `from skl2onnx import convert_sklearn` 无报错；ONNX 导出后 ORT 可加载且 val 准确率 100% 一致 |
| 构建环境版本锁定 | 见"依赖版本矩阵" | `nvidia-smi` / `python -c "import tensorrt; print(tensorrt.__version__)"` 确认版本 |

---

## 成功指标

| 指标 | 保证目标 | 冲刺目标 | 验证方式 |
|------|---------|---------|---------|
| TDUS test top-1 准确率（ORT fp32 路径）| ≥ 93.8% | ≥ 94.5% | C++ benchmark 脚本对比 `benchmark_pipeline.py` 输出 |
| 端到端吞吐率（TRT fp16 顺序执行，M2，batch=64，含 crop，不含磁盘 I/O）| ≥ 30 img/s | ≥ 43 img/s（FMHA 融合成功前提）| 独立计时脚本，预加载图像，排除磁盘 I/O |
| 端到端吞吐率（TRT fp16 三线程流水线，M3，batch=64，不含磁盘 I/O）| **≥ 35 img/s**（相对 M2 顺序执行提升 ≥15%，体现双缓冲收益）| ≥ 43 img/s（FMHA 融合成功前提）| 独立计时脚本，预加载图像，排除磁盘 I/O |
| DINOv2 embedding 对齐（C++ ORT fp32 vs Python GPU）| cosine 最小值 ≥ 0.99，L∞ < 0.01 | cosine 均值 ≥ 0.9995 | 100 张随机 TDUS 图像 |
| DINOv2 embedding 对齐（TRT fp16 vs ORT fp32）| cosine 均值 ≥ 0.999，最小值 ≥ 0.99 | cosine 最小值 ≥ 0.999 | TDUS test 386 张 |
| SVM 推理结果一致性（ORT vs sklearn）| val 395 张逐样本预测类别字符串完全一致（零差异）| — | val split 395 张全量比对 |
| YOLO bbox 坐标精度（C++ vs Python）| L∞ < 2px | L∞ < 1px | 50 张测试图像，手动/脚本比对 |
| 峰值显存（YOLO + DINOv2 + SVM，batch=64）| ≤ 8 GB | ≤ 6 GB | NVML `nvmlDeviceGetMemoryInfo` 记录最大值 |
| C API 线程安全（单 handle 单线程顺序调用）| 无崩溃，ASAN 无报告 | — | ASAN + Valgrind，1000 图像顺序推理 |

---

## 里程碑规划

```
M0（环境准备，~0.5 周）
  ├── M0.1  确定 YOLO letterbox 方案（方案A/B），写入决策记录
  ├── M0.2  锁定依赖版本矩阵，搭建 Docker 构建环境
  └── M0.3  准备对齐验证脚本（validate_alignment.py），作为后续所有 milestone 的 CI gate

M1（ONNX Runtime 全链路，~2 周）
  ├── M1.1  导出三个 ONNX 模型（YOLO / DINOv2 / SVM）并完成单模型验证
  │          - DINOv2: 验证导出无报错，ORT 可加载，dummy input 输出 shape (1,384)
  │          - SVM: skl2onnx 导出，val 395 张 100% 一致（否则回退 libsvm）
  ├── M1.2  实现 C++ 预处理（OpenCV）+ 运行对齐验证脚本（门控：cosine≥0.9995）
  ├── M1.3  串联 ONNX Runtime 三阶段（无流水线，顺序执行）
  └── M1.4  TDUS test 准确率验证（目标 ≥93.8%）

M2（TensorRT fp16，~2 周）
  ├── M2.1  YOLO TRT 引擎构建 + 单独验证（bbox L∞ < 2px）
  ├── M2.2  DINOv2 TRT 引擎构建
  │          - **先检查 FMHA 融合状态（trtexec verbose grep mha）**
  │          - 若融合失败：执行缓解方案（降级 PyTorch / per-layer fp32）
  │          - embedding cosine similarity 验证（均值≥0.999，最小≥0.99）
  └── M2.3  TRT 串联 + 吞吐率测试（保证目标 ≥30 img/s，冲刺 ≥43 img/s）

M3（三线程双缓冲流水线，~1 周）
  ├── M3.1  I/O Thread（libjpeg-turbo + OpenCV）+ GPU Thread + CPU Thread
  ├── M3.2  CUDA stream 重叠（H2D 异步 + GPU 计算并行）
  ├── M3.3  （可选）CUDA Graph capture for DINOv2 ORT session
  │          注：plan 中 M3.3 与 M3.2 合并为 Stage 17-B，属于合理简化（CUDA Graph 与 stream 重叠高度耦合）
  └── M3.4  吞吐率最终测试 + 峰值显存验证
             保证目标：≥35 img/s（相对 M2 顺序执行 30 img/s 提升 ≥15%，体现双缓冲收益）
             冲刺目标：≥43 img/s（FMHA 融合成功前提下）

M4（C API + 部署打包，~1 周）
  ├── M4.1  封装 `urban_tree_id.h` C API，暴露共享库（`.so`）
  ├── M4.2  ASAN / Valgrind 内存检查
  ├── M4.3  Docker 镜像打包（含 engine build 脚本）
             注：plan 中新增 TRT 引擎版本 header 检查逻辑的端到端验证（Stage 18-C 验收标准第6条）
  └── M4.4  多线程（单 handle 顺序）压测 + 最终文档
```

---

## 参考

- 当前批量推理主路径：`scripts/benchmark_pipeline.py` `_process_batch`（第 171–184 行）调用 `make_crops_gpu_batch` + `dinov2.forward_features` + `predict_species_batch`
- GPU crop 实现：`scripts/predict_pipeline.py` `make_crops_gpu_batch`（第 306–358 行）
- SVM 批量推理：`scripts/predict_pipeline.py` `predict_species_batch`（第 361–373 行）
- 双缓冲检测接口：`scripts/detection/base_detector.py` `preprocess_cpu` / `forward_preprocessed`（第 35–44 行）
- YOLO 不走双缓冲：`scripts/detection/yolo_detector.py`（`supports_pipeline` 未重写，继承 `False`）
- 准确率与吞吐率基准：`gdino_dinov2_svm/RESULTS.md`（推荐配置 imgsz=960，batch=64，21.5 img/s timed）
- SVM 训练脚本：`gdino_dinov2_svm/04_train_validate.py`
- TRT FMHA 融合失败已知问题：[NVIDIA/TensorRT #4537](https://github.com/NVIDIA/TensorRT/issues/4537)（TRT 10.8.0.43 + PyTorch 2.7.1）
- 参考 C++ 实现：[YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP)、[YOLOs-CPP](https://github.com/Geekgineer/YOLOs-CPP)（YOLO ONNX Runtime C++，支持 YOLO5–YOLO12）
- ONNX Runtime TRT EP 文档：[onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- skl2onnx SVC 历史问题：[sklearn-onnx #478](https://github.com/onnx/sklearn-onnx/issues/478)、[onnxruntime #11284](https://github.com/microsoft/onnxruntime/issues/11284)
