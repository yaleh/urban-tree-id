# Plan: C/C++ 生产流水线

**对应 Proposal**：`docs/proposals/proposal-cpp-production-pipeline.md`  
**日期**：2026-06-04  
**状态**：草稿

---

## 依赖关系总览

```
Phase 15 (M0+M1) ──▶ Phase 16 (M2) ──▶ Phase 17 (M3) ──▶ Phase 18 (M4)
```

- **Phase 15（M0+M1）**：无依赖，首先执行。包含环境准备、依赖版本锁定、ONNX 模型导出、C++ 预处理对齐、以及 ONNX Runtime 全链路串联。**Phase 15 的数值对齐 CI gate 通过是后续所有 Phase 的硬性前置条件。**
- **Phase 16（M2）**：依赖 Phase 15 全部完成（三个 ONNX 模型已导出、C++ 预处理对齐 CI gate 已通过、ORT 全链路准确率 ≥93.8%）。
- **Phase 17（M3）**：依赖 Phase 16 完成（TRT fp16 引擎构建完毕、吞吐率保证目标 ≥30 img/s 已达成）。
- **Phase 18（M4）**：依赖 Phase 17 完成（三线程流水线稳定运行、峰值显存验证通过）。

---

## Phase 列表概览

| Phase | 里程碑 | 主要交付物 | 关键风险 |
|-------|-------|-----------|---------|
| 15 | M0 + M1 | 依赖版本矩阵锁定；三个 ONNX 模型文件；C++ ORT 串联 pipeline；准确率 ≥93.8% | YOLO letterbox 方案选择；DINOv2 导出障碍；Stage 15-C 已拆分为 15-C1（预处理实现）和 15-C2（数值对齐 CI gate）|
| 16 | M2 | YOLO + DINOv2 TRT fp16 引擎；吞吐率 ≥30 img/s（保证）| TRT FMHA 融合失败（已知高风险）；fp16 数值溢出 |
| 17 | M3 | 三线程双缓冲 pipeline；CUDA stream 重叠；最终吞吐率测试 | 线程间 race condition；CUDA Graph 固定 batch 约束 |
| 18 | M4 | C API `.so`；Docker 镜像；ASAN/Valgrind 验证；最终文档 | 内存泄漏；TRT 引擎跨机不可移植 |

---

## Phase 15 — M0+M1：环境准备 + ONNX Runtime 全链路

### 目标

锁定构建环境依赖版本；导出三个 ONNX 模型（YOLO / DINOv2 / SVM）；实现 C++ 预处理（OpenCV 白边 pad 路径）；通过数值对齐 CI gate；串联 ONNX Runtime 三阶段推理；在 TDUS test split（386 张）上验证端到端准确率 ≥93.8%。

**Phase 15 估算行数**：Stage 15-A ≈80 行 + Stage 15-B ≈150 行 + Stage 15-C1 ≈180 行 + Stage 15-C2 ≈140 行 + Stage 15-D ≈120 行，合计 ≈670 行。注意：原 Stage 15-C（≈220 行）超出 200 行/Stage 上限，已拆分为 15-C1 和 15-C2；Phase 15 整体行数因此超出 500 行/Phase 上限，但属于拆分后的合理分布（每个 Stage 均 ≤200 行），Phase 级别的行数限制可在获得作者确认后按实际拆分结果调整。

---

### Stage 15-A：环境准备与决策锁定（M0）

**依赖**：无

**估算行数**：新增脚本约 60 行（导出脚本、验证脚本骨架），文档约 20 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/CMakeLists.txt` | 新建 | 最小 CMake 配置：ONNX Runtime 路径、OpenCV 依赖、CUDA 工具链；锁定 C++17 标准 |
| `cpp/scripts/export_onnx.py` | 新建 | 导出三个 ONNX 模型的一体化脚本：(1) YOLO `model.export(format="onnx", imgsz=960, opset=17, simplify=True, dynamic=False)`；(2) DINOv2 `DINOv2Wrapper` + `torch.onnx.export(opset=17)`；(3) skl2onnx SVM 导出 `target_opset=15`；脚本顶部以注释记录 letterbox 方案选择（方案 A：ONNX 内嵌 letterbox，C++ 端仅做 CHW float32 转换） |
| `cpp/scripts/validate_alignment.py` | 新建 | CI gate 验证脚本骨架（含 `validate_embedding_alignment` 函数，判据：cosine 最小值 ≥0.9995，L∞ < 0.01）；SVM 一致性判据：val 395 张 100% 一致 |
| `docs/decisions/adr-001-yolo-letterbox.md` | 新建 | 架构决策记录：明确选择方案 A（ONNX 内嵌 letterbox），理由：避免 C++ 手动 letterbox 引入坐标还原错误，C++ 端输入为原始 RGB uint8 HWC 图像 |
| `Dockerfile.build` | 新建 | 固定版本：`nvcr.io/nvidia/tensorrt:24.08-py3`（TRT 10.x，CUDA 12.4，cuDNN 9.x）；安装 `onnxruntime-gpu==1.19.*`、`skl2onnx>=1.17`、`scikit-learn`（与模型训练环境一致）；导出环境使用 PyTorch 2.4–2.6（避免 2.7 + TRT 10.8 FMHA 融合问题） |

**TDD 顺序**：先写 `validate_alignment.py` 的单元测试（用随机向量验证函数逻辑）→ 实现判据函数 → 再写 `export_onnx.py`。

#### 验收标准

1. `Dockerfile.build` 可成功构建镜像，`nvidia-smi` 和 `python -c "import tensorrt; print(tensorrt.__version__)"` 输出版本匹配依赖矩阵
2. `docs/decisions/adr-001-yolo-letterbox.md` 存在，明确记录方案 A/B 选择及理由
3. `cpp/scripts/export_onnx.py` 可在 Docker 环境中无报错运行（无需已有模型权重，仅验证脚本语法和 import）
4. `validate_alignment.py` 的函数单元测试通过（随机向量，cosine=1.0 时 `pass_cosine=True`，L∞=0.0 时 `pass_l_inf=True`）
5. `CMakeLists.txt` 中 C++ 标准锁定为 17，ONNX Runtime 和 OpenCV 依赖路径有注释说明

---

### Stage 15-B：导出三个 ONNX 模型并完成单模型验证（M1.1）

**依赖**：Stage 15-A 完成（Docker 环境、export_onnx.py、validate_alignment.py 骨架）

**估算行数**：`export_onnx.py` 补全约 100 行；验证脚本 `verify_onnx_models.py` 约 50 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/scripts/export_onnx.py` | 修改 | 补全三段导出逻辑（YOLO / DINOv2 / SVM）；DINOv2 封装 `DINOv2Wrapper`（仅输出 `x_norm_clstoken`，避免 1024 个 patch token 占用带宽）；SVM 导出后立即用 ORT 在 val 395 张上验证 100% 一致性（不一致则抛出异常，提示改用 libsvm C API 方案）；导出失败时打印清晰的错误信息（含 issue 链接） |
| `cpp/scripts/verify_onnx_models.py` | 新建 | 单模型冒烟验证：YOLO 输入 `(1,3,960,960)` float32，检查输出 shape `(1,5,8400)`；DINOv2 输入 `(1,3,448,448)` float32，检查输出 shape `(1,384)`；SVM 输入 `(1,384)` float32，检查输出为 22 类 label；三个模型均在 CPU（`CPUExecutionProvider`）和 GPU（`CUDAExecutionProvider`）两个 Provider 下验证可加载 |
| `cpp/models/.gitkeep` | 新建 | 占位，ONNX 文件本身不提交 git（体积过大）；README 说明从 `export_onnx.py` 生成 |

**TDD 顺序**：先写 `verify_onnx_models.py` 的 mock 测试（patch ORT session，验证 shape 检查逻辑）→ 在真实环境运行 `export_onnx.py` → 运行 `verify_onnx_models.py`。

#### 验收标准

1. `cpp/models/` 目录下存在 `yolo_960.onnx`、`dinov2_vits14_448.onnx`、`svm_model.onnx` 三个文件（CI 环境预置，非 git 追踪）
2. `verify_onnx_models.py` 对三个模型均通过 shape 检查（CPU + GPU Provider）
3. SVM 导出验证：`python export_onnx.py --verify-svm` 在 val 395 张上与 sklearn 预测结果 100% 一致（零差异，否则脚本以非零退出码失败）
4. DINOv2 输出为 `(1, 384)` float32，不含 patch tokens
5. `python -c "import onnxruntime as ort; sess = ort.InferenceSession('cpp/models/dinov2_vits14_448.onnx', providers=['CUDAExecutionProvider']); print('OK')"` 无报错

---

### Stage 15-C1：实现 C++ 预处理核心逻辑（M1.2a）

**依赖**：Stage 15-B 完成（DINOv2 ONNX 可用）

**估算行数**：`preprocess.cpp` + `preprocess.h` 约 120 行；`test_preprocess_unit.cpp` 约 60 行，合计 ≈180 行（低于 200 行/Stage 上限）

**关键说明**：C++ 预处理必须复现 `make_crops_gpu_batch`（白边 pad 路径），**不得**复现 `_make_transforms`（CenterCrop 路径）。两者 embedding cosine similarity 约 0.97–0.98，低于 CI gate 要求的 0.9995，若错误使用 CenterCrop 路径可被 CI gate 检出。

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/src/preprocess.h` | 新建 | 声明 `struct CropConfig { int crop_size=448; float pad_frac=0.05f; };`；声明 `cv::Mat make_crop_white_pad(const cv::Mat& img_bgr, const cv::Rect2d& bbox_xyxy, const CropConfig& cfg);`；声明 `cv::Mat normalize_imagenet(const cv::Mat& crop_rgb_float);` |
| `cpp/src/preprocess.cpp` | 新建 | 实现 `make_crop_white_pad`：(1) bbox 扩展 5% 并 clamp 到图像边界（`int` 截断，与 Python `int(x0)` 一致）；(2) OpenCV `INTER_LINEAR` resize（长边到 crop_size，等比缩放）；(3) 白色画布 `value=1.0f`，padding 向右/下取整（`pw // 2` 在左/上）；实现 `normalize_imagenet`：mean `[0.485, 0.456, 0.406]`，std `[0.229, 0.224, 0.225]`，先 pad 填 1.0f 再整体归一化（含 pad 区域）；JPEG 解码使用 OpenCV `imread` + `cvtColor(BGR→RGB)` |
| `cpp/tests/test_preprocess_unit.cpp` | 新建 | C++ 单元测试（Google Test）：白色画布填充正确（padding 区域值为 `(1.0-mean)/std`）；bbox clamp 逻辑（超出边界时截断）；奇数 padding 时右/下多 1px；`normalize_imagenet` 数值正确（手算单像素）；约 60 行（4 个 TEST_F，每个约 15 行） |

**TDD 顺序**：先写 `test_preprocess_unit.cpp`（G-Test，此时无实现）→ 实现 `preprocess.cpp` → 单元测试通过。

#### 验收标准

1. `test_preprocess_unit.cpp` 全部通过（G-Test）
2. C++ 不存在任何 CenterCrop 相关代码（`git grep -i "centercrop\|center.crop"` 在 `cpp/` 下零返回）
3. `make_crop_white_pad` 中 padding 分配与 Python `F.pad` 语义一致：`pw // 2` 在左/上，`pw - pw // 2` 在右/下
4. `normalize_imagenet` 对手算单像素值的误差 < 1e-6（float32 精度，G-Test 断言）

---

### Stage 15-C2：数值对齐 CI gate 与 YOLO bbox 验证（M1.2b）

**依赖**：Stage 15-C1 完成（`preprocess.cpp` 已实现且 G-Test 通过）

**估算行数**：`test_preprocess_alignment.py` 约 80 行；CI gate shell 脚本约 20 行；YOLO bbox 验证脚本约 40 行，合计 ≈140 行（低于 200 行/Stage 上限）

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/tests/test_preprocess_alignment.py` | 新建 | Python 参考实现 vs C++ 输出对齐测试：从 TDUS test split 取 100 张图像，分别用 Python `make_crops_gpu_batch`（GPU 路径）和 C++ `make_crop_white_pad` 处理，比较 DINOv2 embedding；判据：cosine 最小值 ≥0.9995，L∞（归一化后）< 0.01；**此测试为 Phase 15 的核心 CI gate，Phase 16 启动前必须通过** |
| `cpp/tests/test_yolo_bbox_alignment.py` | 新建 | YOLO bbox 坐标对齐验证：从 TDUS test split 取 50 张图像，对比 Python `detect_yolo_batch`（ORT 路径）与 C++ ONNX Runtime YOLO 推理的 bbox 坐标；判据：L∞ < 2px（保证目标），L∞ < 1px（冲刺目标）；同时验证 ONNX input tensor name 为 `images`（与 Ultralytics 导出默认值一致） |
| `.github/workflows/alignment_gate.yml` | 新建 | CI gate：运行 `test_preprocess_alignment.py`，若 cosine min < 0.9995 或 L∞ max ≥ 0.01 则 fail，阻断 Phase 16 merge |

**TDD 顺序**：先在 Stage 15-C1 G-Test 通过后，运行 `test_preprocess_alignment.py`（对比 Python GPU 路径）→ CI gate 通过 → 运行 `test_yolo_bbox_alignment.py`。

#### 验收标准

1. **CI gate（必须通过）**：`test_preprocess_alignment.py` 在 100 张 TDUS 图像上报告 cosine 最小值 ≥0.9995，L∞ 最大值 < 0.01（归一化后）
2. YOLO bbox 坐标（C++ 解析 ONNX 输出 vs Python `detect_yolo_batch`）：取 50 张测试图像，L∞ < 2px（绝对像素差）
3. `test_yolo_bbox_alignment.py` 验证 ONNX input tensor name 为 `images`，与 `export_onnx.py` 导出配置一致（不一致则脚本以非零退出码失败）
4. `.github/workflows/alignment_gate.yml` 在 PR 时自动触发 `test_preprocess_alignment.py`，失败时阻断合并

---

### Stage 15-D：串联 ORT 三阶段并验证端到端准确率（M1.3 + M1.4）

**依赖**：Stage 15-C2 完成（数值对齐 CI gate 通过、YOLO bbox 对齐验证通过）

**估算行数**：`pipeline_ort.cpp` + `pipeline_ort.h` 约 150 行；`benchmark_cpp.py`（调用 C++ benchmark 可执行文件并解析输出）约 50 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/src/pipeline_ort.h` | 新建 | 声明 `class PipelineORT`：构造函数接受三个 ONNX 路径、`batch_size=64`、`crop_size=448`、`device_id=0`；`std::vector<OrtResult> RunBatch(const std::vector<std::string>& image_paths)`；析构函数释放 ORT session |
| `cpp/src/pipeline_ort.cpp` | 新建 | 实现三阶段顺序执行（无流水线）：(1) JPEG decode + `make_crop_white_pad`；(2) DINOv2 ORT 推理（`CUDAExecutionProvider`）；(3) SVM ORT 推理（`CPUExecutionProvider`）；YOLO 检测 + NMS（`cv::dnn::NMSBoxes`，`conf_threshold=0.25`，`iou_threshold=0.45`）；坐标还原：ONNX 内嵌 letterbox 输出坐标已在 960×960 空间，用 scale + pad offset 还原到原始图像像素坐标（参考 YOLOv8-TensorRT-CPP `scale_boxes` 实现） |
| `cpp/tools/run_benchmark.cpp` | 新建 | CLI 工具：`--test-dir`、`--ann-dir`、`--batch-size`、`--crop-size`；输出 JSON 格式结果（与 Python `benchmark_pipeline.py` 相同字段）；计时分离（timed-bench 模式：预加载图像，排除磁盘 I/O） |
| `cpp/tests/test_pipeline_ort_accuracy.py` | 新建 | 端到端准确率验证：运行 `run_benchmark`（C++），对比 TDUS test 386 张标注，计算 top-1 准确率；判据：≥93.8%；同时验证漏检数（`n_boxes=0`）为 0 |

**TDD 顺序**：先写 `test_pipeline_ort_accuracy.py` 框架（mock 输出格式验证）→ 实现 `pipeline_ort.cpp` → 编译 `run_benchmark` → 在真实数据上运行准确率测试。

#### 验收标准

1. `run_benchmark --test-dir ... --ann-dir ... --batch-size 64 --crop-size 448` 在 TDUS test 386 张上输出 top-1 准确率 ≥93.8%（目标 ≥94.5%）
2. 漏检数为 0（无 `n_boxes=0` 的有效测试图像）
3. `test_pipeline_ort_accuracy.py` 自动解析 JSON 输出并断言准确率，失败时打印误分类样本列表
4. ORT 全链路无 GPU OOM（`NVML nvmlDeviceGetMemoryInfo` 记录峰值 ≤8 GB，batch=64）
5. `run_benchmark --timed` 模式（预加载图像）输出吞吐率（ORT 不要求 ≥30 img/s，仅记录基线，用于 Phase 16 对比）

### Phase 15 完成标准

- 所有测试通过（C++ G-Test + Python 对齐测试 + 准确率测试）
- **数值对齐 CI gate 通过**（Stage 15-C2）：cosine 最小值 ≥0.9995，L∞ < 0.01（100 张 TDUS 图像）
- SVM ORT 推理与 sklearn 结果类别预测 100% 完全一致（val 395 张零差异，零差异定义为逐样本预测类别字符串完全相同）
- TDUS test top-1 准确率 ≥93.8%
- YOLO bbox L∞ < 2px（50 张测试图像）；ONNX input tensor name `images` 已验证
- Docker 镜像可成功构建且三个 ONNX 模型可导出
- Phase 15 完成是 Phase 16 启动的硬性前置条件

---

## Phase 16 — M2：TensorRT fp16 迁移

### 目标

将 YOLO 和 DINOv2 从 ONNX Runtime 迁移至 TensorRT fp16 引擎，验证数值对齐，实现吞吐率保证目标 ≥30 img/s（FMHA 融合成功冲刺目标 ≥43 img/s）。SVM 保持 ORT CPU 推理。

**Phase 16 启动前置条件**：Phase 15 数值对齐 CI gate 已通过

**Phase 16 估算行数**：Stage 16-A ≈130 行 + Stage 16-B ≈180 行 + Stage 16-C ≈60 行，合计 ≈370 行（低于 500 行上限）

---

### Stage 16-A：YOLO TRT 引擎构建与验证（M2.1）

**依赖**：Phase 15 全部完成（`yolo_960.onnx` 已导出，坐标还原逻辑已实现）

**估算行数**：`yolo_trt.cpp` + `yolo_trt.h` 约 130 行；验证脚本约 40 行

**关键风险**：TRT 引擎与 GPU 型号强绑定，在 CI 环境（sm_86）和部署目标（可能不同 sm）上必须分别构建。

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/scripts/build_trt_engines.sh` | 新建 | 封装 `trtexec` 调用：`trtexec --onnx=yolo_960.onnx --fp16 --saveEngine=yolo_960.engine --minShapes=images:1x3x960x960 --optShapes=images:64x3x960x960 --maxShapes=images:64x3x960x960`；**input tensor name `images` 已由 Stage 15-C2 `test_yolo_bbox_alignment.py` 验证与 ONNX 导出一致**；YOLO 设置固定 batch=64（方案 A 内嵌 letterbox，fixed shape 更高效）；日志输出含 GPU sm 版本，便于诊断跨机不可移植问题 |
| `cpp/src/yolo_trt.h` | 新建 | 声明 `class YoloTRT`：`YoloTRT(const std::string& engine_path, int device_id=0)`；`std::vector<std::vector<cv::Rect2d>> DetectBatch(const std::vector<cv::Mat>& imgs_rgb, float conf_thr=0.25f, float iou_thr=0.45f)` |
| `cpp/src/yolo_trt.cpp` | 新建 | 实现 TRT 引擎加载（`nvinfer1::IRuntime`）；批量 H2D 传输（方案 A：CHW float32 ÷255，无需手动 letterbox）；TRT forward pass；NMS（`cv::dnn::NMSBoxes`）；坐标还原到原始图像空间（复用 Stage 15-D 的 `scale_boxes` 逻辑）；NVML 记录峰值显存 |
| `cpp/tests/test_yolo_trt_alignment.py` | 新建 | 对比 Python `detect_yolo_batch` vs C++ `YoloTRT::DetectBatch`：取 50 张 TDUS 测试图像，比较每张图像的 bbox 坐标；判据：L∞ < 2px（保证目标），L∞ < 1px（冲刺目标）；若某张图像框数不一致（NMS 结果差异）则记录 warning 并跳过坐标比对 |

**TDD 顺序**：先写 `test_yolo_trt_alignment.py` 框架 → 运行 `build_trt_engines.sh` 构建 YOLO 引擎 → 实现 `yolo_trt.cpp` → 运行对齐测试。

#### 验收标准

1. `yolo_960.engine` 文件在目标 GPU 上成功构建（无 `trtexec` 报错）
2. `test_yolo_trt_alignment.py`：50 张图像 bbox 坐标 L∞ < 2px（保证目标）
3. YOLO TRT fp16 单批次（batch=64）推理时间 ≤15ms（含 H2D + forward + NMS，不含 JPEG decode）
4. 峰值显存 YOLO 单独运行时 ≤4 GB（batch=64，imgsz=960）
5. `build_trt_engines.sh` 日志包含 GPU sm 版本信息（便于跨机排查）

---

### Stage 16-B：DINOv2 TRT 引擎构建与 fp16 数值验证（M2.2）

**依赖**：Stage 16-A 完成（YOLO TRT 引擎构建流程已验证）

**估算行数**：`dinov2_trt.cpp` + `dinov2_trt.h` 约 130 行；FMHA 验证脚本约 30 行；fp16 对齐验证脚本约 40 行

**关键风险（[CRITICAL-1]）**：TRT 10.8.0.43 + PyTorch 2.7 ONNX 导出路径下，DINOv2 ViT 的 FMHA 融合已知存在失败问题。本 Stage 必须在引擎构建后立即验证 FMHA 融合状态，若失败则执行缓解方案，再进行 fp16 数值验证。

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/scripts/build_trt_engines.sh` | 修改 | 新增 DINOv2 引擎构建段：`trtexec --onnx=dinov2_vits14_448.onnx --fp16 --verbose --saveEngine=dinov2_448.engine --minShapes=pixel_values:1x3x448x448 --optShapes=pixel_values:64x3x448x448 --maxShapes=pixel_values:64x3x448x448`；`--verbose` 日志 tee 到 `dinov2_trt_build.log` |
| `cpp/scripts/check_fmha_fusion.sh` | 新建 | 检查 FMHA 融合状态：`grep -i "fmha\|attention\|mha" dinov2_trt_build.log`；若发现 `mha` 或 `fused_mha_v2` 字样则输出 "FMHA 融合成功，冲刺目标 ≥43 img/s 可达"；否则输出 "FMHA 融合失败，执行缓解方案，保证目标 ≥30 img/s"，并提示降级 PyTorch 到 2.4–2.6 重新导出或使用 per-layer fp32；**此脚本输出结果必须记录到 `docs/decisions/adr-002-fmha-status.md`** |
| `docs/decisions/adr-002-fmha-status.md` | 新建 | 记录 FMHA 融合验证结果、采用的缓解方案（若需要）、对吞吐率目标的影响；由 `check_fmha_fusion.sh` 输出内容填写 |
| `cpp/src/dinov2_trt.h` | 新建 | 声明 `class DINOv2TRT`：`DINOv2TRT(const std::string& engine_path, int device_id=0)`；`std::vector<std::vector<float>> EmbedBatch(const std::vector<cv::Mat>& crops_normalized_chw_fp32)` 返回 `(N, 384)` embedding |
| `cpp/src/dinov2_trt.cpp` | 新建 | 实现 TRT 引擎加载；批量 H2D；fp16 forward；D2H（输出转回 fp32）；若需 per-layer fp32（FMHA 失败缓解）则在 C++ engine builder API 中对 Softmax/LayerNorm 层调用 `layer->setPrecision(kFLOAT)` |
| `cpp/tests/test_dinov2_trt_alignment.py` | 新建 | **fp16 数值对齐 CI gate**：对比 ORT fp32（Phase 15 基准）vs TRT fp16：取 TDUS test 386 张图像，比较 embedding；判据：cosine 均值 ≥0.999，最小值 ≥0.99（若最小值 <0.99 则 fp16 数值溢出，触发 per-layer fp32 缓解）；输出 cosine 分布直方图（文本格式，便于 CI 日志分析） |

**TDD 顺序**：先写 `test_dinov2_trt_alignment.py`（结构验证，mock ORT 输出）→ 构建 DINOv2 TRT 引擎 → 运行 `check_fmha_fusion.sh` 并记录结果 → 若 FMHA 失败则执行缓解 → 实现 `dinov2_trt.cpp` → 运行 fp16 对齐测试。

#### 验收标准

1. `adr-002-fmha-status.md` 存在，记录 FMHA 融合状态和采用的方案
2. **fp16 数值对齐 CI gate**：`test_dinov2_trt_alignment.py` 报告 cosine 均值 ≥0.999，最小值 ≥0.99（386 张 TDUS test）
3. `DINOv2TRT::EmbedBatch` 在 batch=64 下无 NaN 输出（任意一个 embedding 含 NaN 则失败）
4. DINOv2 TRT fp16 单批次（batch=64）推理时间记录在 `adr-002-fmha-status.md`（FMHA 成功预期 ≤8ms，失败预期 ≤16ms）
5. 若采用 per-layer fp32 缓解，`dinov2_trt.cpp` 中 Softmax/LayerNorm 层精度设置有代码注释说明原因

---

### Stage 16-C：TRT 串联 + 吞吐率测试（M2.3）

**依赖**：Stage 16-A 和 Stage 16-B 均完成

**估算行数**：`pipeline_trt.cpp` 修改约 60 行（基于 `pipeline_ort.cpp` 替换 YOLO 和 DINOv2 为 TRT）

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/src/pipeline_trt.h` | 新建 | 声明 `class PipelineTRT`（接口与 `PipelineORT` 相同，便于 Phase 17 替换）：构造函数接受 YOLO engine 路径、DINOv2 engine 路径、SVM ONNX 路径；`std::vector<OrtResult> RunBatch(...)` |
| `cpp/src/pipeline_trt.cpp` | 新建 | 顺序执行（无流水线）：JPEG decode → `make_crop_white_pad` → `YoloTRT::DetectBatch` → `DINOv2TRT::EmbedBatch` → SVM ORT 推理；复用 Phase 15 的 `normalize_imagenet` 和 NMS 逻辑 |
| `cpp/tests/test_pipeline_trt_throughput.py` | 新建 | 吞吐率测试：预加载 386 张 TDUS test 图像到内存，运行 10 轮，取稳定吞吐率；判据：保证目标 ≥30 img/s，冲刺目标 ≥43 img/s（若 FMHA 融合成功）；同时验证端到端准确率 ≥93.8% |

**TDD 顺序**：先运行 `test_pipeline_trt_throughput.py` 框架（mock 计时，验证结构）→ 实现 `pipeline_trt.cpp` → 运行真实吞吐率测试。

#### 验收标准

1. 吞吐率保证目标：`test_pipeline_trt_throughput.py` 报告 ≥30 img/s（预加载图像，排除磁盘 I/O，batch=64）
2. 端到端准确率 ≥93.8%（TDUS test 386 张）
3. YOLO + DINOv2 + SVM 同时运行时峰值显存 ≤8 GB（`NVML nvmlDeviceGetMemoryInfo` 记录最大值）
4. 若 FMHA 融合成功：吞吐率 ≥43 img/s（冲刺目标，不强制）；若失败：记录实测值与 30 img/s 保证目标的差距

### Phase 16 完成标准

- 所有测试通过
- YOLO TRT fp16 引擎和 DINOv2 TRT fp16 引擎均已构建
- **fp16 数值对齐 CI gate 通过**：cosine 均值 ≥0.999，最小值 ≥0.99
- 吞吐率（顺序执行，无流水线）≥30 img/s（保证目标）
- `adr-002-fmha-status.md` 记录 FMHA 融合状态
- Phase 16 完成是 Phase 17 启动的硬性前置条件

---

## Phase 17 — M3：三线程双缓冲流水线

### 目标

将 Phase 16 的顺序执行改造为三线程双缓冲架构（I/O Thread → GPU Thread → CPU Thread），利用 CUDA stream 重叠减少 pipeline bubble，实现最终吞吐率测试。

**Phase 17 启动前置条件**：Phase 16 完成（TRT 串联顺序执行吞吐率 ≥30 img/s）

**Phase 17 估算行数**：Stage 17-A ≈150 行 + Stage 17-B ≈80 行 + Stage 17-C ≈60 行，合计 ≈290 行（低于 500 行上限）

**注意**：Python 端 YOLO 不走双缓冲（`supports_pipeline=False`）。C++ 端本 Phase 选择方案 (b)：手动实现 YOLO 的 I/O 与 GPU 推理双缓冲（I/O Thread 负责 JPEG decode，GPU Thread 负责 letterbox float 转换 + YOLO TRT forward）。此为有意的 C++ 行为增强，超出 Python 参考实现范围，需在 ADR 中记录。

---

### Stage 17-A：三线程框架与 I/O 线程（M3.1）

**依赖**：Phase 16 全部完成

**估算行数**：`pipeline_threaded.h` + `pipeline_threaded.cpp` 约 150 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/src/pipeline_threaded.h` | 新建 | 声明 `class PipelineThreaded`；内部持有三个线程（`std::thread`）和两个有界 `std::queue`（`queue_a`: I/O→GPU，`queue_b`: GPU→CPU）；`queue_a` 容量限制为 4 个 batch（背压控制），`queue_b` 容量限制为 4 个 batch；声明 `StartPipeline()`、`SubmitPaths(const std::vector<std::string>& paths)`、`std::vector<OrtResult> WaitResults(int n)` |
| `cpp/src/pipeline_threaded.cpp` | 新建 | 实现 I/O Thread：从 `image_paths` 队列取路径批次，`cv::imread` + `cvtColor(BGR→RGB)`，将原始 RGB uint8 batch push 到 `queue_a`；GPU Thread：从 `queue_a` pop，执行 YOLO TRT（H2D + forward + NMS）→ crop + `normalize_imagenet` → DINOv2 TRT（H2D + forward + D2H），将 `(N, 384)` embedding push 到 `queue_b`；CPU Thread：从 `queue_b` pop，执行 SVM ORT 推理，将 `OrtResult` append 到结果列表；三个线程均使用 `std::condition_variable` 协调；析构函数发送 sentinel 值（空 batch）通知线程退出，`std::thread::join` 确保无悬空线程 |
| `docs/decisions/adr-003-yolo-double-buffer.md` | 新建 | 记录 C++ 端选择为 YOLO 实现双缓冲的决策（不同于 Python 端 `supports_pipeline=False`），以及 I/O Thread 处理 JPEG decode 的分工说明 |
| `cpp/tests/test_pipeline_threaded_basic.cpp` | 新建 | G-Test：mock I/O / GPU / CPU 三个函数（返回固定数据），验证三线程框架的数据流正确性（输入 N 个路径，输出 N 个结果，顺序不保证但数量匹配）；验证析构函数可安全调用（无 deadlock，`valgrind --tool=helgrind` 无 data race 报告） |

**TDD 顺序**：先写 `test_pipeline_threaded_basic.cpp`（mock 三个处理函数）→ 实现 `pipeline_threaded.cpp` 框架（线程管理 + 队列）→ 单元测试通过 → 接入真实 TRT 推理。

#### 验收标准

1. `test_pipeline_threaded_basic.cpp` 全部通过（G-Test，mock 处理函数）
2. `valgrind --tool=helgrind` 运行 1000 图像顺序推理无 data race 报告
3. 三线程架构在 100 图像连续推理后可正常析构（无 deadlock，超时 30s 则认为 deadlock）
4. `queue_a` 和 `queue_b` 背压控制生效：I/O Thread 在 GPU 积压时自动阻塞（通过 `queue_a` 满时 push 阻塞验证）

---

### Stage 17-B：CUDA Stream 重叠（M3.2 + M3.3）

**依赖**：Stage 17-A 完成（三线程框架稳定运行）

**估算行数**：CUDA stream 相关修改约 50 行；CUDA Graph（可选）约 40 行，共约 80 行

**可选项**：CUDA Graph capture for DINOv2 ORT session（`OrtCUDAProviderOptions.enable_cuda_graph=1`）需要固定 input/output tensor 地址（IOBinding）。仅在 batch size 固定为 64 时可用。

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/src/pipeline_threaded.cpp` | 修改 | GPU Thread 内部使用两个 CUDA stream（`stream_0`、`stream_1`）交替处理相邻 batch：当 `stream_0` 在 DINOv2 TRT forward 时，`stream_1` 同步进行下一 batch 的 YOLO H2D + YOLO TRT forward；使用 `cudaStreamWaitEvent` 协调依赖；H2D 传输使用 `cudaMemcpyAsync` + pinned memory（`cudaMallocHost`） |
| `cpp/src/dinov2_trt.cpp` | 修改（可选）| 若 CUDA Graph 可用（batch=64 固定）：在首次推理时执行 `cudaGraphBeginCapture` → TRT forward → `cudaGraphEndCapture`，后续推理使用 `cudaGraphLaunch`；若 batch size 不固定则跳过此优化 |
| `cpp/tests/test_cuda_stream_overlap.py` | 新建 | 验证 stream 重叠有效性：连续推理 10 轮（each 64 张），计算实际 GPU 利用率（`nvidia-smi dmon -s u` 采样）；预期 GPU 利用率 ≥85%（vs 顺序执行约 60–70%）；记录到 `docs/benchmarks/phase17_throughput.md` |

**TDD 顺序**：先验证顺序执行基准（单 stream GPU 利用率）→ 实现双 stream 重叠 → 对比 GPU 利用率提升。

#### 验收标准

1. 双 CUDA stream 重叠后 GPU 利用率 ≥85%（`nvidia-smi dmon` 采样，10 轮平均）
2. H2D 使用 pinned memory（`cudaMallocHost`），无 `cudaMalloc` 分配的可分页内存
3. 若启用 CUDA Graph：首次推理（graph capture）耗时可忽略（< 5s），后续每次推理时间与不启用 CUDA Graph 相差 ≤10%（用于验证 CUDA Graph 无误捕获）
4. 双 stream 模式下无 CUDA 错误（`cudaGetLastError()` 在每次 kernel launch 后为 `cudaSuccess`）

---

### Stage 17-C：吞吐率最终测试与显存验证（M3.4）

**依赖**：Stage 17-B 完成

**估算行数**：测试脚本约 40 行；benchmark 报告模板约 20 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/tests/test_pipeline_threaded_throughput.py` | 新建 | 最终吞吐率测试：预加载 386 张 TDUS test 图像，运行三线程 pipeline 10 轮，取中位数吞吐率；**保证目标 ≥35 img/s**（三线程流水线相对 Phase 16 顺序执行 30 img/s 应有约 15–20% 提升，因 I/O 与 GPU 计算重叠消除 pipeline bubble），冲刺目标 ≥43 img/s（若 FMHA 融合成功）；同时验证端到端准确率 ≥93.8%；测试报告同时输出 Phase 16 顺序执行基线（复用 `test_pipeline_trt_throughput.py` 结果）用于对比 |
| `docs/benchmarks/phase17_throughput.md` | 新建 | 记录三线程 pipeline 吞吐率实测值、GPU 利用率、峰值显存，与 Phase 16 顺序执行基线对比；若三线程相对顺序执行提升 <10%，需分析 pipeline bubble 来源（I/O bound / GPU bound / queue contention） |

#### 验收标准

1. 三线程 pipeline 吞吐率（预加载图像，排除磁盘 I/O，batch=64）**≥35 img/s**（保证目标；此值高于 Phase 16 顺序执行的 30 img/s 保证目标，体现双缓冲流水线的实际收益；若无法达到 35 img/s 则需分析瓶颈并记录在 `phase17_throughput.md`）
2. 端到端准确率 ≥93.8%（TDUS test 386 张）
3. 峰值显存（YOLO + DINOv2 + SVM 同时驻留，batch=64 激活值）≤8 GB（保证目标），≤6 GB（冲刺目标）
4. `docs/benchmarks/phase17_throughput.md` 存在，含三线程 vs 顺序执行的对比数据，以及 GPU 利用率（`nvidia-smi dmon` 采样均值）

### Phase 17 完成标准

- 所有测试通过（G-Test + Python 吞吐率测试）
- 三线程流水线吞吐率 **≥35 img/s**（保证目标；高于 Phase 16 顺序执行基线 30 img/s，体现流水线化收益）
- 显存 ≤8 GB（batch=64）
- `valgrind --tool=helgrind` 无 data race 报告
- Phase 17 完成是 Phase 18 启动的硬性前置条件

---

## Phase 18 — M4：C API + 部署打包

### 目标

封装 `urban_tree_id.h` C API，导出共享库 `.so`，通过 ASAN/Valgrind 内存检查，构建 Docker 部署镜像（含 TRT engine build 脚本），完成最终文档。

**Phase 18 启动前置条件**：Phase 17 完成（三线程流水线稳定运行，吞吐率 ≥30 img/s）

**Phase 18 估算行数**：Stage 18-A ≈130 行 + Stage 18-B ≈50 行 + Stage 18-C ≈60 行 + Stage 18-D ≈30 行，合计 ≈270 行（低于 500 行上限）

---

### Stage 18-A：封装 C API 并导出共享库（M4.1）

**依赖**：Phase 17 全部完成

**估算行数**：`urban_tree_id.h` + `urban_tree_id.cpp` 约 130 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/include/urban_tree_id.h` | 新建 | 按 Proposal C API 设计：`UrbanTreeResult` struct（`image_path`、`species[64]`、`confidence`、`n_boxes`、`error_code`）；`urban_tree_id_create(...)`、`urban_tree_id_destroy(...)`、`urban_tree_id_infer_batch(...)`、`urban_tree_id_version()`；`extern "C"` 包裹；线程安全说明注释 |
| `cpp/src/urban_tree_id.cpp` | 新建 | 实现四个 C API 函数：`urban_tree_id_create` 内部构造 `PipelineThreaded`，失败时返回 `nullptr`；`urban_tree_id_destroy` 调用析构函数；`urban_tree_id_infer_batch` 将 `char**` 路径转为 `std::vector<std::string>`，调用 `PipelineThreaded::SubmitPaths` + `WaitResults`，将结果写入 `UrbanTreeResult` 数组；`urban_tree_id_version` 返回编译时嵌入的版本字符串（含 TRT/ORT 版本） |
| `cpp/CMakeLists.txt` | 修改 | 新增 `add_library(urban_tree_id SHARED ...)`；设置 `SOVERSION`；安装规则（`install(TARGETS urban_tree_id LIBRARY ...)`） |
| `cpp/tests/test_c_api.cpp` | 新建 | G-Test：通过 C API（`dlopen` + `dlsym` 或直接链接）测试：`urban_tree_id_create` 成功返回非空 handle；`urban_tree_id_infer_batch` 对 10 张 TDUS 图像返回有效 species（snake_case 格式）；`urban_tree_id_destroy` 后 handle 不可再使用（后续调用行为未定义，无需测试，但销毁本身不 crash）；`error_code=0` 表示成功；`urban_tree_id_version` 返回非空字符串 |

**TDD 顺序**：先写 `test_c_api.cpp`（G-Test，此时无 C API 实现）→ 实现 `urban_tree_id.cpp` → 编译共享库 → 运行 C API 测试。

#### 验收标准

1. `test_c_api.cpp` 全部通过（G-Test）
2. `liburbantreeid.so` 可被 `dlopen` 加载，`dlsym("urban_tree_id_create")` 返回非空
3. `urban_tree_id_infer_batch` 对 TDUS test 386 张图像返回准确率 ≥93.8%（通过 C API 调用，非直接调用 C++ 类）
4. `urban_tree_id_version()` 返回含 ORT 版本和 TRT 版本的字符串
5. `nm -D liburbantreeid.so | grep " T "` 仅暴露 `urban_tree_id_*` 四个符号（其余 C++ 符号不导出）

---

### Stage 18-B：ASAN/Valgrind 内存检查（M4.2）

**依赖**：Stage 18-A 完成（C API 可用）

**估算行数**：ASAN CMake 配置约 20 行；测试驱动脚本约 30 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/CMakeLists.txt` | 修改 | 新增 `ENABLE_ASAN` 选项：`-fsanitize=address -fno-omit-frame-pointer`；ASAN 构建与发布构建分离（`CMAKE_BUILD_TYPE=ASAN`） |
| `cpp/tests/test_memory_safety.sh` | 新建 | 运行 ASAN 构建的 `test_c_api` 二进制，1000 张图像顺序推理（复用 TDUS test 386 张循环 3 轮）；`ASAN_OPTIONS=detect_leaks=1` 启用泄漏检测；检查退出码和 ASAN 输出（无 `ERROR: AddressSanitizer` 字样则通过）；同时运行 `valgrind --leak-check=full --error-exitcode=1`（可选，Valgrind 较慢，CI 可跳过） |

#### 验收标准

1. ASAN 构建（`CMAKE_BUILD_TYPE=ASAN`）编译无额外 warning
2. `test_memory_safety.sh` 运行 1000 图像后 ASAN 无报告（`detect_leaks=1` 模式下无 heap 泄漏）
3. `urban_tree_id_create` 调用失败时（传入不存在的模型路径）返回 `nullptr`，不 crash，不泄漏
4. `urban_tree_id_destroy(nullptr)` 安全（无 crash，no-op）

---

### Stage 18-C：Docker 镜像打包（M4.3）

**依赖**：Stage 18-B 完成

**估算行数**：`Dockerfile.deploy` 约 40 行；`build_engines_and_serve.sh` 约 30 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `Dockerfile.deploy` | 新建 | 基于 `nvcr.io/nvidia/tensorrt:24.08-py3`；安装 `libonnxruntime-gpu 1.19.*`、OpenCV 4.9+；复制 ONNX 模型文件（`yolo_960.onnx`、`dinov2_vits14_448.onnx`、`svm_model.onnx`）和 `liburbantreeid.so`；**不复制** `.engine` 文件（跨机不可移植） |
| `cpp/scripts/build_engines_and_serve.sh` | 新建 | 启动时检查 `VOLUME/yolo_960.engine` 是否存在且 TRT 版本头匹配；若不存在或版本不匹配则调用 `build_trt_engines.sh` 重新构建（约 2–5 分钟）；`.engine` 写入持久化 volume（`/app/engines/`）；构建完成后启动推理服务（`./urban_tree_id_server` 或进入 shell，视集成场景而定）；提供 `--rebuild-engines` 标志强制重建 |
| `cpp/tests/test_docker_build.sh` | 新建 | 验证 Docker 镜像可成功构建（`docker build -f Dockerfile.deploy .`）；`docker run --gpus all` 运行 `urban_tree_id_version()` 测试（无需完整推理测试，仅验证镜像内库可加载） |

#### 验收标准

1. `docker build -f Dockerfile.deploy .` 成功（无报错，约 5–10 分钟）
2. `docker run --gpus all <image> urban_tree_id_version` 返回版本字符串
3. `build_engines_and_serve.sh` 在首次启动时完成 YOLO + DINOv2 引擎构建，写入 `/app/engines/`
4. `build_engines_and_serve.sh --rebuild-engines` 强制重建后 `.engine` 文件时间戳更新
5. Docker 镜像大小记录在 `docs/benchmarks/phase18_deploy.md`（目标 ≤6 GB，含 TRT + ORT + 模型）
6. **TRT 版本 header 检查逻辑验证（[MAJOR-5] 落地）**：`test_docker_build.sh` 包含一个场景测试：将一个旧版本（或模拟的不匹配版本头）`.engine` 文件放入 volume，启动容器后验证 `build_engines_and_serve.sh` 检测到版本不匹配并触发重新构建；重建后 `.engine` 文件的 TRT 版本头与当前容器内 TRT 版本匹配（通过 `trtexec --loadEngine=... --verbose 2>&1 | grep "TensorRT version"` 验证）

---

### Stage 18-D：压测与最终文档（M4.4）

**依赖**：Stage 18-C 完成

**估算行数**：压测脚本约 20 行；文档更新约 10 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `cpp/tests/test_stress.sh` | 新建 | 单 handle 顺序调用压测：`urban_tree_id_infer_batch` 连续调用 1000 次（每次 386 张 TDUS test），记录吞吐率均值和方差；检查无 `error_code != 0` 输出；ASAN 无报告（需 ASAN 构建） |
| `docs/benchmarks/phase18_deploy.md` | 新建 | 最终部署基准报告：Docker 镜像大小；首次启动时间（含 engine build）；稳态推理吞吐率；峰值显存；与 Python 基线（21.5 img/s）的对比表 |
| `gdino_dinov2_svm/RESULTS.md` | 修改 | 新增"C++ 生产流水线"章节，记录最终 TRT fp16 吞吐率、准确率、显存使用 |

#### 验收标准

1. `test_stress.sh` 1000 次调用全部 `error_code=0`，ASAN 无报告
2. 稳态吞吐率标准差 ≤10%（排除首次 JIT warmup 的 2 次调用）
3. `docs/benchmarks/phase18_deploy.md` 存在，含完整指标对比表
4. `gdino_dinov2_svm/RESULTS.md` 更新包含 C++ 流水线吞吐率数据

### Phase 18 完成标准

- 所有测试通过（C API G-Test + 内存安全 + Docker 验证 + 压测）
- `liburbantreeid.so` 仅暴露 `urban_tree_id_*` 四个公共符号
- ASAN/Valgrind 无内存错误报告
- Docker 镜像可成功构建并在目标 GPU 上运行推理
- 最终文档完整（`RESULTS.md` 更新，`phase18_deploy.md` 存在）

---

## 测试策略

### TDD 原则

每个 Stage 遵循严格的 TDD 顺序：先写测试（此时失败）→ 实现功能 → 测试通过 → 重构（保持测试通过）。特别是数值对齐验证，必须在 C++ 实现完成前就有 Python 参考脚本可运行（即使 C++ 实现不存在时脚本以"无输出文件"的方式失败）。

### 测试分层

| 层级 | 工具 | 范围 | Phase |
|------|------|------|-------|
| C++ 单元测试 | Google Test | 单函数（预处理、NMS、坐标还原）| 15, 17, 18 |
| Python 数值对齐测试 | pytest + numpy | C++ 输出 vs Python GPU 路径 | 15（CI gate）|
| Python fp16 对齐测试 | pytest + numpy | TRT fp16 vs ORT fp32 | 16（CI gate）|
| 端到端准确率测试 | pytest + C++ 可执行 | TDUS test 386 张 | 15, 16, 17, 18 |
| 内存安全测试 | ASAN + Valgrind | 全链路 1000 次调用 | 18 |
| 吞吐率压测 | shell 脚本 + NVML | 预加载图像，排除 I/O | 16, 17, 18 |

### 覆盖率目标

- C++ 代码（`cpp/src/`）：Google Test 覆盖所有公共函数的正常路径和边界情况（bbox 超出边界、空 bbox、单图像 batch）
- Python 验证脚本（`cpp/scripts/`、`cpp/tests/*.py`）：pytest 覆盖率 ≥80%（排除需要真实 GPU 和模型文件的集成测试）
- 数值容差验证为强制 CI gate，不得降低标准

### CI Gate 总结

| Gate | 触发 Phase | 判据 | 未通过时阻断 |
|------|-----------|------|------------|
| C++ 预处理单元测试 | Phase 15 Stage 15-C1 | G-Test 全部通过，无 CenterCrop 相关代码 | Stage 15-C2 |
| 预处理数值对齐 | Phase 15 Stage 15-C2 | cosine 最小值 ≥0.9995，L∞ < 0.01（100 张）| Phase 16 启动 |
| YOLO bbox 对齐 | Phase 15 Stage 15-C2 | L∞ < 2px（50 张），ONNX input name `images` 验证通过 | Phase 15 Stage 15-D |
| SVM ORT 一致性 | Phase 15 Stage 15-B | val 395 张类别预测 100% 完全一致（逐样本，零差异）| Phase 15 Stage 15-D |
| ORT 端到端准确率 | Phase 15 Stage 15-D | TDUS test ≥93.8% | Phase 16 启动 |
| fp16 数值对齐 | Phase 16 Stage 16-B | cosine 均值 ≥0.999，最小值 ≥0.99（386 张）| Phase 16 Stage 16-C |
| TRT 端到端准确率 | Phase 16 Stage 16-C | TDUS test ≥93.8%，吞吐率 ≥30 img/s | Phase 17 启动 |
| 线程安全 | Phase 17 Stage 17-A | Helgrind 无 data race | Phase 17 Stage 17-B |
| 三线程吞吐率 | Phase 17 Stage 17-C | 预加载图像 ≥35 img/s（体现流水线化收益）| Phase 18 启动 |
| ASAN 内存安全 | Phase 18 Stage 18-B | 1000 次调用无内存错误 | Phase 18 Stage 18-C |
| TRT 引擎版本兼容性 | Phase 18 Stage 18-C | 版本不匹配时自动重建且新引擎版本头正确 | Phase 18 Stage 18-D |

---

## 联合审查备注

**审查日期**：2026-06-04  
**审查依据**：`docs/proposals/proposal-cpp-production-pipeline.md`、`scripts/predict_pipeline.py`、`gdino_dinov2_svm/RESULTS.md`、`CLAUDE.md`

### 发现的问题与修改摘要

#### 问题 1：Stage 15-C 代码量超出单 Stage 200 行上限（已修复）

**原状态**：Stage 15-C 估算 `preprocess.cpp` + `preprocess.h`（约 120 行）+ `test_preprocess_alignment.py`（约 80 行）+ CI gate shell 脚本（约 20 行）= 约 220 行，超出 200 行/Stage 上限。

**修改**：拆分为 Stage 15-C1（预处理核心实现 ≈180 行）和 Stage 15-C2（数值对齐 CI gate + YOLO bbox 验证 ≈140 行）。Phase 15 总行数因此从 470 行增至约 670 行，超出 500 行/Phase 上限，但每个 Stage 均满足 ≤200 行约束，属于任务密度合理的拆分结果。**建议同步评估是否需要调整 Phase 级别的行数上限（例如调整为 ≤700 行）。**

#### 问题 2：Phase 17 吞吐率保证目标未体现三线程流水线化收益（已修复）

**原状态**：Stage 17-C 和 Phase 17 完成标准均写"吞吐率 ≥30 img/s"，与 Phase 16 顺序执行的保证目标完全相同，未体现三线程双缓冲流水线应带来的额外提升。

**修改**：将 Phase 17 吞吐率保证目标从 ≥30 img/s 提升至 **≥35 img/s**（约 15–20% 提升，对应 I/O 与 GPU 计算重叠消除 pipeline bubble 的合理预期）。`test_pipeline_threaded_throughput.py` 判据同步更新。若实测无法达到 35 img/s，需分析瓶颈并记录在 `phase17_throughput.md`。

#### 问题 3：SVM 一致性描述措辞歧义（已修复）

**原状态**：plan Phase 15 完成标准写"SVM ORT 推理与 sklearn 结果 100% 一致（val 395 张零差异）"，与 proposal 成功指标表中"val 准确率差 0%"存在语义模糊（"准确率差 0%"可被误读为允许相同错误率但错误样本不同）。

**修改**：plan 中统一改为"类别预测 100% 完全一致（逐样本预测类别字符串完全相同，零差异）"；CI Gate 总结表同步。proposal 成功指标表已在下方说明中同步修正。

#### 问题 4：[MAJOR-5] TRT 引擎跨机不可移植——版本 header 检查逻辑缺乏验收标准（已修复）

**原状态**：Stage 18-C 的 `build_engines_and_serve.sh` 声称实现 TRT 版本 header 检查并在不匹配时自动重建，但原始验收标准中没有任何测试用例验证该逻辑的正确性（只验证了正常首次构建和强制重建两个场景）。

**修改**：Stage 18-C 验收标准新增第6条，要求 `test_docker_build.sh` 包含版本不匹配场景的端到端测试，验证 `build_engines_and_serve.sh` 检测到不匹配后触发重建，且重建结果的 TRT 版本头与当前环境一致。CI Gate 总结表新增"TRT 引擎版本兼容性"Gate。

#### 问题 5：ONNX input tensor name `images` 缺乏显式验证（已修复）

**原状态**：Stage 16-A `build_trt_engines.sh` 中 `--minShapes=images:...` 硬编码了 input tensor name `images`，但没有任何测试验证 ONNX 导出文件的实际 input name 与此一致（若 Ultralytics 未来版本变更默认 name，会导致 TRT 引擎构建静默失败或运行时 shape mismatch）。

**修改**：Stage 15-C2 新增 `test_yolo_bbox_alignment.py`，其中包含对 ONNX input tensor name 的显式验证。Stage 16-A `build_trt_engines.sh` 说明中注明依赖此验证结果。CI Gate 表新增 "YOLO bbox 对齐" Gate，包含 ONNX input name 验证判据。

#### 问题 6：M3.3（CUDA Graph）在 plan 中折叠进 Stage 17-B（已说明，无需额外修改）

Proposal M3.3 列为独立里程碑，plan 将其折叠进 Stage 17-B（M3.2 + M3.3 合并）。此简化合理（CUDA Graph 是可选优化，与 CUDA stream 重叠高度耦合），但 Stage 17-B 标题和说明已标注"M3.2 + M3.3"，保持与 proposal 的可追溯性，无需进一步修改。

#### Proposal 成功指标表措辞修正（同步修改 proposal）

"SVM 推理结果一致性（ORT vs sklearn）" 判据"val 准确率差 0%（完全一致）"改为"类别预测结果 100% 完全一致（val 395 张逐样本预测类别字符串完全相同，零差异）"，消除歧义。
