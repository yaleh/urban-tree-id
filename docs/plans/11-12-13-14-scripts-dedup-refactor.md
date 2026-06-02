# Plan: scripts/ 去重与封装重构

**对应 Proposal**：`docs/proposals/proposal-scripts-dedup-refactor.md`  
**日期**：2026-06-02  
**状态**：已完成（2026-06-02）

---

## 依赖关系总览

```
Phase 11 (P0) ──┐
                ├──▶ Phase 13 (P2)
Phase 12 (P1) ──┘
Phase 14 (P3) ── 独立，无依赖（可在任意阶段后执行）
```

- **Phase 11（P0）**：无依赖，可首先执行
- **Phase 12（P1）**：无依赖，可与 Phase 11 并行；建议顺序执行（先 11 后 12）
- **Phase 13（P2）**：依赖 Phase 11（`gdino_preprocess_with_size` 已可用）和 Phase 12（`image_utils.py`、`yolo_io.py` 已可用）
- **Phase 14（P3）**：独立，无硬性依赖；建议在 Phase 13 完成后执行，以避免 CLI 重命名与 P2 重构同时进行时的冲突

---

## Phase 11 — P0：直接重复，零成本修复

### 目标

消除 `gdino_preprocess` 的多份拷贝，以及 `benchmark_pipeline.py` 中基于运行时动态绑定的 `_ensure_pp_imports` 反模式，使核心函数有唯一的定义入口、import 路径对 IDE 和类型检查器可见。

**估算行数（Phase 11 合计）**：Stage 11-A 净变更约 95 行 + Stage 11-B 净变更约 80 行，Phase 总计 ≈175 行（远低于 500 行上限）

---

### Stage 11-A：新增 `gdino_preprocess_with_size` 并迁移 `classify_multi_tree`

**依赖**：无

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/gdino_utils.py` | 修改 | 新增私有辅助函数 `_compute_resize(pil_img) -> tuple[int, int]`；重构 `gdino_preprocess` 调用 `_compute_resize`；新增 `gdino_preprocess_with_size(pil_img) -> tuple[Tensor, int, int]` |
| `tests/test_gdino_utils.py` | 修改 | 新增 `gdino_preprocess_with_size` 的单元测试：正常图像返回正确 `(new_h, new_w)`；返回的 tensor 与 `gdino_preprocess` 输出等价；极端宽高比（宽 >> 高 和 高 >> 宽）；SHORTEST_EDGE/LONGEST_EDGE 边界触发路径 |
| `scripts/classify_multi_tree.py` | 修改 | 删除第 62–71 行本地 `gdino_preprocess` 定义；添加 `from gdino_utils import gdino_preprocess_with_size as gdino_preprocess`；验证第 201 行 `tensor, prep_h, prep_w = gdino_preprocess(pil)` 调用签名与新函数匹配 |

**TDD 顺序**：先在 `test_gdino_utils.py` 写测试（此时会失败）→ 再在 `gdino_utils.py` 实现 `_compute_resize` 和 `gdino_preprocess_with_size` →  最后更新 `classify_multi_tree.py` import。

#### 验收标准

1. `pytest tests/test_gdino_utils.py` 全部通过，包含新增的 `gdino_preprocess_with_size` 用例
2. `pytest tests/test_classify_multi_tree.py` 全部通过（无需修改该测试文件）
3. `gdino_utils.py` 中不存在重复的 resize 计算逻辑（`_compute_resize` 被 `gdino_preprocess` 和 `gdino_preprocess_with_size` 共用）
4. `classify_multi_tree.py` 中不再定义本地 `gdino_preprocess` 函数
5. `gdino_dinov2_svm/03_extract_embeddings.py` 的本地 `gdino_preprocess` 定义暂未删除（留待 Stage 11-B 或作为低优先级后续）

---

### Stage 11-B：迁移 `03_extract_embeddings.py` 并移除 `_ensure_pp_imports`

**依赖**：Stage 11-A 完成

**估算行数**：删除约 70 行（benchmark_pipeline 的全局变量声明 + `_ensure_pp_imports` 函数体 + 三处调用点），新增约 8 行（顶层 import 语句），`03_extract_embeddings.py` 删除约 10 行并新增 2 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `gdino_dinov2_svm/03_extract_embeddings.py` | 修改 | 删除第 57–66 行本地 `gdino_preprocess` 定义；在文件顶部（`sys.path` 调整之后）添加 `from scripts.gdino_utils import gdino_preprocess`（保留现有 `sys.path` 调整行不动） |
| `scripts/benchmark_pipeline.py` | 修改 | 删除第 47–52 行的模块级全局变量声明（`gdino_forward_batch = None` 等五个）；删除第 55–86 行的 `_ensure_pp_imports` 函数；在文件顶部添加 `from predict_pipeline import (gdino_forward_batch, gdino_preprocess_batch, detect_yolo_batch, make_crops_gpu_batch, predict_species_batch)`；删除 `_run_detector_loop`（第 246 行）、`_preload_batches`（第 449 行）、`_run_timed_bench`（第 511 行）中对 `_ensure_pp_imports()` 的调用 |
| `tests/test_benchmark_pipeline.py` | 修改 | 删除任何直接调用或断言 `_ensure_pp_imports` 存在的代码；确认现有 `patch("benchmark_pipeline.gdino_forward_batch", ...)` 等 patch 路径无需修改（直接 import 后名字同样绑定在 `benchmark_pipeline` 模块命名空间） |

**TDD 顺序**：先审查 `test_benchmark_pipeline.py` 中所有 `_ensure_pp_imports` 相关代码 → 更新测试（删除对该函数的测试）→ 再修改 `benchmark_pipeline.py`。

#### 验收标准

1. `pytest tests/test_benchmark_pipeline.py` 全部通过
2. `benchmark_pipeline.py` 文件中不存在字符串 `_ensure_pp_imports`
3. `benchmark_pipeline.py` 文件中不存在 `gdino_forward_batch = None`（或类似全局 None 初始化）
4. `benchmark_pipeline.py` 顶部存在 `from predict_pipeline import` 直接导入语句
5. `03_extract_embeddings.py` 中不再定义本地 `gdino_preprocess` 函数
6. `pytest tests/` 全部通过（整体回归）

### Phase 11 完成标准

- 所有测试通过，覆盖率不低于执行前水平（不得低于 65.3%）
- `gdino_preprocess` 在整个代码库中仅在 `scripts/gdino_utils.py` 定义一次
- `_ensure_pp_imports` 在整个代码库中零出现
- `git grep "def gdino_preprocess"` 只返回 `scripts/gdino_utils.py` 一处结果（含 `gdino_preprocess_with_size`）

---

## Phase 12 — P1：提取共享工具模块

### 目标

新建 `scripts/yolo_io.py` 和 `scripts/image_utils.py` 两个纯工具模块，将散落在多处的 YOLO 标签 I/O 函数和图像裁剪函数迁移至统一入口。

**估算行数**：新增约 120 行（两个新文件 + 两个新测试文件框架），删除约 90 行（各脚本中的本地定义），净变更 ≈210 行（低于 500 行上限）

---

### Stage 12-A：新建 `yolo_io.py` 并迁移 YOLO 标签 I/O

**依赖**：无（可与 Phase 11 并行启动）

**估算行数**：新增 `yolo_io.py` 约 50 行，新增 `test_yolo_io.py` 约 60 行，删除三处本地定义约 60 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/yolo_io.py` | 新建 | 定义 `load_yolo_xyxy(label_path, img_w, img_h) -> torch.Tensor`；`xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h) -> tuple[float, float, float, float]`；`write_yolo_labels(label_path, boxes_xyxy, img_w, img_h) -> None` |
| `tests/test_yolo_io.py` | 新建 | 测试用例：`write_yolo_labels` → `load_yolo_xyxy` round-trip（坐标误差 < 0.5px）；空 tensor 写出空文件；空文件读入空 tensor；`xyxy_to_yolo` 数值正确性（边框覆盖整图时 cx=cy=0.5, w=h=1.0）；5 列格式验证（class_id 列为 0） |
| `scripts/generate_pseudo_labels.py` | 修改 | 删除第 84–103 行的 `xyxy_to_yolo` 和 `write_label_file`；添加 `from yolo_io import xyxy_to_yolo, write_yolo_labels`；更新调用处使用新函数签名 |
| `scripts/self_train.py` | 修改 | 删除第 67–98 行的 `load_yolo_xyxy` 和 `write_yolo_from_xyxy`；添加 `from yolo_io import load_yolo_xyxy, write_yolo_labels`；更新调用处 |
| `scripts/diff_gdino_yolo.py` | 修改 | 删除第 25–39 行的 `load_yolo_xyxy`；添加 `from yolo_io import load_yolo_xyxy` |
| `tests/test_self_train.py` | 修改 | 将 `load_yolo_xyxy`、`write_yolo_from_xyxy` 的 import 路径更新为 `from yolo_io import ...`；验证测试仍通过 |

**TDD 顺序**：先写 `test_yolo_io.py`（此时无 `yolo_io.py`，全部失败）→ 实现 `yolo_io.py` →  → 迁移各脚本 → 更新 `test_self_train.py`。

#### 验收标准

1. `pytest tests/test_yolo_io.py` 全部通过
2. `pytest tests/test_self_train.py` 全部通过
3. `git grep "def load_yolo_xyxy"` 只返回 `scripts/yolo_io.py` 一处
4. `git grep "def write_label_file\|def write_yolo_from_xyxy"` 零返回
5. `scripts/yolo_io.py` 模块顶部有 docstring 注明"所有脚本应从此模块 import，不得自行实现"

---

### Stage 12-B：新建 `image_utils.py` 并迁移 `make_crop`

**依赖**：无（可与 Stage 12-A 并行）

**估算行数**：新增 `image_utils.py` 约 45 行，新增 `test_image_utils.py` 约 70 行，删除三处本地定义约 55 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/image_utils.py` | 新建 | 定义 `make_crop_xyxy(pil_img, box_xyxy, pad_frac=0.05, target_size=448) -> Image.Image`（合并 `predict_pipeline.py:59–76` 和 `03_extract_embeddings.py:69–84` 的逻辑）；定义 `make_crop_norm(pil_img, box_norm, pad_frac=0.05, target_size=448) -> Image.Image`（反归一化后调用 `make_crop_xyxy`） |
| `tests/test_image_utils.py` | 新建 | 测试用例：`make_crop_xyxy` 和 `make_crop_norm` 输出等价性（等价归一化坐标应得到像素级相同输出）；边缘 box（x0=0, y0=0 时 pad 被截断）；极小框（宽/高 < 1px 时 `max(1, ...)` 保护不抛异常）；输出尺寸始终为 `(target_size, target_size)`；`pad_frac=0` 时不扩张 |
| `scripts/predict_pipeline.py` | 修改 | 删除第 59–76 行本地 `make_crop`；添加 `from image_utils import make_crop_xyxy as make_crop`；`TARGET_SIZE = 448` 常量可删除（已参数化为默认值） |
| `gdino_dinov2_svm/03_extract_embeddings.py` | 修改 | 删除第 69–84 行本地 `make_crop`；在 `sys.path` 调整之后添加 `from scripts.image_utils import make_crop_xyxy as make_crop` |
| `scripts/classify_multi_tree.py` | 修改 | 删除第 76–95 行本地 `make_crop`；添加 `from image_utils import make_crop_norm as make_crop`；`CROP_TARGET = 448` 常量可删除 |

**TDD 顺序**：先写 `test_image_utils.py` → 实现 `image_utils.py` → 迁移各脚本。

#### 验收标准

1. `pytest tests/test_image_utils.py` 全部通过
2. `pytest tests/test_predict_pipeline.py` 和 `pytest tests/test_classify_multi_tree.py` 全部通过
3. `git grep "def make_crop"` 只返回 `scripts/image_utils.py` 中的两处定义（`make_crop_xyxy` 和 `make_crop_norm`）
4. `predict_pipeline.py` 和 `classify_multi_tree.py` 中不存在 `TARGET_SIZE` 或 `CROP_TARGET` 常量
5. `make_crop_norm` 和 `make_crop_xyxy` 用相同坐标调用时输出像素级相同

### Phase 12 完成标准

- 所有测试通过
- `scripts/yolo_io.py` 和 `scripts/image_utils.py` 均存在且有完整 docstring
- `pytest --cov=scripts --cov-report=term-missing` 覆盖率 ≥ 75%（新模块无历史包袱，易覆盖）
- `git grep "def load_yolo_xyxy\|def xyxy_to_yolo\|def write_label_file\|def write_yolo_from_xyxy\|def make_crop"` 仅在对应工具模块中返回结果，其余文件零出现

---

## Phase 13 — P2：检测逻辑重新整合

### 目标

让 `classify_multi_tree.py` 复用 `predict_pipeline` 的检测函数族，消除手写 NMS 副本；提取 `generate_pseudo_labels` 内部的批量推断内核 `_run_gdino_batch`，消除 `run_inference` 与 `run_inference_flat` 的结构性重复。

**前置条件**：Phase 11（`gdino_preprocess_with_size` 可用）和 Phase 12（`image_utils.py` 可用）均已完成

**估算行数**：重构约 200 行（`classify_multi_tree` 检测入口 + `generate_pseudo_labels` 批量内核），新增/修改测试约 80 行，总计 ≈280 行（低于 500 行上限）

**关键风险提示（必读）**：
- `predict_pipeline.detect_gdino` 返回 xyxy **像素坐标**，而 `classify_multi_tree` 现有流程全程使用**归一化坐标**。直接替换而不做坐标转换将导致裁剪坐标完全错误（静默 bug）。本 Phase 采用方案 A：`classify_detections` 改为接受 xyxy 像素坐标，内部使用 `make_crop_xyxy`，结果字典的 `box_norm` 字段由像素坐标除以 W/H 反算。
- `classify_multi_tree` 的手写 NMS 使用 IoU 阈值 0.5，而 `predict_pipeline.detect_gdino` 的 `torchvision.ops.nms` 使用 `GDINO_IOU_THR=0.45`，**这是有意的行为变更**，PR 描述中必须明确说明。
- `classify_multi_tree.detect_and_classify` 使用 `score_thr=0.3`，而 `predict_pipeline.detect_gdino` 默认使用 `GDINO_SCORE_THR=0.35`，迁移后分数阈值将从 0.3 升为 0.35，可能导致部分低置信度框被过滤。**这同样是有意的行为变更**，必须在 Stage 13-A 的 PR 描述中与 NMS 阈值变更一并说明。

---

### Stage 13-A：重构 `classify_multi_tree.py` 检测入口

**依赖**：Phase 11（`gdino_preprocess_with_size`），Phase 12-B（`image_utils.make_crop_xyxy`）

**估算行数**：删除约 120 行（手写 NMS + 旧检测流程），新增约 40 行（新 `classify_detections` + `run_image`）

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/classify_multi_tree.py` | 修改 | 添加 `from predict_pipeline import detect_gdino, detect_yolo`；添加 `from image_utils import make_crop_xyxy`；重构 `detect_and_classify`（第 195–298 行）：删除手写 GDino 前向推断和手写 NMS（第 241–260 行），改为调用 `detect_gdino(pil, device)` 返回 xyxy 像素坐标；新增 `classify_detections(pil_img, boxes_xyxy, dinov2, clf, device) -> list[dict]` 函数，内部使用 `make_crop_xyxy`，结果字典的 `box_norm` 字段由 `(x0/W, y0/H, x1/W, y1/H)` 计算；重构 `detect_and_classify_yolo`（第 301–341 行）：改为调用 `detect_yolo(pil, device)` |
| `tests/test_classify_multi_tree.py` | 修改 | 新增 `classify_detections` 的单元测试：mock `detect_gdino` 返回已知 xyxy boxes，验证裁剪调用 `make_crop_xyxy` 而非 `make_crop_norm`；验证结果字典中 `box_norm` 值正确（像素坐标/图像尺寸）；验证坐标系不匹配不会发生（box_norm 值在 [0, 1] 范围内） |

**TDD 顺序**：先写新测试（验证像素坐标路径和 `box_norm` 计算）→ 再重构实现。

#### 验收标准

1. `pytest tests/test_classify_multi_tree.py` 全部通过
2. `classify_multi_tree.py` 中不存在手写 IoU 循环（`git grep "iou\|intersection\|union"` 在该文件中零返回，或仅在注释中出现）
3. `classify_multi_tree.py` 中不存在对 GDino 模型的直接前向调用（不再有 `processor(images=..., text=...)` 或等价调用，检测委托给 `predict_pipeline.detect_gdino`）
4. 结果字典中的 `box_norm` 字段值在 [0, 1] 范围内（可用断言验证）
5. PR 描述中明确说明两处行为变更：NMS IoU 阈值 0.5 → 0.45，分数阈值 score_thr 0.3 → 0.35（仅在 PR 描述中说明，不接受"仅注释"替代）

---

### Stage 13-B：提取 `generate_pseudo_labels._run_gdino_batch`

**依赖**：Phase 11（`gdino_utils.gdino_preprocess` 已统一），Phase 12-A（`yolo_io` 可用）

**估算行数**：新增 `_run_gdino_batch` 约 40 行，重构 `run_inference` 和 `run_inference_flat` 各节省约 50 行，净约 60 行变更

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/generate_pseudo_labels.py` | 修改 | 提取私有函数 `_run_gdino_batch(dataset, processor, model, device, batch_size, box_threshold, text_threshold, num_workers: int = 0) -> Iterator[tuple[list[str], list[Image.Image], list[Tensor], list[Tensor]]]`（逐批 yield `(paths, pil_imgs, boxes_list, scores_list)`；`num_workers` 由调用方传入以保留原有差异：`run_inference` 传 `num_workers=2`，`run_inference_flat` 传 `num_workers=0`）；重构 `run_inference`（第 108–190 行）：推断循环内部改为迭代 `_run_gdino_batch(... num_workers=2)`，保留亮度过滤和目录布局逻辑；重构 `run_inference_flat`（第 195–271 行）：改为迭代 `_run_gdino_batch(... num_workers=0)`，保留无亮度过滤、失败列表逻辑 |
| `tests/test_generate_pseudo_labels.py` | 修改（若存在）或新建 | 新增 `_run_gdino_batch` 的单元测试：mock DataLoader，验证 yield 结构正确；验证 `run_inference` 亮度过滤路径和 `run_inference_flat` 无亮度过滤路径独立正确 |

**TDD 顺序**：先写/更新测试 → 提取 `_run_gdino_batch` → 重构两个 `run_inference` 函数。

#### 验收标准

1. 如存在 `tests/test_generate_pseudo_labels.py`，全部通过；若新建，覆盖 `_run_gdino_batch` 主路径
2. `generate_pseudo_labels.py` 中 `run_inference` 和 `run_inference_flat` 每个函数长度不超过 40 行（不含注释空行）
3. `_run_gdino_batch` 函数存在且有类型注解
4. `pytest tests/` 全部通过（整体回归）

### Phase 13 完成标准

- 所有测试通过
- `pytest --cov=scripts --cov-report=term-missing` 覆盖率 ≥ 78%
- `classify_multi_tree.py` 中不再维护独立的 GDino 前向推断和 NMS 实现
- `generate_pseudo_labels.py` 中 `run_inference` 和 `run_inference_flat` 不再有重复的批量推断循环

---

## Phase 14 — P3：结构性改进（CLI 统一 + 可选 Dataset 基类）

### 目标

通过共享工厂函数统一 4 个 CLI 脚本的公共参数解析，消除参数名称漂移（`--yolo-model` vs `--yolo-checkpoint`）。Dataset 基类为可选项，需团队评估后决定是否实施。

**估算行数**：新建 `cli_common.py` 约 40 行，修改 4 个 CLI 脚本各约 15 行，测试修改约 5 行，总计 ≈120 行

---

### Stage 14-A：新建 `cli_common.py` 并迁移公共参数

**依赖**：无硬性依赖；建议 Phase 13 完成后执行

**估算行数**：新增 `cli_common.py` 约 40 行，修改 4 个脚本共约 60 行，修改测试约 5 行

#### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/cli_common.py` | 新建 | 定义 `add_detector_args(parser) -> None`（添加 `--detector`、`--yolo-checkpoint`、`--rf-detr-checkpoint`、`--device`）；定义 `add_gdino_threshold_args(parser) -> None`（添加 `--box-threshold`、`--text-threshold`、`--score-thr`、`--iou-thr`）；所有参数默认值集中在此处 |
| `scripts/predict_pipeline.py` | 修改 | 在 `build_parser()` 中用 `add_detector_args(parser)` 替换现有重复的 `add_argument` 调用 |
| `scripts/benchmark_pipeline.py` | 修改 | 同上；同时添加 `add_gdino_threshold_args(parser)` |
| `scripts/classify_multi_tree.py` | 修改 | 在 `build_arg_parser()` 中将 `--yolo-model` 重命名为 `--yolo-checkpoint`（`dest="yolo_checkpoint"`）；保留 `--yolo-model` 作为 deprecated alias（`add_argument("--yolo-model", dest="yolo_checkpoint", help="已废弃，请使用 --yolo-checkpoint")`）；使用 `add_detector_args(parser)` 替换重复参数 |
| `scripts/generate_pseudo_labels.py` | 修改 | 在 `parse_args_flat()` 中用 `add_gdino_threshold_args(parser)` 替换重复参数 |
| `tests/test_classify_multi_tree.py` | 修改 | 将第 23 行的 `args.yolo_model` 改为 `args.yolo_checkpoint`（因 `dest` 已更改） |

**关键提示**：`tests/test_classify_multi_tree.py:23` 的 `assert "tree_yolo26s_halfres" in args.yolo_model` 必须在同一 PR 内更新，否则测试会 `AttributeError` 失败。

**TDD 顺序**：先更新 `test_classify_multi_tree.py:23`（使测试在重命名后仍通过）→ 新建 `cli_common.py` → 迁移各脚本。

#### 验收标准

1. `pytest tests/test_classify_multi_tree.py` 全部通过（包括第 23 行断言改为 `args.yolo_checkpoint`）
2. `pytest tests/test_predict_pipeline.py` 和 `pytest tests/test_benchmark_pipeline.py` 全部通过
3. `scripts/cli_common.py` 存在，包含 `add_detector_args` 和 `add_gdino_threshold_args` 两个函数
4. `classify_multi_tree.py` 中 `--yolo-checkpoint` 和 `--yolo-model`（deprecated alias）并存，均解析到 `dest="yolo_checkpoint"`
5. `git grep "\-\-device.*default\|\"--detector\".*default"` 在 `scripts/` 中仅在 `cli_common.py` 中定义默认值（其他脚本通过调用工厂函数获得）

---

### Stage 14-B：Dataset 基类（可选）

**依赖**：Stage 14-A 完成；**实施前需团队明确决定**

**决策标准**：仅在以下条件全部满足时实施：
- 团队计划在 `scripts/` 中新增更多 Dataset 类（收益随类数量线性增长）
- 愿意承担字段名统一（`self.samples` → `self.paths`）的改动成本
- `gdino_dinov2_svm/` 的 `TreeDataset`/`TDUSDataset` 暂不迁移（保持独立）

**若跳过此 Stage，Phase 14 仍视为完成。**

**估算行数**：新增 `datasets.py` 约 30 行，修改 `benchmark_pipeline.py` 和 `generate_pseudo_labels.py` 各约 10 行

#### 文件变更清单（如实施）

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/datasets.py` | 新建 | 定义 `_PathsDataset(Dataset)`（字段 `self.paths: list`，方法 `__len__`）；定义 `FrameDataset(_PathsDataset)` 和 `_ImageDataset(_PathsDataset)` 继承自基类 |
| `scripts/generate_pseudo_labels.py` | 修改 | 将 `FrameDataset` 迁移至 `datasets.py`；添加 `from datasets import FrameDataset` |
| `scripts/benchmark_pipeline.py` | 修改 | 将 `_ImageDataset` 迁移至 `datasets.py`；添加 `from datasets import _ImageDataset` |
| `tests/test_datasets.py` | 新建（如实施） | 测试 `_PathsDataset.__len__` 正确；`FrameDataset` 和 `_ImageDataset` 可实例化并通过 DataLoader |

#### 验收标准（如实施）

1. `pytest tests/test_datasets.py` 全部通过
2. `pytest tests/` 全部通过（整体回归）
3. `generate_pseudo_labels.py` 和 `benchmark_pipeline.py` 中不再定义 `FrameDataset`/`_ImageDataset` 类体

### Phase 14 完成标准

- 所有测试通过（含 `test_classify_multi_tree.py` 第 23 行修复）
- `pytest --cov=scripts --cov-report=term-missing` 覆盖率 ≥ 80%（达到目标）
- `--yolo-checkpoint` 和 `--yolo-model`（deprecated）并存，向后兼容
- 公共参数默认值仅在 `cli_common.py` 中维护一份
- Dataset 基类状态：已实施或已明确记录为"暂不做"

---

## 总体验收标准

| 指标 | 目标 |
|------|------|
| 测试覆盖率（`scripts/` 包） | ≥ 80%（当前 65.3%） |
| `gdino_preprocess` 定义数量 | 1（仅 `gdino_utils.py`） |
| `make_crop` 定义数量 | 2（`make_crop_xyxy` 和 `make_crop_norm`，均在 `image_utils.py`） |
| `load_yolo_xyxy` 定义数量 | 1（仅 `yolo_io.py`） |
| `_ensure_pp_imports` 出现次数 | 0 |
| 手写 NMS 实现数量 | 0（均委托给 `torchvision.ops.nms`） |
| CLI `--yolo-model` 处理 | deprecated alias，解析到 `yolo_checkpoint` |
| 所有测试（`pytest tests/`） | 全部通过 |
