# Proposal: scripts/ 去重与封装重构

**状态**：草稿  
**日期**：2026-06-02  
**作者**：Yale Huang  

---

## 1. 背景

`urban-tree-id` 是一个城市树木物种识别流水线，使用 GroundingDINO 检测树木边界框、DINOv2 提取嵌入、SVM 分类树种。代码库包含：

- `scripts/`：20 个业务脚本（检测、训练、推断、数据集构建等）
- `tests/`：13 个测试文件
- `gdino_dinov2_svm/`：3 个早期实验脚本（编号文件名 `02_`/`03_`/`04_`）

项目在快速迭代过程中，多处核心逻辑被逐文件复制而非模块化复用。随着脚本数量增长，同一函数的多份拷贝开始出现不一致的签名、不同的阈值默认值和微小的行为差异，提升了维护成本，也增大了引入静默 bug 的风险。

---

## 2. 目标

完成本提案后，代码库应满足以下结构性目标：

1. **零冗余核心函数**：每个被多处使用的函数只在一处定义，其他模块直接 import。
2. **YOLO 标签 I/O 统一入口**：所有 YOLO 格式读写通过 `scripts/yolo_io.py` 进行。
3. **图像裁剪逻辑统一入口**：`make_crop`（xyxy 坐标）及其归一化坐标变体合并到一处。
4. **直接 import 替代 importlib 动态加载**：`benchmark_pipeline.py` 中的 `_ensure_pp_imports` 反模式被消除。
5. **Dataset 基类**：4 个 Dataset 类共享最小基类，消除重复的 `__init__`/`__len__`/`__getitem__` 样板代码。
6. **CLI 参数解析集中管理**：公共参数（`--detector`、`--device`、`--conf` 等）通过共享工厂函数或父 parser 定义，默认值只维护一份。

---

## 3. 问题清单与方案设计

### P0 — 直接重复，零成本可修复

#### 3.1 `gdino_preprocess` 三处定义

**现状**

| 文件 | 行号 | 备注 |
|------|------|------|
| `scripts/gdino_utils.py` | 19–34 | 应为唯一来源；已正确定义 |
| `scripts/classify_multi_tree.py` | 62–71 | 逐行拷贝，但**多返回 `new_h, new_w`**（签名不同） |
| `gdino_dinov2_svm/03_extract_embeddings.py` | 57–66 | 逐行拷贝，签名与 `gdino_utils.py` 一致 |

`classify_multi_tree.py` 的拷贝还把额外的 `(new_h, new_w)` 返回值传给后续逻辑（第 201 行 `tensor, prep_h, prep_w = gdino_preprocess(pil)`），这与 `gdino_utils.py` 的单返回值签名存在静默分叉。

**方案**

1. 在 `gdino_utils.py` 中保留现有 `gdino_preprocess`（单返回 tensor），并新增：
   ```python
   def gdino_preprocess_with_size(pil_img) -> tuple[torch.Tensor, int, int]:
       """同 gdino_preprocess，额外返回 (new_h, new_w)，供需要 pixel_mask 的调用方使用。"""
       # 注意：必须重新执行 resize 逻辑以获得 new_h, new_w；
       # 不能从已归一化的 tensor 的 shape 反推，因为 shape 与 resize 后相同，
       # 但 gdino_preprocess 内部的 int(round(...)) 会造成舍入误差，无法逆向还原。
       w, h = pil_img.size
       scale = SHORTEST_EDGE / min(h, w)
       new_h, new_w = int(round(h * scale)), int(round(w * scale))
       if max(new_h, new_w) > LONGEST_EDGE:
           scale = LONGEST_EDGE / max(new_h, new_w)
           new_h, new_w = int(round(new_h * scale)), int(round(new_w * scale))
       t = gdino_preprocess(pil_img)
       return t, new_h, new_w
   ```
   **实现注意**：不能写 `return t, t.shape[1], t.shape[2]`——`t` 的 shape 已经是 resize 后的尺寸，
   与 `new_h, new_w` 一致，但必须显式重算才能保证语义清晰和未来实现变化时的正确性。
   建议将 resize 计算提取为私有函数 `_compute_resize(pil_img) -> tuple[int,int]` 供两个公开函数共用。

2. `classify_multi_tree.py:62–71` 删除本地定义，改为：
   ```python
   from gdino_utils import gdino_preprocess_with_size as gdino_preprocess
   ```
3. `gdino_dinov2_svm/03_extract_embeddings.py:57–66` 删除本地定义，改为：
   ```python
   from scripts.gdino_utils import gdino_preprocess  # 或调整 sys.path
   ```
   由于 `03_` 是实验脚本，可保留一行 `sys.path` 调整后直接 import。

#### 3.2 `benchmark_pipeline._ensure_pp_imports`（inert-slot 反模式）

**现状**

`scripts/benchmark_pipeline.py:55–86` 使用模块级全局变量（初始为 `None`）加 `_ensure_pp_imports()` 函数，在运行时通过普通 `from predict_pipeline import ...` 动态绑定五个函数到模块全局变量。

**根因澄清**：这里并非循环依赖，也不是路径问题（`predict_pipeline` 与 `benchmark_pipeline` 同在 `scripts/` 目录，无任何循环引用）。源代码注释明确写道："Tests can patch these names directly on this module"——该设计的唯一目的是为测试提供一种 monkeypatch 入口：测试可以在调用 `_ensure_pp_imports()` 之前把模块全局变量设为 Mock，函数会检测到非 `None` 就跳过覆盖。这是一种非标准的测试可插拔设计，并非为了解决任何 import 技术问题。

> **验证**：`benchmark_pipeline.py` 的 `_ensure_pp_imports` 函数（第 75 行）使用的是普通 `from predict_pipeline import ...`，不是 `importlib.import_module`；路径已通过 `sys.path` 加入，但这在直接 import 时同样有效。

这一设计造成：
- 函数引用在文件顶部不可见，IDE 无法跳转，类型检查器无法推断；
- 任何名称拼写错误要到运行时才暴露；
- 代码复杂度高，阅读者需要理解该惰性初始化机制（而正常 import 用标准 `patch` 即可）。

**方案**

直接在文件顶部正常 import：

```python
from predict_pipeline import (
    gdino_forward_batch,
    gdino_preprocess_batch,
    detect_yolo_batch,
    make_crops_gpu_batch,
    predict_species_batch,
)
```

测试中需要 mock 时，改用 `unittest.mock.patch` 在调用路径上打桩（`patch("benchmark_pipeline.gdino_forward_batch", ...)`），这是标准 Python 测试惯例，无需运行时动态绑定。

同时删除：
- 第 47–52 行的模块级全局变量声明；
- 第 55–86 行的 `_ensure_pp_imports` 函数；
- `_preload_batches`（第 449 行）和 `_run_timed_bench`（第 511 行）中对 `_ensure_pp_imports()` 的调用；
- `_run_detector_loop`（第 246 行）中对 `_ensure_pp_imports()` 的调用。

---

### P1 — 散落的业务逻辑，需提取共享模块

#### 3.3 `make_crop` 三处实现

**现状**

| 文件 | 行号 | 坐标系 |
|------|------|--------|
| `scripts/predict_pipeline.py` | 59–76 | xyxy 像素坐标 |
| `gdino_dinov2_svm/03_extract_embeddings.py` | 69–84 | xyxy 像素坐标（与上完全相同） |
| `scripts/classify_multi_tree.py` | 76–95 | 归一化 [0,1] 坐标，需先乘以 W/H |

`predict_pipeline.py` 和 `03_extract_embeddings.py` 的实现逐行相同（pad → crop → aspect-ratio-preserving resize → white canvas paste）。`classify_multi_tree.py` 的变体仅多了一个坐标反归一化步骤，其余逻辑完全一致。

> **实现验证**：通过阅读源码确认，`classify_multi_tree.py:76–95` 的 `make_crop` 接受 `box_norm`（归一化 [0,1] 坐标），函数内部先乘以 `W`/`H` 反归一化，再执行与 `predict_pipeline.py:59–76` 完全相同的 pad→crop→resize→paste 逻辑。两者唯一差异是输入坐标系；`Image.LANCZOS` 调用、`max(1, ...)` 保护、白色画布填充均相同。

**方案**

新建 `scripts/image_utils.py`（推荐，保持关注点分离），定义单一主函数并提供薄包装：

```python
def make_crop_xyxy(pil_img, box_xyxy, pad_frac=0.05, target_size=448) -> Image.Image:
    """裁剪并填充到 target_size×target_size 白色画布（xyxy 像素坐标）。"""
    ...  # 合并 predict_pipeline.py:59–76 与 03_extract_embeddings.py:69–84

def make_crop_norm(pil_img, box_norm, pad_frac=0.05, target_size=448) -> Image.Image:
    """同 make_crop_xyxy，接受归一化 [0,1] 坐标。"""
    W, H = pil_img.size
    x0, y0, x1, y1 = box_norm[0]*W, box_norm[1]*H, box_norm[2]*W, box_norm[3]*H
    return make_crop_xyxy(pil_img, (x0, y0, x1, y1), pad_frac, target_size)
```

迁移后：
- `predict_pipeline.py:59–76` → `from image_utils import make_crop_xyxy as make_crop`
- `03_extract_embeddings.py:69–84` → 同上
- `classify_multi_tree.py:76–95` → `from image_utils import make_crop_norm as make_crop`

`target_size` 参数化可消除 `classify_multi_tree.py` 中的 `CROP_TARGET = 448` 常量与 `predict_pipeline.py` 中的 `TARGET_SIZE = 448` 常量之间的重复。

**测试同步要求**：迁移后需在 `tests/test_image_utils.py`（新建）中补充 `make_crop_xyxy` 和 `make_crop_norm` 的单元测试，重点覆盖：边缘贴合裁剪、pad_frac 溢出边界截断、极小框（宽/高 < 1px 时的 `max(1, ...)` 保护）、归一化与像素坐标两种调用的输出等价性。

#### 3.4 YOLO 标签 I/O 散落三处

**现状**

| 文件 | 函数 | 行号 |
|------|------|------|
| `scripts/generate_pseudo_labels.py` | `xyxy_to_yolo()` | 84–89 |
| `scripts/generate_pseudo_labels.py` | `write_label_file()` | 92–103 |
| `scripts/self_train.py` | `load_yolo_xyxy()` | 67–82 |
| `scripts/self_train.py` | `write_yolo_from_xyxy()` | 85–98 |
| `scripts/diff_gdino_yolo.py` | `load_yolo_xyxy()` | 25–39 |

`self_train.py` 和 `diff_gdino_yolo.py` 的 `load_yolo_xyxy` 逐行相同。`generate_pseudo_labels.py` 的 `write_label_file` 与 `self_train.py` 的 `write_yolo_from_xyxy` 功能等价（接受 tensor vs. 接受逐行 list，但核心变换相同）。

**方案**

新建 `scripts/yolo_io.py`：

```python
"""YOLO 标签格式读写工具。所有脚本应从此模块 import，不得自行实现。"""

def load_yolo_xyxy(label_path: Path, img_w: int, img_h: int) -> torch.Tensor:
    """读取 YOLO txt → (N, 4) xyxy 像素坐标 tensor。"""
    ...

def xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h) -> tuple[float, float, float, float]:
    """xyxy 像素坐标 → YOLO 归一化 (cx, cy, w, h)。"""
    ...

def write_yolo_labels(label_path: Path, boxes_xyxy: torch.Tensor,
                      img_w: int, img_h: int) -> None:
    """将 (N, 4) xyxy tensor 写入 YOLO txt 文件。"""
    ...
```

迁移后：
- `generate_pseudo_labels.py:84–103` 删除，改为 `from yolo_io import xyxy_to_yolo, write_yolo_labels`
- `self_train.py:67–98` 删除，改为 `from yolo_io import load_yolo_xyxy, write_yolo_labels`
- `diff_gdino_yolo.py:25–39` 删除，改为 `from yolo_io import load_yolo_xyxy`

---

### P2 — 检测逻辑重新实现

#### 3.5 `classify_multi_tree.py` 绕过 `predict_pipeline` 检测函数族

**现状**

`predict_pipeline.py` 已提供完整的检测函数族：`detect_gdino`（81–135）、`detect_yolo`（138–152）、`detect_rf_detr`（155–164）。但 `classify_multi_tree.py` 中：

- `detect_and_classify`（195–298，104 行）完全绕过 `detect_gdino`，自行实现了一套 GDino 前向推断 + 手写 NMS 逻辑（241–260 行的纯 Python IoU 循环），使用不同的分数阈值（`score_thr=0.3`，而 `predict_pipeline.py` 默认 `GDINO_SCORE_THR=0.35`）。
- `detect_and_classify_yolo`（301–341）同样未复用 `detect_yolo`，直接调用 `yolo_model.predict`。

此外，`generate_pseudo_labels.py` 的 `run_inference`（108–190）也自行内嵌了完整的 GDino 前向推断，而不是调用 `predict_pipeline.detect_gdino`。

**接口兼容性问题（P2 可行性障碍）**

在实施此方案前，必须解决一个严重的接口不匹配：

| 函数 | 返回格式 |
|------|---------|
| `predict_pipeline.detect_gdino` | `list[list[float]]`，xyxy **像素坐标** |
| `classify_multi_tree.detect_and_classify` 内部流程 | 全程使用 **归一化坐标**（`box_norm`），传入 `make_crop(pil, box_norm)` 并最终存入结果字典的 `box_norm` 字段 |

因此，直接 `boxes = detect_gdino(pil, device)` 后传入现有 `classify_detections` 会导致裁剪坐标错误（像素值被当成 [0,1] 归一化坐标使用）。

**必须在接口层做出明确选择之一**：
- **方案 A（推荐）**：`detect_gdino` 返回值不变（像素坐标），`classify_detections` 接受 `boxes_xyxy`，内部改用 `make_crop_xyxy`；结果字典中的 `box_norm` 字段在分类后由像素坐标反算：`(x0/W, y0/H, x1/W, y1/H)`。
- **方案 B**：新增 `detect_gdino_norm` 变体返回归一化坐标；但这会增加 API 曲面，不推荐。

**修正后的方案**

```python
# classify_multi_tree.py 重构后的检测入口
from predict_pipeline import detect_gdino, detect_yolo
from image_utils import make_crop_xyxy

def classify_detections(pil_img, boxes_xyxy, dinov2, clf, device):
    """对给定 xyxy 像素坐标 boxes 裁剪、嵌入、分类，返回 list[dict]。
    结果字典中 box_norm 由像素坐标归一化得到。
    """
    W, H = pil_img.size
    ...

def run_image(img_path, detector_fn, dinov2, clf, device):
    pil = Image.open(img_path).convert("RGB")
    boxes = detector_fn(pil, device)          # detect_gdino 或 detect_yolo → xyxy 像素坐标
    return classify_detections(pil, boxes, dinov2, clf, device)
```

这样 `classify_multi_tree` 的检测阈值、NMS 逻辑与 `predict_pipeline` 保持一致，不再维护独立副本。

`generate_pseudo_labels.py` 的 `run_inference` 因需要批量推断（DataLoader + brightness filter）和特定的输出格式（YOLO txt），不适合直接复用 `detect_gdino`（单图接口）；但可以复用 `gdino_utils.gdino_preprocess` 和 `yolo_io.write_yolo_labels`，将推断循环本身保留（已在 P0/P1 处理了其直接依赖）。

#### 3.6 `generate_pseudo_labels.run_inference` 与 `run_inference_flat` 结构高度重复

**现状**

`run_inference`（108–190，83 行）与 `run_inference_flat`（195–271，77 行）结构几乎完全相同：加载模型 → 构建 Dataset/DataLoader → 前向推断 → filter_boxes → write_label_file。主要差异：

| 维度 | `run_inference` | `run_inference_flat` |
|------|----------------|----------------------|
| 亮度过滤 | 有（per-image `is_too_dark`） | 无 |
| 输出结构 | 按视频目录组织，同时复制图像 | 仅输出标签，记录失败列表 |
| DataLoader workers | 2 | 0 |

**方案**

提取共享的批量推断内核：

```python
def _run_gdino_batch(
    dataset: FrameDataset,
    processor,
    model,
    device: str,
    batch_size: int,
    box_threshold: float,
    text_threshold: float,
) -> Iterator[tuple[list[str], list[Image.Image], list[torch.Tensor], list[torch.Tensor]]]:
    """逐批 yield (paths, pil_imgs, boxes_list, scores_list)。"""
    ...
```

`run_inference` 和 `run_inference_flat` 各自保留，但推断循环内部改为调用 `_run_gdino_batch`，差异逻辑（亮度过滤、文件布局）在调用方中处理。这将每个函数的长度从 ~80 行压缩到 ~30 行。

---

### P3 — 结构性改进

#### 3.7 Dataset 基类

**现状**

项目有 4 个 Dataset 类，各自实现完整的 `__init__`/`__len__`/`__getitem__`：

| 类 | 文件 | 行号 | 路径字段名 |
|---|------|------|-----------|
| `FrameDataset` | `scripts/generate_pseudo_labels.py` | 49–60 | `self.paths` |
| `TreeDataset` | `gdino_dinov2_svm/03_extract_embeddings.py` | 103–122 | `self.samples` |
| `TDUSDataset` | `gdino_dinov2_svm/03_extract_embeddings.py` | 125–149 | `self.samples` |
| `_ImageDataset` | `scripts/benchmark_pipeline.py` | 138–164 | `self.paths` |

注意：`FrameDataset` 和 `_ImageDataset` 使用 `self.paths`，`TreeDataset`/`TDUSDataset` 使用 `self.samples`，字段名不统一。

所有类的 `__len__` 均为一行 `return len(self.paths/self.samples)`，`__getitem__` 逻辑各不相同，基类能共享的仅此而已。

**成本收益分析**

PyTorch 的 `torch.utils.data.Dataset` 已是所有自定义 Dataset 的基类（提供 `__add__`、`__getitem__` 抽象接口），本项目四个类均继承自它。在此之上再抽 `_SamplesDataset` 只能共享一行 `__len__`，但引入了：
- 字段名需统一（`paths` vs `samples`），有可能改变现有代码的属性引用；
- `datasets.py` 新文件需要测试覆盖；
- `gdino_dinov2_svm/` 的 `TreeDataset`/`TDUSDataset` 因 import 路径复杂，迁移收益极低。

**结论**：Dataset 基类抽取的收益（消除 1 行 `__len__` 重复 × 4 处）远小于改动成本。**建议将此项移出 P3，降级为"暂不做"或合并到更大的模块重组中**。若仍要实施，至少应先统一字段名（全部改为 `self.paths`），再提取基类。

**若决定实施**，在 `scripts/datasets.py` 中定义（统一字段名为 `paths`）：

```python
from torch.utils.data import Dataset

class _PathsDataset(Dataset):
    """最小化基类：持有 self.paths 列表，提供 __len__。子类必须实现 __getitem__。"""
    paths: list

    def __len__(self):
        return len(self.paths)

class FrameDataset(_PathsDataset): ...
class _ImageDataset(_PathsDataset): ...
```

`TreeDataset` 和 `TDUSDataset` 因属于实验脚本（`gdino_dinov2_svm/`），迁移优先级极低，可暂不处理。

#### 3.8 CLI 参数解析公共参数漂移

**现状**

4 个 CLI 入口各自定义公共参数：

| 文件 | 函数 | 公共参数 |
|------|------|---------|
| `scripts/predict_pipeline.py` | `build_parser()` | `--detector`, `--yolo-checkpoint`, `--rf-detr-checkpoint` |
| `scripts/benchmark_pipeline.py` | `build_parser()` | `--detector`, `--yolo-checkpoint`, `--rf-detr-checkpoint`, `--device` |
| `scripts/classify_multi_tree.py` | `build_arg_parser()` | `--detector`, `--yolo-model`（注意：名称与其他脚本不同！）, `--yolo-conf`, `--yolo-imgsz` |
| `scripts/generate_pseudo_labels.py` | `parse_args_flat()` | `--device`, `--batch-size`, `--box-threshold`, `--score-thr` |

注意 `classify_multi_tree.py` 使用 `--yolo-model` 而其他脚本使用 `--yolo-checkpoint`，这是一个已有的不一致。

**方案**

**模式选择**：argparse 共享参数有两种主流方案：
- **`parents=` 模式**：创建 `add_help=False` 的父 parser，子 parser 通过 `parents=[base_parser]` 继承。适合子命令（subparsers）场景，或多个 CLI 确实需要完全相同的参数集合。
- **工厂函数模式**（`add_detector_args(parser)`）：将 `add_argument(...)` 调用封装为函数，各 parser 主动调用。适合参数集有差异的独立 CLI 脚本。

本项目 4 个 CLI 脚本参数集各有差异（`classify_multi_tree` 有 `--yolo-conf`/`--yolo-imgsz`，`generate_pseudo_labels` 有 `--batch-size` 等），且不构成子命令关系，**工厂函数是正确选择**，`parents=` 会引入过多强制耦合。

在 `scripts/cli_common.py`（新建）中定义共享参数工厂：

```python
def add_detector_args(parser: argparse.ArgumentParser) -> None:
    """向 parser 添加检测器相关公共参数（所有脚本统一默认值）。"""
    parser.add_argument("--detector", default="gdino",
                        choices=["gdino", "yolo", "rf-detr"])
    parser.add_argument("--yolo-checkpoint", default=None)
    parser.add_argument("--rf-detr-checkpoint", default=None)
    parser.add_argument("--device", default=None,
                        help="推断设备（默认自动检测 cuda/cpu）")

def add_gdino_threshold_args(parser: argparse.ArgumentParser) -> None:
    """GDino 阈值参数（generate_pseudo_labels / benchmark 共用）。"""
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--iou-thr", type=float, default=0.45)
```

同时将 `classify_multi_tree.py` 的 `--yolo-model` 重命名为 `--yolo-checkpoint` 以统一接口（需同步更新调用方脚本和文档）。

**测试同步要求（遗漏项）**：`tests/test_classify_multi_tree.py:23` 中有如下断言：
```python
assert "tree_yolo26s_halfres" in args.yolo_model
```
重命名为 `--yolo-checkpoint` 后，该断言将访问 `args.yolo_model`（不存在的属性），测试会 `AttributeError` 失败。**必须在同一 PR 内同步更新此测试**，将其改为 `args.yolo_checkpoint`。

---

## 4. 权衡分析

### 收益

| 优先级 | 改动 | 收益 |
|--------|------|------|
| P0 | 消除 `gdino_preprocess` 冗余 | 消除签名分叉风险；修改一处即可全局生效 |
| P0 | 移除 `_ensure_pp_imports` | 代码可读性大幅提升；IDE 可跳转；消除运行时绑定失败风险 |
| P1 | 新建 `yolo_io.py` | 标签 I/O 逻辑只需测试一次；`diff_gdino_yolo` 和 `self_train` 的同名函数保持同步 |
| P1 | 统一 `make_crop` | 消除归一化/像素坐标分支的隐性不一致；易于添加 augmentation |
| P2 | `classify_multi_tree` 复用检测函数族 | 阈值和 NMS 逻辑保持一致，不再有两套配置 |
| P2 | `run_inference` 与 `run_inference_flat` 共享内核 | 减少重复；bugfix 只需改一处 |
| P3 | Dataset 基类 | 样板代码减少；`__len__` 只有一份 |
| P3 | CLI 公共参数 | 默认值只有一份，参数名称保持一致 |

### 风险

| 风险 | 严重性 | 缓解 |
|------|--------|------|
| `classify_multi_tree.gdino_preprocess` 签名与 `gdino_utils` 不同（多返回 size），改后行为可能变化 | 中 | 新增 `gdino_preprocess_with_size`（见 3.1 节实现注意）；逐函数测试 |
| `_ensure_pp_imports` 被移除后，现有测试中的 monkeypatch 目标路径需更新 | 中 | 统一改为 `patch("benchmark_pipeline.<fn_name>")`；需审查 `test_benchmark_pipeline.py` 全文，确认所有 mock 路径 |
| **[新增] `classify_multi_tree` 复用 `detect_gdino` 的接口不匹配**：`detect_gdino` 返回 xyxy 像素坐标，现有分类流程全程使用归一化坐标（`make_crop(pil, box_norm)` 和结果字典的 `box_norm` 字段）；直接替换会导致裁剪坐标完全错误（静默 bug） | **高** | 实施方案 A（见 3.5 节）：`classify_detections` 改为接受像素坐标，内部统一使用 `make_crop_xyxy`，结果字典中的 `box_norm` 从像素坐标反算归一化值 |
| `classify_multi_tree` 的手写 NMS（IoU 阈值 0.5，纯 Python）与 `predict_pipeline.detect_gdino` 的 `torchvision.ops.nms`（IoU 阈值 `GDINO_IOU_THR=0.45`）行为不完全相同 | 中 | 用相同测试图像对比两种实现的输出，确认等价后再迁移；**阈值差异（0.5 vs 0.45）会导致 `classify_multi_tree` 历史结果与迁移后不完全一致**，需要在 PR 描述中明确说明此行为变更 |
| `classify_multi_tree.detect_and_classify` 使用 `score_thr=0.3`，而 `predict_pipeline.detect_gdino` 默认 `GDINO_SCORE_THR=0.35`，迁移后低置信度框（0.3–0.35 分段）将被过滤 | 中 | 迁移后分数阈值从 0.3 升为 0.35 是**有意的对齐**，但会影响历史输出结果；必须在同一 PR 描述中与 NMS 阈值变更并列说明 |
| `--yolo-model` → `--yolo-checkpoint` 重命名：**`tests/test_classify_multi_tree.py:23` 的 `args.yolo_model` 访问会 AttributeError 失败** | 中（遗漏） | 必须在同一 PR 内将测试中的 `args.yolo_model` 改为 `args.yolo_checkpoint`；同时保留 `--yolo-model` 作为 deprecated alias |
| `03_extract_embeddings.py` 属实验脚本，import path 修改可能影响独立运行 | 低 | 保留 `sys.path` 调整片段，仅删除函数体 |
| **[新增] Dataset 基类字段名不统一**：`FrameDataset`/`_ImageDataset` 用 `self.paths`，`TreeDataset`/`TDUSDataset` 用 `self.samples`；若强行抽取基类必须统一字段名，影响所有属性访问 | 低-中 | 建议暂不实施 Dataset 基类（见 3.7 节结论），或先统一字段名为单独 PR |

---

## 5. 风险与迁移策略

### 5.1 测试覆盖现状

当前测试文件：`tests/test_benchmark_pipeline.py`、`tests/test_classify_multi_tree.py`、`tests/test_gdino_utils.py`、`tests/test_predict_pipeline.py` 等 13 个。

在执行本提案之前，应确认（经阅读测试文件后的实际状态）：
- `test_gdino_utils.py`：**已覆盖** `gdino_preprocess`（含 7 个用例，包括 round-trip 与参考实现对比）；`gdino_preprocess_with_size` 新增后需补充测试。
- `test_predict_pipeline.py`：测试范围为 argparse 层（mock 了所有重型依赖），**未覆盖** `make_crop` 逻辑本身；迁移到 `image_utils.py` 后必须在 `tests/test_image_utils.py` 中补充。
- `test_benchmark_pipeline.py`：通过模块级 `patch` 替换 `gdino_forward_batch` 等全局变量实现 mock；移除 `_ensure_pp_imports` 后，这些 patch 路径（`patch("benchmark_pipeline.gdino_forward_batch", ...)`）仍然有效（直接 import 同样会把名字绑定到 `benchmark_pipeline` 模块命名空间），**无需修改 patch 路径**，但需要删除测试中任何直接调用或验证 `_ensure_pp_imports` 存在的代码。
- `test_classify_multi_tree.py:23`：断言 `"tree_yolo26s_halfres" in args.yolo_model`，**需在 P3 重命名 PR 中同步更新**。
- `test_self_train.py`：**已覆盖** `load_yolo_xyxy` 和 `write_yolo_from_xyxy`；迁移到 `yolo_io.py` 后需更新 import 路径。

新建模块需补充的测试：
- `tests/test_yolo_io.py`：覆盖 `write_yolo_labels` → `load_yolo_xyxy` round-trip，空文件/空 tensor 边界，5 列格式验证。
- `tests/test_image_utils.py`：覆盖 `make_crop_xyxy` 和 `make_crop_norm` 的等价性、边界截断、极小框保护。

### 5.2 迁移顺序（建议）

迁移应按优先级顺序进行，每步独立 PR，保持主干可运行：

1. **Step 1（P0-a）**：在 `gdino_utils.py` 新增 `gdino_preprocess_with_size`（含 `_compute_resize` 私有辅助）；补充 `test_gdino_utils.py` 中对应测试；更新 `classify_multi_tree.py` 的 import，删除其本地定义。确认 `test_classify_multi_tree.py` 全部通过。
2. **Step 2（P0-b）**：将 `benchmark_pipeline.py` 的全局 `None` 变量和 `_ensure_pp_imports` 函数替换为顶层直接 import；删除 `_preload_batches`（第 449 行）、`_run_timed_bench`（第 511 行）、`_run_detector_loop`（第 246 行）中的 `_ensure_pp_imports()` 调用；patch 路径无需修改，但需删除测试中任何断言 `_ensure_pp_imports` 存在的代码。
3. **Step 3（P1-a）**：新建 `scripts/yolo_io.py`；迁移 `generate_pseudo_labels`（第 84–103 行）、`self_train`（第 67–98 行）、`diff_gdino_yolo`（第 25–39 行）的 I/O 函数；更新 `test_self_train.py` 的 import 路径；补充 `tests/test_yolo_io.py`（round-trip + 边界）。
4. **Step 4（P1-b）**：新建 `scripts/image_utils.py`；迁移 `make_crop`；更新 `03_extract_embeddings.py`；补充 `tests/test_image_utils.py`（等价性 + 边界）。
5. **Step 5（P2）**：重构 `classify_multi_tree.py` 检测入口（注意接口不匹配问题，参见 3.5 节方案 A）；提取 `generate_pseudo_labels._run_gdino_batch` 内核（`num_workers` 参数化，`run_inference` 传 2，`run_inference_flat` 传 0，保留原有差异）。**此步涉及两处行为变更，必须在 PR 描述中同时说明：① NMS IoU 阈值从 0.5 变为 0.45；② 分数阈值 `score_thr` 从 0.3 变为 `GDINO_SCORE_THR=0.35`。**
6. **Step 6（P3-a）**：新建 `scripts/cli_common.py`；迁移 CLI 公共参数；将 `classify_multi_tree.py` 的 `--yolo-model` 重命名为 `--yolo-checkpoint`（保留 deprecated alias）；**同步更新 `tests/test_classify_multi_tree.py:23`**。
7. **Step 7（P3-b，可选）**：若决定实施 Dataset 基类，先统一字段名为 `self.paths`，再新建 `scripts/datasets.py`；否则跳过。

### 5.3 向后兼容

- 所有脚本的 CLI 接口在 P0/P1/P2 阶段保持不变。
- P3 的 `--yolo-model` 重命名建议在一个独立 PR 中处理，并保留 deprecated alias 至少一个月。
- `gdino_dinov2_svm/03_extract_embeddings.py` 的改动限于删除本地函数定义和添加 import，不影响其 CLI 和输出格式。

---

## 6. 改后模块结构

```
scripts/
├── gdino_utils.py          # gdino_preprocess, gdino_preprocess_with_size, _compute_resize
├── image_utils.py          # make_crop_xyxy, make_crop_norm                   [新建 P1-b]
├── yolo_io.py              # load_yolo_xyxy, xyxy_to_yolo, write_yolo_labels  [新建 P1-a]
├── cli_common.py           # add_detector_args, add_gdino_threshold_args       [新建 P3-a]
├── datasets.py             # _PathsDataset, FrameDataset, _ImageDataset        [新建 P3-b，可选]
├── predict_pipeline.py     # detect_gdino/yolo/rf_detr, make_crop → image_utils
├── benchmark_pipeline.py   # 直接 import from predict_pipeline（无 _ensure_pp_imports）
├── classify_multi_tree.py  # 复用 detect_gdino/detect_yolo（注意坐标系转换），复用 make_crop_xyxy
├── generate_pseudo_labels.py  # 复用 yolo_io, gdino_utils; 提取 _run_gdino_batch
├── self_train.py           # 复用 yolo_io
└── diff_gdino_yolo.py      # 复用 yolo_io
```

新增文件均为纯工具模块，无 CLI 入口，无副作用，易于单独测试。`datasets.py` 标记为可选，仅在团队判断收益足够时实施（见 3.7 节分析）。
