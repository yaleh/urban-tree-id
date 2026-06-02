"""共享 CLI 参数工厂函数。

所有 CLI 脚本的公共参数应从此处添加，不得在各脚本中重复定义默认值。

默认值说明：
- --detector: 所有脚本均默认 "gdino"
- --yolo-checkpoint: 各脚本此前各有不同默认值：
    classify_multi_tree.py 用 "runs/detect/tree_yolo26s_halfres/weights/best.pt"，
    predict_pipeline.py / benchmark_pipeline.py 用 None。
    classify_multi_tree.py 通过独立的 --yolo-checkpoint 参数保留自身默认值，
    工厂函数此处统一为 None（预测/评估场景的合理值）。
- --device: generate_pseudo_labels.py 的 main() 和 parse_args_flat() 均在
    argparse 默认值中使用 torch.cuda.is_available() 动态判断。
    predict_pipeline / benchmark_pipeline 在运行时检测，默认 None。
    工厂函数统一为 None（运行时自动检测），避免在 import 时触发 CUDA 初始化。
- --box-threshold / --text-threshold / --score-thr / --iou-thr:
    所有脚本一致：0.30 / 0.25 / 0.35 / 0.45。
"""
import argparse


def add_detector_args(parser: argparse.ArgumentParser) -> None:
    """添加检测器选择和模型路径参数。

    覆盖参数：--detector, --yolo-checkpoint, --rf-detr-checkpoint, --device
    """
    parser.add_argument(
        "--detector",
        choices=["gdino", "yolo", "rf-detr"],
        default="gdino",
        help="检测后端（默认：gdino）",
    )
    parser.add_argument(
        "--yolo-checkpoint",
        dest="yolo_checkpoint",
        default=None,
        help="YOLO 模型权重路径",
    )
    parser.add_argument(
        "--rf-detr-checkpoint",
        dest="rf_detr_checkpoint",
        default=None,
        help="RF-DETR 模型权重路径",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="推断设备（cuda/cpu），默认自动检测",
    )


def add_gdino_threshold_args(parser: argparse.ArgumentParser) -> None:
    """添加 GDino 推断阈值参数。

    覆盖参数：--box-threshold, --text-threshold, --score-thr, --iou-thr
    默认值与 generate_pseudo_labels.py 及 predict_pipeline.py 中的常量保持一致。
    """
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--iou-thr", type=float, default=0.45)
