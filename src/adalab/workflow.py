"""
实验工作流与后端统一入口。

本模块负责：
- 解析实验配置文件
- 构建实验目录结构
- 调度数据准备、模型构造与训练流程
- 统一保存模型、监控数据与评估结果

该模块是 adalab 后端对外暴露的主要接口，
前端与实验脚本应通过本模块启动完整实验流程。
"""

import json
import os
import joblib
import time

from typing import Optional, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from .monitor import BoostMonitor
from .data import DataPreparation
from .patch import AdaBoostClfWithMonitor
from .evaluation import val_after_train_parallel
from .io import dump_compressed

# 兼容旧 joblib 路径
import sys

try:
    from adalab.monitor import BoostMonitor

    sys.modules["src.monitor"] = sys.modules["adalab.monitor"]
except Exception:
    pass


@dataclass
class DataSplit:
    """数据划分结果的数据结构。

    用于统一封装一次实验中使用的训练/测试数据及其噪声标注信息，
    作为 workflow 各阶段之间传递的数据载体。

    Attributes:
        X_train (np.ndarray): 训练集特征矩阵，形状为 (n_train, d)。
        X_test (np.ndarray): 测试集特征矩阵，形状为 (n_test, d)。
        y_train (np.ndarray): 训练集标签向量。
        y_test (np.ndarray): 测试集标签向量。
        noise_idx (np.ndarray): 训练集中被标记为噪声样本的索引。
        clean_idx (np.ndarray): 训练集中被标记为干净样本的索引。
        prep (DataPreparation): 生成该数据划分的 DataPreparation 实例。
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    noise_idx: np.ndarray
    clean_idx: np.ndarray
    prep: DataPreparation


@dataclass(frozen=True)
class ExperimentPaths:
    """实验目录布局的只读描述对象。

    该类用于集中管理一次实验的目录结构与核心文件路径，
    便于前端或其他模块按约定位置读取实验产物。

    Attributes:
        exp_dir (Path): 实验根目录路径。
        ckpt_dir (Path): 训练过程 checkpoint 存放目录。
        result_dir (Path): 最终结果与模型文件存放目录。
        config_path (Path): 保存实验配置文件的路径。
        result_csv (Path): 最终导出的监控结果 CSV 文件路径。
    """

    exp_dir: Path
    ckpt_dir: Path
    result_dir: Path
    config_path: Path
    result_csv: Path

    @staticmethod
    def create(exp_name: str, base_dir: str = "experiments") -> "ExperimentPaths":
        """创建实验目录结构并返回路径描述对象。

        若目标实验目录已存在且不为空，则自动添加时间戳以避免覆盖。

        Args:
            exp_name (str): 实验名称。
            base_dir (str, optional): 实验根目录，默认 "experiments"。

        Returns:
            ExperimentPaths: 创建完成的实验路径布局对象。
        """
        exp_dir = Path(safe_exp_dir(exp_name, base_dir))
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Workflow] experiment dir created: {exp_dir}")

        ckpt_dir = exp_dir / "checkpoints"
        result_dir = exp_dir / "results"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        return ExperimentPaths(
            exp_dir=exp_dir,
            ckpt_dir=ckpt_dir,
            result_dir=result_dir,
            config_path=exp_dir / "config.json",
            result_csv=result_dir / "final_results.csv",
        )


@dataclass(frozen=True)
class ArtifactPaths:
    """实验产物文件路径的统一描述。

    用于描述模型与监控器在磁盘上的保存位置，
    同时区分未压缩与压缩版本，方便前端加载。

    Attributes:
        raw_clf (Path): 未压缩的模型 joblib 文件路径。
        compressed_clf (Path): 压缩后的模型 joblib.xz 文件路径。
        raw_monitor (Optional[Path]): 未压缩的 BoostMonitor joblib 路径（若存在）。
        compressed_monitor (Optional[Path]): 压缩后的 BoostMonitor joblib.xz 路径（若存在）。
        monitor_csv (Optional[Path]): 监控结果导出的 CSV 文件路径（若存在）。
    """

    raw_clf: Path
    compressed_clf: Path
    raw_monitor: Optional[Path]
    compressed_monitor: Optional[Path]
    monitor_csv: Optional[Path]

    @staticmethod
    def from_layout(layout: ExperimentPaths, has_monitor: bool) -> "ArtifactPaths":
        """根据实验目录布局生成产物路径描述。

        Args:
            layout (ExperimentPaths): 实验目录布局对象。
            has_monitor (bool): 是否启用了 BoostMonitor。

        Returns:
            ArtifactPaths: 对应当前实验的产物路径集合。
        """
        raw_clf = layout.result_dir / "model.joblib"
        compressed_clf = Path(str(raw_clf) + ".xz")

        if has_monitor:
            raw_monitor = layout.result_dir / "monitor.joblib"
            compressed_monitor = Path(str(raw_monitor) + ".xz")
            monitor_csv = layout.result_csv
        else:
            raw_monitor = None
            compressed_monitor = None
            monitor_csv = None

        return ArtifactPaths(
            raw_clf=raw_clf,
            compressed_clf=compressed_clf,
            raw_monitor=raw_monitor,
            compressed_monitor=compressed_monitor,
            monitor_csv=monitor_csv,
        )


def load_config(config_path):
    """加载实验配置文件。

    Args:
        config_path (str or Path): 配置文件（JSON）路径。

    Returns:
        dict: 解析后的配置字典。
    """
    with open(config_path, "r") as f:
        return json.load(f)


def is_effectively_empty(dir_path):
    """判断实验目录是否在逻辑上为空。

    若目录不存在，或仅包含 config.json 文件，则认为目录为空，
    可安全复用为新的实验目录。

    Args:
        dir_path (str): 目录路径。

    Returns:
        bool: 若目录可视为“空”，返回 True，否则返回 False。
    """
    if not os.path.exists(dir_path):
        return True

    for root, dirs, files in os.walk(dir_path):
        if root == dir_path and len(files) == 1 and files[0] == "config.json":
            continue
        else:
            if files:
                return False
    return True


def safe_exp_dir(exp_name, base_dir="experiments"):
    """生成安全的实验目录路径。

    当目标实验目录已存在且不为空时，
    自动在实验名称后附加时间戳以避免覆盖原有实验。

    Args:
        exp_name (str): 实验名称。
        base_dir (str, optional): 实验根目录，默认 "experiments"。

    Returns:
        str: 可安全使用的实验目录路径。
    """
    exp_dir = os.path.join(base_dir, exp_name)

    if is_effectively_empty(exp_dir):
        # 目录不存在 or “逻辑上空”，直接使用
        return exp_dir

    # 否则自动加 timestamp
    ts = time.strftime("%Y%m%d_%H%M%S")
    new_exp_name = f"{exp_name}_{ts}"
    new_exp_dir = os.path.join(base_dir, new_exp_name)

    print(f"[WARNING] 实验目录 '{exp_dir}' 非空，将自动切换至新目录 '{new_exp_dir}'")

    return new_exp_dir


def prep_data_from_config(config):
    """根据配置文件准备实验数据。

    该函数负责解析数据相关配置，构造 DataPreparation，
    并生成训练/测试数据及噪声索引。

    Args:
        config (dict): 实验配置字典。

    Returns:
        DataSplit: 包含数据划分结果及相关元信息的对象。
    """
    data_cfg = config["data"]

    # 读取 HOG 参数
    hog_cfg = data_cfg.get("hog_params", {})
    hog_orient = hog_cfg.get("orientations", 9)
    hog_ppc = tuple(hog_cfg.get("pixels_per_cell", (4, 4)))
    hog_cpb = tuple(hog_cfg.get("cells_per_block", (2, 2)))

    # 读取 HU 参数
    hu_cfg = data_cfg.get("hu_params", {})
    hu_log_scale = hu_cfg.get("log_scale", True)

    # 构建 DataPreparation
    prep = DataPreparation(
        noise_config=data_cfg["noise_config"],
        test_size=data_cfg["test_size"],
        use_feature=data_cfg.get("use_feature", "original"),
        random_state=data_cfg["random_state"],
        hog_orientations=hog_orient,
        hog_pixels_per_cell=hog_ppc,
        hog_cells_per_block=hog_cpb,
        hu_log_scale=hu_log_scale,
    )
    X_train, X_test, y_train, y_test, noise_idx, clean_idx = prep.prepare()
    return DataSplit(X_train, X_test, y_train, y_test, noise_idx, clean_idx, prep)


def build_experiment(config_path):
    """构建一次完整的实验对象。

    包括：
    - 加载配置文件
    - 创建实验目录结构
    - 准备数据
    - 构造模型与（可选）BoostMonitor

    Args:
        config_path (str or Path): 实验配置文件路径。

    Returns:
        tuple:
            - clf: 构造好的分类模型（未训练）。
            - monitor: BoostMonitor 实例，若未启用则为 None。
            - split (DataSplit): 数据划分结果。
            - layout (ExperimentPaths): 实验目录布局对象。
    """
    config = load_config(config_path)

    exp_name = config["experiment"]["name"]
    base_dir = config["experiment"].get("base_dir", "experiments")
    layout = ExperimentPaths.create(exp_name, base_dir=base_dir)
    with open(layout.config_path, "w") as fw:
        json.dump(config, fw, indent=4)

    split = prep_data_from_config(config)

    # === 构造 Monitor 和 模型===
    monitor_cfg = config["monitor"]
    use_monitor = monitor_cfg.get("use_monitor", True)
    if not use_monitor:
        print("[MODEL] Using original AdaBoost without BoostMonitor")
        model_cfg = config["model"]
        base = DecisionTreeClassifier(**model_cfg["estimator"])

        clf = AdaBoostClassifier(
            estimator=base,
            n_estimators=model_cfg["n_estimators"],
            learning_rate=model_cfg["learning_rate"],
            random_state=model_cfg["random_state"],
        )

        return (
            clf,
            None,
            split,
            layout,
        )

    print("[MODEL] Using AdaBoost with BoostMonitor enabled")
    monitor = BoostMonitor(
        noise_indices=split.noise_idx,
        clean_indices=split.clean_idx,
        is_data_noisy=monitor_cfg["is_data_noisy"],
        checkpoint_interval=monitor_cfg["checkpoint_interval"],
        checkpoint_prefix=str(layout.ckpt_dir),
    )

    model_cfg = config["model"]
    base = DecisionTreeClassifier(**model_cfg["estimator"])

    clf = AdaBoostClfWithMonitor(
        _monitor=monitor,
        X_val=split.X_test,
        y_val=split.y_test,
        estimator=base,
        n_estimators=model_cfg["n_estimators"],
        learning_rate=model_cfg["learning_rate"],
        random_state=model_cfg["random_state"],
    )

    return (clf, monitor, split, layout)


def train_and_save(
    config_path: str,
) -> tuple[
    Union[AdaBoostClassifier, AdaBoostClfWithMonitor],
    Optional[BoostMonitor],
    DataSplit,
    ExperimentPaths,
    ArtifactPaths,
]:
    """执行完整的实验流程并保存所有产物。

    该函数是后端对外暴露的主要入口，完成以下步骤：
    - 构建实验与数据
    - 训练模型
    - （可选）训练后统一验证
    - 保存模型与监控器（含压缩与未压缩版本）
    - 导出监控结果 CSV

    Args:
        config_path (str): 实验配置文件路径。

    Returns:
        tuple:
            - clf: 训练完成的分类模型。
            - monitor: BoostMonitor 对象，若未启用则为 None。
            - split (DataSplit): 数据划分结果。
            - layout (ExperimentPaths): 实验目录布局对象。
            - result_paths (ArtifactPaths): 所有实验产物的路径描述。
    """
    # 构建实验
    (clf, monitor, split, layout) = build_experiment(config_path)
    has_monitor = True if monitor else False
    result_paths = ArtifactPaths.from_layout(layout, has_monitor)

    # Training
    print("[Workflow] \033[33mTraining Started...\033[0m")
    clf.fit(split.X_train, split.y_train)
    print("[Workflow] \033[33mTraining Finished!\033[0m")

    config = load_config(config_path)
    monitor_conf = config["monitor"]
    val_freq = monitor_conf.get("val_freq", 10)
    val_n_jobs = monitor_conf.get("val_n_jobs", 4)

    # val and save monitor results
    if monitor is not None:
        alphas = np.asarray(monitor.alpha_history)

        print(
            "[workflow] \033[33mValidiating on training data after model fitted...\033[0m"
        )
        acc_curv_train, f1_curv_train, val_idx_train = val_after_train_parallel(
            clf, alphas, split.X_train, split.y_train, val_freq, val_n_jobs
        )
        monitor.acc_on_train_data = acc_curv_train.tolist()
        monitor.f1_on_training_data = f1_curv_train.tolist()
        monitor.val_idx = val_idx_train.tolist()
        print("[workflow] \033[33mValidiation on training data finished!\033[0m")

        print(
            "[workflow] \033[33mValidiating on testing data after model fitted...\033[0m"
        )
        acc_curv_test, f1_curv_test, _ = val_after_train_parallel(
            clf, alphas, split.X_test, split.y_test, val_freq, val_n_jobs
        )
        monitor.val_acc_history = acc_curv_test.tolist()
        monitor.val_f1_history = f1_curv_test.tolist()
        print("[workflow] \033[33mValidiation on testing data finished!\033[0m")

        monitor.dump(str(layout.result_csv))
        # raw_monitor_path = os.path.join(layout.result_dir, "monitor.joblib")
        joblib.dump(monitor, result_paths.raw_monitor)
        print(
            f"[Workflow] Uncompressed monitor joblib saved to : {result_paths.raw_monitor}"
        )
        compressed_monitor_path = dump_compressed(
            monitor, str(result_paths.compressed_monitor)
        )
        print(
            f"[Workflow] compressed monitor joblib saved to : {compressed_monitor_path}"
        )

    # 保存未压缩 joblib
    joblib.dump(clf, result_paths.raw_clf)

    print(f"[Workflow] Uncompressed model joblib saved to : {result_paths.raw_clf}")

    # 保存压缩版
    compressed_clf_path = dump_compressed(clf, str(result_paths.compressed_clf))

    print(f"[Workflow] compressed model joblib saved to : {compressed_clf_path}")

    return (clf, monitor, split, layout, result_paths)
