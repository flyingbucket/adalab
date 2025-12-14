import json
import os
import joblib
import time

from typing import Optional
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
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    noise_idx: np.ndarray
    clean_idx: np.ndarray
    prep: DataPreparation


@dataclass(frozen=True)
class ExperimentPaths:
    exp_dir: Path
    ckpt_dir: Path
    result_dir: Path
    config_path: Path
    result_csv: Path

    @staticmethod
    def create(exp_name: str, base_dir: str = "experiments") -> "ExperimentPaths":
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
    raw_clf: Path
    compressed_clf: Path
    raw_monitor: Optional[Path]
    compressed_monitor: Optional[Path]
    monitor_csv: Optional[Path]

    @staticmethod
    def from_layout(layout: ExperimentPaths, has_monitor: bool) -> "ArtifactPaths":
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
    with open(config_path, "r") as f:
        return json.load(f)


def is_effectively_empty(dir_path):
    if not os.path.exists(dir_path):
        return True

    for root, dirs, files in os.walk(dir_path):
        if files:
            return False
    return True


def safe_exp_dir(exp_name, base_dir="experiments"):
    """
    根据 exp_name 构造实验目录，如果已有内容则自动添加时间戳
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


def train_and_save(config_path: str):
    """
    构建实验、训练模型、保存 model / monitor（含未压缩和压缩版）。

    返回:
        clf: 训练好的分类模型
        monitor: BoostMonitor 对象
        data:Tuple of (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        paths: 包含所有输出文件路径的字典
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

    # 保存 monitor 结果
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
