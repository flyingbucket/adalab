import json
import os
import joblib
import lzma
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.monitor import BoostMonitor
from src.patch import AdaBoostClfWithMonitor


def prepare_data(noise_ratio=0.05, test_size=0.2, random_state=42):
    """
    下载 MNIST，并按指定比例添加标签噪声。
    自动返回：
        - X_train, X_test
        - y_train (含噪声) , y_test
        - train_noise_indices  (训练集内部噪声索引)
        - train_clean_indices  (训练集内部干净索引)
    若 noise_ratio=0，则返回完全干净的数据。

    Parameters
    ----------
    noise_ratio : float
        噪声比例（0 ~ 1），表示标签噪声的比例。
        若为 0，则不添加标签噪声。

    test_size : float
        train_test_split 的测试集占比

    random_state : int
        随机种子

    Returns
    -------
    X_train, X_test : ndarray
    y_train, y_test : ndarray
    train_noise_indices : ndarray (训练集内部的噪声样本位置)
    train_clean_indices : ndarray
    """

    print("Downloading MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.int64)
    X = X / 255.0

    n_samples = len(y)

    # -----------------------------------------
    # Case 1: 不添加噪声，返回原始数据
    # -----------------------------------------
    if noise_ratio <= 0:
        print("No noise added, returning clean dataset.")

        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, np.arange(n_samples), test_size=test_size, random_state=random_state
        )

        # 训练集全部是 clean
        train_noise_indices = np.array([], dtype=int)
        train_clean_indices = np.arange(len(y_train))

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            train_noise_indices,
            train_clean_indices,
        )

    # Case 2: 添加噪声
    n_noisy = int(n_samples * noise_ratio)
    rng = np.random.default_rng(random_state)

    noise_indices = rng.choice(n_samples, n_noisy, replace=False)

    y_noisy = y.copy()
    y_noisy[noise_indices] = rng.integers(0, 10, size=n_noisy)

    print(f"Injected label noise: {noise_ratio * 100:.1f}% ({n_noisy} samples)")

    # train/test split，保留原始索引
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y_noisy, np.arange(n_samples), test_size=test_size, random_state=random_state
    )

    # 计算训练集内部噪声位置
    train_noise_mask = np.isin(train_idx, noise_indices)
    train_noise_indices = np.where(train_noise_mask)[0]
    train_clean_indices = np.where(~train_noise_mask)[0]

    print(f"Training set noise samples = {len(train_noise_indices)}")

    return (X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices)


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


def safe_exp_dir(exp_name):
    """
    根据 exp_name 构造实验目录，如果已有内容则自动添加时间戳
    """
    base_dir = "experiments"
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


def build_experiment(config_path):
    config = load_config(config_path)

    exp_name = config["experiment"]["name"]
    raw_exp_dir = os.path.join("experiments", exp_name)
    exp_dir = safe_exp_dir(raw_exp_dir)

    # === 自动创建目录结构 ===
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    result_dir = os.path.join(exp_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print(f"实验目录已创建: {exp_dir}")

    # === 1. 数据准备 ===
    data_cfg = config["data"]
    (
        X_train,
        X_test,
        y_train,
        y_test,
        noise_idx,
        clean_idx,
    ) = prepare_data(
        noise_ratio=data_cfg["noise_ratio"],
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )

    # === 2. 构造 Monitor（自动拼 checkpoint 前缀） ===
    monitor_cfg = config["monitor"]

    monitor = BoostMonitor(
        noise_indices=noise_idx,
        clean_indices=clean_idx,
        is_data_noisy=monitor_cfg["is_data_noisy"],
        checkpoint_interval=monitor_cfg["checkpoint_interval"],
        checkpoint_prefix=ckpt_dir,
    )

    # === 3. 构造模型 ===
    model_cfg = config["model"]
    base = DecisionTreeClassifier(**model_cfg["estimator"])

    clf = AdaBoostClfWithMonitor(
        _monitor=monitor,
        X_val=X_test,
        y_val=y_test,
        estimator=base,
        n_estimators=model_cfg["n_estimators"],
        learning_rate=model_cfg["learning_rate"],
        random_state=model_cfg["random_state"],
    )

    # 返回路径供保存
    result_csv = os.path.join(result_dir, "final_results.csv")

    return (
        clf,
        monitor,
        (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        result_csv,
        result_dir,
    )


def dump_compressed(obj, filepath: str):
    """
    使用 lzma 压缩并保存任意 Python 对象
    """
    compressed_path = filepath + ".xz"

    with lzma.open(compressed_path, "wb") as f:
        joblib.dump(obj, f)

    print(f"已保存并压缩到: {compressed_path}")
    return compressed_path


def load_compressed(filepath: str):
    """
    自动读取 .xz 压缩的 joblib 文件
    """
    with lzma.open(filepath, "rb") as f:
        obj = joblib.load(f)
    print(f"已成功读取: {filepath}")
    return obj


def train_and_save(config_path: str):
    """
    构建实验、训练模型、保存 model / monitor（含未压缩和压缩版）。

    返回:
        clf: 训练好的分类模型
        monitor: BoostMonitor 对象
        data:Tuple of (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        paths: 包含所有输出文件路径的字典
    """
    # ========= 1. 构建实验 =========
    (
        clf,
        monitor,
        (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        result_csv,
        result_dir,
    ) = build_experiment(config_path)

    print("开始训练...")
    clf.fit(X_train, y_train)
    print("训练完成！")

    # 保存 monitor 结果 CSV
    monitor.dump(result_csv)

    # ========= 2. 保存未压缩 joblib =========
    raw_clf_path = os.path.join(result_dir, "model.joblib")
    raw_monitor_path = os.path.join(result_dir, "monitor.joblib")

    joblib.dump(clf, raw_clf_path)
    joblib.dump(monitor, raw_monitor_path)

    print(f"未压缩模型保存到: {raw_clf_path}")
    print(f"未压缩监控器保存到: {raw_monitor_path}")

    # ========= 3. 保存压缩版 =========
    compressed_clf_path = dump_compressed(clf, raw_clf_path)
    compressed_monitor_path = dump_compressed(monitor, raw_monitor_path)

    print(f"压缩模型保存到: {compressed_clf_path}")
    print(f"压缩监控器保存到: {compressed_monitor_path}")

    # ========= 4. 返回结果 =========
    paths = {
        "raw_clf": raw_clf_path,
        "raw_monitor": raw_monitor_path,
        "compressed_clf": compressed_clf_path,
        "compressed_monitor": compressed_monitor_path,
        "monitor_csv": result_csv,
        "result_dir": result_dir,
    }

    return clf, monitor, (X_train, X_test, y_train, y_test, noise_idx, clean_idx), paths
