import json
import os
import joblib
import lzma
import time
import warnings

import numpy as np
import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from .monitor import BoostMonitor
from .data import DataPreparation
from .patch import AdaBoostClfWithMonitor


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
    exp_dir = safe_exp_dir(exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    config_dump_path = os.path.join(exp_dir, "config.json")
    with open(config_dump_path, "w") as fw:
        json.dump(config, fw, indent=4)

    # === 自动创建目录结构 ===
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    result_dir = os.path.join(exp_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print(f"[Workflow] experiment dir created: {exp_dir}")

    # === 1. 数据准备 ===
    data_cfg = config["data"]

    # === 读取 HOG 参数 ===
    hog_cfg = data_cfg.get("hog_params", {})
    hog_orient = hog_cfg.get("orientations", 9)
    hog_ppc = tuple(hog_cfg.get("pixels_per_cell", (4, 4)))
    hog_cpb = tuple(hog_cfg.get("cells_per_block", (2, 2)))

    # === 读取 HU 参数 ===
    hu_cfg = data_cfg.get("hu_params", {})
    hu_log_scale = hu_cfg.get("log_scale", True)

    # === 构建 DataPreparation ===
    prep = DataPreparation(
        # noise_ratio=data_cfg["noise_ratio"],
        noise_config=data_cfg["noise_config"],
        test_size=data_cfg["test_size"],
        use_feature=data_cfg.get("use_feature", "original"),
        random_state=data_cfg["random_state"],
        hog_orientations=hog_orient,
        hog_pixels_per_cell=hog_ppc,
        hog_cells_per_block=hog_cpb,
        hu_log_scale=hu_log_scale,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        noise_idx,
        clean_idx,
    ) = prep.prepare()

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

        # 返回路径供保存
        result_csv = os.path.join(result_dir, "final_results.csv")

        return (
            clf,
            None,
            prep,
            (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
            result_csv,
            exp_dir,
            result_dir,
        )

    print("[MODEL] Using AdaBoost with BoostMonitor enabled")
    monitor = BoostMonitor(
        noise_indices=noise_idx,
        clean_indices=clean_idx,
        is_data_noisy=monitor_cfg["is_data_noisy"],
        checkpoint_interval=monitor_cfg["checkpoint_interval"],
        checkpoint_prefix=ckpt_dir,
    )

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
        prep,
        (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        result_csv,
        exp_dir,
        result_dir,
    )


def dump_compressed_chunks(obj, filepath: str, chunk_size_mb=50):
    """
    压缩并切片保存 Python 对象。
    chunk_size_mb: 每块大小（默认 50MB）
    """
    compressed_path = filepath + ".xz"

    # 1. 压缩到 .xz
    with lzma.open(compressed_path, "wb") as f:
        joblib.dump(obj, f)
    print(f"[Workflow] compressing finished: {compressed_path}")

    # 2. 切片
    chunk_size = chunk_size_mb * 1024 * 1024
    chunks = []

    with open(compressed_path, "rb") as f:
        idx = 0
        while True:
            data = f.read(chunk_size)
            if not data:
                break

            part_path = f"{compressed_path}.part{idx:03d}"
            with open(part_path, "wb") as pf:
                pf.write(data)

            chunks.append(part_path)
            idx += 1

    print(f"[Workflow] cut into {len(chunks)} chunks，chunk size ≤ {chunk_size_mb}MB：")
    for p in chunks:
        print("  ", p)

    return chunks


def load_compressed_chunks(basepath: str):
    """
    basepath: 完整路径，如 /path/to/model.joblib.xz
    自动在同目录查找 model.joblib.xz.part000, part001, ...
    忽略原始文件、merged 文件
    """
    directory = os.path.dirname(basepath)
    filename = os.path.basename(basepath)

    # 在同一目录查找 .part 文件
    parts = sorted(
        [
            os.path.join(directory, p)
            for p in os.listdir(directory)
            if p.startswith(filename + ".part")
        ]
    )

    if not parts:
        raise FileNotFoundError(f"未找到分片：{filename}.part*** 在目录 {directory}")

    merged_path = os.path.join(directory, filename + ".merged")

    # 合并
    with open(merged_path, "wb") as fout:
        for p in parts:
            with open(p, "rb") as fin:
                fout.write(fin.read())

    # 解压并加载对象
    with lzma.open(merged_path, "rb") as f:
        obj = joblib.load(f)

    return obj


def dump_compressed(obj, filepath: str):
    """
    使用 lzma 压缩并保存任意 Python 对象
    """
    compressed_path = filepath + ".xz"

    with lzma.open(compressed_path, "wb") as f:
        joblib.dump(obj, f)

    print(f"[Workflow] compressed and saved to : {compressed_path}")
    return compressed_path


def load_compressed(filepath: str):
    """
    自动读取 .xz 压缩的 joblib 文件
    """
    with lzma.open(filepath, "rb") as f:
        obj = joblib.load(f)
    print(f"[Workflow] loaded compressed joblib from : {filepath}")
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
        prep,
        (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        result_csv,
        exp_dir,
        result_dir,
    ) = build_experiment(config_path)

    print("[Workflow] \033[33mTraining Started...\033[0m")
    clf.fit(X_train, y_train)
    print("[Workflow] \033[33mTraining Finished...\033[0m")

    # 保存 monitor 结果
    if monitor is not None:
        monitor.dump(result_csv)
        raw_monitor_path = os.path.join(result_dir, "monitor.joblib")
        joblib.dump(monitor, raw_monitor_path)
        print(f"[Workflow] Uncompressed monitor joblib saved to : {raw_monitor_path}")
        compressed_monitor_path = dump_compressed(monitor, raw_monitor_path)
        print(
            f"[Workflow] compressed monitor joblib saved to : {compressed_monitor_path}"
        )
    else:
        raw_monitor_path = None
        compressed_monitor_path = None
        result_csv = None

    # ========= 2. 保存未压缩 joblib =========
    raw_clf_path = os.path.join(result_dir, "model.joblib")

    joblib.dump(clf, raw_clf_path)

    print(f"[Workflow] Uncompressed model joblib saved to : {raw_clf_path}")

    # ========= 3. 保存压缩版 =========
    compressed_clf_path = dump_compressed(clf, raw_clf_path)

    print(f"[Workflow] compressed model joblib saved to : {compressed_clf_path}")

    # ========= 4. 返回结果 =========
    paths = {
        "raw_clf": raw_clf_path,
        "raw_monitor": raw_monitor_path,
        "compressed_clf": compressed_clf_path,
        "compressed_monitor": compressed_monitor_path,
        "monitor_csv": result_csv,
        "experiment_dir": exp_dir,
        "result_dir": result_dir,
    }

    return (
        clf,
        monitor,
        prep,
        (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        paths,
    )
