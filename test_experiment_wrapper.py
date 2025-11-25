import os
import json
import joblib
import lzma
import hashlib
import io
import numpy as np
from src.monitor import BoostMonitor
from src.utils import build_experiment, dump_compressed, load_compressed


def _to_serializable(obj):
    """递归地把 numpy 类型、数组等转成 Python 原生类型，方便 json.dumps"""
    # numpy 标量 -> Python 标量
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # numpy 数组 -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # list / tuple -> 逐个转换
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]

    # dict -> 递归转换 value，key 统一成 str
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    # 其它类型（比如 str、bool、None、普通 int/float）直接返回
    return obj


def semantic_hash_monitor(monitor: BoostMonitor) -> str:
    """只对 monitor 的语义字段进行哈希，自动处理 numpy 类型"""

    payload = {
        "noise_indices": monitor.noise_indices,
        "clean_indices": monitor.clean_indices,
        "is_data_noisy": monitor.is_data_noisy,
        "sample_weights_history": monitor.sample_weights_history,
        "noisy_weight_history": monitor.noisy_weight_history,
        "clean_weight_history": monitor.clean_weight_history,
        "error_without_weight_history": monitor.error_without_weight_history,
        "error_history": monitor.error_history,
        "alpha_history": monitor.alpha_history,
        "val_acc_history": monitor.val_acc_history,
        "val_f1_history": monitor.val_f1_history,
        "acc_on_train_data": monitor.acc_on_train_data,
        "f1_on_training_data": monitor.f1_on_training_data,
        "checkpoint_interval": monitor.checkpoint_interval,
        "checkpoint_prefix": monitor.checkpoint_prefix,
    }

    # 关键一步：把里面所有 numpy 类型统统转成原生 Python
    payload = _to_serializable(payload)

    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


def hash_object_bytes(obj):
    """用于比较 dump 到字节流的原始序列化内容（仅用于内部）"""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_model_and_monitor(
    clf,
    monitor,
    X_test,
    raw_clf_path,
    raw_monitor_path,
    compressed_clf_path,
    compressed_monitor_path,
):
    print("\n============================")
    print("   开始对象一致性测试")
    print("============================\n")

    # --------------------------
    # 1. 原始对象 hash (序列化行为哈希)
    # --------------------------
    clf_hash_original = hash_object_bytes(clf)
    monitor_hash_original = semantic_hash_monitor(monitor)

    print("原始 clf 行为哈希:", clf_hash_original)
    print("原始 monitor 语义哈希:", monitor_hash_original)

    # --------------------------
    # 2. 加载直接 joblib.dump 的对象
    # --------------------------
    clf_raw_loaded = joblib.load(raw_clf_path)
    monitor_raw_loaded = joblib.load(raw_monitor_path)

    clf_hash_raw = hash_object_bytes(clf_raw_loaded)
    monitor_hash_raw = semantic_hash_monitor(monitor_raw_loaded)

    print("\n未压缩 load clf 行为哈希:", clf_hash_raw)
    print("未压缩 load monitor 语义哈希:", monitor_hash_raw)

    # --------------------------
    # 3. 加载压缩 dump/load 的对象
    # --------------------------
    with lzma.open(compressed_clf_path, "rb") as f:
        clf_compressed_loaded = joblib.load(f)

    with lzma.open(compressed_monitor_path, "rb") as f:
        monitor_compressed_loaded = joblib.load(f)

    clf_hash_compressed = hash_object_bytes(clf_compressed_loaded)
    monitor_hash_compressed = semantic_hash_monitor(monitor_compressed_loaded)

    print("\n压缩 load clf 行为哈希:", clf_hash_compressed)
    print("压缩 load monitor 语义哈希:", monitor_hash_compressed)

    # --------------------------
    # 4. 行为一致性验证（model）
    # --------------------------
    print("\n====== 行为一致性检测（clf.predict）======")
    pred_consistent_raw = np.array_equal(
        clf.predict(X_test), clf_raw_loaded.predict(X_test)
    )
    pred_consistent_compressed = np.array_equal(
        clf.predict(X_test), clf_compressed_loaded.predict(X_test)
    )

    print("原始 vs 未压缩   predict 一致性:", pred_consistent_raw)
    print("原始 vs 压缩     predict 一致性:", pred_consistent_compressed)

    # --------------------------
    # 5. monitor 语义哈希一致性检测
    # --------------------------
    print("\n====== monitor 语义哈希一致性 ======")
    print(
        "原始 vs 未压缩   monitor 哈希一致:", monitor_hash_original == monitor_hash_raw
    )
    print(
        "原始 vs 压缩     monitor 哈希一致:",
        monitor_hash_original == monitor_hash_compressed,
    )

    print("\n============================")
    print("       一致性测试完成")
    print("============================\n")


if __name__ == "__main__":
    (
        clf,
        monitor,
        (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        result_csv,
        result_dir,
    ) = build_experiment("./configs/test_experiment_wrapper.json")

    print("开始训练...")
    clf.fit(X_train, y_train)

    monitor.dump(result_csv)
    print("训练完成！")

    # dump 成 joblib
    raw_clf_path = os.path.join(result_dir, "model.joblib")
    raw_monitor_path = os.path.join(result_dir, "monitor.joblib")

    joblib.dump(clf, raw_clf_path)
    joblib.dump(monitor, raw_monitor_path)

    # 读取未压缩版本
    clf_raw_loaded = joblib.load(raw_clf_path)
    monitor_raw_loaded = joblib.load(raw_monitor_path)

    # 压缩后 dump 再 load
    compressed_clf_path = dump_compressed(clf, raw_clf_path)
    compressed_monitor_path = dump_compressed(monitor, raw_monitor_path)

    clf_compressed_loaded = load_compressed(compressed_clf_path)
    monitor_compressed_loaded = load_compressed(compressed_monitor_path)

    test_model_and_monitor(
        clf,
        monitor,
        X_test,
        raw_clf_path=os.path.join(result_dir, "model.joblib"),
        raw_monitor_path=os.path.join(result_dir, "monitor.joblib"),
        compressed_clf_path=os.path.join(result_dir, "model.joblib.xz"),
        compressed_monitor_path=os.path.join(result_dir, "monitor.joblib.xz"),
    )
