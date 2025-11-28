import joblib
import lzma
import hashlib
import numpy as np

from src.utils import load_compressed, dump_compressed_chunks, load_compressed_chunks


def file_md5(path):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def verify_behavior(original, recovered, name="obj"):
    """
    验证对象的行为一致性：
    - 对有 predict() 方法的模型：验证预测结果一致
    - 对 monitor：验证关键属性曲线一致
    """
    ok = True

    # ① 预测一致性（适用于模型）
    if hasattr(original, "predict"):
        X = np.random.randn(128, getattr(original, "n_features_in_", 4))
        y1 = original.predict(X)
        y2 = recovered.predict(X)
        if np.array_equal(y1, y2):
            print(f"[OK] {name} 预测行为一致")
        else:
            print(f"[ERR] {name} 预测行为不一致")
            ok = False

    # ② 监控器行为一致性
    # 例如 Monitor().loss_curve, weights_history 等
    for k, v in original.__dict__.items():
        if isinstance(v, np.ndarray):
            if not np.array_equal(v, recovered.__dict__.get(k)):
                print(f"[ERR] Monitor 数组字段 {k} 不一致")
                ok = False

        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            if v != recovered.__dict__.get(k):
                print(f"[ERR] Monitor 列表字段 {k} 不一致")
                ok = False

    if ok:
        print(f"[OK] {name} 行为一致")

    return ok


def verify_md5(original_path, merged_path, name="obj"):
    md5_orig = file_md5(original_path)
    md5_merged = file_md5(merged_path)

    if md5_orig == md5_merged:
        print(f"[OK] {name} 二进制完全一致（最高等级验证）")
        return True
    else:
        print(f"[ERR] {name} 二进制不一致")
        return False


def verify_chunk_correctness(original_path, chunks_loader_func, name="obj"):
    print(f"\n===== 校验 {name} =====")

    # 加载原始对象
    with lzma.open(original_path, "rb") as f:
        original = joblib.load(f)

    # 加载分片恢复对象
    recovered = chunks_loader_func(original_path)

    # 获取 merged 路径
    merged_path = original_path + ".merged"

    ok_md5 = verify_md5(original_path, merged_path, name=name)
    ok_behavior = verify_behavior(original, recovered, name=name)

    return ok_md5 and ok_behavior


if __name__ == "__main__":
    clf_path = "./experiments/baseline_est500_depth2/results/model.joblib.xz"
    monitor_path = "./experiments/baseline_est500_depth2/results/monitor.joblib.xz"

    # 先切片
    # dump_compressed_chunks(load_compressed(clf_path), clf_path[:-3])  # 去掉 .xz
    # dump_compressed_chunks(load_compressed(monitor_path), monitor_path[:-3])

    # 校验模型
    ok_clf = verify_chunk_correctness(
        clf_path, lambda p: load_compressed_chunks(p), name="Classifier"
    )

    # 校验监控器
    ok_monitor = verify_chunk_correctness(
        monitor_path, lambda p: load_compressed_chunks(p), name="Monitor"
    )

    if ok_clf and ok_monitor:
        print("🎉 所有校验均通过！分片后的对象与原始对象完全一致。")
    else:
        print("❌ 校验未通过，请检查分片和合并流程！")
