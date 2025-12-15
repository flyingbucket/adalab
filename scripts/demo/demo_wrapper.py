from src.utils import train_and_save, load_compressed

if __name__ == "__main__":
    # === 用包装函数训练并保存 ===
    (
        clf,
        monitor,
        prep,
        (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        paths,
    ) = train_and_save("./configs/test_experiment_wrapper.json")

    # === 加载压缩版模型和 monitor ===
    clf_loaded = load_compressed(paths["compressed_clf"])
    monitor_loaded = load_compressed(paths["compressed_monitor"])

    print("模型加载成功")
