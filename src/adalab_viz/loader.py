import os
import pandas as pd
import joblib


def load_from_csv(csv_path):
    """
    从 CSV 文件加载监控数据

    Parameters
    ----------
    csv_path : str
        CSV 文件路径

    Returns
    -------
    dict
        包含所有监控数据的字典
    """
    print(f"📂 Loading from CSV: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rounds of training data")

    # 构建与 BoostMonitor 相同的数据结构
    data = {
        "rounds": df["round"].tolist(),
        "error_history": df["weighted_error"].tolist(),
        "alpha_history": df["alpha"].tolist(),
        "error_without_weight_history": df["unweighted_error"].tolist()
        if "unweighted_error" in df.columns
        else [],
        "noisy_weight_history": df["noisy_weight"].tolist()
        if "noisy_weight" in df.columns
        else [],
        "clean_weight_history": df["clean_weight"].tolist()
        if "clean_weight" in df.columns
        else [],
        "val_acc_history": df["val_acc"].tolist() if "val_acc" in df.columns else [],
        "val_f1_history": df["val_f1"].tolist() if "val_f1" in df.columns else [],
        "acc_on_train_data": df["train_acc"].tolist()
        if "train_acc" in df.columns
        else [],
        "f1_on_training_data": df["train_f1"].tolist()
        if "train_f1" in df.columns
        else [],
        "is_data_noisy": "noisy_weight" in df.columns,
        "n_estimators": len(df),
    }

    print(f"✓ Data fields available:")
    for key, value in data.items():
        if key not in ["rounds", "is_data_noisy", "n_estimators"]:
            status = "✓" if (isinstance(value, list) and len(value) > 0) else "✗"
            print(f"  {status} {key}")

    return data


def load_from_joblib(joblib_path):
    """
    从 joblib 文件加载 BoostMonitor 对象

    Parameters
    ----------
    joblib_path : str
        joblib 文件路径

    Returns
    -------
    dict
        包含所有监控数据的字典
    """
    print(f"📂 Loading from joblib: {joblib_path}")

    if not os.path.exists(joblib_path):
        raise FileNotFoundError(f"Joblib file not found: {joblib_path}")

    monitor = joblib.load(joblib_path)
    print(f"✓ Loaded BoostMonitor object")

    # 从 BoostMonitor 对象提取数据
    data = {
        "rounds": list(range(1, len(monitor.error_history) + 1)),
        "error_history": monitor.error_history,
        "alpha_history": monitor.alpha_history,
        "error_without_weight_history": monitor.error_without_weight_history,
        "noisy_weight_history": monitor.noisy_weight_history,
        "clean_weight_history": monitor.clean_weight_history,
        "val_acc_history": monitor.val_acc_history,
        "val_f1_history": monitor.val_f1_history,
        "acc_on_train_data": monitor.acc_on_train_data,
        "f1_on_training_data": monitor.f1_on_training_data,
        "sample_weights_history": monitor.sample_weights_history,
        "is_data_noisy": monitor.is_data_noisy,
        "n_estimators": len(monitor.error_history),
    }

    return data


def load_from_experiment(experiment_name):
    """
    从实验文件夹加载数据（自动检测 CSV 或 joblib）

    Parameters
    ----------
    experiment_name : str
        实验名称，如 'train_val_500rounds'

    Returns
    -------
    dict
        包含所有监控数据的字典
    """
    exp_dir = os.path.join("experiments", experiment_name)

    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    print(f"📁 Loading from experiment: {experiment_name}")

    # 优先尝试 joblib
    joblib_path = os.path.join(exp_dir, "results", "monitor.joblib")
    if os.path.exists(joblib_path):
        return load_from_joblib(joblib_path)

    # 尝试 CSV
    csv_path = os.path.join(exp_dir, "results", "final_results.csv")
    if os.path.exists(csv_path):
        return load_from_csv(csv_path)

    raise FileNotFoundError(f"No monitor data found in {exp_dir}")
