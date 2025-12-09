import numpy as np
from sklearn.metrics import accuracy_score
from adalab.workflow import load_compressed, load_config, prep_data_from_config


def posthoc_sklearn_style_accuracy(clf, monitor, X, y):
    """
    按 sklearn 的 AdaBoostClassifier.decision_function / predict 逻辑，
    但使用 monitor.alpha_history 中的“即时 alpha_t”进行重建。
    """

    estimators = clf.estimators_
    alphas = np.asarray(
        monitor.alpha_history
    )  # ← 关键修复：用 monitor alpha，而不是 clf.estimator_weights_
    T = len(estimators)
    N = X.shape[0]
    classes = clf.classes_
    n_classes = clf.n_classes_

    # precompute weak predictions
    est_preds = np.zeros((T, N), dtype=classes.dtype)
    for t, est in enumerate(estimators):
        est_preds[t] = est.predict(X)

    # convert y -> indices
    y_indices = np.searchsorted(classes, y)

    acc_curve = np.zeros(T)

    for t in range(T):
        scores = np.zeros((N, n_classes), dtype=float)
        weight_sum = np.sum(
            alphas[: t + 1]
        )  # 重新使用前 t 轮即时 alpha，确保与训练时完全一致

        for k in range(t + 1):
            w = alphas[k]
            pred_k = est_preds[k]

            mask = pred_k[:, None] == classes[None, :]
            contrib = np.where(mask, w, -w / (n_classes - 1))
            scores += contrib

        scores /= weight_sum

        if n_classes == 2:
            scores_bin = scores.copy()
            scores_bin[:, 0] *= -1
            df = scores_bin.sum(axis=1)
            pred_idx = (df > 0).astype(int)
        else:
            pred_idx = np.argmax(scores, axis=1)

        acc_curve[t] = np.mean(pred_idx == y_indices)

    return acc_curve


def compare_monitor_and_posthoc(monitor, acc_posthoc, name="train"):
    if name == "train":
        acc_monitor = np.array(monitor.acc_on_train_data)
    elif name == "val":
        acc_monitor = np.array(monitor.val_acc_history)
    else:
        raise ValueError("name must be 'train' or 'val'")

    # --- 修复 off-by-one ---
    # monitor[1:] 对应 posthoc[:-1]
    acc_monitor_aligned = acc_monitor[1:]
    acc_posthoc_aligned = acc_posthoc[:-1]

    min_len = min(len(acc_monitor_aligned), len(acc_posthoc_aligned))
    diff = acc_monitor_aligned[:min_len] - acc_posthoc_aligned[:min_len]
    max_abs_err = np.max(np.abs(diff))

    print(
        f"\n===== Compare {name} accuracy (aligned, monitor[1:] vs posthoc[:-1]) ====="
    )
    print(f"Monitor length: {len(acc_monitor_aligned)}")
    print(f"Posthoc length: {len(acc_posthoc_aligned)}")
    print(f"Maximum absolute error: {max_abs_err:.12f}")

    if np.allclose(acc_monitor_aligned[:min_len], acc_posthoc_aligned[:min_len]):
        print(f"✔ {name} accuracy curves match EXACTLY after alignment.")
    else:
        print(f"✘ {name} accuracy curves differ even after alignment!")
        bad = np.where(np.abs(diff) > 1e-12)[0]
        print("First mismatched indices:", bad[:10])
        for k in bad[:5]:
            print(
                f"  round {k + 1}: monitor={acc_monitor_aligned[k]}, "
                f"posthoc={acc_posthoc_aligned[k]}"
            )


def test_posthoc_vs_monitor_sklearn(clf, monitor, X_train, y_train, X_val, y_val):
    print("=== Train ===")
    acc_train_posthoc = posthoc_sklearn_style_accuracy(clf, monitor, X_train, y_train)
    compare_monitor_and_posthoc(monitor, acc_train_posthoc, name="train")

    print("\n=== Val ===")
    acc_val_posthoc = posthoc_sklearn_style_accuracy(clf, monitor, X_val, y_val)
    compare_monitor_and_posthoc(monitor, acc_val_posthoc, name="val")

    print("\n=== Done ===")


if __name__ == "__main__":
    clf = load_compressed(
        "/home/flyingbucket/CODE/AdaBoost_numbers/experiments/test_end2end/results/model.joblib.xz"
    )
    monitor = load_compressed(
        "/home/flyingbucket/CODE/AdaBoost_numbers/experiments/test_end2end/results/monitor.joblib.xz"
    )
    config = load_config(
        "/home/flyingbucket/CODE/AdaBoost_numbers/experiments/test_end2end_20251208_220714/config.json"
    )
    X_train, X_test, y_train, y_test, _, _, prep = prep_data_from_config(config)
    test_posthoc_vs_monitor_sklearn(clf, monitor, X_train, y_train, X_test, y_test)
