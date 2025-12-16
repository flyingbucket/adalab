"""
模型训练后评估与性能曲线计算。

本模块负责：
- 对完整训练模型进行推理评估
- 在训练完成后重构各阶段集成模型
- 计算 accuracy / F1 等性能曲线

该模块主要为实验分析与前端可视化提供评估数据。
"""

from __future__ import annotations
from joblib import Parallel, delayed
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


from typing import Any, Dict, Optional
from sklearn.metrics import classification_report, accuracy_score


def evaluate(
    y_true,
    y_pred,
    *,
    labels: Optional[list[int]] = None,
    target_names: Optional[list[str]] = None,
    title: Optional[str] = None,
    zero_division: int = 0,
) -> Dict[str, Any]:
    """
    Basic evaluation for multi-class classification.

    Focus:
    - per-class recall
    - overall accuracy

    Designed as a minimal, research-friendly evaluator.
    """

    if title is not None:
        print(f"\n=== {title} ===")

    # sklearn classification report (structured)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=zero_division,
        output_dict=True,
    )

    accuracy = accuracy_score(y_true, y_pred)

    # ---- extract per-class recall ----
    per_class_recall: Dict[str, float] = {}

    for key, value in report.items():
        # skip aggregate entries
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        per_class_recall[key] = value["recall"]

    # ---- minimal CLI output ----
    print(f"Accuracy: {accuracy:.4f}")
    print("Per-class recall:")
    for cls, rec in per_class_recall.items():
        print(f"  class {cls}: recall = {rec:.4f}")

    return {
        "accuracy": accuracy,
        "per_class_recall": per_class_recall,
        "raw_report": report,  # keep full info for later use
    }


# def evaluate(y_true, y_pred, title="Evaluation"):
#     """对完整模型的预测结果进行评估。
#
#     该函数用于在模型训练完成后，
#     对最终集成模型在指定数据集上的预测结果进行一次性评估，
#     并输出常用分类指标。
#
#     Args:
#         y_true (array-like): 真实标签。
#         y_pred (array-like): 模型预测标签。
#         title (str, optional): 评估结果的标题，用于日志输出。
#
#     Returns:
#         dict: 包含以下键值的评估结果字典：
#             - accuracy: 准确率
#             - precision_macro: macro 平均精确率
#             - recall_macro: macro 平均召回率
#             - f1_macro: macro 平均 F1 分数
#     """
#     print(f"\n=== {title} ===")
#
#     acc = accuracy_score(y_true, y_pred)
#     prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
#     rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
#     f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
#
#     print(f"Accuracy:       {acc:.4f}")
#     print(f"Precision_macro:{prec_macro:.4f}")
#     print(f"Recall_macro:   {rec_macro:.4f}")
#     print(f"F1_macro:       {f1_macro:.4f}")
#     return {
#         "accuracy": acc,
#         "precision_macro": prec_macro,
#         "recall_macro": rec_macro,
#         "f1_macro": f1_macro,
#     }


def _compute_round_metric(t, alphas, est_preds, classes, n_classes, y_indices):
    """计算某一 boost 轮次对应的集成预测性能（内部函数）。

    该函数用于在给定轮次 t 时，
    基于前 t+1 个弱分类器重构集成模型预测结果，
    并计算对应的 accuracy 与 F1 分数。

    Args:
        t (int): boost 轮次索引（从 0 开始）。
        alphas (np.ndarray): 各轮弱分类器的权重系数。
        est_preds (np.ndarray): 所有弱分类器在数据集上的预测结果。
        classes (np.ndarray): 类别标签集合。
        n_classes (int): 类别数量。
        y_indices (np.ndarray): 真实标签对应的类别索引。

    Returns:
        tuple[float, float]: (accuracy, f1_score)。
    """
    N = len(y_indices)
    scores = np.zeros((N, n_classes), dtype=float)
    weight_sum = np.sum(alphas[: t + 1])

    for k in range(t + 1):
        w = alphas[k]
        pred_k = est_preds[k]
        mask = pred_k[:, None] == classes[None, :]
        contrib = np.where(mask, w, -w / (n_classes - 1))
        scores += contrib

    scores /= weight_sum

    # predict
    if n_classes == 2:
        tmp = scores.copy()
        tmp[:, 0] *= -1
        df = tmp.sum(axis=1)
        pred_idx = (df > 0).astype(int)
    else:
        pred_idx = np.argmax(scores, axis=1)

    acc = np.mean(pred_idx == y_indices)
    y_pred_labels = classes[pred_idx]
    f1 = f1_score(y_indices, pred_idx, average="macro", zero_division=0)

    return acc, f1


def val_after_train_parallel(clf, alphas, X, y, val_freq=20, n_jobs=-1):
    """训练后并行评估不同 boost 轮次下的模型性能曲线。

    该函数在模型训练完成后，
    通过重构前若干轮弱分类器组成的集成模型，
    计算其在指定数据集上的 accuracy 与 F1 曲线。

    评估点由 ``val_freq`` 控制，并始终包含最后一轮。

    Args:
        clf: 已训练完成的 AdaBoost 模型。
        alphas (array-like): 各轮弱分类器的权重系数。
        X (np.ndarray): 用于评估的特征数据。
        y (np.ndarray): 对应的真实标签。
        val_freq (int, optional): 每隔多少轮进行一次评估，默认 20。
        n_jobs (int, optional): 并行任务数，默认 -1（使用全部可用核心）。

    Returns:
        tuple:
            - acc_curve (np.ndarray): 各评估点的准确率曲线。
            - f1_curve (np.ndarray): 各评估点的 F1 曲线。
            - val_idx (np.ndarray): 对应的 boost 轮次索引。
    """
    estimators = clf.estimators_
    T = len(estimators)
    N = X.shape[0]
    classes = clf.classes_
    n_classes = clf.n_classes_

    val_idx = np.arange(val_freq - 1, T, val_freq)
    val_idx = np.unique(np.append(val_idx, T - 1))
    num_points = len(val_idx)

    # 预计算弱分类器预测
    est_preds = np.zeros((T, N), dtype=classes.dtype)
    for t, est in enumerate(estimators):
        est_preds[t] = est.predict(X)

    y_indices = np.searchsorted(classes, y)

    # 并行计算每一个 t
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_round_metric)(
            t, alphas, est_preds, classes, n_classes, y_indices
        )
        for t in val_idx
    )

    acc_curve = np.array([r[0] for r in results])
    f1_curve = np.array([r[1] for r in results])

    return acc_curve, f1_curve, val_idx


def val_after_train(clf, alphas, X, y, val_freq=20):
    """训练后评估不同 boost 轮次下的模型性能曲线（串行版本）。

    该函数功能与 ``val_after_train_parallel`` 相同，
    但采用串行方式计算，适用于调试或小规模实验。

    Args:
        clf: 已训练完成的 AdaBoost 模型。
        alphas (array-like): 各轮弱分类器的权重系数。
        X (np.ndarray): 用于评估的特征数据。
        y (np.ndarray): 对应的真实标签。
        val_freq (int, optional): 每隔多少轮进行一次评估，默认 20。

    Returns:
        tuple:
            - acc_curve (np.ndarray): 各评估点的准确率曲线。
            - f1_curve (np.ndarray): 各评估点的 F1 曲线。
            - val_idx (np.ndarray): 对应的 boost 轮次索引。
    """

    estimators = clf.estimators_
    T = len(estimators)
    N = X.shape[0]
    classes = clf.classes_
    n_classes = clf.n_classes_

    # val index
    val_idx = np.arange(val_freq - 1, T, val_freq)
    # always include last
    val_idx = np.unique(np.append(val_idx, T - 1))

    num_points = len(val_idx)

    # initialize results arrs
    acc_curve = np.zeros(num_points)
    f1_curve = np.zeros(num_points)

    # pre calculate pred of each estimator
    est_preds = np.zeros((T, N), dtype=classes.dtype)
    for t, est in enumerate(estimators):
        est_preds[t] = est.predict(X)

    # y → 类别索引
    y_indices = np.searchsorted(classes, y)

    # val kernel ,could be running in SIMD
    for i, t in enumerate(val_idx):
        scores = np.zeros((N, n_classes), dtype=float)
        weight_sum = np.sum(alphas[: t + 1])

        for k in range(t + 1):
            w = alphas[k]
            pred_k = est_preds[k]
            mask = pred_k[:, None] == classes[None, :]
            contrib = np.where(mask, w, -w / (n_classes - 1))
            scores += contrib

        scores /= weight_sum

        # 分类预测
        if n_classes == 2:
            tmp = scores.copy()
            tmp[:, 0] *= -1
            df = tmp.sum(axis=1)
            pred_idx = (df > 0).astype(int)
        else:
            pred_idx = np.argmax(scores, axis=1)

        # accuracy
        acc_curve[i] = np.mean(pred_idx == y_indices)

        # f1
        y_pred_labels = classes[pred_idx]
        f1_curve[i] = f1_score(y, y_pred_labels, average="macro", zero_division=0)

    return acc_curve, f1_curve, val_idx
