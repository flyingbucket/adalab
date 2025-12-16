"""
评估器模块 - 提供统一的模型评估功能
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate(y_true, y_pred, title="Evaluation"):
    """
    基础评估函数 - 计算并打印主要指标
    
    Parameters
    ----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    title : str, optional
        评估标题
    
    Returns
    -------
    dict
        包含各项指标的字典
    """
    print(f"\n=== {title} ===")

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Accuracy:       {acc:.4f}")
    print(f"Precision_macro:{prec_macro:.4f}")
    print(f"Recall_macro:   {rec_macro:.4f}")
    print(f"F1_macro:       {f1_macro:.4f}")
    
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
    }


def evaluate_detailed(y_true, y_pred, title="Evaluation"):
    """
    详细评估函数 - 包含micro指标和混淆矩阵
    
    Parameters
    ----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    title : str, optional
        评估标题
    
    Returns
    -------
    dict
        包含详细指标的字典
    """
    print(f"\n=== {title} ===")

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_micro = precision_score(y_true, y_pred, average="micro")
    rec_micro = recall_score(y_true, y_pred, average="micro")
    f1_micro = f1_score(y_true, y_pred, average="micro")

    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy:       {acc:.4f}")
    print(f"Precision_macro:{prec_macro:.4f}   Precision_micro:{prec_micro:.4f}")
    print(f"Recall_macro:   {rec_macro:.4f}   Recall_micro:   {rec_micro:.4f}")
    print(f"F1_macro:       {f1_macro:.4f}   F1_micro:       {f1_micro:.4f}")

    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
        "confusion_matrix": cm.tolist(),
    }


