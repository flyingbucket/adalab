import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.utils import preprocess_for_mnist


def evaluate(y_true, y_pred, title="Evaluation"):
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


def load_testset(folder="test_data"):
    X_list = []
    y_list = []

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            label = int(os.path.splitext(filename)[0])  # 文件名 = 标签

            path = os.path.join(folder, filename)
            x, _ = preprocess_for_mnist(path)  # x shape = (1, 784)

            X_list.append(x[0])
            y_list.append(label)

    X = np.vstack(X_list)  # (N, 784)
    y = np.array(y_list).astype(np.int64)  # (N,)

    return X, y


if __name__ == "__main__":
    X_train, y_train = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )
    y_train = y_train.astype(np.int64)
    X_train = X_train / 255.0

    X_test, y_test = load_testset("test_data")

    # 训练 AdaBoost
    print("Training AdaBoost...")
    base = DecisionTreeClassifier(max_depth=2)

    clf = AdaBoostClassifier(
        estimator=base,
        n_estimators=200,
        learning_rate=0.5,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    print("Training finished!")

    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    print("===Scores on training data")
    evaluate(y_train, y_pred_train)

    print("===Scores on testing data")
    evaluate(y_test, y_pred)
