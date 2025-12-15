import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.utils import prepare_data


if __name__ == "__main__":
    # get original clean data
    X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices = (
        prepare_data(noise_ratio=0)
    )

    # 训练 AdaBoost
    print("Training AdaBoost...")
    base = DecisionTreeClassifier(max_depth=1)

    clf = AdaBoostClassifier(
        estimator=base,
        n_estimators=200,
        learning_rate=0.5,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    print("Training finished!")

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(f"acc on training data：{accuracy_score(y_train, y_pred_train):.4f}")
    print(f"acc on testing data：{accuracy_score(y_test, y_pred_test):.4f}")
