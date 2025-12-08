import os
import json
import argparse

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from adalab.workflow import train_and_save


def evaluate(y_true, y_pred, title="Evaluation"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to json config file")
    args = parser.parse_args()
    config_path = args.config_path

    (
        clf,
        monitor,
        data_prep,
        (X_train, X_test_mnist, y_train, y_test_mnist, noise_idx, clean_idx),
        paths,
    ) = train_and_save(config_path)

    X_course, y_course = data_prep.prepare_course_data("test_data")
    y_pred_mnist = clf.predict(X_test_mnist)
    y_pred_course = clf.predict(X_course)
    print("\n", "===Scores on test data of MNIST===")
    scores_on_mnist = evaluate(y_true=y_test_mnist, y_pred=y_pred_mnist)

    print("\n", "===Scores on test data of corse data===")
    scores_on_course = evaluate(y_true=y_course, y_pred=y_pred_course)

    scores = {"mnist": scores_on_mnist, "course_data": scores_on_course}
    result_dir = paths["result_dir"]
    score_path = os.path.join(result_dir, "scores.json")
    with open(score_path, "w") as f:
        json.dump(scores, f, indent=4)
