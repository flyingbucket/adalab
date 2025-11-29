from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.utils import train_and_save


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

    print("\nConfusion Matrix:")


if __name__ == "__main__":
    (
        clf,
        monitor,
        data_prep,
        (X_train, X_test_mnist, y_train, y_test_mnist, noise_idx, clean_idx),
        paths,
    ) = train_and_save("./configs/main_hog.json")

    X_course, y_course = data_prep.prepare_course_data("test_data")
    y_pred_mnist = clf.predict(X_test_mnist)
    y_pred_course = clf.predict(X_course)

    print("\n", "===Scores on test data of MNIST===")
    evaluate(y_true=y_test_mnist, y_pred=y_pred_mnist)
    print("\n", "===Scores on test data of MNIST===")
    evaluate(y_true=y_course, y_pred=y_pred_course)
