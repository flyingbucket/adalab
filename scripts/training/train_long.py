import os
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src import monitor
from src.monitor import BoostMonitor
from src.patch import AdaBoostClfWithMonitor
from src.utils import prepare_data
from src.evaluation import quick_evaluate, visualize_overfitting_process
from src.utils import build_experiment


if __name__ == "__main__":
    (
        clf,
        monitor,
        (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        result_csv,
        result_dir,
    ) = build_experiment("configs/long_train_val.json")

    print("开始训练...")
    clf.fit(X_train, y_train)

    monitor.dump(result_csv)
    print("训练完成！")

    joblib.dump(clf, os.path.join(result_dir, "model.joblib"))
    joblib.dump(monitor, os.path.join(result_dir, "monitor.joblib"))

    # # 使用新的评估系统
    # print("开始生成详细评估报告...")
    # evaluator = quick_evaluate(
    #     clf,
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     monitor=monitor,
    #     noise_indices=noise_idx,
    #     clean_indices=clean_idx,
    # )
    # ========== 选项2: 可视化过拟合过程（可选） ==========
    # 取消下面的注释来运行过拟合可视化
    # print("\n" + "="*60)
    # print("开始过拟合可视化分析...")
    # visualize_overfitting_process(
    #     X_train, y_train, X_test, y_test,
    #     base_estimator=DecisionTreeClassifier(max_depth=1),
    #     n_estimators_list=[1, 5, 10, 20, 30, 40, 50, 75, 100],
    #     learning_rate=0.5,
    #     save_path='results/overfitting_process.png'
    # )
