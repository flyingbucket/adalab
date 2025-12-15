from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src import monitor
from src.monitor import BoostMonitor
from src.patch import AdaBoostClfWithMonitor
from src.utils import prepare_data
from src.evaluation import quick_evaluate, visualize_overfitting_process


if __name__ == "__main__":
    # 获取含噪声的数据（默认噪声比例5%）
    X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices = (
        prepare_data()
    )

    # ========== 选项1: 单模型训练 + 完整评估 ==========
    # 创建监控器（启用噪声追踪）
    boost_monitor = BoostMonitor(
        train_noise_indices,
        train_clean_indices,
        is_data_noisy=True,
        checkpoint_interval=10,
    )

    # 训练 AdaBoost
    print("训练 AdaBoost 模型...")
    base = DecisionTreeClassifier(max_depth=1)  # 使用决策树桩作为基学习器

    clf = AdaBoostClfWithMonitor(
        _monitor=boost_monitor,
        X_val=X_test,
        y_val=y_test,
        estimator=base,  # 基学习器
        n_estimators=500,  # 弱学习器数量
        learning_rate=0.5,  # 学习率
        random_state=42,  # 随机种子
    )

    # 绑定监控器到分类器
    # clf._monitor = boost_monitor

    # 开始训练
    clf.fit(X_train, y_train)
    clf._monitor.dump("results/total_log.csv")
    print("训练完成！\n")

    # 使用新的评估系统
    print("开始生成详细评估报告...")
    evaluator = quick_evaluate(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        monitor=boost_monitor,
        noise_indices=train_noise_indices,
        clean_indices=train_clean_indices,
    )

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
