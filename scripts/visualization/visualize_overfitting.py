"""
可视化AdaBoost过拟合过程
简洁的脚本，展示模型随着弱学习器数量增加的过拟合行为

可选功能：启用详细训练监控（参考 docs/monitor.md）
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.evaluation import visualize_overfitting_process
from src.patch import AdaBoostClfWithMonitor
from src.utils import prepare_data

from src.monitor import BoostMonitor


def visualize_monitor_data(monitor, n_estimators, is_noisy):
    """
    可视化 BoostMonitor 记录的训练数据
    参考 docs/monitor.md 中的数据结构

    生成 6 个子图：
    1. 错误率演化（weighted vs unweighted）
    2. Alpha 系数演化
    3. 训练 vs 验证准确率
    4. 噪声样本 vs 干净样本权重（仅噪声数据）
    5. F1 分数演化
    6. 样本权重分布变化
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Detailed Training Monitoring (n_estimators={n_estimators})",
        fontsize=16,
        fontweight="bold",
    )

    rounds = list(range(1, len(monitor.error_history) + 1))

    # 1. 错误率演化
    ax1 = axes[0, 0]
    ax1.plot(rounds, monitor.error_history, "b-", linewidth=2, label="Weighted Error")
    if len(monitor.error_without_weight_history) == len(rounds):
        ax1.plot(
            rounds,
            monitor.error_without_weight_history,
            "r--",
            linewidth=2,
            label="Unweighted Error",
            alpha=0.7,
        )
    ax1.set_xlabel("Boosting Round")
    ax1.set_ylabel("Error Rate")
    ax1.set_title("Error Rate Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Alpha 系数
    ax2 = axes[0, 1]
    ax2.plot(rounds, monitor.alpha_history, "g-", linewidth=2, marker="o", markersize=4)
    ax2.axhline(
        y=np.mean(monitor.alpha_history),
        color="orange",
        linestyle="--",
        label=f"Mean={np.mean(monitor.alpha_history):.3f}",
        alpha=0.7,
    )
    ax2.set_xlabel("Boosting Round")
    ax2.set_ylabel("Alpha (Weak Learner Weight)")
    ax2.set_title("Alpha Coefficient Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 准确率
    ax3 = axes[0, 2]
    if len(monitor.acc_on_train_data) > 0:
        ax3.plot(
            rounds,
            monitor.acc_on_train_data,
            "b-",
            linewidth=2,
            label="Train Accuracy",
            marker="o",
            markersize=4,
        )
    if len(monitor.val_acc_history) > 0:
        ax3.plot(
            rounds,
            monitor.val_acc_history,
            "r-",
            linewidth=2,
            label="Val Accuracy",
            marker="s",
            markersize=4,
        )
    ax3.set_xlabel("Boosting Round")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Accuracy Evolution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 噪声 vs 干净样本权重
    ax4 = axes[1, 0]
    if is_noisy and len(monitor.noisy_weight_history) > 0:
        ax4.plot(
            rounds,
            monitor.noisy_weight_history,
            "r-",
            linewidth=2,
            label="Noisy Samples",
            marker="o",
            markersize=4,
        )
        ax4.plot(
            rounds,
            monitor.clean_weight_history,
            "g-",
            linewidth=2,
            label="Clean Samples",
            marker="s",
            markersize=4,
        )
        ax4.axhline(y=0.5, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax4.set_xlabel("Boosting Round")
        ax4.set_ylabel("Total Weight")
        ax4.set_title("Noisy vs Clean Sample Weights")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "N/A\n(Clean Data)",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax4.transAxes,
        )
        ax4.set_xticks([])
        ax4.set_yticks([])

    # 5. F1 分数
    ax5 = axes[1, 1]
    if len(monitor.f1_on_training_data) > 0:
        ax5.plot(
            rounds,
            monitor.f1_on_training_data,
            "b-",
            linewidth=2,
            label="Train F1",
            marker="o",
            markersize=4,
        )
    if len(monitor.val_f1_history) > 0:
        ax5.plot(
            rounds,
            monitor.val_f1_history,
            "r-",
            linewidth=2,
            label="Val F1",
            marker="s",
            markersize=4,
        )
    ax5.set_xlabel("Boosting Round")
    ax5.set_ylabel("F1 Score")
    ax5.set_title("F1 Score Evolution")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 样本权重分布
    ax6 = axes[1, 2]
    if len(monitor.sample_weights_history) > 0:
        # 选择关键轮次
        key_rounds = [0, len(rounds) // 3, len(rounds) * 2 // 3, len(rounds) - 1]
        positions = []
        data_to_plot = []
        labels = []

        for i, idx in enumerate(key_rounds):
            if idx < len(monitor.sample_weights_history):
                positions.append(i + 1)
                data_to_plot.append(monitor.sample_weights_history[idx])
                labels.append(f"R{idx + 1}")

        bp = ax6.boxplot(
            data_to_plot,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            labels=labels,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        ax6.set_ylabel("Sample Weight")
        ax6.set_title("Sample Weight Distribution")
        ax6.grid(True, alpha=0.3, axis="y")
    else:
        ax6.text(
            0.5,
            0.5,
            "N/A",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax6.transAxes,
        )
        ax6.set_xticks([])
        ax6.set_yticks([])

    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    """主函数：可视化过拟合过程"""

    print("\n" + "█" * 60)
    print("AdaBoost 过拟合可视化".center(56))
    print("█" * 60)

    # ========== 1. 选择数据类型 ==========
    print("\n选择数据类型:")
    print("1. 干净数据（无噪声）")
    print("2. 含噪声数据（5%噪声）")
    print("3. 含噪声数据（10%噪声）")

    # 默认使用选项2
    choice = 2  # 可以修改为1或3

    if choice == 1:
        noise_ratio = 0
        data_type = "干净数据"
    elif choice == 2:
        noise_ratio = 0.05
        data_type = "5%噪声数据"
    else:
        noise_ratio = 0.10
        data_type = "10%噪声数据"

    print(f"\n使用: {data_type}")
    print("-" * 60)

    # ========== 2. 准备数据 ==========
    print("\n准备数据...")
    X_train, X_test, y_train, y_test, noise_idx, clean_idx = prepare_data(
        noise_ratio=noise_ratio
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    if noise_ratio > 0:
        print(f"噪声样本: {len(noise_idx)}")
        print(f"干净样本: {len(clean_idx)}")

    # ========== 3. 选择配置 ==========
    print("\n" + "=" * 60)
    print("配置选项".center(56))
    print("=" * 60)

    # 配置1: 快速测试（推荐）
    config = {
        "base_estimator": DecisionTreeClassifier(max_depth=1),  # 决策树桩
        "n_estimators_list": [1, 5, 10, 20, 30, 40, 50, 75, 100],  # 测试点
        "learning_rate": 0.5,  # 学习率
        "random_state": 42,
    }

    # 配置2: 精细分析（更多测试点，需要更长时间）
    # config = {
    #     "base_estimator": DecisionTreeClassifier(max_depth=1),
    #     "n_estimators_list": list(range(1, 101, 5)),  # [1, 6, 11, ..., 96]
    #     "learning_rate": 0.5,
    #     "random_state": 42,
    # }

    # 配置3: 深树测试（观察更复杂基学习器的影响）
    # config = {
    #     "base_estimator": DecisionTreeClassifier(max_depth=3),
    #     "n_estimators_list": [1, 5, 10, 20, 30, 40, 50],
    #     "learning_rate": 0.5,
    #     "random_state": 42,
    # }

    print(f"基学习器: 决策树 (max_depth={config['base_estimator'].max_depth})")
    print(f"测试点数量: {len(config['n_estimators_list'])}")
    print(
        f"弱学习器范围: {config['n_estimators_list'][0]} - {config['n_estimators_list'][-1]}"
    )
    print(f"学习率: {config['learning_rate']}")

    # ========== 4. 可视化过拟合 ==========
    print("\n开始训练和可视化...")
    print("-" * 60)

    results = visualize_overfitting_process(
        X_train,
        y_train,
        X_test,
        y_test,
        base_estimator=config["base_estimator"],
        n_estimators_list=config["n_estimators_list"],
        learning_rate=config["learning_rate"],
        random_state=config["random_state"],
        save_path=None,  # 设为路径可保存图表，如 'overfitting.png'
    )

    # ========== 5. 额外分析（可选） ==========
    print("\n" + "=" * 60)
    print("建议".center(56))
    print("=" * 60)

    best_idx = results["test_accuracy"].index(max(results["test_accuracy"]))
    best_n = results["n_estimators"][best_idx]
    final_n = results["n_estimators"][-1]
    final_overfit = results["overfitting_degree"][-1]

    # 根据结果给出建议
    if final_overfit > 0.15:
        print("\n⚠️  严重过拟合警告:")
        print(f"   - 当前过拟合程度: {final_overfit:.2%}")
        print(f"   - 建议减少弱学习器数量至 {best_n} 左右")
        print("   - 或使用更小的学习率（如 0.1）")
    elif final_overfit > 0.10:
        print("\n⚠️  中度过拟合:")
        print(f"   - 当前过拟合程度: {final_overfit:.2%}")
        print(f"   - 建议使用早停，在 n={best_n} 处停止训练")
    elif final_overfit < 0.05:
        print("\n✓ 模型拟合良好:")
        print(f"   - 过拟合程度低: {final_overfit:.2%}")
        print("   - 可以考虑增加弱学习器数量以提升性能")
    else:
        print("\n✓ 模型表现良好:")
        print(f"   - 过拟合程度: {final_overfit:.2%} (可接受)")
        print(f"   - 建议使用 n={best_n} 个弱学习器")

    # 噪声数据的额外建议
    if noise_ratio > 0:
        print("\n💡 噪声数据建议:")
        print(f"   - 当前数据有 {noise_ratio * 100:.0f}% 噪声")
        print("   - AdaBoost 对噪声敏感，容易过拟合")
        print("   - 建议:")
        print(f"     1. 使用较少的弱学习器（{best_n} 左右）")
        print("     2. 降低学习率（从 0.5 到 0.3）")
        print("     3. 考虑数据清洗或噪声鲁棒方法")

    # ========== 6. 可选：详细训练监控 ==========
    # 取消下面的注释来启用详细监控可视化
    enable_detailed_monitoring = False  # 设为 True 启用详细监控

    if enable_detailed_monitoring:
        print("\n" + "=" * 60)
        print("详细训练监控".center(56))
        print("=" * 60)
        print(f"\n重新训练最佳模型 (n={best_n})，启用监控...")

        # 创建监控器（参考 docs/monitor.md）
        monitor = BoostMonitor(
            noise_indices=noise_idx,
            clean_indices=clean_idx,
            is_data_noisy=(noise_ratio > 0),
            checkpoint_interval=999,
            checkpoint_prefix="temp",
        )

        # 使用监控器训练
        clf_monitored = AdaBoostClfWithMonitor(
            estimator=config["base_estimator"],
            n_estimators=best_n,
            learning_rate=config["learning_rate"],
            random_state=config["random_state"],
            monitor=monitor,
        )
        clf_monitored.fit(X_train, y_train)

        # 生成详细可视化（6个子图）
        print("\n生成详细训练过程可视化...")
        visualize_monitor_data(monitor, best_n, noise_ratio > 0)

        print("\n✓ 详细监控可视化完成！")
        print("\n📊 监控数据包含:")
        print(f"   - 错误率历史: {len(monitor.error_history)} 轮")
        print(f"   - Alpha系数: {len(monitor.alpha_history)} 轮")
        print(f"   - 样本权重演化: {len(monitor.sample_weights_history)} 轮")

    print("\n" + "=" * 60)
    print("\n✓ 可视化完成！")
    print("\n💡 提示:")
    print("   - 图表会自动显示（关闭窗口继续）")
    print("   - 要保存图表，设置 save_path='overfitting.png'")
    print("   - 要测试不同配置，修改脚本中的 config 字典")
    print("   - 要启用详细监控，设置 enable_detailed_monitoring=True")
    print("=" * 60)


if __name__ == "__main__":
    main()
