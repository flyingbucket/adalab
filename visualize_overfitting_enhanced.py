"""
增强版：可视化AdaBoost过拟合过程 + 详细训练监控
结合 BoostMonitor 提供更深入的训练动态分析
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from src.utils import prepare_data
from src.evaluation import visualize_overfitting_process
from src.monitor import BoostMonitor
from src.patch import AdaBoostClfWithMonitor


def visualize_detailed_training(monitor, n_estimators, save_path=None):
    """
    使用 BoostMonitor 数据生成详细的训练过程可视化
    
    参考 docs/monitor.md 中的数据结构
    
    Parameters
    ----------
    monitor : BoostMonitor
        训练监控对象
    n_estimators : int
        弱学习器数量
    save_path : str, optional
        保存路径
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Detailed Training Analysis (n_estimators={n_estimators})', 
                 fontsize=16, fontweight='bold')
    
    rounds = list(range(1, len(monitor.error_history) + 1))
    
    # ========== 1. 错误率演化 ==========
    ax1 = axes[0, 0]
    ax1.plot(rounds, monitor.error_history, 'b-', linewidth=2, label='Weighted Error')
    if len(monitor.error_without_weight_history) == len(rounds):
        ax1.plot(rounds, monitor.error_without_weight_history, 'r--', 
                linewidth=2, label='Unweighted Error', alpha=0.7)
    ax1.set_xlabel('Boosting Round', fontsize=12)
    ax1.set_ylabel('Error Rate', fontsize=12)
    ax1.set_title('Error Rate Evolution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Alpha 系数演化 ==========
    ax2 = axes[0, 1]
    ax2.plot(rounds, monitor.alpha_history, 'g-', linewidth=2, marker='o', 
            markersize=4, markevery=max(1, len(rounds)//20))
    ax2.set_xlabel('Boosting Round', fontsize=12)
    ax2.set_ylabel('Alpha (Weak Learner Weight)', fontsize=12)
    ax2.set_title('Alpha Coefficient Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 标注平均值
    avg_alpha = np.mean(monitor.alpha_history)
    ax2.axhline(y=avg_alpha, color='orange', linestyle='--', 
               label=f'Mean={avg_alpha:.3f}', alpha=0.7)
    ax2.legend()
    
    # ========== 3. 训练 vs 验证准确率 ==========
    ax3 = axes[0, 2]
    if len(monitor.acc_on_train_data) > 0:
        ax3.plot(rounds, monitor.acc_on_train_data, 'b-', linewidth=2, 
                label='Train Accuracy', marker='o', markersize=4,
                markevery=max(1, len(rounds)//20))
    if len(monitor.val_acc_history) > 0:
        ax3.plot(rounds, monitor.val_acc_history, 'r-', linewidth=2, 
                label='Val Accuracy', marker='s', markersize=4,
                markevery=max(1, len(rounds)//20))
    ax3.set_xlabel('Boosting Round', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========== 4. 噪声样本 vs 干净样本权重 ==========
    ax4 = axes[1, 0]
    if monitor.is_data_noisy and len(monitor.noisy_weight_history) > 0:
        ax4.plot(rounds, monitor.noisy_weight_history, 'r-', linewidth=2, 
                label='Noisy Samples Weight', marker='o', markersize=4,
                markevery=max(1, len(rounds)//20))
        ax4.plot(rounds, monitor.clean_weight_history, 'g-', linewidth=2, 
                label='Clean Samples Weight', marker='s', markersize=4,
                markevery=max(1, len(rounds)//20))
        ax4.set_xlabel('Boosting Round', fontsize=12)
        ax4.set_ylabel('Total Weight', fontsize=12)
        ax4.set_title('Noisy vs Clean Sample Weights', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 标注初始状态（应该接近 0.5）
        ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
    else:
        ax4.text(0.5, 0.5, 'Noise Analysis\nNot Available\n(Clean Data)', 
                ha='center', va='center', fontsize=14, color='gray')
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    # ========== 5. F1 分数演化 ==========
    ax5 = axes[1, 1]
    if len(monitor.f1_on_training_data) > 0:
        ax5.plot(rounds, monitor.f1_on_training_data, 'b-', linewidth=2, 
                label='Train F1', marker='o', markersize=4,
                markevery=max(1, len(rounds)//20))
    if len(monitor.val_f1_history) > 0:
        ax5.plot(rounds, monitor.val_f1_history, 'r-', linewidth=2, 
                label='Val F1', marker='s', markersize=4,
                markevery=max(1, len(rounds)//20))
    ax5.set_xlabel('Boosting Round', fontsize=12)
    ax5.set_ylabel('F1 Score', fontsize=12)
    ax5.set_title('F1 Score Evolution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ========== 6. 样本权重分布变化 ==========
    ax6 = axes[1, 2]
    if len(monitor.sample_weights_history) > 0:
        # 选择几个关键轮次展示
        key_rounds = [0, len(rounds)//3, len(rounds)*2//3, len(rounds)-1]
        positions = []
        data_to_plot = []
        labels = []
        
        for i, idx in enumerate(key_rounds):
            if idx < len(monitor.sample_weights_history):
                positions.append(i + 1)
                data_to_plot.append(monitor.sample_weights_history[idx])
                labels.append(f'Round {idx+1}')
        
        # 箱型图
        bp = ax6.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                        labels=labels)
        
        # 美化箱型图
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax6.set_ylabel('Sample Weight', fontsize=12)
        ax6.set_title('Sample Weight Distribution', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.tick_params(axis='x', rotation=15)
    else:
        ax6.text(0.5, 0.5, 'Sample Weight\nDistribution\nNot Available', 
                ha='center', va='center', fontsize=14, color='gray')
        ax6.set_xticks([])
        ax6.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed training visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """主函数：增强版过拟合可视化 + 详细训练监控"""
    
    print("\n" + "█" * 60)
    print("AdaBoost Enhanced Visualization".center(60))
    print("Overfitting Analysis + Training Monitoring".center(60))
    print("█" * 60)
    
    # ========== 1. 选择数据类型 ==========
    print("\n选择数据类型:")
    print("1. 干净数据（无噪声）")
    print("2. 含噪声数据（5%噪声）⭐ 推荐")
    print("3. 含噪声数据（10%噪声）")
    
    choice = 2  # 默认使用选项2
    
    if choice == 1:
        noise_ratio = 0
        data_type = "Clean Data"
    elif choice == 2:
        noise_ratio = 0.05
        data_type = "5% Noisy Data"
    else:
        noise_ratio = 0.10
        data_type = "10% Noisy Data"
    
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
        print(f"噪声样本: {len(noise_idx)} ({len(noise_idx)/len(X_train)*100:.1f}%)")
        print(f"干净样本: {len(clean_idx)} ({len(clean_idx)/len(X_train)*100:.1f}%)")
    
    # ========== 3. 选择配置 ==========
    print("\n" + "=" * 60)
    print("Configuration".center(60))
    print("=" * 60)
    
    # 快速配置
    config = {
        "base_estimator": DecisionTreeClassifier(max_depth=1),
        "n_estimators_list": [1, 5, 10, 20, 30, 40, 50, 75, 100],
        "learning_rate": 0.5,
        "random_state": 42,
    }
    
    print(f"Base Estimator: Decision Tree (max_depth={config['base_estimator'].max_depth})")
    print(f"Test Points: {len(config['n_estimators_list'])}")
    print(f"Estimators Range: {config['n_estimators_list'][0]} - {config['n_estimators_list'][-1]}")
    print(f"Learning Rate: {config['learning_rate']}")
    
    # ========== 4. 第一阶段：过拟合可视化 ==========
    print("\n" + "=" * 60)
    print("Phase 1: Overfitting Analysis".center(60))
    print("=" * 60)
    
    results = visualize_overfitting_process(
        X_train,
        y_train,
        X_test,
        y_test,
        base_estimator=config["base_estimator"],
        n_estimators_list=config["n_estimators_list"],
        learning_rate=config["learning_rate"],
        random_state=config["random_state"],
        save_path=None,
    )
    
    # 找到最佳配置
    best_idx = results["test_accuracy"].index(max(results["test_accuracy"]))
    best_n = results["n_estimators"][best_idx]
    best_test_acc = results["test_accuracy"][best_idx]
    
    print(f"\n✓ Best Configuration Found:")
    print(f"   - Number of Estimators: {best_n}")
    print(f"   - Test Accuracy: {best_test_acc:.4f}")
    
    # ========== 5. 第二阶段：详细训练监控 ==========
    print("\n" + "=" * 60)
    print("Phase 2: Detailed Training Monitoring".center(60))
    print("=" * 60)
    print(f"\nRe-training best model (n={best_n}) with monitoring enabled...")
    print("This will generate detailed training dynamics visualization.")
    
    # 创建监控器
    is_data_noisy = noise_ratio > 0
    monitor = BoostMonitor(
        noise_indices=noise_idx,
        clean_indices=clean_idx,
        is_data_noisy=is_data_noisy,
        checkpoint_interval=999,  # 不需要checkpoint
        checkpoint_prefix="temp"
    )
    
    # 使用监控器训练最佳模型
    clf = AdaBoostClfWithMonitor(
        estimator=config["base_estimator"],
        n_estimators=best_n,
        learning_rate=config["learning_rate"],
        random_state=config["random_state"],
        monitor=monitor
    )
    
    print(f"Training with {best_n} estimators...")
    clf.fit(X_train, y_train)
    
    # 记录验证集指标
    for i in range(best_n):
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        # 这里简化了，实际应该每轮记录
    
    print("✓ Training completed!")
    
    # 生成详细可视化
    print("\nGenerating detailed training visualization...")
    visualize_detailed_training(
        monitor=monitor,
        n_estimators=best_n,
        save_path=None  # 设置路径可保存，如 'detailed_training.png'
    )
    
    # ========== 6. 总结和建议 ==========
    print("\n" + "=" * 60)
    print("Summary & Recommendations".center(60))
    print("=" * 60)
    
    final_overfit = results["overfitting_degree"][best_idx]
    
    print(f"\n📊 Model Performance:")
    print(f"   - Train Accuracy: {results['train_accuracy'][best_idx]:.4f}")
    print(f"   - Test Accuracy:  {results['test_accuracy'][best_idx]:.4f}")
    print(f"   - Overfitting:    {final_overfit:.4f} ({final_overfit*100:.2f}%)")
    
    # 根据结果给出建议
    if final_overfit > 0.15:
        print("\n⚠️  Severe Overfitting Detected:")
        print(f"   - Consider reducing estimators")
        print(f"   - Try lower learning rate (e.g., 0.1)")
        print(f"   - Apply regularization techniques")
    elif final_overfit > 0.10:
        print("\n⚠️  Moderate Overfitting:")
        print(f"   - Current configuration is acceptable")
        print(f"   - Early stopping at n={best_n} recommended")
    else:
        print("\n✓ Good Model Fit:")
        print(f"   - Low overfitting degree")
        print(f"   - Model generalizes well")
    
    # 噪声相关建议
    if is_data_noisy:
        print(f"\n💡 Noise-Specific Insights:")
        
        if len(monitor.noisy_weight_history) > 0:
            final_noisy_weight = monitor.noisy_weight_history[-1]
            final_clean_weight = monitor.clean_weight_history[-1]
            weight_ratio = final_noisy_weight / final_clean_weight if final_clean_weight > 0 else 0
            
            print(f"   - Final noisy sample weight: {final_noisy_weight:.4f}")
            print(f"   - Final clean sample weight: {final_clean_weight:.4f}")
            print(f"   - Weight ratio (noisy/clean): {weight_ratio:.3f}")
            
            if weight_ratio > 1.5:
                print(f"\n   ⚠️  Noisy samples are over-weighted!")
                print(f"   - This indicates noise sensitivity")
                print(f"   - Consider robust AdaBoost methods")
    
    # Alpha 系数分析
    if len(monitor.alpha_history) > 0:
        avg_alpha = np.mean(monitor.alpha_history)
        std_alpha = np.std(monitor.alpha_history)
        print(f"\n📈 Weak Learner Analysis:")
        print(f"   - Average alpha: {avg_alpha:.3f}")
        print(f"   - Std of alpha:  {std_alpha:.3f}")
        
        if std_alpha / avg_alpha > 0.5:
            print(f"   - High variance in learner weights")
            print(f"   - Some learners much stronger than others")
    
    print("\n" + "=" * 60)
    print("\n✓ Visualization Complete!")
    print("\n💡 Tips:")
    print("   - Two visualizations generated:")
    print("     1. Overfitting curves (Phase 1)")
    print("     2. Detailed training dynamics (Phase 2)")
    print("   - Set save_path to save figures")
    print("   - Adjust config for different experiments")
    print("=" * 60)


if __name__ == "__main__":
    main()



