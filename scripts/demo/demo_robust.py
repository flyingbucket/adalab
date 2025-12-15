"""
鲁棒AdaBoost快速演示
展示如何使用鲁棒方法解决噪声和过拟合问题
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from src.utils import prepare_data
from src.robust_adaboost import create_robust_adaboost


def quick_demo():
    """快速演示"""
    
    print("\n" + "█" * 60)
    print("鲁棒AdaBoost快速演示".center(56))
    print("█" * 60)
    
    # 准备含噪声的数据
    print("\n准备数据（5%标签噪声）...")
    X_train, X_test, y_train, y_test, noise_idx, clean_idx = prepare_data(
        noise_ratio=0.05
    )
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"噪声样本: {len(noise_idx)} ({len(noise_idx)/len(X_train)*100:.1f}%)")
    
    # ========== 1. 标准AdaBoost ==========
    print("\n" + "=" * 60)
    print("1. 标准AdaBoost（基准）")
    print("=" * 60)
    
    clf_standard = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=0.5,
        random_state=42
    )
    
    clf_standard.fit(X_train, y_train)
    
    train_acc_std = clf_standard.score(X_train, y_train)
    test_acc_std = clf_standard.score(X_test, y_test)
    overfit_std = train_acc_std - test_acc_std
    
    print(f"训练集准确率: {train_acc_std:.4f} ({train_acc_std*100:.2f}%)")
    print(f"测试集准确率: {test_acc_std:.4f} ({test_acc_std*100:.2f}%)")
    print(f"过拟合程度: {overfit_std:.4f} ({overfit_std*100:.2f}%)")
    
    # 噪声样本分析
    y_pred_std = clf_standard.predict(X_train)
    noise_acc_std = (y_train[noise_idx] == y_pred_std[noise_idx]).mean()
    clean_acc_std = (y_train[clean_idx] == y_pred_std[clean_idx]).mean()
    
    print(f"噪声样本准确率: {noise_acc_std:.4f}")
    print(f"干净样本准确率: {clean_acc_std:.4f}")
    print(f"准确率差距: {clean_acc_std - noise_acc_std:.4f}")
    
    # ========== 2. 鲁棒AdaBoost（平衡配置）==========
    print("\n" + "=" * 60)
    print("2. 鲁棒AdaBoost - 平衡配置")
    print("=" * 60)
    print("改进策略: 权重裁剪 + 早停")
    
    clf_robust = create_robust_adaboost(strategy="balanced", random_state=42)
    clf_robust.fit(X_train, y_train)
    
    train_acc_rob = clf_robust.score(X_train, y_train)
    test_acc_rob = clf_robust.score(X_test, y_test)
    overfit_rob = train_acc_rob - test_acc_rob
    
    print(f"\n结果:")
    print(f"训练集准确率: {train_acc_rob:.4f} ({train_acc_rob*100:.2f}%)")
    print(f"测试集准确率: {test_acc_rob:.4f} ({test_acc_rob*100:.2f}%)")
    print(f"过拟合程度: {overfit_rob:.4f} ({overfit_rob*100:.2f}%)")
    print(f"使用弱学习器: {clf_robust.best_n_estimators_}/{clf_robust.n_estimators}")
    
    # 噪声样本分析
    y_pred_rob = clf_robust.predict(X_train)
    noise_acc_rob = (y_train[noise_idx] == y_pred_rob[noise_idx]).mean()
    clean_acc_rob = (y_train[clean_idx] == y_pred_rob[clean_idx]).mean()
    
    print(f"噪声样本准确率: {noise_acc_rob:.4f}")
    print(f"干净样本准确率: {clean_acc_rob:.4f}")
    print(f"准确率差距: {clean_acc_rob - noise_acc_rob:.4f}")
    
    # ========== 3. 改进效果对比 ==========
    print("\n" + "█" * 60)
    print("改进效果对比".center(56))
    print("█" * 60)
    
    test_improve = test_acc_rob - test_acc_std
    overfit_improve = overfit_std - overfit_rob
    noise_gap_improve = (clean_acc_std - noise_acc_std) - (clean_acc_rob - noise_acc_rob)
    
    print(f"\n相比标准AdaBoost:")
    print(f"  测试准确率提升: {test_improve:+.4f} ({test_improve*100:+.2f}%)")
    print(f"  过拟合减少: {overfit_improve:+.4f} ({overfit_improve*100:+.2f}%)")
    print(f"  噪声差距缩小: {noise_gap_improve:+.4f} ({noise_gap_improve*100:+.2f}%)")
    
    # ========== 4. 结论 ==========
    print("\n" + "=" * 60)
    print("结论".center(56))
    print("=" * 60)
    
    if test_improve > 0:
        print("\n✅ 测试准确率显著提升")
    
    if overfit_improve > 0.02:
        print("✅ 过拟合显著减少")
    
    if noise_gap_improve > 0:
        print("✅ 噪声鲁棒性提升")
    
    print(f"\n💡 推荐:")
    print(f"   - 对于含噪声数据，使用鲁棒AdaBoost")
    print(f"   - 自动早停找到最佳弱学习器数量 (n={clf_robust.best_n_estimators_})")
    print(f"   - 权重裁剪防止噪声样本权重爆炸")
    
    print("\n" + "=" * 60)
    print("\n✓ 演示完成！")
    print("\n更多配置请查看: docs/robust_adaboost_guide.md")
    print("完整对比请运行: python compare_robust_methods.py")
    print("=" * 60)


if __name__ == "__main__":
    quick_demo()






