"""
快速演示脚本：生成包含 val_idx 的监控数据
用于测试严格模式的可视化功能
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.utils import DataPreparation
from src.monitor import BoostMonitor
from src.patch import boost_with_monitor

print("=" * 60)
print("快速演示：生成包含 val_idx 的监控数据".center(60))
print("=" * 60)

# 1. 准备数据
print("\n📊 准备数据...")
data_prep = DataPreparation(
    noise_ratio=0.0,
    test_size=0.2,
    random_state=42
)

X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices = data_prep.prepare()
print(f"✓ 训练集: {X_train.shape[0]} 样本")
print(f"✓ 测试集: {X_test.shape[0]} 样本")

# 2. 初始化监控器（包含 val_idx 支持）
print("\n📈 初始化监控器...")
monitor = BoostMonitor(
    noise_indices=train_noise_indices,
    clean_indices=train_clean_indices,
    is_data_noisy=False
)
print(f"✓ BoostMonitor 初始化完成")
print(f"✓ val_idx 字段存在: {hasattr(monitor, 'val_idx')}")

# 3. 训练模型并模拟监控数据
print("\n🚀 模拟训练过程（50个弱学习器）...")

# 模拟50轮训练的监控数据
n_estimators = 50
for i in range(n_estimators):
    # 模拟错误率和alpha值（递减趋势）
    error = 0.35 - 0.003 * i + np.random.normal(0, 0.01)
    error = np.clip(error, 0.1, 0.5)
    alpha = np.log((1 - error) / error)
    
    monitor.error_history.append(error)
    monitor.alpha_history.append(alpha)
    monitor.error_without_weight_history.append(error)
    
    # 模拟样本权重
    sample_weights = np.random.uniform(0.0001, 0.002, len(X_train))
    sample_weights /= sample_weights.sum()
    monitor.sample_weights_history.append(sample_weights)

print(f"✓ 模拟完成 {n_estimators} 轮训练数据")

# 4. 训练一个真实模型用于验证
print("\n🎯 训练真实模型用于验证...")
clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42,
    algorithm='SAMME'
)
clf.fit(X_train, y_train)
print("✓ 模型训练完成")

# 5. val-after-train 模式：每10轮验证一次
print("\n🔍 执行 val-after-train 验证...")
val_rounds = [10, 20, 30, 40, 50]
for round_idx in val_rounds:
    # 使用训练好的模型的前N个弱学习器
    clf_temp = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=round_idx,
        learning_rate=1.0,
        random_state=42,
        algorithm='SAMME'
    )
    clf_temp.estimators_ = clf.estimators_[:round_idx]
    clf_temp.estimator_weights_ = clf.estimator_weights_[:round_idx]
    clf_temp.estimator_errors_ = clf.estimator_errors_[:round_idx]
    clf_temp.classes_ = clf.classes_
    clf_temp.n_classes_ = clf.n_classes_
    
    # 在测试集上验证
    y_pred = clf_temp.predict(X_test)
    val_acc = accuracy_score(y_test, y_pred)
    val_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 记录验证结果（注意：round_idx 是1-based的轮次）
    monitor.record_validation(round_idx - 1, val_acc, val_f1)
    print(f"  Round {round_idx:2d}: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

print(f"\n✓ 验证完成，共 {len(monitor.val_idx)} 个验证点")
print(f"✓ val_idx: {monitor.val_idx}")
print(f"✓ val_acc: {[f'{acc:.4f}' for acc in monitor.val_acc_history]}")

# 5. 保存监控数据
print("\n💾 保存监控数据...")
output_dir = "experiments/strict_mode_demo/results"
os.makedirs(output_dir, exist_ok=True)

joblib_path = os.path.join(output_dir, "monitor.joblib")
joblib.dump(monitor, joblib_path)
print(f"✓ 已保存到: {joblib_path}")

# 验证保存的数据
print("\n🔍 验证保存的数据...")
loaded_monitor = joblib.load(joblib_path)
print(f"✓ 加载成功")
print(f"✓ val_idx 存在: {hasattr(loaded_monitor, 'val_idx')}")
print(f"✓ val_idx 长度: {len(loaded_monitor.val_idx)}")
print(f"✓ val_acc_history 长度: {len(loaded_monitor.val_acc_history)}")
print(f"✓ 长度匹配: {len(loaded_monitor.val_idx) == len(loaded_monitor.val_acc_history)}")

print("\n" + "=" * 60)
print("✅ 演示数据生成完成！")
print("=" * 60)
print("\n现在可以运行可视化：")
print(f"python scripts/visualization/visualize_from_results.py \\")
print(f"    --joblib {joblib_path}")
print("=" * 60)

