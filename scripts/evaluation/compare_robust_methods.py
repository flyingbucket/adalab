"""
对比标准AdaBoost和鲁棒方法
清楚展示改进效果
"""

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from src.robust_adaboost import create_robust_adaboost
from src.utils import prepare_data

# 初始化字体
# init()  # 如需中文支持可取消注释
# matplotlib.rcParams["font.family"] = "Source Han Sans CN"  # 中文字体
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # 英文字体（跨平台兼容）
matplotlib.rcParams["axes.unicode_minus"] = False


def train_and_evaluate(clf, X_train, y_train, X_test, y_test, name):
    """
    训练并评估模型

    Parameters
    ----------
    clf : 分类器
    X_train, y_train : 训练集
    X_test, y_test : 测试集
    name : 模型名称

    Returns
    -------
    结果字典
    """
    print(f"\n{'=' * 60}")
    print(f"训练: {name}")
    print(f"{'=' * 60}")

    # 训练
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    # 评估
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    overfit = train_acc - test_acc

    # 获取实际使用的弱学习器数量
    if hasattr(clf, "best_n_estimators_"):
        n_used = clf.best_n_estimators_
    elif hasattr(clf, "n_estimators_"):
        n_used = clf.n_estimators_
    else:
        n_used = clf.n_estimators

    print(f"训练时间: {train_time:.2f} 秒")
    print(f"使用弱学习器数量: {n_used}")
    print(f"训练集准确率: {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"过拟合程度: {overfit:.4f} ({overfit * 100:.2f}%)")

    return {
        "name": name,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "overfit": overfit,
        "train_time": train_time,
        "n_used": n_used,
        "model": clf,
    }


def compare_on_noisy_samples(results, X_train, y_train, noise_idx, clean_idx):
    """
    对比在噪声样本上的表现

    Parameters
    ----------
    results : 结果列表
    X_train, y_train : 训练集
    noise_idx : 噪声样本索引
    clean_idx : 干净样本索引
    """
    if len(noise_idx) == 0:
        print("\n无噪声样本，跳过噪声分析")
        return

    print("\n" + "=" * 60)
    print("噪声样本分析".center(56))
    print("=" * 60)

    for result in results:
        clf = result["model"]
        y_pred = clf.predict(X_train)

        # 噪声样本准确率
        noise_acc = accuracy_score(y_train[noise_idx], y_pred[noise_idx])

        # 干净样本准确率
        clean_acc = accuracy_score(y_train[clean_idx], y_pred[clean_idx])

        # 差距
        gap = clean_acc - noise_acc

        print(f"\n{result['name']}:")
        print(f"  噪声样本准确率: {noise_acc:.4f} ({noise_acc * 100:.2f}%)")
        print(f"  干净样本准确率: {clean_acc:.4f} ({clean_acc * 100:.2f}%)")
        print(f"  准确率差距: {gap:.4f} ({gap * 100:.2f}%)")

        result["noise_acc"] = noise_acc
        result["clean_acc"] = clean_acc
        result["noise_gap"] = gap


def plot_comparison(results, save_path=None):
    """
    可视化对比结果

    Parameters
    ----------
    results : 结果列表
    save_path : 保存路径
    """
    n_models = len(results)
    names = [r["name"] for r in results]

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 训练 vs 测试准确率
    ax1 = axes[0, 0]
    x = np.arange(n_models)
    width = 0.35

    train_accs = [r["train_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]

    ax1.bar(x - width / 2, train_accs, width, label="训练集", color="skyblue")
    ax1.bar(x + width / 2, test_accs, width, label="测试集", color="lightcoral")

    ax1.set_ylabel("准确率", fontsize=12)
    ax1.set_title("训练集 vs 测试集准确率", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim([0.5, 1.0])

    # 子图2: 过拟合程度
    ax2 = axes[0, 1]
    overfits = [r["overfit"] for r in results]
    colors = [
        "red" if o > 0.15 else "orange" if o > 0.10 else "green" for o in overfits
    ]

    bars = ax2.bar(x, overfits, color=colors, alpha=0.7)
    ax2.axhline(y=0.10, color="orange", linestyle="--", linewidth=1, label="阈值:10%")
    ax2.axhline(y=0.15, color="red", linestyle="--", linewidth=1, label="阈值:15%")

    ax2.set_ylabel("过拟合程度", fontsize=12)
    ax2.set_title("过拟合程度对比", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # 子图3: 噪声 vs 干净样本准确率（如果有）
    ax3 = axes[1, 0]
    if "noise_acc" in results[0]:
        noise_accs = [r["noise_acc"] for r in results]
        clean_accs = [r["clean_acc"] for r in results]

        ax3.bar(x - width / 2, noise_accs, width, label="噪声样本", color="salmon")
        ax3.bar(x + width / 2, clean_accs, width, label="干净样本", color="lightgreen")

        ax3.set_ylabel("准确率", fontsize=12)
        ax3.set_title("噪声样本 vs 干净样本准确率", fontsize=14)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=15, ha="right")
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "无噪声数据",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax3.transAxes,
        )
        ax3.set_xticks([])
        ax3.set_yticks([])

    # 子图4: 训练时间和弱学习器数量
    ax4 = axes[1, 1]
    train_times = [r["train_time"] for r in results]
    n_used_list = [r["n_used"] for r in results]

    ax4_twin = ax4.twinx()

    bars1 = ax4.bar(
        x - width / 2, train_times, width, label="训练时间(秒)", color="steelblue"
    )
    bars2 = ax4_twin.bar(
        x + width / 2, n_used_list, width, label="弱学习器数量", color="darkorange"
    )

    ax4.set_ylabel("训练时间 (秒)", fontsize=12, color="steelblue")
    ax4_twin.set_ylabel("弱学习器数量", fontsize=12, color="darkorange")
    ax4.set_title("训练效率对比", fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=15, ha="right")

    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax4.grid(axis="y", alpha=0.3)

    plt.suptitle("AdaBoost 方法对比", fontsize=16, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n对比图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def print_summary(results):
    """
    打印总结报告

    Parameters
    ----------
    results : 结果列表
    """
    print("\n" + "█" * 60)
    print("对比总结".center(56))
    print("█" * 60)

    # 找出最佳模型
    best_test = max(results, key=lambda x: x["test_acc"])
    best_overfit = min(results, key=lambda x: x["overfit"])

    print("\n🏆 最佳测试准确率:")
    print(f"   {best_test['name']}: {best_test['test_acc']:.4f}")

    print("\n✓ 最小过拟合:")
    print(f"   {best_overfit['name']}: {best_overfit['overfit']:.4f}")

    if "noise_gap" in results[0]:
        best_noise = min(results, key=lambda x: x["noise_gap"])
        print("\n💡 最佳噪声鲁棒性:")
        print(f"   {best_noise['name']}: 噪声差距 = {best_noise['noise_gap']:.4f}")

    # 改进幅度
    standard = next((r for r in results if "标准" in r["name"]), None)
    if standard:
        print("\n📈 相比标准AdaBoost的改进:")
        for result in results:
            if result["name"] == standard["name"]:
                continue

            test_improve = result["test_acc"] - standard["test_acc"]
            overfit_improve = standard["overfit"] - result["overfit"]

            print(f"\n   {result['name']}:")
            print(f"     测试准确率: {test_improve:+.4f} ({test_improve * 100:+.2f}%)")
            print(
                f"     过拟合减少: {overfit_improve:+.4f} ({overfit_improve * 100:+.2f}%)"
            )

    print("\n" + "=" * 60)


def main():
    """主函数"""
    print("\n" + "█" * 60)
    print("AdaBoost 鲁棒方法对比实验".center(56))
    print("█" * 60)

    # ========== 1. 准备数据 ==========
    print("\n步骤1: 准备数据")
    print("-" * 60)

    # 使用含噪声的数据（更能体现改进效果）
    noise_ratio = 0.05  # 可以改为 0.10 测试更高噪声
    X_train, X_test, y_train, y_test, noise_idx, clean_idx = prepare_data(
        noise_ratio=noise_ratio
    )

    print("数据集: MNIST")
    print(f"噪声比例: {noise_ratio * 100:.0f}%")
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"噪声样本: {len(noise_idx)}")

    # ========== 2. 定义要对比的模型 ==========
    print("\n步骤2: 准备对比模型")
    print("-" * 60)

    base = DecisionTreeClassifier(max_depth=1)

    models = [
        # 标准AdaBoost（基准）
        (
            AdaBoostClassifier(
                estimator=base, n_estimators=50, learning_rate=0.5, random_state=42
            ),
            "标准AdaBoost",
        ),
        # 鲁棒方法1: 平衡配置
        (create_robust_adaboost("balanced", random_state=42), "鲁棒-平衡"),
        # 鲁棒方法2: 激进裁剪（最适合高噪声）
        (create_robust_adaboost("aggressive_clip", random_state=42), "鲁棒-激进裁剪"),
        # 鲁棒方法3: 重点早停
        (create_robust_adaboost("early_stop", random_state=42), "鲁棒-早停"),
    ]

    print(f"对比模型数量: {len(models)}")
    for _, name in models:
        print(f"  - {name}")

    # ========== 3. 训练和评估 ==========
    print("\n步骤3: 训练和评估模型")
    print("-" * 60)

    results = []
    for clf, name in models:
        result = train_and_evaluate(clf, X_train, y_train, X_test, y_test, name)
        results.append(result)

    # ========== 4. 噪声样本分析 ==========
    compare_on_noisy_samples(results, X_train, y_train, noise_idx, clean_idx)

    # ========== 5. 可视化对比 ==========
    print("\n步骤4: 生成对比可视化")
    print("-" * 60)
    plot_comparison(results, save_path="results/robust_comparison.png")

    # ========== 6. 打印总结 ==========
    print_summary(results)

    # ========== 7. 建议 ==========
    print("\n💡 建议:")
    best_result = max(results, key=lambda x: x["test_acc"])

    if best_result["overfit"] < 0.10:
        print(f"   ✅ 推荐使用: {best_result['name']}")
        print(f"      - 测试准确率最高: {best_result['test_acc']:.4f}")
        print(f"      - 过拟合程度低: {best_result['overfit']:.4f}")
    elif best_result["overfit"] < 0.15:
        print(f"   ⚠️ 可以使用: {best_result['name']}")
        print(f"      - 测试准确率: {best_result['test_acc']:.4f}")
        print(f"      - 过拟合程度中等: {best_result['overfit']:.4f}")
        print("      - 建议进一步调整参数")
    else:
        print(f"   ⚠️ {best_result['name']} 仍有明显过拟合")
        print("      - 建议使用'激进裁剪'或'保守'配置")
        print("      - 或降低学习率到 0.1-0.3")

    if noise_ratio > 0:
        print("\n   💡 噪声数据建议:")
        print(f"      - 当前噪声: {noise_ratio * 100:.0f}%")
        if noise_ratio >= 0.1:
            print("      - 推荐使用: '鲁棒-激进裁剪' 或 '保守' 配置")
        else:
            print("      - 推荐使用: '鲁棒-平衡' 配置")

    print("\n" + "=" * 60)
    print("\n✓ 对比实验完成！")


if __name__ == "__main__":
    main()
