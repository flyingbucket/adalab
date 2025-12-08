import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_training_data(data, save_path=None):
    """
    可视化训练数据（6个子图）

    Parameters
    ----------
    data : dict
        监控数据字典
    save_path : str, optional
        保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Training Analysis from Saved Results (n={data['n_estimators']})",
        fontsize=16,
        fontweight="bold",
    )

    rounds = data["rounds"]

    # ========== 1. 错误率演化 ==========
    ax1 = axes[0, 0]
    ax1.plot(rounds, data["error_history"], "b-", linewidth=2, label="Weighted Error")
    if len(data["error_without_weight_history"]) > 0:
        ax1.plot(
            rounds,
            data["error_without_weight_history"],
            "r--",
            linewidth=2,
            label="Unweighted Error",
            alpha=0.7,
        )
    ax1.set_xlabel("Boosting Round", fontsize=12)
    ax1.set_ylabel("Error Rate", fontsize=12)
    ax1.set_title("Error Rate Evolution", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ========== 2. Alpha 系数演化 ==========
    ax2 = axes[0, 1]
    ax2.plot(
        rounds,
        data["alpha_history"],
        "g-",
        linewidth=2,
        marker="o",
        markersize=4,
        markevery=max(1, len(rounds) // 20),
    )
    avg_alpha = np.mean(data["alpha_history"])
    ax2.axhline(
        y=avg_alpha,
        color="orange",
        linestyle="--",
        label=f"Mean={avg_alpha:.3f}",
        alpha=0.7,
    )
    ax2.set_xlabel("Boosting Round", fontsize=12)
    ax2.set_ylabel("Alpha (Weak Learner Weight)", fontsize=12)
    ax2.set_title("Alpha Coefficient Evolution", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ========== 3. 训练 vs 验证准确率 ==========
    ax3 = axes[0, 2]
    if len(data["acc_on_train_data"]) > 0:
        ax3.plot(
            rounds,
            data["acc_on_train_data"],
            "b-",
            linewidth=2,
            label="Train Accuracy",
            marker="o",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    if len(data["val_acc_history"]) > 0:
        ax3.plot(
            rounds,
            data["val_acc_history"],
            "r-",
            linewidth=2,
            label="Val Accuracy",
            marker="s",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    ax3.set_xlabel("Boosting Round", fontsize=12)
    ax3.set_ylabel("Accuracy", fontsize=12)
    ax3.set_title("Accuracy Evolution", fontsize=14, fontweight="bold")

    # 如果没有训练集准确率数据
    if len(data["acc_on_train_data"]) == 0 and len(data["val_acc_history"]) > 0:
        ax3.text(
            0.5,
            0.05,
            "Training accuracy not recorded in CSV\n(only validation accuracy available)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="gray",
            transform=ax3.transAxes,
            style="italic",
        )

    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ========== 4. 噪声样本 vs 干净样本权重 ==========
    ax4 = axes[1, 0]
    if data["is_data_noisy"] and len(data["noisy_weight_history"]) > 0:
        ax4.plot(
            rounds,
            data["noisy_weight_history"],
            "r-",
            linewidth=2,
            label="Noisy Samples",
            marker="o",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
        ax4.plot(
            rounds,
            data["clean_weight_history"],
            "g-",
            linewidth=2,
            label="Clean Samples",
            marker="s",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
        ax4.axhline(y=0.5, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax4.set_xlabel("Boosting Round", fontsize=12)
        ax4.set_ylabel("Total Weight", fontsize=12)
        ax4.set_title("Noisy vs Clean Sample Weights", fontsize=14, fontweight="bold")
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

    # ========== 5. F1 分数演化 ==========
    ax5 = axes[1, 1]
    if len(data["f1_on_training_data"]) > 0:
        ax5.plot(
            rounds,
            data["f1_on_training_data"],
            "b-",
            linewidth=2,
            label="Train F1",
            marker="o",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    if len(data["val_f1_history"]) > 0:
        ax5.plot(
            rounds,
            data["val_f1_history"],
            "r-",
            linewidth=2,
            label="Val F1",
            marker="s",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    ax5.set_xlabel("Boosting Round", fontsize=12)
    ax5.set_ylabel("F1 Score", fontsize=12)
    ax5.set_title("F1 Score Evolution", fontsize=14, fontweight="bold")

    # 如果没有训练F1数据
    if len(data["f1_on_training_data"]) == 0 and len(data["val_f1_history"]) > 0:
        ax5.text(
            0.5,
            0.05,
            "Training F1 not recorded in CSV\n(only validation F1 available)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="gray",
            transform=ax5.transAxes,
            style="italic",
        )

    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ========== 6. 样本权重分布变化 ==========
    ax6 = axes[1, 2]
    if "sample_weights_history" in data and len(data["sample_weights_history"]) > 0:
        # 选择关键轮次
        key_rounds = [0, len(rounds) // 3, len(rounds) * 2 // 3, len(rounds) - 1]
        positions = []
        data_to_plot = []
        labels = []

        for i, idx in enumerate(key_rounds):
            if idx < len(data["sample_weights_history"]):
                positions.append(i + 1)
                data_to_plot.append(data["sample_weights_history"][idx])
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

        ax6.set_ylabel("Sample Weight", fontsize=12)
        ax6.set_title("Sample Weight Distribution", fontsize=14, fontweight="bold")
        ax6.grid(True, alpha=0.3, axis="y")
    else:
        ax6.text(
            0.5,
            0.5,
            "N/A\n(Not in CSV)",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax6.transAxes,
        )
        ax6.text(
            0.5,
            0.35,
            "Sample weights only available\nin joblib format",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
            transform=ax6.transAxes,
            style="italic",
        )
        ax6.set_xticks([])
        ax6.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close()
