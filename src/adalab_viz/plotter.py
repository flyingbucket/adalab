import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_training_data(
    data, save_path=None, save_individual=False, output_dir="dummy_output"
):
    """
    可视化训练数据（重新布局：噪声 → 权重分布 → 误差 → alpha → acc → f1）
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Training Analysis from Saved Results (n={data['n_estimators']})",
        fontsize=16,
        fontweight="bold",
    )

    rounds = data["rounds"]

    # ----------------------------------------------------------------------
    # 1. 左上：噪声样本 vs 干净样本权重（核心：AdaBoost 嘘声放大机制）
    # ----------------------------------------------------------------------
    ax1 = axes[0, 0]
    if data["is_data_noisy"] and len(data["noisy_weight_history"]) > 0:
        ax1.plot(
            rounds,
            data["noisy_weight_history"],
            "r-",
            linewidth=2,
            label="Noisy Samples",
            marker="o",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
        ax1.plot(
            rounds,
            data["clean_weight_history"],
            "g-",
            linewidth=2,
            label="Clean Samples",
            marker="s",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
        ax1.axhline(0.5, color="black", linestyle="--", alpha=0.3)
        ax1.set_title("Noisy vs Clean Sample Weights", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Boosting Round")
        ax1.set_ylabel("Total Weight")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "N/A\n(Clean Data)", ha="center", va="center", fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])

    # ----------------------------------------------------------------------
    # 2. 左下：样本权重分布（权重集中情况→过拟合特征）
    # ----------------------------------------------------------------------
    ax2 = axes[1, 0]
    if "sample_weights_history" in data and len(data["sample_weights_history"]) > 0:
        key_rounds = [0, len(rounds) // 3, len(rounds) * 2 // 3, len(rounds) - 1]
        data_to_plot, labels = [], []

        for idx in key_rounds:
            if idx < len(data["sample_weights_history"]):
                data_to_plot.append(data["sample_weights_history"][idx])
                labels.append(f"R{idx + 1}")

        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
        for box in bp["boxes"]:
            box.set_facecolor("lightblue")
            box.set_alpha(0.7)

        ax2.set_title("Sample Weight Distribution", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Sample Weight")
        ax2.grid(True, axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "N/A\n(Not in CSV)", ha="center", va="center", fontsize=14)
        ax2.set_xticks([])
        ax2.set_yticks([])

    # ----------------------------------------------------------------------
    # 3. 中上：错误率演化
    # ----------------------------------------------------------------------
    ax3 = axes[0, 1]
    ax3.plot(rounds, data["error_history"], "b-", linewidth=2, label="Weighted Error")
    if len(data["error_without_weight_history"]) > 0:
        ax3.plot(
            rounds,
            data["error_without_weight_history"],
            "r--",
            linewidth=2,
            label="Unweighted Error",
            alpha=0.7,
        )
    ax3.set_title("Error Rate Evolution", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Boosting Round")
    ax3.set_ylabel("Error Rate")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ----------------------------------------------------------------------
    # 4. 中下：Alpha 系数演化（弱学习器强度）
    # ----------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.plot(
        rounds,
        data["alpha_history"],
        "g-",
        linewidth=2,
        marker="o",
        markersize=4,
        markevery=max(1, len(rounds) // 20),
    )
    avg_alpha = np.mean(data["alpha_history"])
    ax4.axhline(
        avg_alpha,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Mean={avg_alpha:.3f}",
    )
    ax4.set_title("Alpha Coefficient Evolution", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Boosting Round")
    ax4.set_ylabel("Alpha")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ----------------------------------------------------------------------
    # 5. 右上：训练 vs 验证准确率
    # ----------------------------------------------------------------------
    ax5 = axes[0, 2]
    if len(data["acc_on_train_data"]) > 0:
        ax5.plot(
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
        ax5.plot(
            rounds,
            data["val_acc_history"],
            "r-",
            linewidth=2,
            label="Val Accuracy",
            marker="s",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    ax5.set_title("Accuracy Evolution", fontsize=14, fontweight="bold")
    ax5.set_xlabel("Boosting Round")
    ax5.set_ylabel("Accuracy")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ----------------------------------------------------------------------
    # 6. 右下：F1 演化
    # ----------------------------------------------------------------------
    ax6 = axes[1, 2]
    if len(data["f1_on_training_data"]) > 0:
        ax6.plot(
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
        ax6.plot(
            rounds,
            data["val_f1_history"],
            "r-",
            linewidth=2,
            label="Val F1",
            marker="s",
            markersize=4,
            markevery=max(1, len(rounds) // 20),
        )
    ax6.set_title("F1 Score Evolution", fontsize=14, fontweight="bold")
    ax6.set_xlabel("Boosting Round")
    ax6.set_ylabel("F1 Score")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # ----------------------------------------------------------------------
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n[Viz] Figure saved to: {save_path}")
    else:
        plt.show()

    # save every subplot individually
    if save_individual:
        os.makedirs(output_dir, exist_ok=True)

        subplot_titles = [
            "noisy_vs_clean",
            "sample_weight_distribution",
            "error_evolution",
            "alpha_evolution",
            "accuracy_evolution",
            "f1_evolution",
        ]

        # 原图宽度（用于计算缩放比例）
        orig_fig_width = fig.get_figwidth()

        for ax, name in zip(axes.flatten(), subplot_titles):
            fig_single = plt.figure(figsize=(6, 4), dpi=300)
            new_ax = fig_single.add_subplot(111)

            # 缩放因子
            scale = fig_single.get_figwidth() / orig_fig_width

            for line in ax.get_lines():
                x = line.get_xdata()
                y = line.get_ydata()

                # --- 缩放 markevery（防止 marker 密集） ---
                orig_markevery = line.get_markevery()
                if isinstance(orig_markevery, int):
                    markevery_small = max(
                        1,
                        int(
                            orig_markevery
                            * (orig_fig_width / fig_single.get_figwidth())
                        ),
                    )
                else:
                    markevery_small = orig_markevery

                # --- 缩放 marker 大小 ---
                orig_ms = line.get_markersize()
                new_ms = orig_ms * scale if orig_ms else None

                new_ax.plot(
                    x,
                    y,
                    linestyle=line.get_linestyle(),
                    marker=line.get_marker(),
                    color=line.get_color(),
                    linewidth=line.get_linewidth() * scale,
                    markersize=new_ms,
                    markevery=markevery_small,
                )

            new_ax.set_title(ax.get_title(), fontsize=14, fontweight="bold")
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.grid(True, alpha=0.3)

            single_path = os.path.join(output_dir, f"{name}.png")
            fig_single.savefig(single_path, dpi=300, bbox_inches="tight")
            plt.close(fig_single)
            print(f"[Viz] saved: {single_path}")
    plt.close()
