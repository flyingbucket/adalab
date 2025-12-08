"""
评估模块 - 提供详细的模型性能分析和可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mplfonts.bin.cli import init  # 导入初始化函数
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import seaborn as sns

# 初始化字体支持
# init()  # 如需中文支持可取消注释
# matplotlib.rcParams["font.family"] = "Source Han Sans CN"  # 中文字体
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # 英文字体（跨平台兼容）
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class ModelEvaluator:
    """模型评估器 - 提供全面的性能评估功能"""

    def __init__(self, clf, X_train, y_train, X_test, y_test, class_names=None):
        """
        初始化评估器

        Parameters
        ----------
        clf : 训练好的分类器
        X_train : 训练集特征
        y_train : 训练集标签
        X_test : 测试集特征
        y_test : 测试集标签
        class_names : 类别名称列表，默认为 0-9
        """
        self.clf = clf
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names if class_names else [str(i) for i in range(10)]

        # 进行预测
        self.y_train_pred = clf.predict(X_train)
        self.y_test_pred = clf.predict(X_test)

    def print_basic_metrics(self):
        """打印基本评估指标"""
        print("\n" + "=" * 60)
        print("基本性能指标".center(56))
        print("=" * 60)

        # 训练集准确率
        train_acc = accuracy_score(self.y_train, self.y_train_pred)
        print(f"训练集准确率: {train_acc:.4f} ({train_acc * 100:.2f}%)")

        # 测试集准确率
        test_acc = accuracy_score(self.y_test, self.y_test_pred)
        print(f"测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")

        # 过拟合程度
        overfit = train_acc - test_acc
        print(f"过拟合程度: {overfit:.4f} ({overfit * 100:.2f}%)")
        print("=" * 60)

    def print_classification_report(self):
        """打印详细的分类报告"""
        print("\n" + "=" * 60)
        print("测试集分类报告".center(56))
        print("=" * 60)

        # 生成分类报告
        report = classification_report(
            self.y_test,
            self.y_test_pred,
            target_names=self.class_names,
            digits=4,
        )
        print(report)

    def print_per_class_metrics(self):
        """打印每个类别的详细指标"""
        print("\n" + "=" * 60)
        print("各类别详细指标".center(56))
        print("=" * 60)

        # 计算精确率、召回率、F1分数
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, self.y_test_pred
        )

        # 打印表头
        print(f"{'类别':<6} {'精确率':>8} {'召回率':>8} {'F1分数':>8} {'样本数':>8}")
        print("-" * 60)

        # 打印每个类别
        for i, class_name in enumerate(self.class_names):
            print(
                f"{class_name:<6} {precision[i]:>8.4f} {recall[i]:>8.4f} "
                f"{f1[i]:>8.4f} {support[i]:>8d}"
            )

        # 打印平均值
        print("-" * 60)
        print(
            f"{'平均':<6} {precision.mean():>8.4f} {recall.mean():>8.4f} "
            f"{f1.mean():>8.4f} {support.sum():>8d}"
        )
        print("=" * 60)

    def analyze_noise_impact(self, noise_indices, clean_indices):
        """
        分析噪声样本的影响（仅适用于训练集）

        Parameters
        ----------
        noise_indices : 噪声样本在训练集中的索引
        clean_indices : 干净样本在训练集中的索引
        """
        if len(noise_indices) == 0:
            print("\n训练集无噪声样本，跳过噪声影响分析")
            return

        print("\n" + "=" * 60)
        print("噪声影响分析".center(56))
        print("=" * 60)

        # 噪声样本的准确率
        y_noise_true = self.y_train[noise_indices]
        y_noise_pred = self.y_train_pred[noise_indices]
        noise_acc = accuracy_score(y_noise_true, y_noise_pred)

        # 干净样本的准确率
        y_clean_true = self.y_train[clean_indices]
        y_clean_pred = self.y_train_pred[clean_indices]
        clean_acc = accuracy_score(y_clean_true, y_clean_pred)

        print(f"噪声样本数量: {len(noise_indices)}")
        print(f"干净样本数量: {len(clean_indices)}")
        print(f"噪声比例: {len(noise_indices) / len(self.y_train) * 100:.2f}%")
        print("-" * 60)
        print(f"噪声样本准确率: {noise_acc:.4f} ({noise_acc * 100:.2f}%)")
        print(f"干净样本准确率: {clean_acc:.4f} ({clean_acc * 100:.2f}%)")
        print(f"准确率差距: {clean_acc - noise_acc:.4f} ({(clean_acc - noise_acc) * 100:.2f}%)")
        print("=" * 60)

        return {
            "noise_accuracy": noise_acc,
            "clean_accuracy": clean_acc,
            "accuracy_gap": clean_acc - noise_acc,
        }

    def plot_confusion_matrix(self, save_path=None):
        """
        绘制混淆矩阵

        Parameters
        ----------
        save_path : 保存路径，如果为None则显示图像
        """
        # 计算混淆矩阵
        cm = confusion_matrix(self.y_test, self.y_test_pred)

        # 创建图形
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "样本数量"},
        )
        plt.title("测试集混淆矩阵", fontsize=16, pad=20)
        plt.xlabel("预测标签", fontsize=12)
        plt.ylabel("真实标签", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"混淆矩阵已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_normalized_confusion_matrix(self, save_path=None):
        """
        绘制归一化混淆矩阵（显示百分比）

        Parameters
        ----------
        save_path : 保存路径
        """
        # 计算混淆矩阵
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # 创建图形
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "比例"},
        )
        plt.title("归一化混淆矩阵（按行归一化）", fontsize=16, pad=20)
        plt.xlabel("预测标签", fontsize=12)
        plt.ylabel("真实标签", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"归一化混淆矩阵已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_class_performance(self, save_path=None):
        """
        绘制各类别性能柱状图

        Parameters
        ----------
        save_path : 保存路径
        """
        # 计算指标
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, self.y_test_pred
        )

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 精确率
        axes[0, 0].bar(self.class_names, precision, color="steelblue", alpha=0.8)
        axes[0, 0].set_title("各类别精确率", fontsize=14)
        axes[0, 0].set_ylabel("精确率", fontsize=12)
        axes[0, 0].set_ylim([0, 1.05])
        axes[0, 0].axhline(y=precision.mean(), color="r", linestyle="--", label="平均值")
        axes[0, 0].legend()
        axes[0, 0].grid(axis="y", alpha=0.3)

        # 子图2: 召回率
        axes[0, 1].bar(self.class_names, recall, color="darkorange", alpha=0.8)
        axes[0, 1].set_title("各类别召回率", fontsize=14)
        axes[0, 1].set_ylabel("召回率", fontsize=12)
        axes[0, 1].set_ylim([0, 1.05])
        axes[0, 1].axhline(y=recall.mean(), color="r", linestyle="--", label="平均值")
        axes[0, 1].legend()
        axes[0, 1].grid(axis="y", alpha=0.3)

        # 子图3: F1分数
        axes[1, 0].bar(self.class_names, f1, color="seagreen", alpha=0.8)
        axes[1, 0].set_title("各类别F1分数", fontsize=14)
        axes[1, 0].set_xlabel("类别", fontsize=12)
        axes[1, 0].set_ylabel("F1分数", fontsize=12)
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].axhline(y=f1.mean(), color="r", linestyle="--", label="平均值")
        axes[1, 0].legend()
        axes[1, 0].grid(axis="y", alpha=0.3)

        # 子图4: 样本数量
        axes[1, 1].bar(self.class_names, support, color="mediumpurple", alpha=0.8)
        axes[1, 1].set_title("各类别测试样本数量", fontsize=14)
        axes[1, 1].set_xlabel("类别", fontsize=12)
        axes[1, 1].set_ylabel("样本数量", fontsize=12)
        axes[1, 1].grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"类别性能图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_training_history(self, monitor, save_path=None):
        """
        绘制训练历史曲线

        Parameters
        ----------
        monitor : BoostMonitor 对象
        save_path : 保存路径
        """
        if not hasattr(monitor, "error_history"):
            print("监控器中没有训练历史数据")
            return

        iterations = range(1, len(monitor.error_history) + 1)

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 弱学习器错误率
        axes[0, 0].plot(iterations, monitor.error_history, "b-", linewidth=2)
        axes[0, 0].set_title("弱学习器错误率变化", fontsize=14)
        axes[0, 0].set_xlabel("迭代轮次", fontsize=12)
        axes[0, 0].set_ylabel("错误率", fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, max(monitor.error_history) * 1.1])

        # 子图2: 弱学习器权重α
        axes[0, 1].plot(iterations, monitor.alpha_history, "g-", linewidth=2)
        axes[0, 1].set_title("弱学习器权重α变化", fontsize=14)
        axes[0, 1].set_xlabel("迭代轮次", fontsize=12)
        axes[0, 1].set_ylabel("权重α", fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        # 子图3: 样本权重分布变化（箱线图）
        if len(monitor.sample_weights_history) > 0:
            # 选择几个关键时刻的权重分布
            key_iterations = [0, len(monitor.sample_weights_history) // 2, -1]
            data_to_plot = [monitor.sample_weights_history[i] for i in key_iterations]
            labels = [
                f"第{key_iterations[0]+1}轮",
                f"第{key_iterations[1]+1}轮",
                f"第{len(monitor.sample_weights_history)}轮",
            ]

            axes[1, 0].boxplot(data_to_plot, labels=labels)
            axes[1, 0].set_title("样本权重分布变化", fontsize=14)
            axes[1, 0].set_ylabel("样本权重", fontsize=12)
            axes[1, 0].grid(axis="y", alpha=0.3)
            axes[1, 0].set_yscale("log")  # 使用对数刻度

        # 子图4: 噪声样本 vs 干净样本权重对比（如果有）
        if monitor.is_data_noisy and len(monitor.noisy_weight_history) > 0:
            axes[1, 1].plot(
                iterations,
                monitor.noisy_weight_history,
                "r-",
                linewidth=2,
                label="噪声样本",
            )
            axes[1, 1].plot(
                iterations,
                monitor.clean_weight_history,
                "b-",
                linewidth=2,
                label="干净样本",
            )
            axes[1, 1].set_title("噪声样本 vs 干净样本平均权重", fontsize=14)
            axes[1, 1].set_xlabel("迭代轮次", fontsize=12)
            axes[1, 1].set_ylabel("平均权重", fontsize=12)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale("log")  # 使用对数刻度
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "无噪声数据\n或未记录噪声权重",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"训练历史图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_feature_importance(self, save_path=None, top_n=20):
        """
        绘制特征重要性

        Parameters
        ----------
        save_path : 保存路径
        top_n : 显示前N个最重要的特征
        """
        # 检查模型是否支持feature_importances_
        if not hasattr(self.clf, "feature_importances_"):
            print("该模型不支持特征重要性分析")
            return

        # 获取特征重要性
        importances = self.clf.feature_importances_
        n_features = len(importances)

        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 子图1: Top N 特征重要性柱状图
        indices = np.argsort(importances)[::-1][:top_n]
        top_importances = importances[indices]

        axes[0].barh(range(top_n), top_importances[::-1], color="steelblue", alpha=0.8)
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels([f"特征 {i}" for i in indices[::-1]])
        axes[0].set_xlabel("重要性分数", fontsize=12)
        axes[0].set_title(f"Top {top_n} 最重要的特征", fontsize=14)
        axes[0].grid(axis="x", alpha=0.3)

        # 子图2: 特征重要性分布
        axes[1].hist(importances, bins=50, color="darkorange", alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("重要性分数", fontsize=12)
        axes[1].set_ylabel("特征数量", fontsize=12)
        axes[1].set_title("特征重要性分布", fontsize=14)
        axes[1].axvline(
            importances.mean(), color="r", linestyle="--", linewidth=2, label="平均值"
        )
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"特征重要性图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

        # 打印统计信息
        print("\n" + "=" * 60)
        print("特征重要性统计".center(56))
        print("=" * 60)
        print(f"总特征数: {n_features}")
        print(f"非零重要性特征数: {np.sum(importances > 0)}")
        print(f"最大重要性: {importances.max():.6f}")
        print(f"平均重要性: {importances.mean():.6f}")
        print(f"最小重要性: {importances.min():.6f}")
        print("=" * 60)

    def plot_feature_importance_heatmap(self, image_shape=(28, 28), save_path=None):
        """
        将特征重要性可视化为热力图（适用于图像数据如MNIST）

        Parameters
        ----------
        image_shape : tuple
            图像形状，默认 (28, 28) 适用于MNIST
        save_path : 保存路径
        """
        # 检查模型是否支持feature_importances_
        if not hasattr(self.clf, "feature_importances_"):
            print("该模型不支持特征重要性分析")
            return

        # 获取特征重要性
        importances = self.clf.feature_importances_

        # 检查特征数是否匹配
        expected_features = image_shape[0] * image_shape[1]
        if len(importances) != expected_features:
            print(
                f"特征数 ({len(importances)}) 与图像形状 {image_shape} "
                f"不匹配 (期望 {expected_features})"
            )
            return

        # 重塑为图像形状
        importance_map = importances.reshape(image_shape)

        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 子图1: 热力图
        im1 = axes[0].imshow(importance_map, cmap="hot", interpolation="nearest")
        axes[0].set_title("特征重要性热力图", fontsize=14)
        axes[0].set_xlabel("像素列", fontsize=12)
        axes[0].set_ylabel("像素行", fontsize=12)
        plt.colorbar(im1, ax=axes[0], label="重要性分数")

        # 子图2: 叠加网格的热力图
        im2 = axes[1].imshow(importance_map, cmap="YlOrRd", interpolation="nearest")
        axes[1].set_title("特征重要性热力图（高亮显示）", fontsize=14)
        axes[1].set_xlabel("像素列", fontsize=12)
        axes[1].set_ylabel("像素行", fontsize=12)

        # 标记最重要的区域
        threshold = np.percentile(importances, 95)  # 前5%最重要的特征
        y_coords, x_coords = np.where(importance_map > threshold)
        axes[1].scatter(
            x_coords, y_coords, c="blue", s=10, alpha=0.5, label="Top 5% 重要特征"
        )
        axes[1].legend()

        plt.colorbar(im2, ax=axes[1], label="重要性分数")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"特征重要性热力图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

        # 分析重要性分布
        print("\n" + "=" * 60)
        print("像素重要性分析".center(56))
        print("=" * 60)

        # 计算各区域的平均重要性
        h, w = image_shape
        regions = {
            "左上角": importance_map[: h // 2, : w // 2].mean(),
            "右上角": importance_map[: h // 2, w // 2 :].mean(),
            "左下角": importance_map[h // 2 :, : w // 2].mean(),
            "右下角": importance_map[h // 2 :, w // 2 :].mean(),
            "中心区域": importance_map[
                h // 4 : 3 * h // 4, w // 4 : 3 * w // 4
            ].mean(),
            "边缘区域": np.concatenate(
                [
                    importance_map[0, :],  # 上边缘
                    importance_map[-1, :],  # 下边缘
                    importance_map[:, 0],  # 左边缘
                    importance_map[:, -1],  # 右边缘
                ]
            ).mean(),
        }

        print("各区域平均重要性:")
        print("-" * 60)
        for region, importance in sorted(
            regions.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"{region:<12}: {importance:.6f}")
        print("=" * 60)

    def plot_per_class_feature_importance(
        self, X_data, y_data, image_shape=(28, 28), save_path=None
    ):
        """
        绘制各类别的平均特征图（显示每个数字的典型特征）

        Parameters
        ----------
        X_data : 数据特征（通常使用测试集）
        y_data : 数据标签
        image_shape : 图像形状
        save_path : 保存路径
        """
        n_classes = len(self.class_names)

        # 创建图形
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i, class_name in enumerate(self.class_names):
            # 获取该类别的所有样本
            class_mask = y_data == int(class_name)
            class_samples = X_data[class_mask]

            if len(class_samples) == 0:
                axes[i].text(
                    0.5, 0.5, "无样本", ha="center", va="center", fontsize=12
                )
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                continue

            # 计算平均特征图
            avg_features = class_samples.mean(axis=0).reshape(image_shape)

            # 显示
            im = axes[i].imshow(avg_features, cmap="viridis", interpolation="nearest")
            axes[i].set_title(f"数字 {class_name}", fontsize=12)
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.suptitle("各数字的平均特征图", fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"各类别平均特征图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def analyze_misclassified_samples(
        self, n_samples=10, image_shape=(28, 28), save_path=None
    ):
        """
        分析和可视化被错误分类的样本

        Parameters
        ----------
        n_samples : 显示的错误样本数量
        image_shape : 图像形状
        save_path : 保存路径
        """
        # 找出错误分类的样本
        misclassified_mask = self.y_test != self.y_test_pred
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            print("测试集中没有错误分类的样本！")
            return

        # 随机选择n_samples个错误样本
        n_samples = min(n_samples, len(misclassified_indices))
        selected_indices = np.random.choice(
            misclassified_indices, n_samples, replace=False
        )

        # 创建图形
        rows = (n_samples + 4) // 5
        fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows))
        axes = axes.ravel() if n_samples > 1 else [axes]

        for idx, sample_idx in enumerate(selected_indices):
            # 获取样本
            sample = self.X_test[sample_idx].reshape(image_shape)
            true_label = self.y_test[sample_idx]
            pred_label = self.y_test_pred[sample_idx]

            # 显示
            axes[idx].imshow(sample, cmap="gray")
            axes[idx].set_title(
                f"真实: {true_label}\n预测: {pred_label}", fontsize=10, color="red"
            )
            axes[idx].axis("off")

        # 隐藏多余的子图
        for idx in range(n_samples, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"错误分类样本 (共 {len(misclassified_indices)} 个错误)",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"错误分类样本图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

        # 打印错误分类统计
        print("\n" + "=" * 60)
        print("错误分类统计".center(56))
        print("=" * 60)
        print(f"总错误数: {len(misclassified_indices)}")
        print(f"错误率: {len(misclassified_indices) / len(self.y_test) * 100:.2f}%")

        # 统计哪些类别最容易被错误分类
        print("\n最容易被误判的数字 (真实标签):")
        print("-" * 60)
        unique, counts = np.unique(self.y_test[misclassified_indices], return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        for i in sorted_idx[:5]:
            print(
                f"数字 {unique[i]}: {counts[i]} 次 "
                f"({counts[i] / np.sum(self.y_test == unique[i]) * 100:.2f}%)"
            )

        print("\n最容易被误判成的数字 (预测标签):")
        print("-" * 60)
        unique, counts = np.unique(
            self.y_test_pred[misclassified_indices], return_counts=True
        )
        sorted_idx = np.argsort(counts)[::-1]
        for i in sorted_idx[:5]:
            print(f"数字 {unique[i]}: {counts[i]} 次")

        print("=" * 60)

    def generate_full_report(self, monitor=None, noise_indices=None, clean_indices=None):
        """
        生成完整的评估报告

        Parameters
        ----------
        monitor : BoostMonitor 对象（可选）
        noise_indices : 噪声样本索引（可选）
        clean_indices : 干净样本索引（可选）
        """
        print("\n" + "█" * 60)
        print("完整模型评估报告".center(56))
        print("█" * 60)

        # 1. 基本指标
        self.print_basic_metrics()

        # 2. 分类报告
        self.print_classification_report()

        # 3. 各类别详细指标
        self.print_per_class_metrics()

        # 4. 噪声影响分析（如果提供了噪声索引）
        if noise_indices is not None and clean_indices is not None:
            self.analyze_noise_impact(noise_indices, clean_indices)

        # 5. 可视化
        print("\n" + "=" * 60)
        print("生成可视化图表...".center(56))
        print("=" * 60)

        self.plot_confusion_matrix()
        self.plot_normalized_confusion_matrix()
        self.plot_class_performance()

        if monitor is not None:
            self.plot_training_history(monitor)

        # 6. 特征重要性可视化
        if hasattr(self.clf, "feature_importances_"):
            print("\n生成特征重要性分析...")
            self.plot_feature_importance()
            self.plot_feature_importance_heatmap()
            self.plot_per_class_feature_importance(self.X_test, self.y_test)
            self.analyze_misclassified_samples()

        print("\n评估报告生成完成！")


def visualize_overfitting_process(
    X_train,
    y_train,
    X_test,
    y_test,
    base_estimator,
    n_estimators_list=None,
    learning_rate=0.5,
    random_state=42,
    save_path=None,
):
    """
    可视化AdaBoost的过拟合过程

    Parameters
    ----------
    X_train, y_train : 训练集
    X_test, y_test : 测试集
    base_estimator : 基学习器（如 DecisionTreeClassifier(max_depth=1)）
    n_estimators_list : 要测试的弱学习器数量列表，默认 [1, 5, 10, 20, 30, 40, 50, 75, 100]
    learning_rate : 学习率
    random_state : 随机种子
    save_path : 保存路径
    """
    from sklearn.ensemble import AdaBoostClassifier
    from tqdm import tqdm

    # 默认测试点
    if n_estimators_list is None:
        n_estimators_list = [1, 5, 10, 20, 30, 40, 50, 75, 100]

    print("\n" + "=" * 60)
    print("AdaBoost 过拟合过程可视化".center(56))
    print("=" * 60)
    print(f"测试弱学习器数量: {n_estimators_list}")
    print(f"学习率: {learning_rate}")
    print("-" * 60)

    # 记录结果
    results = {
        "n_estimators": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "overfitting_degree": [],
    }

    # 训练多个模型
    for n_est in tqdm(n_estimators_list, desc="Training Progress"):
        clf = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_est,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # 计算准确率
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        # 记录
        results["n_estimators"].append(n_est)
        results["train_accuracy"].append(train_acc)
        results["test_accuracy"].append(test_acc)
        results["overfitting_degree"].append(train_acc - test_acc)

    print("Training completed!")

    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: 学习曲线
    ax1 = axes[0]
    n_est = results["n_estimators"]
    train_acc = results["train_accuracy"]
    test_acc = results["test_accuracy"]

    ax1.plot(n_est, train_acc, "b-o", linewidth=2, markersize=8, label="Train Accuracy")
    ax1.plot(n_est, test_acc, "r-s", linewidth=2, markersize=8, label="Test Accuracy")
    ax1.fill_between(n_est, train_acc, test_acc, alpha=0.3, color="orange", label="Overfitting Gap")

    # 标记最佳点
    best_idx = np.argmax(test_acc)
    best_n = n_est[best_idx]
    best_test_acc = test_acc[best_idx]
    ax1.plot(best_n, best_test_acc, "g*", markersize=20, label=f"Best (n={best_n})")
    ax1.axvline(x=best_n, color="green", linestyle="--", alpha=0.5)

    ax1.set_xlabel("Number of Weak Learners", fontsize=14)
    ax1.set_ylabel("Accuracy", fontsize=14)
    ax1.set_title("Learning Curves: Accuracy vs Number of Estimators", fontsize=16, pad=15)
    ax1.legend(fontsize=12, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(test_acc) - 0.05, 1.0])

    # 子图2: 过拟合程度
    ax2 = axes[1]
    overfit = results["overfitting_degree"]

    ax2.plot(n_est, overfit, "purple", linewidth=3, marker="o", markersize=8)
    ax2.fill_between(
        n_est, 0, overfit, where=np.array(overfit) > 0, alpha=0.3, color="red", label="Overfitting Region"
    )
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # 标记最小过拟合点
    min_overfit_idx = np.argmin(np.abs(overfit))
    min_overfit_n = n_est[min_overfit_idx]
    min_overfit_val = overfit[min_overfit_idx]
    ax2.plot(
        min_overfit_n,
        min_overfit_val,
        "g*",
        markersize=20,
        label=f"Min Overfitting (n={min_overfit_n})",
    )
    ax2.axvline(x=min_overfit_n, color="green", linestyle="--", alpha=0.5)

    ax2.set_xlabel("Number of Weak Learners", fontsize=14)
    ax2.set_ylabel("Overfitting Degree (Train Acc - Test Acc)", fontsize=14)
    ax2.set_title("Overfitting vs Number of Estimators", fontsize=16, pad=15)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"过拟合可视化已保存到: {save_path}")
    else:
        plt.show()

    plt.close()

    # 打印总结
    print("\n" + "=" * 60)
    print("过拟合分析总结".center(56))
    print("=" * 60)

    print(f"\n最佳模型:")
    print(f"  弱学习器数量: {best_n}")
    print(f"  测试集准确率: {best_test_acc:.4f} ({best_test_acc * 100:.2f}%)")
    print(f"  训练集准确率: {train_acc[best_idx]:.4f} ({train_acc[best_idx] * 100:.2f}%)")
    print(f"  过拟合程度: {overfit[best_idx]:.4f} ({overfit[best_idx] * 100:.2f}%)")

    print(f"\n最小过拟合模型:")
    print(f"  弱学习器数量: {min_overfit_n}")
    print(f"  过拟合程度: {min_overfit_val:.4f} ({min_overfit_val * 100:.2f}%)")
    print(f"  测试集准确率: {test_acc[min_overfit_idx]:.4f}")

    print(f"\n趋势分析:")
    print(f"  初始 (n={n_est[0]}): 测试准确率 = {test_acc[0]:.4f}, 过拟合 = {overfit[0]:.4f}")
    print(f"  最终 (n={n_est[-1]}): 测试准确率 = {test_acc[-1]:.4f}, 过拟合 = {overfit[-1]:.4f}")

    if test_acc[-1] < best_test_acc:
        print(f"  ⚠️ 警告: 测试准确率在 n={best_n} 后开始下降，建议使用早停")
    else:
        print(f"  ✓ 测试准确率持续提升，可以考虑增加弱学习器数量")

    print("=" * 60)

    return results


def quick_evaluate(
    clf,
    X_train,
    y_train,
    X_test,
    y_test,
    monitor=None,
    noise_indices=None,
    clean_indices=None,
):
    """
    快速评估函数 - 一键生成完整报告

    Parameters
    ----------
    clf : 训练好的分类器
    X_train : 训练集特征
    y_train : 训练集标签
    X_test : 测试集特征
    y_test : 测试集标签
    monitor : BoostMonitor 对象（可选）
    noise_indices : 噪声样本索引（可选）
    clean_indices : 干净样本索引（可选）
    """
    evaluator = ModelEvaluator(clf, X_train, y_train, X_test, y_test)
    evaluator.generate_full_report(monitor, noise_indices, clean_indices)
    return evaluator

