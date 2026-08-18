"""
AdaBoost 训练过程监控与日志记录。

本模块负责：
- 记录 AdaBoost 训练过程中每一轮的关键统计信息
- 跟踪噪声样本与干净样本的权重变化
- 保存训练后与验证后的性能曲线
- 导出结构化结果供可视化模块使用

该模块为前端可视化提供稳定的数据接口。
"""

import warnings

import pandas as pd


class BoostMonitor:
    """AdaBoost 训练过程监控器。

    BoostMonitor 用于在 AdaBoost 训练过程中持续记录关键统计信息，
    并在训练结束后以结构化形式（CSV / joblib）提供给前端可视化模块使用。

    该类本身不参与模型训练决策，仅负责：
    - 记录每一轮 boost 的误差、权重与系数
    - 跟踪噪声样本与干净样本的权重变化
    - 保存训练后与验证后的性能曲线
    - 定期输出 checkpoint，防止长实验中断造成数据丢失

    前端可视化模块（adalab_viz）应将本类视为
    「实验训练过程的唯一权威数据来源」。

    Attributes:
        noise_indices (array-like): 训练集中被标记为噪声样本的索引。
        clean_indices (array-like): 训练集中被标记为干净样本的索引。
        is_data_noisy (bool): 是否启用噪声相关统计。
        sample_weights_history (list[np.ndarray]): 每一轮 boost 前的样本权重快照。
        noisy_weight_history (list[float]): 每一轮 boost 前噪声样本权重之和。
        clean_weight_history (list[float]): 每一轮 boost 前干净样本权重之和。
        error_without_weight_history (list[float]): 每一轮未加权错误率。
        error_history (list[float]): 每一轮加权错误率（AdaBoost 使用）。
        alpha_history (list[float]): 每一轮弱分类器的权重系数。
        val_acc_history (list[float]): 训练后在验证集上的准确率曲线。
        val_f1_history (list[float]): 训练后在验证集上的 F1 曲线。
        acc_on_train_data (list[float]): 训练后在训练集上的准确率曲线。
        f1_on_training_data (list[float]): 训练后在训练集上的 F1 曲线。
        val_idx (list[int]): 对应性能曲线的 boost 轮次索引。
        checkpoint_interval (int): 自动保存 checkpoint 的轮次间隔。
        checkpoint_prefix (str): checkpoint 文件保存目录。
    """

    def __init__(
        self,
        noise_indices,
        clean_indices,
        is_data_noisy=False,
        checkpoint_interval=50,
        checkpoint_prefix="monitor_checkpoint",
    ):
        """初始化 BoostMonitor。

        Args:
            noise_indices (array-like): 噪声样本在训练集中的索引。
            clean_indices (array-like): 干净样本在训练集中的索引。
            is_data_noisy (bool, optional): 是否启用噪声相关统计，默认 False。
            checkpoint_interval (int, optional): checkpoint 保存间隔轮次，默认 50。
            checkpoint_prefix (str, optional): checkpoint 保存目录路径。
        """
        # data
        self.noise_indices = noise_indices
        self.clean_indices = clean_indices
        self.is_data_noisy = is_data_noisy

        # model history
        self.sample_weights_history = []
        self.noisy_weight_history = []
        self.clean_weight_history = []
        self.error_without_weight_history = []
        self.error_history = []
        self.alpha_history = []

        # validation history
        self.val_acc_history = []
        self.val_f1_history = []

        # scores on training data
        self.acc_on_train_data = []
        self.f1_on_training_data = []

        # validation index
        self.val_idx = []
        # checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_prefix = checkpoint_prefix

    def record_before_boost(self, sample_weight):
        """记录每一轮 boost 开始前的样本权重信息。

        该方法通常在弱分类器训练之前调用，
        用于捕获样本权重在本轮更新前的状态。

        Args:
            sample_weight (np.ndarray): 当前轮次的样本权重向量。
        """
        self.sample_weights_history.append(sample_weight.copy())
        if self.is_data_noisy:
            self.noisy_weight_history.append(sample_weight[self.noise_indices].sum())
            self.clean_weight_history.append(sample_weight[self.clean_indices].sum())

    def record_after_boost(
        self,
        estimator_error,
        estimator_weight,
        iboost,
        total,
        error_without_weight=None,
    ):
        """记录每一轮 boost 结束后的核心统计量。

        该方法在弱分类器训练完成并计算权重后调用，
        主要用于记录误差指标与弱分类器系数。

        Args:
            estimator_error (float): 当前轮的加权错误率。
            estimator_weight (float): 当前弱分类器的权重系数（alpha）。
            iboost (int): 当前 boost 轮次（从 0 开始）。
            total (int): 总 boost 轮数。
            error_without_weight (float, optional): 未加权错误率。
        """
        if estimator_error is not None:
            self.error_history.append(estimator_error)
            self.alpha_history.append(estimator_weight)
        if error_without_weight is not None:
            self.error_without_weight_history.append(error_without_weight)

        # 统一走事件日志
        self._log_event(
            event_type="boost",
            iboost=iboost,
            total=total,
            error=estimator_error,
            alpha=estimator_weight,
            unweighted_err=error_without_weight,
            noisy_w=self.noisy_weight_history[-1] if self.is_data_noisy else None,
        )

    def record_validation(self, iboost, acc, f1):
        """记录单轮验证集性能指标。

        .. deprecated::
            该方法已弃用，不再推荐使用。

        该接口原用于在训练过程中按轮次记录验证集性能。
        当前版本中，所有验证评估统一在模型训练完成后，
        通过 ``val_after_train`` 或 ``val_after_train_parallel`` 执行，
        并将结果直接写入 ``val_acc_history`` 与 ``val_f1_history``。

        保留该方法仅用于兼容早期实验代码。

        Args:
            iboost (int): 当前 boost 轮次。
            acc (float): 验证集准确率。
            f1 (float): 验证集 F1 分数。
        """
        self.val_acc_history.append(acc)
        self.val_f1_history.append(f1)

        self._log_event(
            event_type="val",
            iboost=iboost,
            acc=acc,
            f1=f1,
        )

    def record_training_scores(self, iboost, acc, f1):
        """记录单轮训练集预测性能。

        .. deprecated::
            该方法已弃用，不再推荐使用。

        该接口原用于在训练过程中记录训练集预测表现。
        当前版本中，训练集与测试集的性能评估均在
        模型训练完成后统一执行，
        通过 ``val_after_train`` 或 ``val_after_train_parallel`` 生成完整曲线。

        保留该方法仅用于兼容早期实验代码。

        Args:
            iboost (int): 当前 boost 轮次。
            acc (float): 训练集准确率。
            f1 (float): 训练集 F1 分数。
        """
        self.acc_on_train_data.append(acc)
        self.f1_on_training_data.append(f1)

        self._log_event(
            event_type="train",
            iboost=iboost,
            acc=acc,
            f1=f1,
        )

    def _log_event(self, event_type, **kwargs):
        """
        统一打印日志函数。
        event_type 取值：'boost' / 'val' / 'train'
        """

        # ---- Boost 事件 ----
        if event_type == "boost":
            iboost = kwargs["iboost"]
            total = kwargs["total"]
            error = kwargs["error"]
            alpha = kwargs["alpha"]
            unweighted_err = kwargs.get("unweighted_err")
            noisy_w = kwargs.get("noisy_w")

            # 每 5 轮打印一次
            if (iboost + 1) % 5 != 0:
                return

            msg = (
                f"[BOOST] {iboost + 1}/{total} | error={error:.4f} | alpha={alpha:.4f}"
            )
            if unweighted_err is not None:
                msg += f" | unweighted_err={unweighted_err:.4f}"
            if noisy_w is not None:
                msg += f" | noisy_w={noisy_w:.6f}"

            print(msg)
            return

        if event_type in ("val", "train"):
            iboost = kwargs["iboost"]
            acc = kwargs["acc"]
            f1 = kwargs["f1"]

            tag = "[VAL]" if event_type == "val" else "[TRAIN]"

            # 统一格式
            msg = (
                f"{tag.ljust(7)}"  # 保证 [VAL] / [TRAIN] 左对齐，占 7 字符
                f"round={iboost:03d} | "
                f"acc={acc:8.4f} | "  # 保证 acc 列对齐
                f"f1={f1:8.4f}"  # 保证 f1 列对齐
            )

            print(msg)
            return

    def auto_checkpoint(self, iboost):
        """按固定间隔自动保存训练过程的 checkpoint CSV。

        该方法用于在长时间实验中定期落盘关键监控数据，
        以降低实验中断导致的数据丢失风险。

        Args:
            iboost (int): 当前 boost 轮次（从 0 开始）。
        """

        # 只有在达到间隔时才保存
        if (iboost + 1) % self.checkpoint_interval != 0:
            return

        rounds = len(self.error_history)

        data = {
            "round": list(range(1, rounds + 1)),
            "weighted_error": self.error_history,
            "alpha": self.alpha_history,
            # "acc_on_training_data": self.acc_on_train_data,
            # "f1_on_training_data": self.f1_on_training_data,
            # "val_acc_history": self.val_acc_history,
            # "val_f1_history": self.val_f1_history,
        }

        # 普通错误率
        if len(self.error_without_weight_history) == rounds:
            data["unweighted_error"] = self.error_without_weight_history
        else:
            data["unweighted_error"] = [None] * rounds

        # noisy / clean 权重（已经是 before-boost）
        if self.is_data_noisy:
            data["noisy_weight"] = self.noisy_weight_history[:rounds]
            data["clean_weight"] = self.clean_weight_history[:rounds]
        else:
            data["noisy_weight"] = [None] * rounds
            data["clean_weight"] = [None] * rounds

        df = pd.DataFrame(data)

        # 自动生成 checkpoint 文件名
        ckpt_path = f"{self.checkpoint_prefix}/round_{iboost + 1:04d}.csv"
        df.to_csv(ckpt_path, index=False)

        print(f"[CHECKPOINT] Saved '{ckpt_path}' (round={iboost + 1}, rows={len(df)})")

    def dump(self, filename="monitor_log.csv"):
        """将完整的监控历史导出为 CSV 文件。

        该方法会自动对齐所有已记录的历史序列，
        缺失的数据将使用 None 填充。

        生成的 CSV 文件是前端可视化模块推荐使用的输入格式。

        Args:
            filename (str, optional): 输出 CSV 文件路径，默认 "monitor_log.csv"。
        """
        # 使用 error_history 作为主轴（每轮 after_boost 必记一条）
        rounds = len(self.error_history)

        if rounds == 0:
            warnings.warn("没有任何训练记录，dump 取消。")
            return

        # 构建主数据结构
        data = {
            "round": list(range(1, rounds + 1)),
            "weighted_error": self.error_history,
            "alpha": self.alpha_history,
        }

        # 普通错误率（unweighted）
        if len(self.error_without_weight_history) == rounds:
            data["unweighted_error"] = self.error_without_weight_history
        else:
            data["unweighted_error"] = [None] * rounds

        #  noisy / clean 权重均值
        if self.is_data_noisy:
            data["noisy_weight"] = self.noisy_weight_history[:rounds]
            data["clean_weight"] = self.clean_weight_history[:rounds]
        else:
            data["noisy_weight"] = [None] * rounds
            data["clean_weight"] = [None] * rounds

        # 输出 CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

        print(f"Monitor dumped to '{filename}' (rows={len(df)})")
