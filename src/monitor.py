import warnings
import pandas as pd


class BoostMonitor:
    def __init__(
        self,
        noise_indices,
        clean_indices,
        is_data_noisy=False,
        checkpoint_interval=50,
        checkpoint_prefix="monitor_checkpoint",
    ):
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

        # checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_prefix = checkpoint_prefix

    def record_before_boost(self, sample_weight):
        """记录 boost 开始前的信息"""
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
        """记录 boost 结束后的信息"""

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
        """记录验证集指标"""
        self.val_acc_history.append(acc)
        self.val_f1_history.append(f1)

        self._log_event(
            event_type="val",
            iboost=iboost,
            acc=acc,
            f1=f1,
        )

    def record_training_scores(self, iboost, acc, f1):
        """记录训练集预测表现"""
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
        """
        每隔 interval 轮自动保存监控数据，以防实验中断造成数据丢失。
        """

        # 只有在达到间隔时才保存
        if (iboost + 1) % self.checkpoint_interval != 0:
            return

        rounds = len(self.error_history)

        data = {
            "round": list(range(1, rounds + 1)),
            "weighted_error": self.error_history,
            "alpha": self.alpha_history,
            "acc_on_training_data": self.acc_on_train_data,
            "f1_on_training_data": self.f1_on_training_data,
            "val_acc_history": self.val_acc_history,
            "val_f1_history": self.val_f1_history,
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
        """
        将所有监控数据保存为 CSV 文件（最终版）
        自动对齐所有历史记录，缺失部分会用 None 填充。
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
            "acc_on_training_data": self.acc_on_train_data,
            "f1_on_training_data": self.f1_on_training_data,
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

        # validation 指标
        if len(self.val_acc_history) == rounds:
            data["val_acc"] = self.val_acc_history
        else:
            data["val_acc"] = [None] * rounds

        if len(self.val_f1_history) == rounds:
            data["val_f1"] = self.val_f1_history
        else:
            data["val_f1"] = [None] * rounds

        # 输出 CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

        print(f"Monitor dumped to '{filename}' (rows={len(df)})")
