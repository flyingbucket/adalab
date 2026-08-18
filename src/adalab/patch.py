"""
AdaBoost 训练过程插桩与监控扩展。

本模块通过扩展 sklearn 的 AdaBoostClassifier，
在不改变算法行为的前提下，
对每一轮 boost 过程进行监控与数据记录。

该模块属于 adalab 的核心研究实现，
用于分析权重更新、噪声放大与过拟合现象。
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

from .monitor import BoostMonitor

ori_boost = AdaBoostClassifier._boost


def boost_with_monitor(self, iboost, X, y, sample_weight, random_state):
    """在原始 AdaBoost `_boost` 基础上插入训练监控逻辑。

    该函数通过在 boost 轮次前后调用 BoostMonitor，
    记录样本权重、误差与弱分类器权重等信息。

    该接口主要用于兼容或实验性场景，
    当前项目主流程推荐使用 ``AdaBoostClfWithMonitor`` 子类方式。

    Args:
        iboost (int): 当前 boost 轮次（从 0 开始）。
        X (np.ndarray): 训练特征数据。
        y (np.ndarray): 训练标签。
        sample_weight (np.ndarray): 当前轮次的样本权重。
        random_state: 随机状态对象。

    Returns:
        tuple:
            - sample_weight_new: 更新后的样本权重
            - estimator_weight: 当前弱分类器权重（alpha）
            - estimator_error: 当前轮次的加权错误率
    """
    # ---- 记录 Boost 前 ----
    if hasattr(self, "_monitor"):
        self._monitor.record_before_boost(sample_weight)

    # ---- 调用原始 _boost ----
    sample_weight_new, estimator_weight, estimator_error = ori_boost(
        self, iboost, X, y, sample_weight, random_state
    )

    # ---- 记录 Boost 后 ----
    if hasattr(self, "_monitor"):
        self._monitor.record_after_boost(
            estimator_error, estimator_weight, iboost, self.n_estimators
        )

    return sample_weight_new, estimator_weight, estimator_error


class AdaBoostClfWithMonitor(AdaBoostClassifier):
    """带训练监控功能的 AdaBoostClassifier 扩展。

    该类通过重写 ``_boost`` 方法，
    在每一轮 AdaBoost 训练过程中自动记录关键信息到 BoostMonitor，
    包括样本权重变化、误差指标与弱分类器权重。

    该类在行为上与 sklearn 的 ``AdaBoostClassifier`` 保持一致，
    仅在训练过程中增加监控与数据记录能力。

    Attributes:
        _monitor (BoostMonitor): 用于记录训练过程信息的监控器实例。
        X_val (Optional[np.ndarray]): 验证集特征（已弃用，仅兼容旧代码）。
        y_val (Optional[np.ndarray]): 验证集标签（已弃用，仅兼容旧代码）。
    """

    def __init__(
        self,
        _monitor: BoostMonitor,
        X_val=None,
        y_val=None,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1,
        algorithm="deprecated",
        random_state=None,
    ):
        """初始化带监控功能的 AdaBoost 分类器。

        Args:
            _monitor (BoostMonitor): 训练过程监控器实例。
            X_val (np.ndarray, optional): 验证集特征（已弃用）。
            y_val (np.ndarray, optional): 验证集标签（已弃用）。
            estimator: 基学习器实例。
            n_estimators (int, optional): boost 轮数。
            learning_rate (float, optional): 学习率。
            algorithm (str, optional): AdaBoost 算法类型（保留参数）。
            random_state (int, optional): 随机种子。
        """
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )
        self._monitor = _monitor
        self.X_val = X_val
        self.y_val = y_val

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """执行单轮 AdaBoost 训练并记录监控数据。

        该方法重写自 sklearn 的 ``AdaBoostClassifier._boost``，
        在不改变算法行为的前提下，
        在 boost 前后自动调用 BoostMonitor 记录关键信息。

        Args:
            iboost (int): 当前 boost 轮次（从 0 开始）。
            X (np.ndarray): 训练特征数据。
            y (np.ndarray): 训练标签。
            sample_weight (np.ndarray): 当前轮次的样本权重。
            random_state: 随机状态对象。

        Returns:
            tuple:
                - sample_weight: 更新后的样本权重
                - estimator_weight: 当前弱分类器权重（alpha）
                - estimator_error: 当前轮次的加权错误率
        """
        # ===== BEFORE BOOST =====
        if hasattr(self, "_monitor"):
            self._monitor.record_before_boost(sample_weight)

        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y

        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        error_without_weight = float(np.mean(incorrect))

        # ===== Case 1: perfect classifier =====
        if estimator_error <= 0:
            estimator_weight = 1.0

            if hasattr(self, "_monitor"):
                self._monitor.record_after_boost(
                    estimator_error,
                    estimator_weight,
                    iboost,
                    self.n_estimators,
                    error_without_weight,
                )

            return sample_weight, estimator_weight, estimator_error

        n_classes = self.n_classes_

        # ===== Case 2: worse than random guessing =====
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)

            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier ensemble is worse than random."
                )

            if hasattr(self, "_monitor"):
                self._monitor.record_after_boost(
                    estimator_error,
                    None,
                    iboost,
                    self.n_estimators,
                    error_without_weight,
                )

            return None, None, None

        # ===== Normal SAMME estimator weight =====
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        if hasattr(self, "_monitor"):
            self._monitor.record_after_boost(
                estimator_error,
                estimator_weight,
                iboost,
                self.n_estimators,
                error_without_weight,
            )

        # ===== Update sample_weight =====
        if iboost != self.n_estimators - 1:
            sample_weight = np.exp(
                np.log(sample_weight)
                + estimator_weight * incorrect * (sample_weight > 0)
            )
        # validation can be done AFTER training !
        # validation
        # self._run_validation(iboost)
        # self._val_on_train_data(iboost, X, y)
        # save monitor data checkpoint
        self._monitor.auto_checkpoint(iboost)
        return sample_weight, estimator_weight, estimator_error

    def _run_validation(self, iboost):
        """在训练过程中执行单轮验证评估。

        .. deprecated::
            该方法已弃用

        当前版本中，所有验证评估统一在模型训练完成后执行，
        使用 ``val_after_train`` 或 ``val_after_train_parallel`` 生成性能曲线。

        保留该方法仅用于兼容早期实验代码。

        Args:
            iboost (int): 当前 boost 轮次。
        """
        if self.X_val is None or self.y_val is None:
            return

        # 当前模型的集成已经形成，可以直接 predict
        y_pred = self.predict(self.X_val)

        # 计算指标
        acc = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred, average="macro")

        # 写进 monitor
        if hasattr(self, "_monitor") and self._monitor is not None:
            self._monitor.record_validation(iboost, acc, f1)

    def _val_on_train_data(self, iboost, X, y):
        """【已弃用】在训练过程中评估训练集性能。

        .. deprecated::
            该方法已弃用，不再推荐使用。

        当前版本中，训练集性能评估统一在训练完成后执行，
        并以曲线形式写入 BoostMonitor。

        保留该方法仅用于兼容早期实验代码。

        Args:
            iboost (int): 当前 boost 轮次。
            X (np.ndarray): 训练特征数据。
            y (np.ndarray): 训练标签。
        """

        # 当前模型的集成已经形成，可以直接 predict
        y_pred = self.predict(X)

        # 计算指标
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")

        # 写进 monitor
        if hasattr(self, "_monitor") and self._monitor is not None:
            self._monitor.record_training_scores(iboost, acc, f1)
