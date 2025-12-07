"""
鲁棒AdaBoost实现
提供多种策略解决噪声敏感和过拟合问题
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class RobustAdaBoost:
    """
    鲁棒AdaBoost包装器
    实现多种改进策略
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=50,
        learning_rate=0.5,
        random_state=None,
        # 鲁棒性参数
        weight_clip_percentile=95,  # 权重裁剪百分位数
        use_early_stopping=True,  # 是否使用早停
        validation_fraction=0.1,  # 验证集比例
        early_stopping_rounds=10,  # 早停轮数
        use_sample_weight_smoothing=False,  # 是否使用权重平滑
        smoothing_factor=0.5,  # 平滑因子
    ):
        """
        初始化鲁棒AdaBoost

        Parameters
        ----------
        base_estimator : 基学习器，默认为决策树桩
        n_estimators : 最大弱学习器数量
        learning_rate : 学习率
        random_state : 随机种子
        weight_clip_percentile : 权重裁剪百分位数（0-100）
            例如95表示将权重限制在前95%范围内
        use_early_stopping : 是否使用早停
        validation_fraction : 用于早停的验证集比例
        early_stopping_rounds : 多少轮不提升则停止
        use_sample_weight_smoothing : 是否对样本权重进行平滑
        smoothing_factor : 权重平滑因子（0-1），越小平滑越多
        """
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=1)

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        # 鲁棒性参数
        self.weight_clip_percentile = weight_clip_percentile
        self.use_early_stopping = use_early_stopping
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.use_sample_weight_smoothing = use_sample_weight_smoothing
        self.smoothing_factor = smoothing_factor

        # 训练后的信息
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.best_n_estimators_ = n_estimators
        self.train_scores_ = []
        self.val_scores_ = []

    def fit(self, X, y, sample_weight=None):
        """
        训练鲁棒AdaBoost

        Parameters
        ----------
        X : 训练特征
        y : 训练标签
        sample_weight : 初始样本权重（可选）
        """
        from sklearn.model_selection import train_test_split

        # 如果使用早停，划分验证集
        if self.use_early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y,
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        n_samples = len(X_train)

        # 初始化样本权重
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight / sample_weight.sum()

        # 早停相关
        best_val_score = 0
        rounds_without_improvement = 0

        # 逐步训练弱学习器
        for i in range(self.n_estimators):
            # 应用权重裁剪
            sample_weight = self._clip_weights(sample_weight)

            # 应用权重平滑（可选）
            if self.use_sample_weight_smoothing:
                sample_weight = self._smooth_weights(sample_weight)

            # 训练弱学习器
            estimator = self._make_estimator()
            estimator.fit(X_train, y_train, sample_weight=sample_weight)

            # 预测
            y_pred = estimator.predict(X_train)

            # 计算加权错误率
            incorrect = y_pred != y_train
            estimator_error = np.sum(sample_weight * incorrect) / np.sum(sample_weight)

            # 如果错误率太高，停止训练
            # 注意：对于多分类问题，初始错误率可能较高，因此放宽条件
            if estimator_error >= 0.9:  # 放宽到90%，只有在极端情况下才停止
                print(
                    f"轮次 {i+1}: 错误率 {estimator_error:.4f} >= 0.9，提前停止"
                )
                break
            elif estimator_error >= 0.5:
                # 对于多分类，错误率>50%不一定意味着没有用处
                # 继续训练但打印警告
                if i == 0:  # 只在第一轮打印警告
                    print(f"警告：轮次 {i+1} 错误率 {estimator_error:.4f}，继续训练...")

            # 计算弱学习器权重α
            estimator_weight = self.learning_rate * np.log(
                (1 - estimator_error) / max(estimator_error, 1e-10)
            )

            # 更新样本权重
            sample_weight *= np.exp(estimator_weight * incorrect)
            sample_weight /= sample_weight.sum()

            # 保存弱学习器
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
            self.estimator_errors_.append(estimator_error)

            # 计算当前模型性能
            train_score = self.score(X_train, y_train)
            self.train_scores_.append(train_score)

            # 早停检查
            if self.use_early_stopping:
                val_score = self.score(X_val, y_val)
                self.val_scores_.append(val_score)

                if val_score > best_val_score:
                    best_val_score = val_score
                    rounds_without_improvement = 0
                    self.best_n_estimators_ = i + 1
                else:
                    rounds_without_improvement += 1

                # 打印进度（每10轮）
                if (i + 1) % 10 == 0:
                    print(
                        f"轮次 {i+1}/{self.n_estimators}: "
                        f"训练={train_score:.4f}, 验证={val_score:.4f}, "
                        f"最佳轮次={self.best_n_estimators_}"
                    )

                # 检查是否应该早停
                if rounds_without_improvement >= self.early_stopping_rounds:
                    print(
                        f"\n早停: {self.early_stopping_rounds} 轮未提升，"
                        f"停止在第 {i+1} 轮"
                    )
                    print(f"使用前 {self.best_n_estimators_} 个弱学习器")
                    break
            else:
                # 不使用早停时的进度显示
                if (i + 1) % 10 == 0:
                    print(
                        f"轮次 {i+1}/{self.n_estimators}: "
                        f"训练={train_score:.4f}, 错误率={estimator_error:.4f}"
                    )

        return self

    def _clip_weights(self, sample_weight):
        """
        裁剪样本权重，限制极端值

        Parameters
        ----------
        sample_weight : 当前样本权重

        Returns
        -------
        裁剪后的样本权重
        """
        if self.weight_clip_percentile >= 100:
            return sample_weight

        # 计算权重上限（基于百分位数）
        max_weight = np.percentile(sample_weight, self.weight_clip_percentile)

        # 裁剪权重
        clipped_weight = np.clip(sample_weight, 0, max_weight)

        # 重新归一化
        clipped_weight /= clipped_weight.sum()

        return clipped_weight

    def _smooth_weights(self, sample_weight):
        """
        平滑样本权重，减少极端差异

        Parameters
        ----------
        sample_weight : 当前样本权重

        Returns
        -------
        平滑后的样本权重
        """
        # 使用幂函数平滑：w_new = w^smoothing_factor
        smoothed_weight = np.power(sample_weight, self.smoothing_factor)

        # 重新归一化
        smoothed_weight /= smoothed_weight.sum()

        return smoothed_weight

    def _make_estimator(self):
        """创建基学习器的副本"""
        from sklearn.base import clone

        return clone(self.base_estimator)

    def predict(self, X):
        """
        预测

        Parameters
        ----------
        X : 测试特征

        Returns
        -------
        预测标签
        """
        # 使用前 best_n_estimators_ 个学习器
        n_estimators = min(self.best_n_estimators_, len(self.estimators_))

        # 加权投票
        predictions = np.array(
            [est.predict(X) for est in self.estimators_[:n_estimators]]
        )
        weights = np.array(self.estimator_weights_[:n_estimators])

        # 对每个类别计算加权得分
        classes = np.unique(predictions)
        n_samples = X.shape[0]
        class_scores = np.zeros((n_samples, len(classes)))

        for i, cls in enumerate(classes):
            class_scores[:, i] = np.sum(
                weights[:, np.newaxis] * (predictions == cls), axis=0
            )

        # 返回得分最高的类别
        return classes[np.argmax(class_scores, axis=1)]

    def predict_proba(self, X):
        """
        预测概率

        Parameters
        ----------
        X : 测试特征

        Returns
        -------
        预测概率
        """
        # 使用前 best_n_estimators_ 个学习器
        n_estimators = min(self.best_n_estimators_, len(self.estimators_))

        predictions = np.array(
            [est.predict(X) for est in self.estimators_[:n_estimators]]
        )
        weights = np.array(self.estimator_weights_[:n_estimators])

        classes = np.unique(predictions)
        n_samples = X.shape[0]
        class_scores = np.zeros((n_samples, len(classes)))

        for i, cls in enumerate(classes):
            class_scores[:, i] = np.sum(
                weights[:, np.newaxis] * (predictions == cls), axis=0
            )

        # 归一化为概率
        proba = class_scores / class_scores.sum(axis=1, keepdims=True)
        return proba

    def score(self, X, y):
        """
        计算准确率

        Parameters
        ----------
        X : 特征
        y : 真实标签

        Returns
        -------
        准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# 方便的预设配置
def create_robust_adaboost(strategy="balanced", **kwargs):
    """
    创建预设配置的鲁棒AdaBoost

    Parameters
    ----------
    strategy : str
        策略名称，可选:
        - 'balanced': 平衡配置（推荐）
        - 'aggressive_clip': 激进权重裁剪（高噪声环境）
        - 'early_stop': 重点早停（防止过拟合）
        - 'smooth': 权重平滑（温和改进）
        - 'conservative': 保守配置（最鲁棒）

    Returns
    -------
    RobustAdaBoost 实例
    """
    configs = {
        "balanced": {
            "n_estimators": 100,
            "learning_rate": 0.5,
            "weight_clip_percentile": 95,
            "use_early_stopping": True,
            "validation_fraction": 0.1,
            "early_stopping_rounds": 10,
            "use_sample_weight_smoothing": False,
        },
        "aggressive_clip": {
            "n_estimators": 100,
            "learning_rate": 0.3,
            "weight_clip_percentile": 90,  # 更激进裁剪
            "use_early_stopping": True,
            "validation_fraction": 0.15,
            "early_stopping_rounds": 15,
            "use_sample_weight_smoothing": True,
            "smoothing_factor": 0.7,
        },
        "early_stop": {
            "n_estimators": 200,
            "learning_rate": 0.5,
            "weight_clip_percentile": 98,
            "use_early_stopping": True,
            "validation_fraction": 0.2,  # 更大验证集
            "early_stopping_rounds": 5,  # 更快早停
            "use_sample_weight_smoothing": False,
        },
        "smooth": {
            "n_estimators": 100,
            "learning_rate": 0.5,
            "weight_clip_percentile": 98,
            "use_early_stopping": True,
            "validation_fraction": 0.1,
            "early_stopping_rounds": 10,
            "use_sample_weight_smoothing": True,
            "smoothing_factor": 0.5,  # 更强平滑
        },
        "conservative": {
            "n_estimators": 150,
            "learning_rate": 0.1,  # 低学习率
            "weight_clip_percentile": 90,
            "use_early_stopping": True,
            "validation_fraction": 0.15,
            "early_stopping_rounds": 20,
            "use_sample_weight_smoothing": True,
            "smoothing_factor": 0.6,
        },
    }

    if strategy not in configs:
        raise ValueError(
            f"未知策略: {strategy}. 可选: {list(configs.keys())}"
        )

    # 合并用户提供的参数
    config = configs[strategy]
    config.update(kwargs)

    return RobustAdaBoost(**config)




