import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.monitor import BoostMonitor

ori_boost = AdaBoostClassifier._boost


def boost_with_monitor(self, iboost, X, y, sample_weight, random_state):
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

        # validation
        self._run_validation(iboost)
        self._val_on_train_data(iboost, X, y)
        # save monitor data checkpoint
        self._monitor.auto_checkpoint(iboost)
        return sample_weight, estimator_weight, estimator_error

    def _run_validation(self, iboost):
        """Run validation after each boost round."""
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
        """Calculate scores on training data."""

        # 当前模型的集成已经形成，可以直接 predict
        y_pred = self.predict(X)

        # 计算指标
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")

        # 写进 monitor
        if hasattr(self, "_monitor") and self._monitor is not None:
            self._monitor.record_training_scores(iboost, acc, f1)
