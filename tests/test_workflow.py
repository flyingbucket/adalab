import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from adalab import workflow


def _make_dummy_split() -> workflow.DataSplit:
    # 小数据，训练很快
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 5)).astype(np.float32)
    y_train = rng.integers(0, 2, size=(20,), dtype=np.int64)
    X_test = rng.normal(size=(10, 5)).astype(np.float32)
    y_test = rng.integers(0, 2, size=(10,), dtype=np.int64)

    noise_idx = np.array([0, 1, 2], dtype=np.int64)
    clean_idx = np.array([i for i in range(20) if i not in noise_idx], dtype=np.int64)

    # prep 只要占位，不会用到
    prep = MagicMock()

    return workflow.DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        noise_idx=noise_idx,
        clean_idx=clean_idx,
        prep=prep,
    )


def _write_config(path: Path, use_monitor: bool) -> None:
    cfg = {
        "experiment": {"name": "unittest_exp"},
        # data 在本测试里会被 mock 掉，但保留结构防止未来改动出错
        "data": {
            "noise_config": {},
            "test_size": 0.2,
            "random_state": 42,
            "use_feature": "original",
        },
        "monitor": {
            "use_monitor": use_monitor,
            "is_data_noisy": False,
            "checkpoint_interval": 50,
            "val_freq": 2,
            "val_n_jobs": 1,
        },
        "model": {
            "estimator": {"max_depth": 1},
            "n_estimators": 3,
            "learning_rate": 1.0,
            "random_state": 0,
        },
    }
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


class TestWorkflow(unittest.TestCase):
    def test_build_experiment_no_monitor(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            config_path = td_path / "config.json"
            _write_config(config_path, use_monitor=False)

            dummy_split = _make_dummy_split()

            # patch ExperimentPaths.create，把实验目录建到临时目录里（避免污染项目根目录）
            orig_create = workflow.ExperimentPaths.create

            def create_in_tmp(exp_name: str, base_dir: str = "experiments"):
                return orig_create(exp_name, base_dir=str(td_path / "runs"))

            with (
                patch.object(
                    workflow.ExperimentPaths, "create", side_effect=create_in_tmp
                ),
                patch.object(
                    workflow, "prep_data_from_config", return_value=dummy_split
                ),
            ):
                clf, monitor, split, layout = workflow.build_experiment(
                    str(config_path)
                )

                self.assertIsNone(monitor)
                self.assertIs(split, dummy_split)

                # layout 目录 & config.json 必须存在
                self.assertTrue(layout.exp_dir.exists())
                self.assertTrue(layout.ckpt_dir.exists())
                self.assertTrue(layout.result_dir.exists())
                self.assertTrue(layout.config_path.exists())

                # 写入的 config 内容至少包含 experiment.name
                dumped = json.loads(layout.config_path.read_text(encoding="utf-8"))
                self.assertEqual(dumped["experiment"]["name"], "unittest_exp")

                # clf 类型（不要求完全一致也行，但这里顺便校验）
                from sklearn.ensemble import AdaBoostClassifier

                self.assertIsInstance(clf, AdaBoostClassifier)

    def test_train_and_save_no_monitor_calls_save_paths(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            config_path = td_path / "config.json"
            _write_config(config_path, use_monitor=False)

            dummy_split = _make_dummy_split()

            orig_create = workflow.ExperimentPaths.create

            def create_in_tmp(exp_name: str, base_dir: str = "experiments"):
                return orig_create(exp_name, base_dir=str(td_path / "runs"))

            # mock joblib.dump 和 dump_compressed，验证“传入路径是否正确”
            with (
                patch.object(
                    workflow.ExperimentPaths, "create", side_effect=create_in_tmp
                ),
                patch.object(
                    workflow, "prep_data_from_config", return_value=dummy_split
                ),
                patch.object(workflow.joblib, "dump") as m_joblib_dump,
                patch.object(
                    workflow, "dump_compressed", side_effect=lambda obj, out: out
                ) as m_dump_compressed,
            ):
                clf, monitor, split, layout, result_paths = workflow.train_and_save(
                    str(config_path)
                )

                self.assertIsNone(monitor)
                self.assertIs(split, dummy_split)

                # joblib.dump 应该被调用一次：保存 raw_clf
                m_joblib_dump.assert_called_once()
                args, kwargs = m_joblib_dump.call_args
                self.assertEqual(Path(args[1]), result_paths.raw_clf)

                # dump_compressed 应该被调用一次：保存 compressed_clf
                m_dump_compressed.assert_called_once()
                args, kwargs = m_dump_compressed.call_args
                self.assertEqual(args[1], str(result_paths.compressed_clf))

                # monitor 相关路径应为 None
                self.assertIsNone(result_paths.raw_monitor)
                self.assertIsNone(result_paths.compressed_monitor)
                self.assertIsNone(result_paths.monitor_csv)

    def test_build_experiment_with_monitor_checkpoint_prefix_is_str(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            config_path = td_path / "config.json"
            _write_config(config_path, use_monitor=True)

            dummy_split = _make_dummy_split()

            orig_create = workflow.ExperimentPaths.create

            def create_in_tmp(exp_name: str, base_dir: str = "experiments"):
                return orig_create(exp_name, base_dir=str(td_path / "runs"))

            # 用假 BoostMonitor 捕获 checkpoint_prefix
            class DummyMonitor:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.alpha_history = []
                    self.acc_on_train_data = []
                    self.f1_on_training_data = []
                    self.val_idx = []
                    self.val_acc_history = []
                    self.val_f1_history = []

                def dump(self, *args, **kwargs):
                    return None

            class DummyClf:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def fit(self, X, y):
                    return self

            with (
                patch.object(
                    workflow.ExperimentPaths, "create", side_effect=create_in_tmp
                ),
                patch.object(
                    workflow, "prep_data_from_config", return_value=dummy_split
                ),
                patch.object(workflow, "BoostMonitor", DummyMonitor),
                patch.object(workflow, "AdaBoostClfWithMonitor", DummyClf),
            ):
                clf, monitor, split, layout = workflow.build_experiment(
                    str(config_path)
                )

                self.assertIsNotNone(monitor)
                prefix = monitor.kwargs["checkpoint_prefix"]
                self.assertIsInstance(
                    prefix, str
                )  # ✅ 关键：monitor 内部用 f-string 拼路径更安全


if __name__ == "__main__":
    unittest.main(verbosity=2)
