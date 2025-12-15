import unittest
import numpy as np
from adalab.workflow import load_compressed, load_config, prep_data_from_config
from adalab.evaluation import val_after_train, val_after_train_parallel


class TestValAfterTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = load_compressed(
            "/home/flyingbucket/CODE/AdaBoost_numbers/experiments/test_end2end/results/model.joblib.xz"
        )

        cls.monitor = load_compressed(
            "/home/flyingbucket/CODE/AdaBoost_numbers/experiments/test_end2end/results/monitor.joblib.xz"
        )

        cls.config = load_config(
            "/home/flyingbucket/CODE/AdaBoost_numbers/experiments/test_end2end_20251208_220714/config.json"
        )

        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
            _,
            _,
            cls.prep,
        ) = prep_data_from_config(cls.config)

        cls.alphas = np.asarray(cls.monitor.alpha_history)
        cls.val_freq = cls.config["monitor"].get("val_freq", 20)

    def test_sequential_vs_parallel(self):
        acc_seq, f1_seq, idx_seq = val_after_train(
            self.clf, self.alphas, self.X_test, self.y_test, val_freq=self.val_freq
        )

        acc_par, f1_par, idx_par = val_after_train_parallel(
            self.clf,
            self.alphas,
            self.X_test,
            self.y_test,
            val_freq=self.val_freq,
            n_jobs=-1,
        )

        # index 一定要一致
        self.assertTrue(np.array_equal(idx_seq, idx_par))

        # 误差要非常小
        self.assertTrue(np.allclose(acc_seq, acc_par, atol=1e-12))
        self.assertTrue(np.allclose(f1_seq, f1_par, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
