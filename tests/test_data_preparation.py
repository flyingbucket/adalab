import time
import unittest

import numpy as np
from src.utils import DataPreparation


class TestDataPreparation(unittest.TestCase):
    def test_prepare_full_pipeline(self):
        """真实下载 MNIST，测试 DataPreparation 全流程"""

        noise_config = {
            "label_flip": {"ratio": 0.2},
            "gaussian": {"std": 0.05},
            "salt_pepper": {"amount": 0.02},
            "contrast": {"factor_range": [0.7, 1.3]},
            "rotate": {"angle_range": 10},
            "blur": {"kernel_size": 3},
            "brightness": {"shift_range": 0.2},
        }

        dp = DataPreparation(
            noise_config=noise_config,
            test_size=0.2,
            use_feature="original",
            random_state=42,
        )

        # ========== 运行完整 prepare 流程 ==========
        t0 = time.time()
        X_train, X_test, y_train, y_test, noise_idx, clean_idx = dp.prepare()
        t1 = time.time()

        print(f"[Test] Total prepare time: {t1 - t0:.2f} seconds")

        # ========== 基本维度测试 ==========
        self.assertEqual(X_train.shape[0], int(70000 * 0.8))
        self.assertEqual(X_test.shape[0], int(70000 * 0.2))
        self.assertEqual(len(y_train), X_train.shape[0])
        self.assertEqual(len(y_test), X_test.shape[0])

        # ========== label_flip 噪声比例 ==========
        expected_noise = int(len(y_train) * 0.2)
        self.assertTrue(abs(len(noise_idx) - expected_noise) <= 3)  # 允许一点随机偏差

        # ========== 像素噪声是否只作用于噪声样本 ==========
        noisy_pixels = X_train[noise_idx]
        clean_pixels = X_train[clean_idx]

        # 均值应该明显不同（噪声会增加像素变化）
        self.assertNotEqual(np.mean(noisy_pixels), np.mean(clean_pixels))

        # ========== y_train 发生变化 ==========
        self.assertFalse(np.array_equal(y_train, dp.y_train_raw))

        # ========== 测试集保持干净 ==========
        self.assertTrue(np.array_equal(X_test, dp.X_test))
        self.assertTrue(np.array_equal(y_test, dp.y_test))

        print("[Test] prepare() pipeline with real MNIST passed.")

    def test_hog_feature_extraction(self):
        """真实 MNIST + HOG 特征流水线"""

        dp = DataPreparation(
            noise_config={"label_flip": {"ratio": 0.1}},
            test_size=0.2,
            use_feature="hog",
            random_state=42,
        )

        X_train, X_test, y_train, y_test, noise_idx, clean_idx = dp.prepare()

        # HOG 输出应为二维矩阵
        self.assertEqual(X_train.ndim, 2)
        self.assertEqual(X_test.ndim, 2)
        self.assertTrue(X_train.shape[0] > 1000)

        print("[Test] HOG feature extraction passed.")

    def test_hu_feature_extraction(self):
        """真实 MNIST + Hu Moments 特征流水线"""

        dp = DataPreparation(
            noise_config={"label_flip": {"ratio": 0.1}},
            test_size=0.2,
            use_feature="hu",
            random_state=42,
        )

        X_train, X_test, y_train, y_test, noise_idx, clean_idx = dp.prepare()

        # Hu Moments = 7 个系数
        self.assertEqual(X_train.shape[1], 7)
        self.assertEqual(X_test.shape[1], 7)

        print("[Test] Hu feature extraction passed.")


if __name__ == "__main__":
    unittest.main()
