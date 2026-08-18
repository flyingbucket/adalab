import unittest
from unittest.mock import MagicMock

import numpy as np

from adalab.data import DataPreparationForTesting, DataSplitForTesting


class TestDataPreparationForTesting(unittest.TestCase):
    def setUp(self):
        # 创建一个假设的训练分割对象
        self.train_split_mock = MagicMock()
        self.train_split_mock.X_test = np.random.rand(
            100, 28 * 28
        )  # 假设100个28x28的图像
        self.train_split_mock.y_test = np.random.randint(
            0, 10, 100
        )  # 假设100个标签，范围是[0, 9]

        # 创建DataPreparationForTesting对象
        self.test_shift_config = {
            "contrast": {"factor_range": (0.8, 1.2)},
            "brightness": {"shift_range": 0.1},
            "rotate": {"angle_range": 5},
        }
        self.use_feature = "original"  # 假设的特征配置
        self.feature_config = {}  # 可以根据需要填充
        self.data_preparation = DataPreparationForTesting(
            self.test_shift_config,
            self.use_feature,
            self.feature_config,
            self.train_split_mock,
        )

    def test_apply_shift(self):
        # 使用apply_shift方法进行测试，检查返回的数据形状和类型
        X = self.train_split_mock.X_test
        shifted_X = self.data_preparation.apply_shift(X, self.test_shift_config)
        self.assertEqual(shifted_X.shape, X.shape)  # 检查数据维度
        self.assertIsInstance(shifted_X, np.ndarray)  # 确保返回的是numpy数组

    def test_get_shift_x_test(self):
        # 测试get_shift_x_test，确保返回包含所有扰动配置
        shifted_data = self.data_preparation.get_shift_x_test()
        self.assertTrue("contrast" in shifted_data)
        self.assertTrue("brightness" in shifted_data)
        self.assertTrue("rotate" in shifted_data)
        self.assertEqual(
            shifted_data["contrast"].shape, self.train_split_mock.X_test.shape
        )

    def test_prepare_course_data(self):
        # 模拟课程数据的准备工作
        folder = "./data/test_images"  # 模拟的文件夹路径
        X, y = self.data_preparation.prepare_course_data(folder)
        self.assertEqual(
            X.shape[0], 10
        )  # 检查返回的数据数量是否为100（假设课程数据有100个样本）
        self.assertEqual(y.shape[0], 10)  # 标签数量应该也为100
        self.assertEqual(X.shape[1], 28 * 28)  # 每个图像应该被处理成784维的向量

    def test_prepare(self):
        # 测试prepare方法，确保返回的DataSplitForTesting对象包含正确的数据
        folder = "./data/test_images"  # 模拟的文件夹路径
        data_split = self.data_preparation.prepare(folder)

        self.assertIsInstance(data_split, DataSplitForTesting)
        self.assertEqual(
            data_split.X_mnist_ori.shape, self.train_split_mock.X_test.shape
        )
        self.assertEqual(len(data_split.X_mnist_shift), len(self.test_shift_config))
        self.assertEqual(data_split.X_course.shape[0], 10)
        self.assertEqual(data_split.y_course.shape[0], 10)


if __name__ == "__main__":
    unittest.main()
