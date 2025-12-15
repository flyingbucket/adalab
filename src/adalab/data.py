"""
MNIST 数据准备、噪声注入与特征提取。

本模块负责：
- 下载与划分 MNIST 数据集
- 按配置向训练集注入标签或像素级噪声
- 提取原始像素、HOG 或 Hu Moments 特征
- 处理外部课程或真实拍照数字数据

该模块为实验提供统一、可复现的数据输入。
"""

from __future__ import annotations
import os
import warnings
from typing import Any, Dict, Tuple, Optional

import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
Int64Array = NDArray[np.int64]


def preprocess_for_mnist(path):
    """将任意单张数字图片预处理为 MNIST 风格输入。

    该函数用于将真实拍照或扫描的数字图像转换为
    与 MNIST 数据集一致的 28×28、黑底白字、归一化格式，
    主要用于课程数据或外部数据的推理测试。

    Args:
        path (str): 输入图像文件路径。

    Returns:
        tuple:
            - x (np.ndarray): 展平后的特征向量，形状为 (1, 784)，取值范围 [0, 1]。
            - canvas (np.ndarray): 28×28 的中间灰度图像（uint8），便于可视化调试。
    """
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 轻度降噪
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu 阈值
    _, binary = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 调整为黑底白字
    bw_ratio = np.mean(binary == 0)  # 所有像素中黑色比例

    if bw_ratio < 0.5:
        binary = 255 - binary

    # bounding box
    ys, xs = np.where(binary == 255)
    if len(xs) == 0:
        # fallback 防止全白/全黑图
        digit = binary
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        digit = binary[y1 : y2 + 1, x1 : x2 + 1]

    # 轻度膨胀（让线条更粗一点，更像 MNIST）
    kernel = np.ones((2, 2), np.uint8)
    digit = cv2.dilate(digit, kernel, iterations=1)

    # 缩放到 20×20（保持比例）
    h, w = digit.shape
    if h > w:
        new_h, new_w = 20, int(20 * w / h)
    else:
        new_w, new_h = 20, int(20 * h / w)

    digit_small = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 居中到 28×28 黑底
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = digit_small

    # 归一化（MNIST 风格）
    arr_final = canvas.astype("float32") / 255.0

    return arr_final.reshape(1, -1), canvas


class DataPreparation:
    """MNIST 数据准备与噪声注入调度器。

    该类负责完成一次实验中所有与数据相关的工作，包括：
    - 下载 MNIST 数据集
    - 划分训练集与测试集
    - 按配置向训练集注入噪声
    - 提取特征（原始像素 / HOG / Hu Moments）

    该类是 workflow 中数据准备阶段的唯一入口。

    Attributes:
        X_train (FloatArray): 训练集特征矩阵。
        X_test (FloatArray): 测试集特征矩阵。
        y_train (Int64Array): 训练集标签（可能包含噪声）。
        y_test (Int64Array): 测试集标签（始终为干净标签）。
        train_noise_indices (Int64Array): 训练集中噪声样本的索引。
        train_clean_indices (Int64Array): 训练集中干净样本的索引。
    """

    X_train: FloatArray
    X_test: FloatArray
    y_train: Int64Array
    y_test: Int64Array

    # 噪声索引
    train_noise_indices: Int64Array
    train_clean_indices: Int64Array

    def __init__(
        self,
        noise_config: Optional[Dict[str, Any]] = None,
        test_size=0.2,
        use_feature="original",
        random_state=42,
        # HOG 参数
        hog_orientations=9,
        hog_pixels_per_cell=(4, 4),
        hog_cells_per_block=(2, 2),
        # Hu Moments 参数
        hu_log_scale=True,
    ):
        """初始化数据准备器。

        Args:
            noise_config (dict, optional): 噪声配置字典，用于控制噪声类型与比例。
            test_size (float, optional): 测试集比例，默认 0.2。
            use_feature (str, optional): 特征类型，可选 "original"、"hog"、"hu"。
            random_state (int, optional): 随机种子。
            hog_orientations (int, optional): HOG 特征方向数。
            hog_pixels_per_cell (tuple, optional): HOG 每个 cell 的像素大小。
            hog_cells_per_block (tuple, optional): HOG 每个 block 的 cell 数。
            hu_log_scale (bool, optional): 是否对 Hu Moments 进行 log 变换。
        """
        self.noise_config = noise_config or {}
        self.test_size = test_size
        self.use_feature = use_feature
        self.random_state = random_state

        # HOG settings
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block

        # Hu settings
        self.hu_log_scale = hu_log_scale

        # empty init
        self.X_train = np.empty((0, 0), dtype=np.float32)
        self.X_test = np.empty((0, 0), dtype=np.float32)
        self.y_train = np.empty((0,), dtype=np.int64)
        self.y_test = np.empty((0,), dtype=np.int64)

        self.train_noise_indices = np.empty((0,), dtype=np.int64)
        self.train_clean_indices = np.empty((0,), dtype=np.int64)

        self.perturber = MNISTPerturber(random_state)

    def download_mnist(self):
        print("[Data] Downloading MNIST...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        y = y.astype(np.int64)
        X = X / 255.0
        self.X_raw = X
        self.y_raw = y

    def split(self):
        """将原始 MNIST 数据划分为训练集与测试集。

        划分结果会同时保存样本在原始数据中的索引，
        以便后续准确标记训练集内部的噪声样本位置。
        """
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            self.X_raw,
            self.y_raw,  # 全干净标签
            np.arange(len(self.y_raw)),
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.test_idx = np.asarray(test_idx, dtype=np.int64)

        self.X_train_raw = np.asarray(X_train, dtype=np.float32)
        self.X_test = np.asarray(X_test, dtype=np.float32)
        self.y_train_raw = np.asarray(y_train, dtype=np.int64)
        self.y_test = np.asarray(y_test, dtype=np.int64)
        print(f"[Data] Split done: Train={len(X_train)}, Test={len(X_test)}")

    def inject_noise(self):
        """向训练集注入噪声。

        根据 ``noise_config`` 中的配置，
        对训练集样本施加标签噪声或像素级扰动。

        噪声仅作用于训练集，测试集始终保持干净标签。
        """
        print("[Data] Applying perturbations...")

        X = self.X_train_raw.copy()
        y = self.y_train_raw.copy()
        pert = self.perturber

        noise_items = [(k, v) for k, v in self.noise_config.items() if k != "ratio"]
        ratio = float(self.noise_config.get("ratio", 0.0))
        # 若无 noise_config，直接返回
        if len(self.noise_config) == 0 or ratio == 0 or len(noise_items) == 0:
            print("[Data] No perturbations applied.")
            self.X_train = X
            self.y_train = y
            self.noise_indices = np.array([], dtype=int)
            return
        n_samples = len(X)
        n_noisy = int(n_samples * ratio)
        noise_indices = pert.rng.choice(n_samples, n_noisy, replace=False)
        print(f"[Data] Random selection: {n_noisy} indices selected")

        self.noise_indices = np.array(noise_indices)
        subset = X[self.noise_indices]  # 只处理噪声样本

        # 将所有像素噪声叠加到相同 noise_indices 样本上
        for noise_type, params in noise_items:
            if noise_type == "label_flip" and params:
                y = pert.flip_labels(y, noise_indices=self.noise_indices)
                print(f"[Data] Label flip: {len(noise_indices)} indices selected")

            elif noise_type == "gaussian":
                std = params.get("std", 0.1)
                subset = pert.add_gaussian_noise(subset, noise_std=std)
                print(f"[Data] Gaussian noise std={std}")

            elif noise_type == "salt_pepper":
                amount = params.get("amount", 0.05)
                subset = pert.add_salt_pepper_noise(subset, amount=amount)
                print(f"[Data] Salt-Pepper amount={amount}")

            elif noise_type == "contrast":
                fr = params.get("factor_range", (0.5, 1.5))
                subset = pert.adjust_contrast(subset, factor_range=fr)
                print(f"[Data] Contrast factor_range={fr}")

            elif noise_type == "brightness":
                sr = params.get("shift_range", 0.3)
                subset = pert.add_brightness_shift(subset, shift_range=sr)
                print(f"[Data] Brightness shift_range={sr}")

            elif noise_type == "rotate":
                ar = params.get("angle_range", 15)
                subset = pert.rotate_slight(subset, angle_range=ar)
                print(f"[Data] Rotate angle_range={ar}")

            elif noise_type == "blur":
                ks = params.get("kernel_size", 3)
                subset = pert.add_blur(subset, kernel_size=ks)
                print(f"[Data] Blur kernel_size={ks}")

            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            # 写回噪声样本
            X[self.noise_indices] = subset

        self.X_train = X
        self.y_train = y
        print(f"[Data] Total noisy samples: {len(self.noise_indices)}")

        # 训练集内部噪声索引
        self.train_noise_indices = self.noise_indices
        self.train_clean_indices = np.array(
            list(set(range(len(self.y_train))) - set(self.train_noise_indices))
        )

        print(
            f"[Data] Noisy Train: {len(self.train_noise_indices)} noise, {len(self.train_clean_indices)} clean"
        )

    # 特征提取
    def extract_hog(self, X):
        """从输入图像中提取 HOG 特征。

        Args:
            X (np.ndarray): 展平的 MNIST 图像数据。

        Returns:
            np.ndarray: HOG 特征矩阵。
        """
        X_reshaped = X.reshape(-1, 28, 28)
        feats = []
        for img in X_reshaped:
            f = hog(
                img,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                block_norm="L2-Hys",
            )
            feats.append(f)
        return np.array(feats)

    def extract_hu(self, X):
        """从输入图像中提取 Hu Moments 特征。

        Args:
            X (np.ndarray): 展平的 MNIST 图像数据。

        Returns:
            np.ndarray: Hu Moments 特征矩阵。
        """
        X_reshaped = X.reshape(-1, 28, 28)
        feats = []
        for img in X_reshaped:
            moments = cv2.HuMoments(cv2.moments(img)).flatten()
            if self.hu_log_scale:
                moments = -np.sign(moments) * np.log10(np.abs(moments))
            feats.append(moments)
        return np.array(feats)

    def apply_feature(self):
        """根据配置对训练集与测试集应用特征提取。

        特征类型由 ``self.use_feature`` 控制：
        - "original": 使用原始像素
        - "hog": 使用 HOG 特征
        - "hu": 使用 Hu Moments
        """
        if self.use_feature == "original":
            print("[Data] No feature extracted,using original images")
            pass

        elif self.use_feature == "hog":
            print("[Data] Extracting HOG features...")
            self.X_train = self.extract_hog(self.X_train)
            self.X_test = self.extract_hog(self.X_test)

        elif self.use_feature == "hu":
            print("[Data] Extracting Hu moments...")
            self.X_train = self.extract_hu(self.X_train)
            self.X_test = self.extract_hu(self.X_test)

        else:
            raise ValueError("[Data] Invalid feature type")

    # 总调度函数

    def prepare(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """执行完整的数据准备流程。

        该方法是数据准备阶段的统一入口，
        会依次执行下载、划分、噪声注入与特征提取。

        Returns:
            tuple:
                - X_train: 训练集特征
                - X_test: 测试集特征
                - y_train: 训练集标签
                - y_test: 测试集标签
                - train_noise_indices: 训练集噪声样本索引
                - train_clean_indices: 训练集干净样本索引
        """
        self.download_mnist()
        self.split()
        self.inject_noise()
        # self.split()
        self.apply_feature()
        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.train_noise_indices,
            self.train_clean_indices,
        )

    def prepare_course_data(self, folder):
        """处理课程提供的真实拍照数字数据。

        该方法用于对外部真实图像数据进行推理测试，
        不进行训练/测试划分，也不注入噪声，
        仅执行 MNIST 风格预处理与特征提取。

        Args:
            folder (str): 图像文件夹路径，文件名需为标签值。

        Returns:
            tuple:
                - X (np.ndarray): 特征矩阵（与训练特征维度一致）
                - y (np.ndarray): 标签数组
        """
        print(f"[Data] Loading course dataset from: {folder}")

        X_list = []
        y_list = []

        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                label = int(os.path.splitext(filename)[0])
                path = os.path.join(folder, filename)

                # MNIST化预处理
                x28, _ = preprocess_for_mnist(path)  # (1, 784)
                X_list.append(x28[0])
                y_list.append(label)

        X_raw = np.array(X_list)
        y = np.array(y_list, dtype=np.int64)

        if self.use_feature == "original":
            X = X_raw

        elif self.use_feature == "hog":
            print("[Data] Extracting HOG features for course data...")
            X = self.extract_hog(X_raw)

        elif self.use_feature == "hu":
            print("[Data] Extracting Hu moments for course data...")
            X = self.extract_hu(X_raw)

        else:
            raise ValueError(f"Invalid feature type: {self.use_feature}")

        return X, y


class MNISTPerturber:
    """MNIST 数据扰动器。

    提供多种用于鲁棒性实验的标签与像素级扰动方法，
    仅作为 DataPreparation 的内部组件使用。
    """

    def __init__(self, random_state=42):
        """
        初始化扰动器

        Parameters
        ----------
        random_state : int
            随机种子
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def flip_labels(self, y, noise_indices, num_classes=10):
        """
        随机翻转标签噪声（对分类标签做对抗扰动）

        Parameters
        ----------
        y : array
            标签数组（真实标签）
        noise_indices:NDArray
            噪声数据索引
        num_classes : int
            分类数（默认为 MNIST 的 10 类）

        Returns
        -------
        y_noisy : ndarray
            添加噪声后的标签
        """
        y_noisy = y.copy()

        y_noisy[noise_indices] = self.rng.randint(
            0, num_classes, size=len(noise_indices)
        )

        return y_noisy

    def add_brightness_shift(self, X, shift_range=0.3):
        """
        添加亮度偏移

        Parameters
        ----------
        X : array
            原始数据 [0, 1]
        shift_range : float
            亮度偏移范围 [-shift_range, shift_range]
        """
        shift = self.rng.uniform(-shift_range, shift_range, size=len(X))
        X_perturbed = X + shift[:, np.newaxis]
        return np.clip(X_perturbed, 0, 1)

    def add_gaussian_noise(self, X, noise_std=0.1):
        """
        添加高斯噪声

        Parameters
        ----------
        X : array
            原始数据
        noise_std : float
            噪声标准差
        """
        noise = self.rng.normal(0, noise_std, X.shape)
        X_perturbed = X + noise
        return np.clip(X_perturbed, 0, 1)

    def add_salt_pepper_noise(self, X, amount=0.05):
        """
        添加椒盐噪声

        Parameters
        ----------
        X : array
            原始数据
        amount : float
            噪声比例
        """
        X_perturbed = X.copy()

        # Salt噪声（白点）
        n_salt = int(amount * X.size * 0.5)
        coords = [self.rng.randint(0, i, n_salt) for i in X.shape]
        X_perturbed[tuple(coords)] = 1

        # Pepper噪声（黑点）
        n_pepper = int(amount * X.size * 0.5)
        coords = [self.rng.randint(0, i, n_pepper) for i in X.shape]
        X_perturbed[tuple(coords)] = 0

        return X_perturbed

    def add_blur(self, X, kernel_size=3):
        """
        添加模糊效果（简单平均滤波）

        Parameters
        ----------
        X : array
            原始数据
        kernel_size : int
            模糊核大小
        """
        from scipy.ndimage import uniform_filter

        X_perturbed = np.zeros_like(X)
        for i in range(len(X)):
            img = X[i].reshape(28, 28)
            blurred = uniform_filter(img, size=kernel_size, mode="constant")
            X_perturbed[i] = blurred.ravel()

        return X_perturbed

    def adjust_contrast(self, X, factor_range=(0.5, 1.5)):
        """
        调整对比度

        Parameters
        ----------
        X : array
            原始数据
        factor_range : tuple
            对比度因子范围
        """
        factors = self.rng.uniform(factor_range[0], factor_range[1], size=len(X))

        X_perturbed = np.zeros_like(X)
        for i in range(len(X)):
            mean = X[i].mean()
            X_perturbed[i] = mean + factors[i] * (X[i] - mean)

        return np.clip(X_perturbed, 0, 1)

    def rotate_slight(self, X, angle_range=15):
        """
        轻微旋转

        Parameters
        ----------
        X : array
            原始数据
        angle_range : float
            旋转角度范围（度）
        """
        from scipy.ndimage import rotate

        X_perturbed = np.zeros_like(X)
        for i in range(len(X)):
            img = X[i].reshape(28, 28)
            angle = self.rng.uniform(-angle_range, angle_range)
            rotated = rotate(img, angle, reshape=False, mode="constant", cval=0)
            X_perturbed[i] = rotated.ravel()

        return X_perturbed

    def apply_perturbation(self, X, perturbation_type, **kwargs):
        """
        应用指定类型的扰动

        Parameters
        ----------
        X : array
            原始数据
        perturbation_type : str
            扰动类型
        """
        if perturbation_type == "brightness":
            return self.add_brightness_shift(X, **kwargs)
        elif perturbation_type == "gaussian_noise":
            return self.add_gaussian_noise(X, **kwargs)
        elif perturbation_type == "salt_pepper":
            return self.add_salt_pepper_noise(X, **kwargs)
        elif perturbation_type == "blur":
            return self.add_blur(X, **kwargs)
        elif perturbation_type == "contrast":
            return self.adjust_contrast(X, **kwargs)
        elif perturbation_type == "rotation":
            return self.rotate_slight(X, **kwargs)
        else:
            raise ValueError(f"未知扰动类型: {perturbation_type}")


def prepare_data(noise_ratio=0.05, test_size=0.2, random_state=42):
    """deprecated,use DataPreparation instead
    下载 MNIST，并按指定比例添加标签噪声。
    自动返回：
        - X_train, X_test
        - y_train (含噪声) , y_test
        - train_noise_indices  (训练集内部噪声索引)
        - train_clean_indices  (训练集内部干净索引)
    若 noise_ratio=0，则返回完全干净的数据。

    Parameters
    ----------
    noise_ratio : float
        噪声比例（0 ~ 1），表示标签噪声的比例。
        若为 0，则不添加标签噪声。

    test_size : float
        train_test_split 的测试集占比

    random_state : int
        随机种子

    Returns
    -------
    X_train, X_test : ndarray
    y_train, y_test : ndarray
    train_noise_indices : ndarray (训练集内部的噪声样本位置)
    train_clean_indices : ndarray
    """
    warnings.warn(
        "prepare_data() is deprecated. Please use DataPreparation instead.",
        FutureWarning,
        stacklevel=2,
    )
    print("Downloading MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.int64)
    X = X / 255.0

    n_samples = len(y)

    # -----------------------------------------
    # Case 1: 不添加噪声，返回原始数据
    # -----------------------------------------
    if noise_ratio <= 0:
        print("No noise added, returning clean dataset.")

        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, np.arange(n_samples), test_size=test_size, random_state=random_state
        )

        # 训练集全部是 clean
        train_noise_indices = np.array([], dtype=int)
        train_clean_indices = np.arange(len(y_train))

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            train_noise_indices,
            train_clean_indices,
        )

    # Case 2: 添加噪声
    n_noisy = int(n_samples * noise_ratio)
    rng = np.random.default_rng(random_state)

    noise_indices = rng.choice(n_samples, n_noisy, replace=False)

    y_noisy = y.copy()
    y_noisy[noise_indices] = rng.integers(0, 10, size=n_noisy)

    print(f"Injected label noise: {noise_ratio * 100:.1f}% ({n_noisy} samples)")

    # train/test split，保留原始索引
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y_noisy, np.arange(n_samples), test_size=test_size, random_state=random_state
    )

    # 计算训练集内部噪声位置
    train_noise_mask = np.isin(train_idx, noise_indices)
    train_noise_indices = np.where(train_noise_mask)[0]
    train_clean_indices = np.where(~train_noise_mask)[0]

    print(f"Training set noise samples = {len(train_noise_indices)}")

    return (X_train, X_test, y_train, y_test, train_noise_indices, train_clean_indices)
