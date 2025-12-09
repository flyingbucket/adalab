import os
import warnings

import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skimage.feature import hog


def preprocess_for_mnist(path):
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
    def __init__(
        self,
        noise_config={},
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
        self.noise_config = noise_config
        self.test_size = test_size
        self.use_feature = use_feature
        self.random_state = random_state

        # HOG settings
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block

        # Hu settings
        self.hu_log_scale = hu_log_scale

        # 保存噪声索引
        self.train_noise_indices = None
        self.train_clean_indices = None

        self.perturber = MNISTPerturber(random_state)

    def download_mnist(self):
        print("[Data] Downloading MNIST...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        y = y.astype(np.int64)
        X = X / 255.0
        self.X_raw = X
        self.y_raw = y

    def inject_noise(self):
        """Add noise to training data
        set self.X_train, self.y_train, self.noise_indices
        """
        print("[Data] Applying perturbations...")

        X = self.X_train_raw.copy()
        y = self.y_train_raw.copy()
        pert = self.perturber

        # 若无 noise_config，直接返回
        if not hasattr(self, "noise_config") or len(self.noise_config) == 0:
            print("[Data] No perturbations applied.")
            self.X_train = X
            self.y_train = y
            self.noise_indices = np.array([], dtype=int)
            return

        # 先确定 noise_indices（标签噪声 > 像素噪声触发）
        if "label_flip" in self.noise_config:
            ratio = self.noise_config["label_flip"].get("ratio", 0.0)
            y, noise_indices = pert.flip_labels(y, noise_ratio=ratio)
            print(f"[Data] Label flip: {len(noise_indices)} indices selected")
        else:
            # 未定义标签噪声，则以任意噪声条目作为依据确定噪声比例
            key = list(self.noise_config.keys())[0]
            ratio = self.noise_config[key].get("ratio", 0.1)  # 默认10%
            n_samples = len(X)
            n_noisy = int(n_samples * ratio)
            noise_indices = pert.rng.choice(n_samples, n_noisy, replace=False)
            print(f"[Data] Random selection: {n_noisy} indices selected")

        self.noise_indices = np.array(noise_indices)

        # 将所有像素噪声叠加到相同 noise_indices 样本上
        for noise_type, params in self.noise_config.items():
            if noise_type == "label_flip":
                continue  # 已处理

            subset = X[self.noise_indices]  # 只处理噪声样本

            if noise_type == "gaussian":
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

    def split(self):
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            self.X_raw,
            self.y_raw,  # 全干净标签
            np.arange(len(self.y_raw)),
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # 保存划分索引，用于 inject_noise 后映射
        self.train_idx = train_idx
        self.test_idx = test_idx

        self.X_train_raw = X_train
        self.X_test = X_test
        self.y_train_raw = y_train
        self.y_test = y_test

        print(f"[Data] Split done: Train={len(X_train)}, Test={len(X_test)}")

    # 特征提取

    def extract_hog(self, X):
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
        X_reshaped = X.reshape(-1, 28, 28)
        feats = []
        for img in X_reshaped:
            moments = cv2.HuMoments(cv2.moments(img)).flatten()
            if self.hu_log_scale:
                moments = -np.sign(moments) * np.log10(np.abs(moments))
            feats.append(moments)
        return np.array(feats)

    def apply_feature(self):
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

    def prepare(self):
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
        """
        处理课程老师提供的真实拍照数字数据。
        不做 train_split，不加噪声，只做预处理 + 特征提取。

        参数
        ----
        folder : str
            文件夹路径，文件名必须为 0.png 1.png ... 这样的格式

        返回
        ----
        X : ndarray
            特征矩阵（维度与训练特征保持一致）
        y : ndarray
            标签（来自文件名）
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
    """MNIST数据扰动器"""

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

    def flip_labels(self, y, noise_ratio=0.0, num_classes=10):
        """
        随机翻转标签噪声（对分类标签做对抗扰动）

        Parameters
        ----------
        y : array
            标签数组（真实标签）
        noise_ratio : float
            噪声比例
        num_classes : int
            分类数（默认为 MNIST 的 10 类）

        Returns
        -------
        y_noisy : ndarray
            添加噪声后的标签
        noise_indices : ndarray
            被修改标签的索引
        """
        y_noisy = y.copy()
        n_samples = len(y)

        if noise_ratio <= 0:
            return y_noisy, np.array([], dtype=int)

        n_noisy = int(n_samples * noise_ratio)
        noise_indices = self.rng.choice(n_samples, n_noisy, replace=False)

        y_noisy[noise_indices] = self.rng.randint(0, num_classes, size=n_noisy)

        return y_noisy, noise_indices

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
