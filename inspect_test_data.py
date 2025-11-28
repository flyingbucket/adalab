import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def inspect_images(folder="test_data"):
    print(f"Scanning folder: {folder}\n")

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = Image.open(path)

        # 基本属性
        width, height = img.size
        mode = img.mode

        # 灰度化后查看像素统计
        gray = img.convert("L")
        arr = np.array(gray)

        mean_val = arr.mean()
        min_val = arr.min()
        max_val = arr.max()

        # 判断大致背景：>128 白底，<128 黑底
        background = "白底" if mean_val > 128 else "黑底"

        print(f"文件名: {filename}")
        print(f"  尺寸: {width} x {height}")
        print(f"  模式: {mode}")
        print(f"  灰度均值: {mean_val:.2f} (判定为 {background})")
        print(f"  最小像素: {min_val}, 最大像素: {max_val}")
        print("-" * 40)


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


if __name__ == "__main__":
    import os

    folder = "test_data"
    images = []
    names = []

    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)

        x, vis = preprocess_for_mnist(path)
        images.append(vis)
        names.append(filename)

    # 拼图显示
    fig, axes = plt.subplots(2, 5, figsize=(10, 2 * 2))

    for ax, img, name in zip(axes.flatten(), images, names):
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
