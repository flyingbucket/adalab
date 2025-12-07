"""
使用TVTK/Mayavi进行3D可视化
将MNIST手写数字转换为3D高度图和其他创意可视化
"""

import numpy as np
from src.utils import prepare_data

# 检查是否安装了mayavi
try:
    from mayavi import mlab
    from tvtk.api import tvtk
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    print("警告: 未安装Mayavi/TVTK")
    print("安装命令: pip install mayavi")
    print("或: conda install -c conda-forge mayavi")


def visualize_digit_as_3d_surface(digit_image, title="手写数字3D高度图"):
    """
    将2D手写数字图像可视化为3D表面（高度图）
    
    Parameters
    ----------
    digit_image : ndarray, shape (28, 28)
        手写数字图像
    title : str
        图表标题
    """
    if not MAYAVI_AVAILABLE:
        print("需要安装Mayavi才能使用此功能")
        return
    
    # 创建网格坐标
    x, y = np.mgrid[0:28, 0:28]
    
    # 使用像素值作为z坐标（高度）
    z = digit_image
    
    # 创建3D可视化
    fig = mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
    
    # 绘制表面
    surf = mlab.surf(x, y, z, colormap='viridis', warp_scale='auto')
    
    # 添加颜色条
    mlab.colorbar(surf, title='像素值', orientation='vertical')
    
    # 设置视角
    mlab.view(azimuth=45, elevation=65, distance='auto')
    
    # 添加坐标轴
    mlab.axes(surf, color=(0, 0, 0), 
              xlabel='X', ylabel='Y', zlabel='像素强度')
    
    # 设置标题
    mlab.title(title, size=0.3, height=0.95, color=(0, 0, 0))
    
    mlab.show()


def visualize_multiple_digits_3d(X_data, y_data, n_samples=5):
    """
    在3D空间中展示多个手写数字
    
    Parameters
    ----------
    X_data : 特征数据
    y_data : 标签
    n_samples : 展示的样本数量
    """
    if not MAYAVI_AVAILABLE:
        print("需要安装Mayavi才能使用此功能")
        return
    
    # 选择不同类别的样本
    fig = mlab.figure(size=(1200, 800), bgcolor=(1, 1, 1))
    
    for i in range(min(n_samples, 10)):
        # 找到该类别的第一个样本
        idx = np.where(y_data == i)[0][0]
        digit = X_data[idx].reshape(28, 28)
        
        # 创建网格
        x, y = np.mgrid[0:28, 0:28]
        z = digit
        
        # 在不同位置绘制
        offset_x = (i % 5) * 35
        offset_y = (i // 5) * 35
        
        surf = mlab.surf(x + offset_x, y + offset_y, z * 2, 
                        colormap='jet', representation='surface')
        
        # 添加文本标签
        mlab.text3d(offset_x + 14, offset_y + 14, digit.max() * 2.5, 
                   str(i), scale=3, color=(0, 0, 0))
    
    mlab.view(azimuth=45, elevation=60, distance=400)
    mlab.title("MNIST数字0-9的3D展示", size=0.3, height=0.95, color=(0, 0, 0))
    
    mlab.show()


def visualize_feature_space_3d(X_data, y_data, n_samples=500):
    """
    在3D空间中可视化降维后的特征空间
    使用PCA降维到3D
    
    Parameters
    ----------
    X_data : 特征数据
    y_data : 标签
    n_samples : 使用的样本数量
    """
    if not MAYAVI_AVAILABLE:
        print("需要安装Mayavi才能使用此功能")
        return
    
    from sklearn.decomposition import PCA
    
    print("使用PCA降维到3D...")
    
    # 选择子集
    indices = np.random.choice(len(X_data), min(n_samples, len(X_data)), replace=False)
    X_subset = X_data[indices]
    y_subset = y_data[indices]
    
    # PCA降维到3D
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_subset)
    
    print(f"解释方差比: {pca.explained_variance_ratio_}")
    
    # 创建3D散点图
    fig = mlab.figure(size=(1000, 800), bgcolor=(1, 1, 1))
    
    # 为每个类别使用不同颜色
    colors = [
        (1, 0, 0),    # 0: 红色
        (0, 1, 0),    # 1: 绿色
        (0, 0, 1),    # 2: 蓝色
        (1, 1, 0),    # 3: 黄色
        (1, 0, 1),    # 4: 品红
        (0, 1, 1),    # 5: 青色
        (1, 0.5, 0),  # 6: 橙色
        (0.5, 0, 1),  # 7: 紫色
        (0.5, 0.5, 0),# 8: 橄榄绿
        (0, 0.5, 0.5),# 9: 青绿
    ]
    
    for digit in range(10):
        mask = y_subset == digit
        points = X_3d[mask]
        
        if len(points) > 0:
            mlab.points3d(
                points[:, 0], 
                points[:, 1], 
                points[:, 2],
                color=colors[digit],
                scale_factor=2.0,
                mode='sphere',
                opacity=0.6
            )
            
            # 添加图例（使用文本）
            center = points.mean(axis=0)
            mlab.text3d(center[0], center[1], center[2], 
                       str(digit), scale=3, color=colors[digit])
    
    # 添加坐标轴
    mlab.axes(color=(0, 0, 0), nb_labels=5)
    mlab.xlabel('PC1', object=None)
    mlab.ylabel('PC2', object=None)
    mlab.zlabel('PC3', object=None)
    
    mlab.title("MNIST特征空间3D可视化 (PCA)", 
              size=0.3, height=0.95, color=(0, 0, 0))
    
    mlab.show()


def visualize_weight_evolution_3d(weight_history, noise_indices, title="样本权重演变3D"):
    """
    在3D空间中可视化样本权重的演变过程
    
    Parameters
    ----------
    weight_history : list of arrays
        每轮的样本权重历史
    noise_indices : array
        噪声样本索引
    title : str
        标题
    """
    if not MAYAVI_AVAILABLE:
        print("需要安装Mayavi才能使用此功能")
        return
    
    n_rounds = len(weight_history)
    n_samples = min(100, len(weight_history[0]))  # 只显示前100个样本
    
    # 创建数据矩阵
    weight_matrix = np.array([w[:n_samples] for w in weight_history])
    
    # 创建网格
    x = np.arange(n_samples)  # 样本索引
    y = np.arange(n_rounds)   # 训练轮次
    X, Y = np.meshgrid(x, y)
    Z = weight_matrix
    
    # 创建3D可视化
    fig = mlab.figure(size=(1000, 800), bgcolor=(1, 1, 1))
    
    # 绘制表面
    surf = mlab.surf(X, Y, Z, colormap='hot', warp_scale='auto')
    
    # 标记噪声样本
    noise_in_range = noise_indices[noise_indices < n_samples]
    if len(noise_in_range) > 0:
        for idx in noise_in_range:
            # 在噪声样本位置画垂直线
            z_vals = weight_matrix[:, idx]
            mlab.plot3d(
                [idx] * n_rounds,
                y,
                z_vals,
                color=(1, 0, 0),
                tube_radius=0.3
            )
    
    mlab.colorbar(surf, title='权重', orientation='vertical')
    mlab.xlabel('样本索引')
    mlab.ylabel('训练轮次')
    mlab.zlabel('权重值')
    mlab.title(title, size=0.3, height=0.95, color=(0, 0, 0))
    
    mlab.view(azimuth=45, elevation=65, distance='auto')
    
    mlab.show()


def demo_3d_visualization():
    """演示各种3D可视化"""
    
    if not MAYAVI_AVAILABLE:
        print("\n" + "=" * 60)
        print("Mayavi/TVTK未安装")
        print("=" * 60)
        print("\n安装方法:")
        print("  pip install mayavi")
        print("  或")
        print("  conda install -c conda-forge mayavi")
        print("\n注意: Mayavi需要Qt作为GUI后端")
        print("  pip install PyQt5")
        print("=" * 60)
        return
    
    print("\n" + "█" * 60)
    print("MNIST 3D可视化演示 (使用Mayavi/TVTK)".center(56))
    print("█" * 60)
    
    # 准备数据
    print("\n加载MNIST数据...")
    X_train, X_test, y_train, y_test, noise_idx, clean_idx = prepare_data(
        noise_ratio=0.05
    )
    
    print(f"数据加载完成: {len(X_train)} 训练样本")
    
    # 菜单
    print("\n选择可视化类型:")
    print("1. 单个数字的3D高度图")
    print("2. 多个数字的3D展示")
    print("3. 特征空间3D可视化 (PCA)")
    print("4. 全部演示")
    
    choice = input("\n请输入选择 (1-4, 默认1): ").strip() or "1"
    
    if choice == "1" or choice == "4":
        print("\n生成单个数字的3D高度图...")
        # 选择一个清晰的数字
        idx = np.where(y_train == 3)[0][0]
        digit = X_train[idx].reshape(28, 28)
        visualize_digit_as_3d_surface(digit, title=f"数字 {y_train[idx]} 的3D高度图")
    
    if choice == "2" or choice == "4":
        print("\n生成多个数字的3D展示...")
        visualize_multiple_digits_3d(X_train, y_train, n_samples=10)
    
    if choice == "3" or choice == "4":
        print("\n生成特征空间3D可视化...")
        visualize_feature_space_3d(X_train, y_train, n_samples=500)
    
    print("\n✓ 3D可视化完成！")
    print("\n说明:")
    print("  - 使用鼠标拖动可以旋转视角")
    print("  - 滚轮可以缩放")
    print("  - 关闭窗口继续下一个可视化")


def create_custom_visualization():
    """
    自定义3D可视化示例
    展示如何使用TVTK底层API
    """
    if not MAYAVI_AVAILABLE:
        print("需要安装Mayavi/TVTK")
        return
    
    from mayavi import mlab
    from tvtk.api import tvtk
    
    print("\n创建自定义TVTK可视化...")
    
    # 准备数据
    X_train, X_test, y_train, y_test, _, _ = prepare_data(noise_ratio=0)
    
    # 选择一个数字
    idx = np.where(y_train == 8)[0][0]
    digit = X_train[idx].reshape(28, 28)
    
    # 使用TVTK API创建结构化网格
    x, y = np.mgrid[0:28, 0:28]
    z = digit
    
    # 创建图形
    fig = mlab.figure(size=(800, 600), bgcolor=(0.9, 0.9, 0.9))
    
    # 使用TVTK创建多个层次的表面
    for level in [0.3, 0.5, 0.7]:
        # 创建等值面
        threshold_value = digit.max() * level
        mask = digit >= threshold_value
        
        if mask.any():
            contour = mlab.contour3d(
                digit,
                contours=[threshold_value],
                opacity=0.3,
                colormap='cool'
            )
    
    # 添加原始表面
    surf = mlab.surf(x, y, z, colormap='viridis', opacity=0.8)
    
    mlab.title("使用TVTK的多层次可视化", size=0.3, height=0.95)
    mlab.show()


if __name__ == "__main__":
    # 运行演示
    demo_3d_visualization()
    
    # 如果想要自定义可视化，取消下面的注释
    # create_custom_visualization()


