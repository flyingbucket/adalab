"""
对比不同可视化工具的效果
展示Matplotlib vs TVTK/Mayavi的适用场景
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from src.utils import prepare_data

# 检查Mayavi是否可用
try:
    from mayavi import mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    print("注意: Mayavi未安装，将只展示Matplotlib可视化")
    print("安装命令: conda install -c conda-forge mayavi")


def demo_matplotlib_3d():
    """使用Matplotlib创建3D可视化"""
    
    print("\n" + "=" * 60)
    print("方法1: Matplotlib 3D可视化 (推荐)")
    print("=" * 60)
    
    # 准备数据
    print("加载数据...")
    X_train, _, y_train, _, _, _ = prepare_data(noise_ratio=0)
    
    # PCA降维到3D
    print("PCA降维到3D...")
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_train[:1000])
    
    # 创建3D散点图
    fig = plt.figure(figsize=(12, 10))
    
    # 子图1: 3D散点图
    ax1 = fig.add_subplot(221, projection='3d')
    
    colors = plt.cm.tab10(range(10))
    for digit in range(10):
        mask = y_train[:1000] == digit
        ax1.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                   c=[colors[digit]], label=str(digit), s=20, alpha=0.6)
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('特征空间3D可视化 (Matplotlib)', fontsize=14)
    ax1.legend(ncol=2, fontsize=8)
    
    # 子图2: 单个数字的3D表面
    ax2 = fig.add_subplot(222, projection='3d')
    
    idx = np.where(y_train == 8)[0][0]
    digit = X_train[idx].reshape(28, 28)
    x, y = np.meshgrid(range(28), range(28))
    
    surf = ax2.plot_surface(x, y, digit, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('像素值')
    ax2.set_title('数字"8"的3D高度图', fontsize=14)
    fig.colorbar(surf, ax=ax2, shrink=0.5)
    
    # 子图3: 2D热力图（对比）
    ax3 = fig.add_subplot(223)
    im = ax3.imshow(digit, cmap='viridis')
    ax3.set_title('2D热力图（更清晰！）', fontsize=14)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3)
    
    # 子图4: 原始图像
    ax4 = fig.add_subplot(224)
    ax4.imshow(digit, cmap='gray')
    ax4.set_title('原始图像（最直观！）', fontsize=14)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/matplotlib_3d_demo.png', dpi=150, bbox_inches='tight')
    print(f"图表已保存到: results/matplotlib_3d_demo.png")
    plt.show()
    
    print("\n优点:")
    print("  ✅ 安装简单")
    print("  ✅ 跨平台兼容")
    print("  ✅ 文档丰富")
    print("  ✅ 易于调试")
    print("  ✅ 可以保存为图片")


def demo_mayavi_3d():
    """使用Mayavi创建3D可视化"""
    
    if not MAYAVI_AVAILABLE:
        print("\n" + "=" * 60)
        print("方法2: Mayavi 3D可视化 (需要安装)")
        print("=" * 60)
        print("\nMayavi未安装，跳过此演示")
        print("\n安装方法:")
        print("  conda install -c conda-forge mayavi")
        print("  pip install mayavi PyQt5")
        return
    
    print("\n" + "=" * 60)
    print("方法2: Mayavi 3D可视化")
    print("=" * 60)
    
    # 准备数据
    print("加载数据...")
    X_train, _, y_train, _, _, _ = prepare_data(noise_ratio=0)
    
    # PCA降维到3D
    print("PCA降维到3D...")
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_train[:1000])
    
    # 创建Mayavi可视化
    print("创建Mayavi可视化（会打开新窗口）...")
    
    fig = mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
    
    colors = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
        (0, 1, 1), (1, 0.5, 0), (0.5, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5)
    ]
    
    for digit in range(10):
        mask = y_train[:1000] == digit
        points = X_3d[mask]
        
        if len(points) > 0:
            mlab.points3d(
                points[:, 0], points[:, 1], points[:, 2],
                color=colors[digit],
                scale_factor=2.0,
                mode='sphere',
                opacity=0.6
            )
    
    mlab.title("特征空间3D可视化 (Mayavi)", size=0.3, height=0.95, color=(0, 0, 0))
    mlab.show()
    
    print("\n优点:")
    print("  ✅ 交互性强")
    print("  ✅ 渲染质量高")
    print("  ✅ 适合复杂3D数据")
    
    print("\n缺点:")
    print("  ⚠️ 安装复杂")
    print("  ⚠️ 依赖Qt")
    print("  ⚠️ 学习曲线陡")


def compare_visualization_approaches():
    """对比总结"""
    
    print("\n" + "█" * 60)
    print("可视化工具对比总结".center(56))
    print("█" * 60)
    
    print("\n📊 对于本项目（MNIST + AdaBoost）:")
    print("-" * 60)
    
    print("\n推荐度排序:")
    print("  1. ⭐⭐⭐⭐⭐ Matplotlib 2D (当前使用)")
    print("     - 学习曲线、混淆矩阵、热力图")
    print("     - 最适合统计分析和分类任务")
    print("     - 简单、可靠、跨平台")
    
    print("\n  2. ⭐⭐⭐⭐ Matplotlib 3D (可选)")
    print("     - 如果需要3D散点图")
    print("     - 简单易用，不需要额外安装")
    print("     - 适合展示降维后的特征空间")
    
    print("\n  3. ⭐⭐⭐ Plotly (如需交互)")
    print("     - 轻量级交互式可视化")
    print("     - 在浏览器中查看")
    print("     - pip install plotly")
    
    print("\n  4. ⭐⭐ Mayavi/TVTK (学习用)")
    print("     - 适合学习3D可视化技术")
    print("     - 对本项目帮助有限")
    print("     - 安装和使用较复杂")
    
    print("\n💡 建议:")
    print("-" * 60)
    print("  ✓ 继续使用Matplotlib 2D - 已经很好了！")
    print("  ✓ 如果想要3D - 用Matplotlib 3D就够了")
    print("  ✓ 如果想学习TVTK - 当作课外学习项目")
    print("  ✗ 不要为了3D而3D - 2D通常更清晰")
    
    print("\n📈 何时真正需要TVTK:")
    print("-" * 60)
    print("  • 医学影像（CT、MRI体数据）")
    print("  • 流体动力学（速度场、压力场）")
    print("  • 地质数据（地形、地震）")
    print("  • 分子结构（蛋白质、化学键）")
    print("  • 有限元分析（应力、变形）")
    
    print("\n  ❌ 不适合: MNIST分类任务")
    
    print("\n" + "=" * 60)


def main():
    """主函数"""
    
    print("\n" + "█" * 60)
    print("可视化工具对比演示".center(56))
    print("█" * 60)
    
    import os
    os.makedirs('results', exist_ok=True)
    
    # 演示Matplotlib
    demo_matplotlib_3d()
    
    # 演示Mayavi（如果可用）
    if MAYAVI_AVAILABLE:
        response = input("\n是否继续Mayavi演示? (y/n, 默认n): ").strip().lower()
        if response == 'y':
            demo_mayavi_3d()
    
    # 对比总结
    compare_visualization_approaches()
    
    print("\n✓ 演示完成！")
    print("\n更多信息请查看:")
    print("  • docs/3d_visualization_guide.md")
    print("  • visualize_3d_tvtk.py")


if __name__ == "__main__":
    main()




