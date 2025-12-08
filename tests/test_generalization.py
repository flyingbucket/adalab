"""
泛化能力测试：在标准MNIST上训练，在带扰动的MNIST上测试
测试模型对视觉扰动的鲁棒性
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mplfonts.bin.cli import init
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm

from src.utils import prepare_data
from src.robust_adaboost import create_robust_adaboost, RobustAdaBoost

# 初始化中文字体
init()
matplotlib.rcParams["font.family"] = "Source Han Sans CN"
matplotlib.rcParams["axes.unicode_minus"] = False


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
            blurred = uniform_filter(img, size=kernel_size, mode='constant')
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
            rotated = rotate(img, angle, reshape=False, mode='constant', cval=0)
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
        if perturbation_type == 'brightness':
            return self.add_brightness_shift(X, **kwargs)
        elif perturbation_type == 'gaussian_noise':
            return self.add_gaussian_noise(X, **kwargs)
        elif perturbation_type == 'salt_pepper':
            return self.add_salt_pepper_noise(X, **kwargs)
        elif perturbation_type == 'blur':
            return self.add_blur(X, **kwargs)
        elif perturbation_type == 'contrast':
            return self.adjust_contrast(X, **kwargs)
        elif perturbation_type == 'rotation':
            return self.rotate_slight(X, **kwargs)
        else:
            raise ValueError(f"未知扰动类型: {perturbation_type}")


def visualize_perturbations(X_original, perturber, save_path=None):
    """
    可视化不同类型的扰动效果
    
    Parameters
    ----------
    X_original : array
        原始数据
    perturber : MNISTPerturber
        扰动器
    save_path : str
        保存路径
    """
    # 选择一些样本
    n_samples = 5
    indices = np.random.choice(len(X_original), n_samples, replace=False)
    samples = X_original[indices]
    
    # 定义扰动类型
    perturbations = [
        ('原始', None, {}),
        ('亮度偏移', 'brightness', {'shift_range': 0.3}),
        ('高斯噪声', 'gaussian_noise', {'noise_std': 0.15}),
        ('椒盐噪声', 'salt_pepper', {'amount': 0.05}),
        ('模糊', 'blur', {'kernel_size': 3}),
        ('对比度', 'contrast', {'factor_range': (0.5, 1.5)}),
        ('旋转', 'rotation', {'angle_range': 15}),
    ]
    
    # 创建图形
    fig, axes = plt.subplots(len(perturbations), n_samples, 
                            figsize=(n_samples*2, len(perturbations)*2))
    
    for i, (name, ptype, params) in enumerate(perturbations):
        if ptype is None:
            perturbed = samples
        else:
            perturbed = perturber.apply_perturbation(samples, ptype, **params)
        
        for j in range(n_samples):
            ax = axes[i, j]
            ax.imshow(perturbed[j].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(name, fontsize=12, rotation=0, 
                            ha='right', va='center', labelpad=50)
    
    plt.suptitle('MNIST数据扰动类型展示', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"扰动示例已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_generalization(clf, X_test, y_test, perturber, model_name="模型"):
    """
    测试模型在不同扰动下的泛化能力
    
    Parameters
    ----------
    clf : 分类器
        训练好的模型
    X_test : array
        测试数据
    y_test : array
        测试标签
    perturber : MNISTPerturber
        扰动器
    model_name : str
        模型名称
    
    Returns
    -------
    results : dict
        测试结果
    """
    print(f"\n{'='*60}")
    print(f"测试 {model_name} 的泛化能力")
    print(f"{'='*60}")
    
    # 定义扰动配置
    perturbation_configs = [
        ('原始（无扰动）', None, {}),
        ('亮度±10%', 'brightness', {'shift_range': 0.1}),
        ('亮度±20%', 'brightness', {'shift_range': 0.2}),
        ('亮度±30%', 'brightness', {'shift_range': 0.3}),
        ('高斯噪声σ=0.05', 'gaussian_noise', {'noise_std': 0.05}),
        ('高斯噪声σ=0.10', 'gaussian_noise', {'noise_std': 0.10}),
        ('高斯噪声σ=0.15', 'gaussian_noise', {'noise_std': 0.15}),
        ('椒盐噪声2%', 'salt_pepper', {'amount': 0.02}),
        ('椒盐噪声5%', 'salt_pepper', {'amount': 0.05}),
        ('椒盐噪声10%', 'salt_pepper', {'amount': 0.10}),
        ('模糊3x3', 'blur', {'kernel_size': 3}),
        ('模糊5x5', 'blur', {'kernel_size': 5}),
        ('对比度±30%', 'contrast', {'factor_range': (0.7, 1.3)}),
        ('对比度±50%', 'contrast', {'factor_range': (0.5, 1.5)}),
        ('旋转±5°', 'rotation', {'angle_range': 5}),
        ('旋转±10°', 'rotation', {'angle_range': 10}),
        ('旋转±15°', 'rotation', {'angle_range': 15}),
    ]
    
    results = {
        'names': [],
        'accuracies': [],
        'accuracy_drops': [],
    }
    
    # 测试每种扰动
    baseline_acc = None
    
    for name, ptype, params in tqdm(perturbation_configs, desc="测试扰动"):
        # 应用扰动
        if ptype is None:
            X_perturbed = X_test
        else:
            X_perturbed = perturber.apply_perturbation(X_test, ptype, **params)
        
        # 预测
        y_pred = clf.predict(X_perturbed)
        acc = accuracy_score(y_test, y_pred)
        
        # 记录基线准确率
        if baseline_acc is None:
            baseline_acc = acc
        
        # 计算准确率下降
        acc_drop = baseline_acc - acc
        
        results['names'].append(name)
        results['accuracies'].append(acc)
        results['accuracy_drops'].append(acc_drop)
        
        print(f"{name:20s}: {acc:.4f} (下降 {acc_drop:.4f})")
    
    return results


def plot_generalization_results(results_dict, save_path=None):
    """
    可视化泛化能力测试结果
    
    Parameters
    ----------
    results_dict : dict
        多个模型的结果字典 {model_name: results}
    save_path : str
        保存路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子图1: 准确率对比
    ax1 = axes[0]
    x = np.arange(len(results_dict[list(results_dict.keys())[0]]['names']))
    width = 0.35
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        offset = width * (i - len(results_dict)/2 + 0.5)
        ax1.bar(x + offset, results['accuracies'], width, 
               label=model_name, alpha=0.8)
    
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.set_title('不同扰动下的模型准确率', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_dict[list(results_dict.keys())[0]]['names'], 
                       rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # 子图2: 准确率下降
    ax2 = axes[1]
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        offset = width * (i - len(results_dict)/2 + 0.5)
        bars = ax2.bar(x + offset, results['accuracy_drops'], width,
                      label=model_name, alpha=0.8)
        
        # 为负值（性能提升）和正值（性能下降）使用不同颜色
        for bar, drop in zip(bars, results['accuracy_drops']):
            if drop > 0:
                bar.set_color('red')
                bar.set_alpha(0.6)
            else:
                bar.set_color('green')
                bar.set_alpha(0.6)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('准确率下降', fontsize=12)
    ax2.set_xlabel('扰动类型', fontsize=12)
    ax2.set_title('准确率下降幅度（正值=性能下降，负值=性能提升）', fontsize=14, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_dict[list(results_dict.keys())[0]]['names'],
                       rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"泛化能力测试结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(results_dict):
    """
    打印泛化能力测试总结
    
    Parameters
    ----------
    results_dict : dict
        多个模型的结果
    """
    print("\n" + "█" * 60)
    print("泛化能力测试总结".center(56))
    print("█" * 60)
    
    for model_name, results in results_dict.items():
        print(f"\n{model_name}:")
        print("-" * 60)
        
        baseline_acc = results['accuracies'][0]
        avg_acc = np.mean(results['accuracies'][1:])  # 排除基线
        avg_drop = np.mean(results['accuracy_drops'][1:])
        max_drop = np.max(results['accuracy_drops'][1:])
        
        # 找出最难的扰动
        worst_idx = np.argmax(results['accuracy_drops'][1:]) + 1
        worst_name = results['names'][worst_idx]
        worst_acc = results['accuracies'][worst_idx]
        
        print(f"  基线准确率（无扰动）: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        print(f"  平均准确率（有扰动）: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        print(f"  平均准确率下降: {avg_drop:.4f} ({avg_drop*100:.2f}%)")
        print(f"  最大准确率下降: {max_drop:.4f} ({max_drop*100:.2f}%)")
        print(f"  最难扰动: {worst_name} (准确率: {worst_acc:.4f})")
    
    # 对比分析
    if len(results_dict) > 1:
        print("\n" + "=" * 60)
        print("模型对比".center(56))
        print("=" * 60)
        
        model_names = list(results_dict.keys())
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                res1 = results_dict[name1]
                res2 = results_dict[name2]
                
                # 计算平均准确率差异
                avg_acc1 = np.mean(res1['accuracies'][1:])
                avg_acc2 = np.mean(res2['accuracies'][1:])
                
                print(f"\n{name1} vs {name2}:")
                print(f"  平均准确率差异: {avg_acc1 - avg_acc2:+.4f}")
                
                if avg_acc1 > avg_acc2:
                    print(f"  ✓ {name1} 在扰动数据上表现更好")
                elif avg_acc2 > avg_acc1:
                    print(f"  ✓ {name2} 在扰动数据上表现更好")
                else:
                    print(f"  = 两者表现相当")
    
    print("\n" + "=" * 60)


def main():
    """主函数"""
    
    print("\n" + "█" * 60)
    print("MNIST泛化能力测试".center(56))
    print("在标准MNIST上训练，在带扰动的MNIST上测试".center(56))
    print("█" * 60)
    
    import os
    os.makedirs('results', exist_ok=True)
    
    # ========== 1. 准备数据 ==========
    print("\n步骤1: 准备数据")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test, _, _ = prepare_data(noise_ratio=0)
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # ========== 2. 创建扰动器 ==========
    print("\n步骤2: 创建数据扰动器")
    print("-" * 60)
    
    perturber = MNISTPerturber(random_state=42)
    
    # 可视化扰动效果
    print("生成扰动示例可视化...")
    visualize_perturbations(X_test, perturber, 
                           save_path='results/perturbation_examples.png')
    
    # ========== 3. 训练模型 ==========
    print("\n步骤3: 训练模型")
    print("-" * 60)
    
    models = {}
    
    # 模型1: 标准AdaBoost
    print("\n训练标准AdaBoost...")
    clf_standard = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),  # 增加树深度以适应高维数据
        n_estimators=100,  # 增加弱分类器数量
        learning_rate=1.0,
        random_state=42
    )
    clf_standard.fit(X_train, y_train)
    models['标准AdaBoost'] = clf_standard
    print(f"训练完成，测试准确率: {clf_standard.score(X_test, y_test):.4f}")
    
    # 模型2: 鲁棒AdaBoost（禁用早停以适应多分类问题）
    print("\n训练鲁棒AdaBoost...")
    clf_robust = RobustAdaBoost(
        base_estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        learning_rate=1.0,
        random_state=42,
        weight_clip_percentile=95,
        use_early_stopping=False,  # 禁用早停
        use_sample_weight_smoothing=True,  # 使用权重平滑
        smoothing_factor=0.5
    )
    clf_robust.fit(X_train, y_train)
    models['鲁棒AdaBoost'] = clf_robust
    print(f"训练完成，测试准确率: {clf_robust.score(X_test, y_test):.4f}")
    
    # ========== 4. 测试泛化能力 ==========
    print("\n步骤4: 测试泛化能力")
    print("-" * 60)
    
    results_dict = {}
    
    for model_name, clf in models.items():
        results = test_generalization(clf, X_test, y_test, perturber, model_name)
        results_dict[model_name] = results
    
    # ========== 5. 可视化结果 ==========
    print("\n步骤5: 生成可视化")
    print("-" * 60)
    
    plot_generalization_results(results_dict, 
                               save_path='results/generalization_test.png')
    
    # ========== 6. 打印总结 ==========
    print_summary(results_dict)
    
    print("\n✓ 泛化能力测试完成！")
    print("\n生成的文件:")
    print("  • results/perturbation_examples.png - 扰动示例")
    print("  • results/generalization_test.png - 泛化测试结果")


if __name__ == "__main__":
    main()

