"""
Generalization Test: Train on standard MNIST, test on perturbed MNIST
Test model robustness against visual perturbations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm

from src.utils import DataPreparation
from src.robust_adaboost import create_robust_adaboost, RobustAdaBoost


class MNISTPerturber:
    """MNIST data perturber"""
    
    def __init__(self, random_state=42):
        """
        Initialize perturber
        
        Parameters
        ----------
        random_state : int
            Random seed
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def add_brightness_shift(self, X, shift_range=0.3):
        """
        Add brightness shift
        
        Parameters
        ----------
        X : array
            Original data [0, 1]
        shift_range : float
            Brightness shift range [-shift_range, shift_range]
        """
        shift = self.rng.uniform(-shift_range, shift_range, size=len(X))
        X_perturbed = X + shift[:, np.newaxis]
        return np.clip(X_perturbed, 0, 1)
    
    def add_gaussian_noise(self, X, noise_std=0.1):
        """
        Add Gaussian noise
        
        Parameters
        ----------
        X : array
            Original data
        noise_std : float
            Noise standard deviation
        """
        noise = self.rng.normal(0, noise_std, X.shape)
        X_perturbed = X + noise
        return np.clip(X_perturbed, 0, 1)
    
    def add_salt_pepper_noise(self, X, amount=0.05):
        """
        Add salt and pepper noise
        
        Parameters
        ----------
        X : array
            Original data
        amount : float
            Noise ratio
        """
        X_perturbed = X.copy()
        
        # Salt noise (white dots)
        n_salt = int(amount * X.size * 0.5)
        coords = [self.rng.randint(0, i, n_salt) for i in X.shape]
        X_perturbed[tuple(coords)] = 1
        
        # Pepper noise (black dots)
        n_pepper = int(amount * X.size * 0.5)
        coords = [self.rng.randint(0, i, n_pepper) for i in X.shape]
        X_perturbed[tuple(coords)] = 0
        
        return X_perturbed
    
    def add_blur(self, X, kernel_size=3):
        """
        Add blur effect (simple average filter)
        
        Parameters
        ----------
        X : array
            Original data
        kernel_size : int
            Blur kernel size
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
        Adjust contrast
        
        Parameters
        ----------
        X : array
            Original data
        factor_range : tuple
            Contrast factor range
        """
        factors = self.rng.uniform(factor_range[0], factor_range[1], size=len(X))
        
        X_perturbed = np.zeros_like(X)
        for i in range(len(X)):
            mean = X[i].mean()
            X_perturbed[i] = mean + factors[i] * (X[i] - mean)
        
        return np.clip(X_perturbed, 0, 1)
    
    def rotate_slight(self, X, angle_range=15):
        """
        Slight rotation
        
        Parameters
        ----------
        X : array
            Original data
        angle_range : float
            Rotation angle range (degrees)
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
        Apply specified perturbation type
        
        Parameters
        ----------
        X : array
            Original data
        perturbation_type : str
            Perturbation type
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
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def visualize_perturbations(X_original, perturber, save_path=None):
    """
    Visualize different perturbation effects
    
    Parameters
    ----------
    X_original : array
        Original data
    perturber : MNISTPerturber
        Perturber instance
    save_path : str
        Save path
    """
    # Select some samples
    n_samples = 5
    indices = np.random.choice(len(X_original), n_samples, replace=False)
    samples = X_original[indices]
    
    # Define perturbation types
    perturbations = [
        ('Original', None, {}),
        ('Brightness', 'brightness', {'shift_range': 0.3}),
        ('Gaussian Noise', 'gaussian_noise', {'noise_std': 0.15}),
        ('Salt & Pepper', 'salt_pepper', {'amount': 0.05}),
        ('Blur', 'blur', {'kernel_size': 3}),
        ('Contrast', 'contrast', {'factor_range': (0.5, 1.5)}),
        ('Rotation', 'rotation', {'angle_range': 15}),
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
    
    plt.suptitle('MNIST Data Perturbation Examples', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Perturbation examples saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_generalization(clf, X_test, y_test, perturber, model_name="Model"):
    """
    Test model generalization under different perturbations
    
    Parameters
    ----------
    clf : classifier
        Trained model
    X_test : array
        Test data
    y_test : array
        Test labels
    perturber : MNISTPerturber
        Perturber instance
    model_name : str
        Model name
    
    Returns
    -------
    results : dict
        Test results
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name} Generalization")
    print(f"{'='*60}")
    
    # Define perturbation configurations
    perturbation_configs = [
        ('Original (No Perturbation)', None, {}),
        ('Brightness ±10%', 'brightness', {'shift_range': 0.1}),
        ('Brightness ±20%', 'brightness', {'shift_range': 0.2}),
        ('Brightness ±30%', 'brightness', {'shift_range': 0.3}),
        ('Gaussian Noise σ=0.05', 'gaussian_noise', {'noise_std': 0.05}),
        ('Gaussian Noise σ=0.10', 'gaussian_noise', {'noise_std': 0.10}),
        ('Gaussian Noise σ=0.15', 'gaussian_noise', {'noise_std': 0.15}),
        ('Salt & Pepper 2%', 'salt_pepper', {'amount': 0.02}),
        ('Salt & Pepper 5%', 'salt_pepper', {'amount': 0.05}),
        ('Salt & Pepper 10%', 'salt_pepper', {'amount': 0.10}),
        ('Blur 3x3', 'blur', {'kernel_size': 3}),
        ('Blur 5x5', 'blur', {'kernel_size': 5}),
        ('Contrast ±30%', 'contrast', {'factor_range': (0.7, 1.3)}),
        ('Contrast ±50%', 'contrast', {'factor_range': (0.5, 1.5)}),
        ('Rotation ±5°', 'rotation', {'angle_range': 5}),
        ('Rotation ±10°', 'rotation', {'angle_range': 10}),
        ('Rotation ±15°', 'rotation', {'angle_range': 15}),
    ]
    
    results = {
        'names': [],
        'accuracies': [],
        'accuracy_drops': [],
    }
    
    # Test each perturbation
    baseline_acc = None
    
    for name, ptype, params in tqdm(perturbation_configs, desc="Testing perturbations"):
        # Apply perturbation
        if ptype is None:
            X_perturbed = X_test
        else:
            X_perturbed = perturber.apply_perturbation(X_test, ptype, **params)
        
        # Predict
        y_pred = clf.predict(X_perturbed)
        acc = accuracy_score(y_test, y_pred)
        
        # Record baseline accuracy
        if baseline_acc is None:
            baseline_acc = acc
        
        # Calculate accuracy drop
        acc_drop = baseline_acc - acc
        
        results['names'].append(name)
        results['accuracies'].append(acc)
        results['accuracy_drops'].append(acc_drop)
        
        print(f"{name:30s}: {acc:.4f} (drop {acc_drop:.4f})")
    
    return results


def plot_generalization_results(results_dict, save_path=None):
    """
    Visualize generalization test results
    
    Parameters
    ----------
    results_dict : dict
        Results dictionary for multiple models {model_name: results}
    save_path : str
        Save path
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Accuracy comparison
    ax1 = axes[0]
    x = np.arange(len(results_dict[list(results_dict.keys())[0]]['names']))
    width = 0.35
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        offset = width * (i - len(results_dict)/2 + 0.5)
        ax1.bar(x + offset, results['accuracies'], width, 
               label=model_name, alpha=0.8)
    
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Under Different Perturbations', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_dict[list(results_dict.keys())[0]]['names'], 
                       rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Subplot 2: Accuracy drop
    ax2 = axes[1]
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        offset = width * (i - len(results_dict)/2 + 0.5)
        bars = ax2.bar(x + offset, results['accuracy_drops'], width,
                      label=model_name, alpha=0.8)
        
        # Use different colors for negative (improvement) and positive (degradation) values
        for bar, drop in zip(bars, results['accuracy_drops']):
            if drop > 0:
                bar.set_color('red')
                bar.set_alpha(0.6)
            else:
                bar.set_color('green')
                bar.set_alpha(0.6)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Accuracy Drop', fontsize=12)
    ax2.set_xlabel('Perturbation Type', fontsize=12)
    ax2.set_title('Accuracy Drop Magnitude (Positive=Degradation, Negative=Improvement)', fontsize=14, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_dict[list(results_dict.keys())[0]]['names'],
                       rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Generalization test results saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(results_dict):
    """
    Print generalization test summary
    
    Parameters
    ----------
    results_dict : dict
        Results for multiple models
    """
    print("\n" + "=" * 60)
    print("Generalization Test Summary".center(60))
    print("=" * 60)
    
    for model_name, results in results_dict.items():
        print(f"\n{model_name}:")
        print("-" * 60)
        
        baseline_acc = results['accuracies'][0]
        avg_acc = np.mean(results['accuracies'][1:])  # Exclude baseline
        avg_drop = np.mean(results['accuracy_drops'][1:])
        max_drop = np.max(results['accuracy_drops'][1:])
        
        # Find the hardest perturbation
        worst_idx = np.argmax(results['accuracy_drops'][1:]) + 1
        worst_name = results['names'][worst_idx]
        worst_acc = results['accuracies'][worst_idx]
        
        print(f"  Baseline accuracy (no perturbation): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        print(f"  Average accuracy (with perturbation): {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        print(f"  Average accuracy drop: {avg_drop:.4f} ({avg_drop*100:.2f}%)")
        print(f"  Maximum accuracy drop: {max_drop:.4f} ({max_drop*100:.2f}%)")
        print(f"  Hardest perturbation: {worst_name} (accuracy: {worst_acc:.4f})")
    
    # Comparison analysis
    if len(results_dict) > 1:
        print("\n" + "=" * 60)
        print("Model Comparison".center(60))
        print("=" * 60)
        
        model_names = list(results_dict.keys())
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                res1 = results_dict[name1]
                res2 = results_dict[name2]
                
                # Calculate average accuracy difference
                avg_acc1 = np.mean(res1['accuracies'][1:])
                avg_acc2 = np.mean(res2['accuracies'][1:])
                
                print(f"\n{name1} vs {name2}:")
                print(f"  Average accuracy difference: {avg_acc1 - avg_acc2:+.4f}")
                
                if avg_acc1 > avg_acc2:
                    print(f"  ✓ {name1} performs better on perturbed data")
                elif avg_acc2 > avg_acc1:
                    print(f"  ✓ {name2} performs better on perturbed data")
                else:
                    print(f"  = Both models perform similarly")
    
    print("\n" + "=" * 60)


def main():
    """Main function"""
    
    print("\n" + "=" * 60)
    print("MNIST Generalization Test".center(60))
    print("Train on standard MNIST, test on perturbed MNIST".center(60))
    print("=" * 60)
    
    import os
    os.makedirs('results', exist_ok=True)
    
    # ========== 1. Prepare data ==========
    print("\nStep 1: Prepare data")
    print("-" * 60)
    
    data_prep = DataPreparation(noise_ratio=0, use_feature='original', random_state=42)
    X_train, X_test, y_train, y_test, _, _ = data_prep.prepare()
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # ========== 2. Create perturber ==========
    print("\nStep 2: Create data perturber")
    print("-" * 60)
    
    perturber = MNISTPerturber(random_state=42)
    
    # Visualize perturbation effects
    print("Generating perturbation examples...")
    visualize_perturbations(X_test, perturber, 
                           save_path='results/perturbation_examples.png')
    
    # ========== 3. Train models ==========
    print("\nStep 3: Train models")
    print("-" * 60)
    
    models = {}
    
    # Model 1: Standard AdaBoost
    print("\nTraining standard AdaBoost...")
    clf_standard = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),  # Increase tree depth to ensure error < 0.5
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )
    clf_standard.fit(X_train, y_train)
    models['Standard AdaBoost'] = clf_standard
    print(f"Training complete, test accuracy: {clf_standard.score(X_test, y_test):.4f}")
    
    # Model 2: Robust AdaBoost (disable early stopping for multi-class problem)
    print("\nTraining robust AdaBoost...")
    clf_robust = RobustAdaBoost(
        base_estimator=DecisionTreeClassifier(max_depth=5),  # Use deeper trees to ensure error < 0.5
        n_estimators=100,
        learning_rate=1.0,
        random_state=42,
        weight_clip_percentile=95,
        use_early_stopping=False,  # Disable early stopping
        use_sample_weight_smoothing=True,  # Use weight smoothing
        smoothing_factor=0.5
    )
    clf_robust.fit(X_train, y_train)
    models['Robust AdaBoost'] = clf_robust
    print(f"Training complete, test accuracy: {clf_robust.score(X_test, y_test):.4f}")
    
    # ========== 4. Test generalization ==========
    print("\nStep 4: Test generalization")
    print("-" * 60)
    
    results_dict = {}
    
    for model_name, clf in models.items():
        results = test_generalization(clf, X_test, y_test, perturber, model_name)
        results_dict[model_name] = results
    
    # ========== 5. Visualize results ==========
    print("\nStep 5: Generate visualization")
    print("-" * 60)
    
    plot_generalization_results(results_dict, 
                               save_path='results/generalization_test.png')
    
    # ========== 6. Print summary ==========
    print_summary(results_dict)
    
    print("\n✓ Generalization test complete!")
    print("\nGenerated files:")
    print("  • results/perturbation_examples.png - Perturbation examples")
    print("  • results/generalization_test.png - Generalization test results")


if __name__ == "__main__":
    main()

