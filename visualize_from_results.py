"""
从已保存的训练结果可视化训练过程
支持读取 CSV 或 joblib 格式的 monitor 数据

用法：
    python visualize_from_results.py --experiment train_val_500rounds
    python visualize_from_results.py --csv experiments/train_val_500rounds/results/final_results.csv
    python visualize_from_results.py --joblib experiments/train_val_500rounds/results/monitor.joblib
"""

import os
import argparse
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt


def load_from_csv(csv_path):
    """
    从 CSV 文件加载监控数据
    
    Parameters
    ----------
    csv_path : str
        CSV 文件路径
    
    Returns
    -------
    dict
        包含所有监控数据的字典
    """
    print(f"📂 Loading from CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rounds of training data")
    
    # 构建与 BoostMonitor 相同的数据结构
    data = {
        'rounds': df['round'].tolist(),
        'error_history': df['weighted_error'].tolist(),
        'alpha_history': df['alpha'].tolist(),
        'error_without_weight_history': df['unweighted_error'].tolist() if 'unweighted_error' in df.columns else [],
        'noisy_weight_history': df['noisy_weight'].tolist() if 'noisy_weight' in df.columns else [],
        'clean_weight_history': df['clean_weight'].tolist() if 'clean_weight' in df.columns else [],
        'val_acc_history': df['val_acc'].tolist() if 'val_acc' in df.columns else [],
        'val_f1_history': df['val_f1'].tolist() if 'val_f1' in df.columns else [],
        'acc_on_train_data': df['train_acc'].tolist() if 'train_acc' in df.columns else [],
        'f1_on_training_data': df['train_f1'].tolist() if 'train_f1' in df.columns else [],
        'is_data_noisy': 'noisy_weight' in df.columns,
        'n_estimators': len(df),
    }
    
    print(f"✓ Data fields available:")
    for key, value in data.items():
        if key not in ['rounds', 'is_data_noisy', 'n_estimators']:
            status = "✓" if (isinstance(value, list) and len(value) > 0) else "✗"
            print(f"  {status} {key}")
    
    return data


def load_from_joblib(joblib_path):
    """
    从 joblib 文件加载 BoostMonitor 对象
    
    Parameters
    ----------
    joblib_path : str
        joblib 文件路径
    
    Returns
    -------
    dict
        包含所有监控数据的字典
    """
    print(f"📂 Loading from joblib: {joblib_path}")
    
    if not os.path.exists(joblib_path):
        raise FileNotFoundError(f"Joblib file not found: {joblib_path}")
    
    monitor = joblib.load(joblib_path)
    print(f"✓ Loaded BoostMonitor object")
    
    # 从 BoostMonitor 对象提取数据
    data = {
        'rounds': list(range(1, len(monitor.error_history) + 1)),
        'error_history': monitor.error_history,
        'alpha_history': monitor.alpha_history,
        'error_without_weight_history': monitor.error_without_weight_history,
        'noisy_weight_history': monitor.noisy_weight_history,
        'clean_weight_history': monitor.clean_weight_history,
        'val_acc_history': monitor.val_acc_history,
        'val_f1_history': monitor.val_f1_history,
        'acc_on_train_data': monitor.acc_on_train_data,
        'f1_on_training_data': monitor.f1_on_training_data,
        'sample_weights_history': monitor.sample_weights_history,
        'is_data_noisy': monitor.is_data_noisy,
        'n_estimators': len(monitor.error_history),
    }
    
    return data


def load_from_experiment(experiment_name):
    """
    从实验文件夹加载数据（自动检测 CSV 或 joblib）
    
    Parameters
    ----------
    experiment_name : str
        实验名称，如 'train_val_500rounds'
    
    Returns
    -------
    dict
        包含所有监控数据的字典
    """
    exp_dir = os.path.join('experiments', experiment_name)
    
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    print(f"📁 Loading from experiment: {experiment_name}")
    
    # 优先尝试 joblib
    joblib_path = os.path.join(exp_dir, 'results', 'monitor.joblib')
    if os.path.exists(joblib_path):
        return load_from_joblib(joblib_path)
    
    # 尝试 CSV
    csv_path = os.path.join(exp_dir, 'results', 'final_results.csv')
    if os.path.exists(csv_path):
        return load_from_csv(csv_path)
    
    raise FileNotFoundError(f"No monitor data found in {exp_dir}")


def visualize_training_data(data, save_path=None):
    """
    可视化训练数据（6个子图）
    
    Parameters
    ----------
    data : dict
        监控数据字典
    save_path : str, optional
        保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Analysis from Saved Results (n={data["n_estimators"]})', 
                 fontsize=16, fontweight='bold')
    
    rounds = data['rounds']
    
    # ========== 1. 错误率演化 ==========
    ax1 = axes[0, 0]
    ax1.plot(rounds, data['error_history'], 'b-', linewidth=2, label='Weighted Error')
    if len(data['error_without_weight_history']) > 0:
        ax1.plot(rounds, data['error_without_weight_history'], 'r--', 
                linewidth=2, label='Unweighted Error', alpha=0.7)
    ax1.set_xlabel('Boosting Round', fontsize=12)
    ax1.set_ylabel('Error Rate', fontsize=12)
    ax1.set_title('Error Rate Evolution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Alpha 系数演化 ==========
    ax2 = axes[0, 1]
    ax2.plot(rounds, data['alpha_history'], 'g-', linewidth=2, marker='o', 
            markersize=4, markevery=max(1, len(rounds)//20))
    avg_alpha = np.mean(data['alpha_history'])
    ax2.axhline(y=avg_alpha, color='orange', linestyle='--', 
               label=f'Mean={avg_alpha:.3f}', alpha=0.7)
    ax2.set_xlabel('Boosting Round', fontsize=12)
    ax2.set_ylabel('Alpha (Weak Learner Weight)', fontsize=12)
    ax2.set_title('Alpha Coefficient Evolution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== 3. 训练 vs 验证准确率 ==========
    ax3 = axes[0, 2]
    if len(data['acc_on_train_data']) > 0:
        ax3.plot(rounds, data['acc_on_train_data'], 'b-', linewidth=2, 
                label='Train Accuracy', marker='o', markersize=4,
                markevery=max(1, len(rounds)//20))
    if len(data['val_acc_history']) > 0:
        ax3.plot(rounds, data['val_acc_history'], 'r-', linewidth=2, 
                label='Val Accuracy', marker='s', markersize=4,
                markevery=max(1, len(rounds)//20))
    ax3.set_xlabel('Boosting Round', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Accuracy Evolution', fontsize=14, fontweight='bold')
    
    # 如果没有训练集准确率数据
    if len(data['acc_on_train_data']) == 0 and len(data['val_acc_history']) > 0:
        ax3.text(0.5, 0.05, 'Training accuracy not recorded in CSV\n(only validation accuracy available)', 
                ha='center', va='bottom', fontsize=10, color='gray', 
                transform=ax3.transAxes, style='italic')
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========== 4. 噪声样本 vs 干净样本权重 ==========
    ax4 = axes[1, 0]
    if data['is_data_noisy'] and len(data['noisy_weight_history']) > 0:
        ax4.plot(rounds, data['noisy_weight_history'], 'r-', linewidth=2, 
                label='Noisy Samples', marker='o', markersize=4,
                markevery=max(1, len(rounds)//20))
        ax4.plot(rounds, data['clean_weight_history'], 'g-', linewidth=2, 
                label='Clean Samples', marker='s', markersize=4,
                markevery=max(1, len(rounds)//20))
        ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax4.set_xlabel('Boosting Round', fontsize=12)
        ax4.set_ylabel('Total Weight', fontsize=12)
        ax4.set_title('Noisy vs Clean Sample Weights', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'N/A\n(Clean Data)', 
                ha='center', va='center', fontsize=14, color='gray', 
                transform=ax4.transAxes)
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    # ========== 5. F1 分数演化 ==========
    ax5 = axes[1, 1]
    if len(data['f1_on_training_data']) > 0:
        ax5.plot(rounds, data['f1_on_training_data'], 'b-', linewidth=2, 
                label='Train F1', marker='o', markersize=4,
                markevery=max(1, len(rounds)//20))
    if len(data['val_f1_history']) > 0:
        ax5.plot(rounds, data['val_f1_history'], 'r-', linewidth=2, 
                label='Val F1', marker='s', markersize=4,
                markevery=max(1, len(rounds)//20))
    ax5.set_xlabel('Boosting Round', fontsize=12)
    ax5.set_ylabel('F1 Score', fontsize=12)
    ax5.set_title('F1 Score Evolution', fontsize=14, fontweight='bold')
    
    # 如果没有训练F1数据
    if len(data['f1_on_training_data']) == 0 and len(data['val_f1_history']) > 0:
        ax5.text(0.5, 0.05, 'Training F1 not recorded in CSV\n(only validation F1 available)', 
                ha='center', va='bottom', fontsize=10, color='gray', 
                transform=ax5.transAxes, style='italic')
    
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ========== 6. 样本权重分布变化 ==========
    ax6 = axes[1, 2]
    if 'sample_weights_history' in data and len(data['sample_weights_history']) > 0:
        # 选择关键轮次
        key_rounds = [0, len(rounds)//3, len(rounds)*2//3, len(rounds)-1]
        positions = []
        data_to_plot = []
        labels = []
        
        for i, idx in enumerate(key_rounds):
            if idx < len(data['sample_weights_history']):
                positions.append(i + 1)
                data_to_plot.append(data['sample_weights_history'][idx])
                labels.append(f'R{idx+1}')
        
        bp = ax6.boxplot(data_to_plot, positions=positions, widths=0.6, 
                        patch_artist=True, labels=labels)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax6.set_ylabel('Sample Weight', fontsize=12)
        ax6.set_title('Sample Weight Distribution', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax6.text(0.5, 0.5, 'N/A\n(Not in CSV)', 
                ha='center', va='center', fontsize=14, color='gray', 
                transform=ax6.transAxes)
        ax6.text(0.5, 0.35, 'Sample weights only available\nin joblib format', 
                ha='center', va='center', fontsize=10, color='gray', 
                transform=ax6.transAxes, style='italic')
        ax6.set_xticks([])
        ax6.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(data):
    """打印数据摘要"""
    print("\n" + "=" * 60)
    print("Training Summary".center(60))
    print("=" * 60)
    
    print(f"\n📊 Basic Info:")
    print(f"   - Total Rounds: {data['n_estimators']}")
    print(f"   - Data Type: {'Noisy' if data['is_data_noisy'] else 'Clean'}")
    
    print(f"\n📈 Final Metrics:")
    if len(data['val_acc_history']) > 0:
        print(f"   - Final Val Accuracy: {data['val_acc_history'][-1]:.4f}")
        print(f"   - Best Val Accuracy:  {max(data['val_acc_history']):.4f} (round {data['val_acc_history'].index(max(data['val_acc_history']))+1})")
    
    if len(data['val_f1_history']) > 0:
        print(f"   - Final Val F1: {data['val_f1_history'][-1]:.4f}")
        print(f"   - Best Val F1:  {max(data['val_f1_history']):.4f} (round {data['val_f1_history'].index(max(data['val_f1_history']))+1})")
    
    print(f"\n🔍 Error Analysis:")
    print(f"   - Initial Error: {data['error_history'][0]:.4f}")
    print(f"   - Final Error:   {data['error_history'][-1]:.4f}")
    print(f"   - Min Error:     {min(data['error_history']):.4f}")
    
    print(f"\n⚖️ Alpha Analysis:")
    alphas = data['alpha_history']
    print(f"   - Mean Alpha: {np.mean(alphas):.3f}")
    print(f"   - Std Alpha:  {np.std(alphas):.3f}")
    print(f"   - Max Alpha:  {max(alphas):.3f} (round {alphas.index(max(alphas))+1})")
    print(f"   - Min Alpha:  {min(alphas):.3f} (round {alphas.index(min(alphas))+1})")
    
    if data['is_data_noisy'] and len(data['noisy_weight_history']) > 0:
        print(f"\n💡 Noise Analysis:")
        final_noisy = data['noisy_weight_history'][-1]
        final_clean = data['clean_weight_history'][-1]
        ratio = final_noisy / final_clean if final_clean > 0 else 0
        
        print(f"   - Initial Noisy Weight: {data['noisy_weight_history'][0]:.4f}")
        print(f"   - Final Noisy Weight:   {final_noisy:.4f}")
        print(f"   - Final Clean Weight:   {final_clean:.4f}")
        print(f"   - Weight Ratio (noisy/clean): {ratio:.3f}")
        
        if ratio > 1.5:
            print(f"   ⚠️  Noisy samples are over-weighted!")
        elif ratio > 1.0:
            print(f"   ⚠️  Noisy samples slightly over-weighted")
        else:
            print(f"   ✓  Weight distribution reasonable")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training results from saved CSV or joblib files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from experiment directory
  python visualize_from_results.py --experiment train_val_500rounds
  
  # Load from CSV file
  python visualize_from_results.py --csv experiments/train_val_500rounds/results/final_results.csv
  
  # Load from joblib file
  python visualize_from_results.py --joblib experiments/my_exp/results/monitor.joblib
  
  # Save figure
  python visualize_from_results.py --experiment train_val_500rounds --save my_plot.png
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment', '-e', type=str, help='Experiment name (e.g., train_val_500rounds)')
    group.add_argument('--csv', '-c', type=str, help='Path to CSV file')
    group.add_argument('--joblib', '-j', type=str, help='Path to joblib file')
    
    parser.add_argument('--save', '-s', type=str, help='Save figure to path')
    parser.add_argument('--no-plot', action='store_true', help='Print summary only, no plot')
    
    args = parser.parse_args()
    
    print("\n" + "█" * 60)
    print("Training Results Visualization".center(60))
    print("█" * 60)
    
    # 加载数据
    try:
        if args.experiment:
            data = load_from_experiment(args.experiment)
        elif args.csv:
            data = load_from_csv(args.csv)
        elif args.joblib:
            data = load_from_joblib(args.joblib)
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return 1
    
    # 打印摘要
    print_summary(data)
    
    # 可视化
    if not args.no_plot:
        print("\n📊 Generating visualization...")
        visualize_training_data(data, save_path=args.save)
        print("\n✓ Visualization complete!")
    
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())



