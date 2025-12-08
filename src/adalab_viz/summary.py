import numpy as np


def print_summary(data):
    """打印数据摘要"""
    print("\n" + "=" * 60)
    print("Training Summary".center(60))
    print("=" * 60)

    print("\n Basic Info:")
    print(f"   - Total Rounds: {data['n_estimators']}")
    print(f"   - Data Type: {'Noisy' if data['is_data_noisy'] else 'Clean'}")

    print("\n Final Metrics:")
    if len(data["val_acc_history"]) > 0:
        print(f"   - Final Val Accuracy: {data['val_acc_history'][-1]:.4f}")
        print(
            f"   - Best Val Accuracy:  {max(data['val_acc_history']):.4f} (round {data['val_acc_history'].index(max(data['val_acc_history'])) + 1})"
        )

    if len(data["val_f1_history"]) > 0:
        print(f"   - Final Val F1: {data['val_f1_history'][-1]:.4f}")
        print(
            f"   - Best Val F1:  {max(data['val_f1_history']):.4f} (round {data['val_f1_history'].index(max(data['val_f1_history'])) + 1})"
        )

    print("\n Error Analysis:")
    print(f"   - Initial Error: {data['error_history'][0]:.4f}")
    print(f"   - Final Error:   {data['error_history'][-1]:.4f}")
    print(f"   - Min Error:     {min(data['error_history']):.4f}")

    print("\n Alpha Analysis:")
    alphas = data["alpha_history"]
    print(f"   - Mean Alpha: {np.mean(alphas):.3f}")
    print(f"   - Std Alpha:  {np.std(alphas):.3f}")
    print(f"   - Max Alpha:  {max(alphas):.3f} (round {alphas.index(max(alphas)) + 1})")
    print(f"   - Min Alpha:  {min(alphas):.3f} (round {alphas.index(min(alphas)) + 1})")

    if data["is_data_noisy"] and len(data["noisy_weight_history"]) > 0:
        print("\n Noise Analysis:")
        final_noisy = data["noisy_weight_history"][-1]
        final_clean = data["clean_weight_history"][-1]
        ratio = final_noisy / final_clean if final_clean > 0 else 0

        print(f"   - Initial Noisy Weight: {data['noisy_weight_history'][0]:.4f}")
        print(f"   - Final Noisy Weight:   {final_noisy:.4f}")
        print(f"   - Final Clean Weight:   {final_clean:.4f}")
        print(f"   - Weight Ratio (noisy/clean): {ratio:.3f}")

        if ratio > 1.5:
            print("  Noisy samples are over-weighted!")
        elif ratio > 1.0:
            print("  Noisy samples slightly over-weighted")
        else:
            print("  Weight distribution reasonable")

    print("=" * 60)
