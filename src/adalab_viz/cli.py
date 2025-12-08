"""
adalab_viz CLI入口

支持：
    python -m adalab_viz --experiment expname
    python -m adalab_viz --csv path/to.csv
    python -m adalab_viz --joblib path/to.joblib
"""

import argparse
from .loader import load_from_csv, load_from_joblib, load_from_experiment
from .summary import print_summary
from .plotter import visualize_training_data


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AdaBoost training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m adalab_viz --experiment noise10_est500_depth2
  python -m adalab_viz --csv path/to/final_results.csv
  python -m adalab_viz --joblib path/to/monitor.joblib
  python -m adalab_viz --experiment foo --save figure.png
""",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment", "-e", type=str, help="Experiment name (from experiments/<name>)"
    )
    group.add_argument("--csv", "-c", type=str, help="Path to CSV results")
    group.add_argument("--joblib", "-j", type=str, help="Path to joblib results")

    parser.add_argument("--save", "-s", type=str, help="Save the generated figure")
    parser.add_argument("--no-plot", action="store_true", help="Only print summary")

    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("AdaLab Visualization".center(60))
    print("█" * 60)

    # ----- 加载数据 -----
    if args.experiment:
        data = load_from_experiment(args.experiment)
    elif args.csv:
        data = load_from_csv(args.csv)
    elif args.joblib:
        data = load_from_joblib(args.joblib)
    else:
        raise ValueError("Unreachable: argparse should enforce one option.")

    # ----- 打印摘要 -----
    print_summary(data)

    # ----- 绘图 -----
    if not args.no_plot:
        print("\n Generating visualization...")
        visualize_training_data(data, save_path=args.save)
        print("\n✓ Visualization complete!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
