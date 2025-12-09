import json
import os
from adalab.workflow import train_and_save
from adalab.data import DataPreparation
from adalab.evaluation import evaluate

# 可视化模块（来自 adalab_viz）
from adalab_viz.loader import load_from_joblib, load_from_csv, load_from_experiment
from adalab_viz.plotter import visualize_training_data
from adalab_viz.summary import print_summary

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to json config file"
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--viz",
        action="store_true",
        help="Visualize after training. 'use_monitor' must be true",
    )
    group.add_argument(
        "--viz-only",
        action="store_true",
        help="Skip training; load monitor results and visualize them",
    )

    return parser.parse_args()


def run_training(config_path, do_viz=False):
    """
    执行训练并在训练后可选进行可视化。
    """

    # 读取配置
    with open(config_path) as f:
        config = json.load(f)

    print("Starting training...\n")

    # 调用工作流
    (
        clf,
        monitor,
        data_prep,
        (X_train, X_test_mnist, y_train, y_test_mnist, noise_idx, clean_idx),
        paths,
    ) = train_and_save(config_path)

    # ------------------------------------------------------------------
    # 训练结束，进行预测和评估
    # ------------------------------------------------------------------
    X_course, y_course = data_prep.prepare_course_data("test_data")

    y_pred_mnist = clf.predict(X_test_mnist)
    y_pred_course = clf.predict(X_course)

    print("\n=== Scores on test data of MNIST ===")
    scores_on_mnist = evaluate(y_true=y_test_mnist, y_pred=y_pred_mnist)

    print("\n=== Scores on test data of course data ===")
    scores_on_course = evaluate(y_true=y_course, y_pred=y_pred_course)

    scores = {"mnist": scores_on_mnist, "course_data": scores_on_course}

    result_dir = paths["result_dir"]
    exp_dir = paths["experiment_dir"]
    score_path = os.path.join(result_dir, "scores.json")

    with open(score_path, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"\nScores saved to: {score_path}")

    # ------------------------------------------------------------------
    # 可视化逻辑
    # ------------------------------------------------------------------
    if do_viz:
        print("\n=== Visualizing training process ===")

        if monitor is None:
            raise TypeError(
                "The monitor instance is None.\n"
                "Did you set 'use_monitor: true' in your config?"
            )

        data = load_from_joblib(monitor)

        print_summary(data)
        viz_dir = os.path.join(exp_dir, "visualization")
        os.makedirs(viz_dir, exist_ok=True)
        total_plot_name = f"{config['experiment']['name']}.png"
        visualize_training_data(
            data,
            save_path=os.path.join(viz_dir, total_plot_name),
            save_individual=True,
            output_dir=viz_dir,
        )

        print(f"Visualization saved under: {result_dir}")

    return paths


def run_viz_only(config_path):
    """
    不训练，只对已有实验结果可视化。
    config_path 用于获取 experiment name 或 result_dir。
    """

    # 获取 config
    with open(config_path) as f:
        config = json.load(f)

    # 实验名称（训练保存目录）
    exp_name = config["experiment"]["name"]
    print(f"\nLoading visualization for existing experiment: {exp_name}")

    # 自动从 experiments/<exp_name> 读取 monitor / csv
    data = load_from_experiment(exp_name)

    print_summary(data)
    exp_dir = os.path.join(
        "experiments",
        exp_name,
    )
    viz_dir = os.path.join(exp_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)
    total_plot_name = f"{config['experiment']['name']}.png"
    visualize_training_data(
        data,
        save_path=os.path.join(viz_dir, total_plot_name),
        save_individual=True,
        output_dir=viz_dir,
    )
    print(f"Visualization saved to: {viz_dir}")


if __name__ == "__main__":
    args = get_args()

    if args.viz_only:
        # 不训练，只可视化
        run_viz_only(args.config_path)

    else:
        # 正常训练 + 可选可视化
        run_training(args.config_path, do_viz=args.viz)
