from __future__ import annotations
import json

from typing import List
import datetime as dt
from pathlib import Path
from argparse import ArgumentParser

from adalab.workflow import train_and_save
from adalab.data import DataPreparation
from adalab.evaluation import evaluate


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


def run_training(
    config_path: str, do_viz: bool = False, course_folder: str = "test_data"
):
    """
    执行训练并在训练后可选进行可视化。

    返回:
        artifacts: ArtifactPaths（你保存下来的产物路径集合）
    """
    config_path_p = Path(config_path)

    # 读取配置（只用于命名/打印）
    config = json.loads(config_path_p.read_text(encoding="utf-8"))

    print("Starting training...\n")

    # training workflow
    clf, monitor, split, layout, artifacts = train_and_save(str(config_path_p))

    # evaluate
    X_course, y_course = split.prep.prepare_course_data(course_folder)

    y_pred_mnist = clf.predict(split.X_test)
    y_pred_course = clf.predict(X_course)

    print("\n=== Scores on test data of MNIST ===")
    scores_on_mnist = evaluate(y_true=split.y_test, y_pred=y_pred_mnist)

    print("\n=== Scores on test data of course data ===")
    scores_on_course = evaluate(y_true=y_course, y_pred=y_pred_course)

    scores = {"mnist": scores_on_mnist, "course_data": scores_on_course}

    score_path = layout.result_dir / "scores.json"
    score_path.write_text(
        json.dumps(scores, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nScores saved to: {score_path}")

    # visualize
    if do_viz:
        print("\n=== Visualizing training process ===")

        if monitor is None:
            raise TypeError(
                "The monitor instance is None.\n"
                "Did you set 'use_monitor: true' in your config?"
            )

        from adalab_viz.loader import load_from_joblib
        from adalab_viz.plotter import visualize_training_data
        from adalab_viz.summary import print_summary

        data = load_from_joblib(monitor)

        print_summary(data)

        viz_dir = layout.exp_dir / "visualization"
        viz_dir.mkdir(parents=True, exist_ok=True)

        total_plot_name = f"{config['experiment']['name']}.png"
        visualize_training_data(
            data,
            save_path=str(viz_dir / total_plot_name),
            save_individual=True,
            output_dir=str(viz_dir),
        )

        print(f"Visualization saved under: {layout.result_dir}")

    return artifacts


def _format_mtime(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _find_experiment_dir(
    exp_name: str,
    base_dir: Path = Path("experiments"),
) -> Path:
    """
    总是进入交互式选择：
    - 候选包含 experiments/<exp_name>（若存在）
    - 也包含 experiments/<exp_name>_*（时间戳目录）
    - 按 mtime 从新到旧排序展示
    """
    candidates: List[Path] = []

    exact = base_dir / exp_name
    if exact.exists() and exact.is_dir():
        candidates.append(exact)

    candidates.extend([p for p in base_dir.glob(f"{exp_name}_*") if p.is_dir()])

    if not candidates:
        raise FileNotFoundError(
            f"No experiment directory found for '{exp_name}' under '{base_dir}'"
        )

    # 去重
    candidates = list(dict.fromkeys(candidates))

    # 按修改时间倒序
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # 展示
    print(f"\nFound {len(candidates)} runs for experiment '{exp_name}'.")
    print("Select one:\n")

    for i, p in enumerate(candidates):
        tag = "(exact)" if p.name == exp_name else ""
        print(f"  [{i}] {p.name:<45} mtime={_format_mtime(p)} {tag}")

    # 交互输入
    while True:
        s = (
            input(f"\nEnter index (0-{len(candidates) - 1}) or 'q' to quit: ")
            .strip()
            .lower()
        )
        if s in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(candidates):
                return candidates[idx]
        print("Invalid input, try again.")


def run_viz_only(config_path: str, base_dir: str = "experiments"):
    """
    不训练，只对已有实验结果可视化。
    config_path 用于获取 experiment name，然后在 experiments/ 下自动定位实验目录。
    """
    config_path_p = Path(config_path)
    config = json.loads(config_path_p.read_text(encoding="utf-8"))

    exp_name = config["experiment"]["name"]
    print(f"\nLoading visualization for existing experiment: {exp_name}")

    exp_dir = _find_experiment_dir(exp_name, base_dir=Path(base_dir))
    print(f"Using experiment {exp_dir}")
    result_dir = exp_dir / "results"

    # 延迟导入,避免没装 adalab[viz] import 直接报错
    from adalab_viz.loader import load_from_joblib, load_from_csv
    from adalab_viz.plotter import visualize_training_data
    from adalab_viz.summary import print_summary

    # 优先用 monitor.joblib（信息更全），否则退化到 csv
    monitor_path = result_dir / "monitor.joblib"
    csv_path = result_dir / "final_results.csv"

    if monitor_path.exists():
        data = load_from_joblib(str(monitor_path))
    elif csv_path.exists():
        data = load_from_csv(str(csv_path))
    else:
        raise FileNotFoundError(
            f"Neither monitor.joblib nor final_results.csv found under: {result_dir}"
        )

    print_summary(data)

    viz_dir = exp_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    total_plot_name = f"{config['experiment']['name']}.png"
    visualize_training_data(
        data,
        save_path=str(viz_dir / total_plot_name),
        save_individual=True,
        output_dir=str(viz_dir),
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
