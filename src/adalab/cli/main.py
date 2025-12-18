from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from adalab.core import ExperimentPipeline


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="adalab", description="AdaLab experiment runner (CLI)")

    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to json config file",
    )
    p.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Base directory that stores experiment runs (default: experiments/)",
    )
    p.add_argument(
        "--course-folder",
        type=str,
        default="./data/test_images",
        help="Course test folder used in evaluation (default: ./data/test_images)",
    )

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--viz",
        action="store_true",
        help="Train + eval + visualize after training (requires use_monitor=true)",
    )
    group.add_argument(
        "--viz-only",
        action="store_true",
        help="Skip training; load existing experiment results then eval + visualize",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipe = ExperimentPipeline(experiments_dir=args.experiments_dir)

    config_path = Path(args.config)

    if args.viz_only:
        pipe.run_eval_viz_only(
            config_path=config_path, course_folder=args.course_folder
        )
        return 0

    # 默认：训练 + 评估；若 --viz 则再可视化
    pipe.run_train_eval(
        config_path=config_path,
        course_folder=args.course_folder,
        do_viz=bool(args.viz),
    )
    return 0
