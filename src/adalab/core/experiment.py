from __future__ import annotations

import re
import json
import datetime as dt
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EvalOutputs:
    scores: Dict[str, Any]
    score_path: Path


class ExperimentPipeline:
    """
    ExperimentPipeline (RunnerPipeline):
    负责训练/评估/可视化的流程编排（胶水层），不承载训练细节。
    """

    def __init__(self, experiments_dir: str | Path = "experiments"):
        self.experiments_dir = Path(experiments_dir)

    # Mode 1/2: train -> eval -> (optional viz)
    def run_train_eval(
        self,
        config_path: str | Path,
        course_folder: str = "test_data",
        do_viz: bool = False,
    ) -> Tuple[Any, Any, Any, Any, Any, EvalOutputs]:
        """
        执行：训练 + 评估 (+ 可选可视化)

        返回：
            (clf, monitor, split, layout, artifacts, eval_outputs)
        """
        from adalab.workflow import train_and_save

        config_path = Path(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))

        print("\033[36m[Pipeline] Starting training...\n\033[0m")

        clf, monitor, split, layout, artifacts = train_and_save(str(config_path))

        # evaluate
        scores = self._run_eval(
            clf=clf,
            split=split,
            course_folder=course_folder,
            result_dir=Path(layout.result_dir),
        )

        eval_outputs = EvalOutputs(
            scores=scores,
            score_path=Path(layout.result_dir) / "scores.json",
        )
        # visualize
        if do_viz:
            self._visualize_after_training(
                monitor=monitor,
                exp_dir=Path(layout.exp_dir),
                result_dir=Path(layout.result_dir),
                experiment_name=config["experiment"]["name"],
            )

        return clf, monitor, split, layout, artifacts, eval_outputs

    # Mode 3: eval + viz for existing experiment ,no training
    def run_eval_viz_only(
        self,
        config_path: str | Path,
        base_dir: str | Path | None = None,
    ) -> Path:
        """
        不重新训练：
        - 根据 config 中 experiment.name，在 experiments_dir 下交互选取一个 run 目录
        - 加载 monitor.joblib 或 final_results.csv
        - 可视化输出到 <exp_dir>/visualization

        返回：viz_dir
        """
        config_path = Path(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))

        exp_name = config["experiment"]["name"]
        base = Path(base_dir) if base_dir is not None else self.experiments_dir

        print(
            f"\033[36m[Pipeline] \nLoading visualization for existing experiment: {exp_name}\033[0m"
        )
        exp_dir = self._find_experiment_dir(exp_name, base_dir=base)
        print(f"\033[36m[Pipeline] Using experiment {exp_dir}\033[0m")

        result_dir = exp_dir / "results"
        data = self._load_viz_data(result_dir)

        viz_dir = exp_dir / "visualization"
        viz_dir.mkdir(parents=True, exist_ok=True)

        self._visualize_data(
            data=data,
            viz_dir=viz_dir,
            experiment_name=exp_name,
        )

        print(f"\033[36m[Pipeline] Visualization saved to: {viz_dir}\033[0m")
        return viz_dir

    def _visualize_after_training(
        self,
        monitor: Any,
        exp_dir: Path,
        result_dir: Path,
        experiment_name: str,
    ) -> None:
        if monitor is None:
            raise TypeError(
                "The monitor instance is None.\n"
                "Did you set 'use_monitor: true' in your config?"
            )

        try:
            from adalab_viz.loader import load_from_joblib
            from adalab_viz.plotter import visualize_training_data
            from adalab_viz.summary import print_summary
        except Exception as e:
            warnings.warn(
                f"[adalab] Visualization skipped: adalab_viz is not available ({e}).",
                RuntimeWarning,
            )
            return

        print("\033[36m[Pipeline] \n[Viz] Visualizing training process\033[0m")
        data = load_from_joblib(monitor)
        print_summary(data)

        viz_dir = exp_dir / "visualization"
        viz_dir.mkdir(parents=True, exist_ok=True)

        total_plot_name = f"{experiment_name}.png"
        visualize_training_data(
            data,
            save_path=str(viz_dir / total_plot_name),
            save_individual=True,
            output_dir=str(viz_dir),
        )

        print(f"\033[36m[Pipeline] Visualization saved under: {result_dir}\033[0m")

    def _load_viz_data(self, result_dir: Path) -> Any:
        try:
            from adalab_viz.loader import load_from_joblib, load_from_csv
        except Exception as e:
            raise ModuleNotFoundError(
                "adalab_viz is required for visualization mode. "
                "Install it (e.g. `pip install adalab[viz]`) then retry."
            ) from e

        monitor_path = result_dir / "monitor.joblib"
        csv_path = result_dir / "final_results.csv"

        if monitor_path.exists():
            return load_from_joblib(str(monitor_path))
        if csv_path.exists():
            return load_from_csv(str(csv_path))

        raise FileNotFoundError(
            f"Neither monitor.joblib nor final_results.csv found under: {result_dir}"
        )

    def _visualize_data(self, data: Any, viz_dir: Path, experiment_name: str) -> None:
        from adalab_viz.plotter import visualize_training_data
        from adalab_viz.summary import print_summary

        print_summary(data)

        total_plot_name = f"{experiment_name}.png"
        visualize_training_data(
            data,
            save_path=str(viz_dir / total_plot_name),
            save_individual=True,
            output_dir=str(viz_dir),
        )

    @staticmethod
    def _extract_run_time(p: Path) -> dt.datetime:
        """
        Priority:
        1. Parse timestamp from directory name: *_YYYYMMDD_HHMMSS
        2. Fallback to filesystem mtime
        """
        _TS_RE = re.compile(r".*_(\d{8}_\d{6})$")
        m = _TS_RE.match(p.name)
        if m:
            try:
                return dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            except ValueError:
                pass

        # fallback
        return dt.datetime.fromtimestamp(p.stat().st_mtime)

    def _find_experiment_dir(self, exp_name: str, base_dir: Path) -> Path:
        candidates: List[Path] = []

        exact = base_dir / exp_name
        if exact.exists() and exact.is_dir():
            candidates.append(exact)

        candidates.extend([p for p in base_dir.glob(f"{exp_name}_*") if p.is_dir()])

        if not candidates:
            raise FileNotFoundError(
                f"No experiment directory found for '{exp_name}' under '{base_dir}'"
            )

        candidates = list(dict.fromkeys(candidates))
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        print(
            f"\033[36m[Pipeline] \nFound {len(candidates)} runs for experiment '{exp_name}'.\033[0m"
        )
        print("\033[36m[Pipeline] Select one:\n\033[0m")
        for i, p in enumerate(candidates):
            tag = "(exact)" if p.name == exp_name else ""
            print(
                f"\033[36m[Pipeline]   [{i}] {p.name:<45} mtime={self._extract_run_time(p)} {tag}\033[0m"
            )

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
            print("\033[36m[Pipeline] Invalid input, try again.\033[0m")

    def _run_eval(
        self,
        *,
        clf,
        split,
        course_folder: str,
        result_dir: Path,
    ) -> dict:
        """
        Run evaluation on:
        - MNIST test split
        - course data

        This method is intentionally lightweight:
        - only requires a classifier with predict()
        - minimal dependency on training artifacts

        Returns:
            scores (dict): structured evaluation results
        """

        from adalab.evaluation import evaluate

        # ---- prepare data ----
        X_course, y_course = split.prep.prepare_course_data(course_folder)

        # ---- predictions ----
        y_pred_mnist = clf.predict(split.X_test)
        y_pred_course = clf.predict(X_course)

        # ---- evaluation ----
        print("\033[36m[Pipeline] \n=== Scores on test data of MNIST ===\033[0m")
        scores_on_mnist = evaluate(
            y_true=split.y_test,
            y_pred=y_pred_mnist,
        )

        print("\033[36m[Pipeline] \n=== Scores on test data of course data ===\033[0m")
        scores_on_course = evaluate(
            y_true=y_course,
            y_pred=y_pred_course,
        )

        scores = {
            "mnist": scores_on_mnist,
            "course_data": scores_on_course,
        }

        # ---- save ----
        score_path = result_dir / "scores.json"
        score_path.write_text(
            json.dumps(scores, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\033[36m[Pipeline] \nScores saved to: {score_path}\033[0m")

        return scores
