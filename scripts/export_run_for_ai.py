from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a complete AIDE run tree for external AI review.",
    )
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("exports"))
    parser.add_argument(
        "--near-submission-rmse-threshold",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--prediction-similarity-sample-size",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--prediction-similarity-min-common-sample-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--skip-near-duplicate-check",
        action="store_true",
        dest="skip_near_duplicate_check",
        help="Skip the expensive near-duplicate submission similarity check.",
    )
    parser.add_argument(
        "--no-near-duplicates",
        action="store_true",
        dest="skip_near_duplicate_check",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from aide.utils.ai_run_export import export_run_for_ai

    progress_callback = _progress_callback() if sys.stderr.isatty() else None
    try:
        result = export_run_for_ai(
            args.log_dir,
            output_dir=args.output_dir,
            near_duplicates=not args.skip_near_duplicate_check,
            near_submission_rmse_threshold=args.near_submission_rmse_threshold,
            prediction_similarity_sample_size=args.prediction_similarity_sample_size,
            prediction_similarity_min_common_sample_size=(
                args.prediction_similarity_min_common_sample_size
            ),
            progress_callback=progress_callback,
        )
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Export directory: {result.export_dir}")
    print(f"Metadata: {result.meta_path}")
    print(f"Nodes: {result.nodes_path}")
    return 0


def _progress_callback() -> Callable[[str, int, int | None], None]:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
    )

    console = Console(stderr=True)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    tasks: dict[str, TaskID] = {}
    started = False

    def callback(stage: str, completed: int, total: int | None) -> None:
        nonlocal started
        if not started:
            progress.start()
            started = True
        if stage not in tasks:
            tasks[stage] = progress.add_task(
                stage,
                total=total,
                completed=completed,
            )
        task_id = tasks[stage]
        if total is None:
            progress.update(task_id, completed=completed, total=1)
        else:
            progress.update(task_id, completed=completed, total=total)
        if total is not None and completed >= total:
            progress.update(task_id, completed=total)
        if stage == "Writing export" and completed >= 1:
            progress.stop()

    return callback


if __name__ == "__main__":
    raise SystemExit(main())
