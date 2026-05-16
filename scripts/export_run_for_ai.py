from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from scripts.render_research_prompt import (
    MODE_TEMPLATE_BY_MODE,
    default_mode_template_path,
    default_output_path,
    default_template_path,
    default_values_path,
    load_values,
    positive_int,
    write_prompt,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a complete AIDE run tree and optional AI review prompt.",
    )
    parser.add_argument("log_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/aideml-ai-review"),
        help="Directory where a timestamped review bundle subdirectory is created.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=sorted(MODE_TEMPLATE_BY_MODE),
        help="Also render a research prompt into the export bundle.",
    )
    parser.add_argument(
        "--task",
        default="playground-series-s6e5",
        help=(
            "Task slug used to find research_hypotheses/<task>/prompt_values.json "
            "when --prompt-values is not provided."
        ),
    )
    parser.add_argument(
        "--prompt-values",
        type=Path,
        help="JSON file with prompt placeholder values.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        help="Base prompt template path.",
    )
    parser.add_argument(
        "--prompt-mode-template",
        type=Path,
        help="Mode-specific prompt block path.",
    )
    parser.add_argument(
        "--hypothesis-count",
        type=positive_int,
        default=10,
        help="Number of hypotheses the rendered prompt should request.",
    )
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
        prompt_path = None
        if args.prompt_mode is not None:
            prompt_path = _write_prompt_to_export_dir(
                mode=args.prompt_mode,
                task=args.task,
                values_path=args.prompt_values,
                template_path=args.prompt_template,
                mode_template_path=args.prompt_mode_template,
                hypothesis_count=args.hypothesis_count,
                export_dir=result.export_dir,
            )
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    _print_result(result.export_dir, result.meta_path, result.nodes_path, prompt_path)
    return 0


def _write_prompt_to_export_dir(
    *,
    mode: str,
    task: str,
    values_path: Path | None,
    template_path: Path | None,
    mode_template_path: Path | None,
    hypothesis_count: int,
    export_dir: Path,
) -> Path:
    resolved_values_path = values_path or default_values_path(task)
    values = load_values(resolved_values_path)
    prompt_name = default_output_path(mode, values).name
    return write_prompt(
        mode=mode,
        values_path=resolved_values_path,
        template_path=template_path or default_template_path(mode),
        mode_template_path=mode_template_path or default_mode_template_path(mode),
        value_overrides={"HYPOTHESIS_COUNT": hypothesis_count},
        out_path=export_dir / prompt_name,
    )


def _print_result(
    export_dir: Path,
    meta_path: Path,
    nodes_path: Path,
    prompt_path: Path | None,
) -> None:
    print(f"Export directory: {export_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Nodes: {nodes_path}")
    if prompt_path is None:
        return
    print(f"Prompt: {prompt_path}")
    print("")
    print("Attach these files to GPT:")
    print(f"- {prompt_path}")
    print(f"- {meta_path}")
    print(f"- {nodes_path}")
    print("")
    print(
        "Then ask: "
        f"Please execute the prompt from {prompt_path.name} using the attached "
        "export files, and return the response in English."
    )


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
