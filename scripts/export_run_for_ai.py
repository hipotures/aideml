from __future__ import annotations

import argparse
from pathlib import Path

from aide.utils.ai_run_export import export_run_for_ai


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
    parser.add_argument("--no-near-duplicates", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = export_run_for_ai(
            args.log_dir,
            output_dir=args.output_dir,
            near_duplicates=not args.no_near_duplicates,
            near_submission_rmse_threshold=args.near_submission_rmse_threshold,
            prediction_similarity_sample_size=args.prediction_similarity_sample_size,
            prediction_similarity_min_common_sample_size=(
                args.prediction_similarity_min_common_sample_size
            ),
        )
    except Exception as exc:
        print(f"Export failed: {exc}")
        return 1
    print(f"Export directory: {result.export_dir}")
    print(f"Metadata: {result.meta_path}")
    print(f"Nodes: {result.nodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
