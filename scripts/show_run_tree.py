from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from aide.run import build_tree_view, load_resume_state, render_tree_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the AIDE solution tree for an existing run.",
    )
    parser.add_argument("run_id", help="Run id under logs/, e.g. 2-skilled-nondescript-lobster")
    parser.add_argument("--logs-dir", default="logs", help="Top-level logs directory")
    parser.add_argument(
        "--workspaces-dir",
        default="workspaces",
        help="Top-level workspaces directory",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=200,
        help="Maximum number of tree lines to print",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=200,
        help="Render width",
    )
    parser.add_argument(
        "--scroll-top",
        type=int,
        default=0,
        help="First tree row to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, journal = load_resume_state(
        run_id=args.run_id,
        top_log_dir=Path(args.logs_dir),
        top_workspace_dir=Path(args.workspaces_dir),
        cli_overrides=[],
    )
    view = build_tree_view(journal, cfg=cfg)
    console = Console(width=args.width, color_system=None)
    console.print(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=max(0, args.scroll_top),
            viewport_height=max(1, args.height),
        )
    )


if __name__ == "__main__":
    main()
