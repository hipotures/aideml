#!/usr/bin/env python3
"""Import standalone legacy Python files as pending nodes in the ``manual`` run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from aide.journal import Journal, Node
from aide.run import allocate_node_artifact_slot, load_resume_state, mark_node_generated_only
from aide.utils.config import (
    _load_cfg,
    copy_solution_helper,
    prep_agent_workspace,
    prep_cfg,
    save_run,
)


@dataclass(frozen=True)
class ManualLegacyRunResult:
    run_id: str
    log_dir: Path
    workspace_dir: Path
    sources: tuple[Path, ...]


def _validate_sources(paths: Sequence[Path]) -> tuple[Path, ...]:
    if not paths:
        raise ValueError("At least one legacy .py file is required.")

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Legacy source file not found: {path}")
        if path.suffix.lower() != ".py":
            raise ValueError(f"Legacy source must be a .py file: {path}")
        if path in seen:
            raise ValueError(f"Duplicate legacy source file: {path}")
        seen.add(path)
        resolved.append(path)
    return tuple(resolved)


def _path_has_entries(path: Path) -> bool:
    """Return whether a path contains state that should not be replaced."""

    if not path.exists():
        return path.is_symlink()
    if not path.is_dir():
        return True
    return any(path.iterdir())


def _configure_manual_runtime(cfg) -> None:
    """Keep the manual run legacy-only and prevent generation beyond its queue."""

    cfg.manual_queue_only = True
    cfg.agent.mode = "legacy"
    cfg.agent.search.code_ahead = 0
    cfg.agent.hypotheses = 0
    cfg.research.enabled = False
    cfg.synthesis.enabled = False
    cfg.refactor.enabled = False


def _new_manual_config(
    *,
    run_id: str,
    logs_dir: Path,
    workspaces_dir: Path,
    data_dir: Path | None,
    desc_file: Path | None,
):
    cfg = _load_cfg(use_cli_args=False)
    if data_dir is not None:
        cfg.data_dir = data_dir
    if desc_file is not None:
        cfg.desc_file = desc_file
    cfg.log_dir = logs_dir
    cfg.workspace_dir = workspaces_dir
    cfg.exp_name = run_id
    _configure_manual_runtime(cfg)

    # prep_cfg normalizes paths and config schema, but prefixes exp_name for
    # ordinary AIDE runs. Restore the explicitly requested manual run layout.
    cfg = prep_cfg(cfg)
    cfg.exp_name = run_id
    cfg.log_dir = logs_dir / run_id
    cfg.workspace_dir = workspaces_dir / run_id
    _configure_manual_runtime(cfg)
    return cfg


def _existing_manual_run(
    *,
    run_id: str,
    logs_dir: Path,
    workspaces_dir: Path,
):
    cfg, journal = load_resume_state(
        run_id=run_id,
        top_log_dir=logs_dir,
        top_workspace_dir=workspaces_dir,
        cli_overrides=[],
    )
    _configure_manual_runtime(cfg)
    return cfg, journal


def seed_manual_legacy_run(
    *,
    sources: Sequence[Path],
    run_id: str = "manual",
    logs_dir: Path = Path("logs"),
    workspaces_dir: Path = Path("workspaces"),
    data_dir: Path | None = None,
    desc_file: Path | None = None,
    prepare_workspace: bool = True,
) -> ManualLegacyRunResult:
    """Append legacy source files as generated-only nodes in a resumable run."""

    source_paths = _validate_sources(sources)
    logs_dir = Path(logs_dir).expanduser().resolve()
    workspaces_dir = Path(workspaces_dir).expanduser().resolve()
    log_dir = logs_dir / run_id
    workspace_dir = workspaces_dir / run_id

    # ``logs/manual`` and ``workspaces/manual`` may have been created as empty
    # parents before this importer runs. Empty roots are not an existing run;
    # initialize them in place. Non-empty partial roots remain protected.
    run_exists = _path_has_entries(log_dir) or _path_has_entries(workspace_dir)
    if run_exists:
        if not (
            log_dir.is_dir()
            and workspace_dir.is_dir()
            and (log_dir / "config.yaml").exists()
            and (log_dir / "journal.json").exists()
        ):
            raise FileExistsError(
                "The manual run exists but is incomplete; refusing to overwrite "
                "non-empty state: "
                f"{log_dir}"
            )
        cfg, journal = _existing_manual_run(
            run_id=run_id,
            logs_dir=logs_dir,
            workspaces_dir=workspaces_dir,
        )
    else:
        cfg = _new_manual_config(
            run_id=run_id,
            logs_dir=logs_dir,
            workspaces_dir=workspaces_dir,
            data_dir=data_dir,
            desc_file=desc_file,
        )
        journal = Journal()
        if prepare_workspace:
            prep_agent_workspace(cfg)
        else:
            (workspace_dir / "input").mkdir(parents=True, exist_ok=True)
            (workspace_dir / "working").mkdir(parents=True, exist_ok=True)
            copy_solution_helper(workspace_dir)

    for source_path in source_paths:
        step = len(journal)
        node_ctime, artifact_dir_name, _artifact_dir = allocate_node_artifact_slot(
            cfg.log_dir,
            step=step,
            workspace_dir=cfg.workspace_dir,
        )
        node = Node(
            code=source_path.read_text(encoding="utf-8"),
            plan=f"Manual legacy draft imported from {source_path}",
            ctime=node_ctime,
            artifact_dir_name=artifact_dir_name,
        )
        mark_node_generated_only(node)
        journal.append(node)

    # A resume should execute exactly the queued drafts and then stop, rather
    # than asking the legacy search policy to generate additional candidates.
    cfg.agent.steps = len(journal)
    _configure_manual_runtime(cfg)
    save_run(cfg, journal)

    return ManualLegacyRunResult(
        run_id=run_id,
        log_dir=Path(cfg.log_dir),
        workspace_dir=Path(cfg.workspace_dir),
        sources=source_paths,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import one or more standalone legacy .py files as pending draft "
            "nodes in logs/manual for execution with aide --resume manual."
        )
    )
    parser.add_argument("sources", nargs="+", type=Path, metavar="SOURCE.py")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--workspaces-dir", type=Path, default=Path("workspaces"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--desc-file", type=Path)
    parser.add_argument(
        "--no-prepare-workspace",
        action="store_true",
        help="Create empty input/working directories instead of preparing task data.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = seed_manual_legacy_run(
        sources=args.sources,
        run_id=args.run_id,
        logs_dir=args.logs_dir,
        workspaces_dir=args.workspaces_dir,
        data_dir=args.data_dir,
        desc_file=args.desc_file,
        prepare_workspace=not args.no_prepare_workspace,
    )
    print(f"Imported {len(result.sources)} legacy draft(s) into {result.log_dir}")
    print(f"Workspace: {result.workspace_dir}")
    print(f"Resume with: uv run aide --resume {result.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
