#!/usr/bin/env python3
"""Create an AIDE run with a seeded root and forced first-layer child queue."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from aide.journal import Journal, Node
from aide.research import (
    FORCED_CHILD_QUEUE_FILE,
    _compatible_manual_hypotheses,
    load_manual_hypothesis_library,
    write_forced_child_hypothesis_queue,
)
from aide.run import mark_node_generated_only
from aide.utils.config import (
    _load_cfg,
    prep_agent_workspace,
    prep_cfg,
    save_run,
)
from aide.utils.metric import MetricValue

@dataclass(frozen=True)
class SeedGeneratedBranchResult:
    run_id: str
    log_dir: Path
    workspace_dir: Path
    root_hypothesis: str
    children: tuple[str, ...]

def _default_data_dir(repo_root: Path, task: str) -> Path:
    return repo_root / "aide" / "example_tasks" / task


def _default_desc_file(repo_root: Path, task: str) -> Path:
    return repo_root / "aide" / "example_tasks" / f"{task}.md"


def _validate_run_id(run_id: str | None) -> None:
    if run_id is None:
        return
    if "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a plain directory name, not a path.")
    if not re.match(r"^\d+-[A-Za-z0-9][A-Za-z0-9._-]*$", run_id):
        raise ValueError("run_id must look like an AIDE run id, for example 2-name.")


def _generated_node(
    *,
    code: str,
    plan: str,
    hypothesis_id: str,
    source_hash: str,
    parent: Node | None = None,
) -> Node:
    node = Node(code=code, plan=plan, parent=parent)
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = [hypothesis_id]
    node.research_source_hash = source_hash
    mark_node_generated_only(node)
    return node


def _manifest_entry_for_code(hypothesis_dir: Path, agent_mode: str, code_file: str) -> dict:
    manifest_path = hypothesis_dir / "code_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    versions = manifest.get("versions", {})
    if not isinstance(versions, dict):
        return {}
    entries = versions.get(agent_mode, [])
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("file") == code_file:
            return entry
    return {}


def _seeded_root_node(
    *,
    code: str,
    plan: str,
    hypothesis_id: str,
    source_hash: str,
    manifest_entry: dict,
) -> Node:
    node = _generated_node(
        code=code,
        plan=plan,
        hypothesis_id=hypothesis_id,
        source_hash=source_hash,
    )
    if manifest_entry.get("buggy") is False and manifest_entry.get("score") is not None:
        node.status = "ok"
        node.metric = MetricValue(manifest_entry["score"], maximize=True)
        node.is_buggy = False
        node.analysis = (
            "Seeded from an existing executed hypothesis manifest; execution skipped."
        )
        node.run_stats = {
            "seeded_from_manifest": True,
            "source_score": manifest_entry["score"],
        }
    return node


def _require_hypotheses(
    *,
    root_ids: set[str],
    child_ids: set[str],
    root_hypothesis: str,
    children: Iterable[str],
) -> None:
    if root_hypothesis not in root_ids:
        raise ValueError(
            f"Unknown, disabled, or incompatible root hypothesis id: {root_hypothesis}"
        )
    missing_children = [
        hypothesis_id for hypothesis_id in tuple(children) if hypothesis_id not in child_ids
    ]
    if missing_children:
        raise ValueError(
            "Unknown or incompatible child hypothesis id(s): "
            + ", ".join(missing_children)
        )


def _child_hypothesis_ids_for_mode(
    *,
    cfg,
    library: ManualHypothesisLibrary,
) -> set[str]:
    agent_mode = cfg.agent.mode
    if getattr(cfg.research, "ignore_hypothesis_agent_modes", False):
        return {hypothesis.id for hypothesis in library.hypotheses}
    return {
        hypothesis.id
        for hypothesis in library.hypotheses
        if agent_mode in hypothesis.agent_modes
    }


def seed_generated_branch(
    *,
    task: str,
    agent_mode: str,
    root_hypothesis: str,
    root_code_file: str,
    children: tuple[str, ...],
    run_id: str | None = None,
    data_dir: Path | None = None,
    desc_file: Path | None = None,
    logs_dir: Path = Path("logs"),
    workspaces_dir: Path = Path("workspaces"),
    repo_root: Path = Path("."),
    prepare_workspace: bool = True,
) -> SeedGeneratedBranchResult:
    repo_root = repo_root.resolve()
    logs_dir = logs_dir.resolve()
    workspaces_dir = workspaces_dir.resolve()
    _validate_run_id(run_id)

    data_dir = (data_dir or _default_data_dir(repo_root, task)).resolve()
    desc_file = (desc_file or _default_desc_file(repo_root, task)).resolve()

    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(data_dir)
    cfg.desc_file = str(desc_file)
    cfg.log_dir = str(logs_dir)
    cfg.workspace_dir = str(workspaces_dir)
    cfg.agent.mode = agent_mode
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.copy_data = False
    cfg.preprocess_data = False
    cfg = prep_cfg(cfg)

    if run_id is not None:
        cfg.exp_name = run_id
        cfg.log_dir = logs_dir / run_id
        cfg.workspace_dir = workspaces_dir / run_id

    if cfg.log_dir.exists():
        raise FileExistsError(f"Run log directory already exists: {cfg.log_dir}")
    if cfg.workspace_dir.exists():
        raise FileExistsError(f"Run workspace directory already exists: {cfg.workspace_dir}")

    library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
    compatible_root_ids = {
        hypothesis.id for hypothesis in _compatible_manual_hypotheses(cfg, library)
    }
    compatible_child_ids = _child_hypothesis_ids_for_mode(cfg=cfg, library=library)
    _require_hypotheses(
        root_ids=compatible_root_ids,
        child_ids=compatible_child_ids,
        root_hypothesis=root_hypothesis,
        children=children,
    )

    root_code_path = library.source_dir / root_hypothesis / root_code_file
    if not root_code_path.exists():
        raise FileNotFoundError(f"Missing root code file: {root_code_path}")
    manifest_entry = _manifest_entry_for_code(
        library.source_dir / root_hypothesis,
        agent_mode,
        root_code_file,
    )

    journal = Journal()
    root_node = _seeded_root_node(
        code=root_code_path.read_text(encoding="utf-8"),
        plan=(
            f"Seeded generated ROOT hypothesis {root_hypothesis} "
            f"from {agent_mode} {root_code_file}."
        ),
        hypothesis_id=root_hypothesis,
        source_hash=library.source_hash,
        manifest_entry=manifest_entry,
    )
    journal.append(root_node)

    if prepare_workspace:
        prep_agent_workspace(cfg)
    else:
        (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
        (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    write_forced_child_hypothesis_queue(
        cfg,
        root_hypothesis=root_hypothesis,
        children=children,
    )

    save_run(cfg, journal)
    return SeedGeneratedBranchResult(
        run_id=str(cfg.exp_name),
        log_dir=Path(cfg.log_dir),
        workspace_dir=Path(cfg.workspace_dir),
        root_hypothesis=root_hypothesis,
        children=children,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a resumable AIDE run whose journal contains a generated "
            "root node from an explicit code file plus a queue of first-layer "
            "child hypotheses. The child code is generated later by normal "
            "AIDE resume using that run's agent.code.model."
        )
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent-mode", required=True, choices=["legacy", "autogluon"])
    parser.add_argument("--root-hypothesis", required=True)
    parser.add_argument("--root-code", required=True)
    parser.add_argument("--children", nargs="+", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--desc-file", type=Path)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--workspaces-dir", type=Path, default=Path("workspaces"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--no-workspace",
        action="store_true",
        help="Only create minimal input/ and working/ directories instead of copying task data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = seed_generated_branch(
        task=args.task,
        agent_mode=args.agent_mode,
        root_hypothesis=args.root_hypothesis,
        root_code_file=args.root_code,
        children=tuple(args.children),
        run_id=args.run_id,
        data_dir=args.data_dir,
        desc_file=args.desc_file,
        logs_dir=args.logs_dir,
        workspaces_dir=args.workspaces_dir,
        repo_root=args.repo_root,
        prepare_workspace=not args.no_workspace,
    )
    print(f"Created generated branch run: {result.run_id}")
    print(f"Log directory: {result.log_dir}")
    print(f"Workspace directory: {result.workspace_dir}")
    print(
        "Queued first-layer children: "
        f"{result.root_hypothesis} -> {', '.join(result.children)}"
    )


if __name__ == "__main__":
    main()
