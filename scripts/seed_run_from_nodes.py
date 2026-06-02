#!/usr/bin/env python3
"""Create a resumable AIDE run seeded from scored nodes in another run."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from aide.journal import Journal, Node
from aide.utils import serialize
from aide.utils.artifact_manifest import RESULT_MANIFEST_NAME
from aide.utils.config import _load_cfg, prep_agent_workspace, prep_cfg, save_run
from aide.utils.node_artifacts import node_artifact_dir
from aide.utils.seed_artifact import (
    SeedArtifactSource,
    SeededArtifactNode,
    find_seed_artifact,
    seed_artifact_source_from_manifest,
    seed_journal_from_artifacts,
    source_is_autogluon,
)


RUN_ID_RE = re.compile(r"^\d+-[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class SeedRunResult:
    run_id: str
    log_dir: Path
    workspace_dir: Path
    seeded: tuple[SeededArtifactNode, ...]


def _cli_sets_key(cli_overrides: Iterable[str], key: str) -> bool:
    prefix = f"{key}="
    return any(arg == key or arg.startswith(prefix) for arg in cli_overrides)


def _split_values(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _split_ints(raw: str | None) -> tuple[int, ...]:
    values = []
    for value in _split_values(raw):
        try:
            values.append(int(value))
        except ValueError as exc:
            raise ValueError(f"Invalid integer step: {value!r}") from exc
    return tuple(values)


def _validate_run_id(run_id: str | None) -> None:
    if run_id is None:
        return
    if "/" in run_id or "\\" in run_id:
        raise ValueError("run id must be a plain directory name, not a path.")
    if not RUN_ID_RE.match(run_id):
        raise ValueError("run id must look like an AIDE run id, for example 2-seeded-run.")


def _source_journal(logs_dir: Path, source_run: str) -> tuple[Path, Journal]:
    log_dir = logs_dir / source_run
    journal_path = log_dir / "journal.json"
    if not journal_path.exists():
        raise FileNotFoundError(f"Missing source journal: {journal_path}")
    return log_dir, serialize.load_json(journal_path, Journal)


def _is_seedable_node(node: Node) -> bool:
    return (
        node.metric is not None
        and node.metric.value is not None
        and not bool(node.is_buggy)
        and node.status != "generated"
    )


def _manifest_path_for_node(source_log_dir: Path, node: Node) -> Path:
    artifact_dir = node_artifact_dir(source_log_dir, node)
    manifest_path = artifact_dir / RESULT_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Node step={node.step} id={node.id} has no artifact manifest: {manifest_path}"
        )
    return manifest_path


def _source_for_node(source_log_dir: Path, node: Node) -> SeedArtifactSource:
    if not _is_seedable_node(node):
        raise ValueError(
            f"Node step={node.step} id={node.id} is not a scored non-buggy node."
        )
    return seed_artifact_source_from_manifest(
        _manifest_path_for_node(source_log_dir, node),
        matched_kind="node",
    )


def _find_unique_node_by_prefix(journal: Journal, prefix: str) -> Node:
    matches = [node for node in journal.nodes if str(node.id).startswith(prefix)]
    if not matches:
        raise ValueError(f"No source node id matches prefix {prefix!r}.")
    if len(matches) > 1:
        ids = ", ".join(node.id for node in matches[:8])
        raise ValueError(f"Source node id prefix {prefix!r} is ambiguous: {ids}")
    return matches[0]


def _select_nodes_by_steps(journal: Journal, steps: tuple[int, ...]) -> list[Node]:
    by_step = {node.step: node for node in journal.nodes}
    missing = [step for step in steps if step not in by_step]
    if missing:
        raise ValueError("Unknown source step(s): " + ", ".join(map(str, missing)))
    return [by_step[step] for step in steps]


def _select_top_nodes(journal: Journal, top_n: int) -> list[Node]:
    if top_n <= 0:
        raise ValueError("--top-n must be greater than zero.")
    candidates = [node for node in journal.nodes if _is_seedable_node(node)]
    if len(candidates) < top_n:
        raise ValueError(
            f"Requested top {top_n} source nodes, but only {len(candidates)} scored "
            "non-buggy node(s) are available."
        )
    return sorted(candidates, key=lambda node: node.metric, reverse=True)[:top_n]


def _dedupe_sources(sources: list[SeedArtifactSource]) -> list[SeedArtifactSource]:
    seen: set[Path] = set()
    deduped: list[SeedArtifactSource] = []
    for source in sources:
        key = source.manifest_path.resolve()
        if key in seen:
            raise ValueError(f"Duplicate selected source artifact: {source.artifact_dir}")
        seen.add(key)
        deduped.append(source)
    return deduped


def select_seed_sources(
    *,
    logs_dir: Path,
    source_run: str,
    top_n: int | None = None,
    steps: tuple[int, ...] = (),
    node_ids: tuple[str, ...] = (),
    shas: tuple[str, ...] = (),
) -> list[SeedArtifactSource]:
    modes = [
        top_n is not None,
        bool(steps),
        bool(node_ids),
        bool(shas),
    ]
    if sum(1 for mode in modes if mode) != 1:
        raise ValueError("Specify exactly one of --top-n, --steps, --node-ids, or --shas.")

    logs_dir = logs_dir.resolve()
    source_log_dir, journal = _source_journal(logs_dir, source_run)

    if top_n is not None:
        nodes = _select_top_nodes(journal, top_n)
        return _dedupe_sources([_source_for_node(source_log_dir, node) for node in nodes])

    if steps:
        nodes = _select_nodes_by_steps(journal, steps)
        return _dedupe_sources([_source_for_node(source_log_dir, node) for node in nodes])

    if node_ids:
        nodes = [_find_unique_node_by_prefix(journal, prefix) for prefix in node_ids]
        return _dedupe_sources([_source_for_node(source_log_dir, node) for node in nodes])

    sources = [
        find_seed_artifact(logs_dir, sha_prefix, source_run=source_run)
        for sha_prefix in shas
    ]
    return _dedupe_sources(sources)


def seed_run_from_nodes(
    *,
    source_run: str,
    cli_overrides: list[str],
    top_n: int | None = None,
    steps: tuple[int, ...] = (),
    node_ids: tuple[str, ...] = (),
    shas: tuple[str, ...] = (),
    run_id: str | None = None,
    prepare_workspace: bool = True,
) -> SeedRunResult:
    _validate_run_id(source_run)
    _validate_run_id(run_id)
    if any(arg == "--resume" or arg.startswith("--resume=") for arg in cli_overrides):
        raise ValueError("Do not pass --resume to the seed script; resume the created run afterwards.")

    cfg = prep_cfg(_load_cfg(cli_args=cli_overrides))
    logs_dir = Path(cfg.log_dir).parent
    workspaces_dir = Path(cfg.workspace_dir).parent

    sources = select_seed_sources(
        logs_dir=logs_dir,
        source_run=source_run,
        top_n=top_n,
        steps=steps,
        node_ids=node_ids,
        shas=shas,
    )

    if run_id is not None:
        cfg.exp_name = run_id
        cfg.log_dir = logs_dir / run_id
        cfg.workspace_dir = workspaces_dir / run_id

    log_dir = Path(cfg.log_dir)
    workspace_dir = Path(cfg.workspace_dir)
    if log_dir.exists():
        raise FileExistsError(f"Run log directory already exists: {log_dir}")
    if workspace_dir.exists():
        raise FileExistsError(f"Run workspace directory already exists: {workspace_dir}")

    if not _cli_sets_key(cli_overrides, "agent.search.num_drafts"):
        cfg.agent.search.num_drafts = len(sources)
    if any(source_is_autogluon(source) for source in sources) and not _cli_sets_key(
        cli_overrides,
        "agent.mode",
    ):
        cfg.agent.mode = "autogluon_preprocess"

    if prepare_workspace:
        prep_agent_workspace(cfg)
    else:
        (workspace_dir / "input").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    journal, seeded = seed_journal_from_artifacts(cfg, sources)
    save_run(cfg, journal)

    return SeedRunResult(
        run_id=str(cfg.exp_name),
        log_dir=log_dir,
        workspace_dir=workspace_dir,
        seeded=tuple(seeded),
    )


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new AIDE run whose root nodes are copied scored artifacts "
            "from another run. Extra arguments are passed as AIDE/OmegaConf overrides."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--steps", help="Comma-separated source journal steps.")
    parser.add_argument("--node-ids", help="Comma-separated source node id prefixes.")
    parser.add_argument("--shas", help="Comma-separated solution/submission SHA prefixes.")
    parser.add_argument(
        "--no-prepare-workspace",
        action="store_true",
        help="Create empty input/working directories instead of preparing data.",
    )
    return parser.parse_known_args(argv)


def _format_seeded_line(item: SeededArtifactNode) -> str:
    source_node = item.source.node_payload
    score = item.node.metric.value if item.node.metric is not None else None
    return (
        f"  step {item.node.step}: {item.node.id} "
        f"score={score} "
        f"source={item.source.run_id}:"
        f"{source_node.get('step') if source_node.get('step') is not None else '?'} "
        f"source_node={source_node.get('id') or '?'} "
        f"artifact={item.artifact_dir.name}"
    )


def main(argv: list[str] | None = None) -> int:
    args, cli_overrides = parse_args(argv)
    result = seed_run_from_nodes(
        source_run=args.source_run,
        run_id=args.run_id,
        top_n=args.top_n,
        steps=_split_ints(args.steps),
        node_ids=_split_values(args.node_ids),
        shas=_split_values(args.shas),
        cli_overrides=cli_overrides,
        prepare_workspace=not args.no_prepare_workspace,
    )

    print(f"Created run: {result.run_id}")
    print(f"Log dir: {result.log_dir}")
    print(f"Workspace dir: {result.workspace_dir}")
    print("Copied nodes:")
    for item in result.seeded:
        print(_format_seeded_line(item))
    print(f"Resume with: uv run aide --resume {result.run_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Seed run failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
