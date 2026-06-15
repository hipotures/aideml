from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path
from typing import Any

from aide.agent import Agent, _add_code_web_search_instruction
from aide.backend.utils import compile_prompt_to_md, write_llm_request_files
from aide.journal import Journal, Node
from aide.run import load_resume_state
from aide.utils.artifact_manifest import reconstruct_journal_from_artifacts
from aide.utils.config import load_task_desc


class PromptRendered(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the agent generation prompt that produced a run step without "
            "calling an LLM, executing code, or modifying the journal."
        )
    )
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Run directory, e.g. logs/3-s6e6-v15-feature-search.",
    )
    parser.add_argument(
        "--step",
        required=True,
        type=int,
        help="Existing node step whose generation prompt should be rendered.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "/tmp/aideml-agent-prompt-<run>-step-<step>."
        ),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help=(
            "Workspace root used for resume loading. Defaults to a workspaces/ "
            "directory next to the logs/ directory."
        ),
    )
    parser.add_argument(
        "--no-data-preview",
        action="store_true",
        help="Do not refresh Data Overview before rendering.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides, e.g. agent.memory_recent_steps=50.",
    )
    return parser.parse_args(argv)


def _resolve_run_dir(run: Path) -> Path:
    if run.exists():
        return run.resolve()
    candidate = Path("logs") / run
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Run directory does not exist: {run}")


def _default_workspace_root(run_dir: Path) -> Path:
    return run_dir.parent.parent / "workspaces"


def _default_out_dir(run_id: str, step: int) -> Path:
    return Path("/tmp") / f"aideml-agent-prompt-{run_id}-step-{step}"


def _find_step(journal_nodes: list[Node], step: int) -> Node:
    for node in journal_nodes:
        if node.step == step:
            return node
    raise ValueError(f"No node with step {step} in journal")


def _journal_before_step(journal: Journal, step: int) -> Journal:
    kept = [
        node
        for node in journal.nodes
        if node.step is not None and node.step < step
    ]
    kept_set = set(kept)
    for node in kept:
        node.children = {child for child in node.children if child in kept_set}
        if node.parent not in kept_set:
            node.parent = None
    return Journal(nodes=kept)


def _restore_research_metadata(reconstructed: Journal, resume_journal: Journal) -> None:
    by_id = {node.id: node for node in resume_journal.nodes}
    by_step = {
        node.step: node
        for node in resume_journal.nodes
        if node.step is not None
    }
    for node in reconstructed.nodes:
        source = by_id.get(node.id) or by_step.get(node.step)
        if source is None:
            continue
        node.research_mode = source.research_mode
        node.research_hypotheses_offered = list(source.research_hypotheses_offered)
        node.research_source_hash = source.research_source_hash
        node.research_runtime_config = source.research_runtime_config
        node.research_hypotheses_llm_claimed_used = list(
            source.research_hypotheses_llm_claimed_used
        )
        node.research_usage_note = source.research_usage_note


def _write_prompt_artifact(
    *,
    prompt: Any,
    out_dir: Path,
    run_id: str,
    target_node: Node,
    parent_node: Node | None,
    agent: Agent,
) -> None:
    system_message = compile_prompt_to_md(prompt)
    context = {
        **agent._generation_log_context(),
        "dry_run": True,
        "rendered_by": "scripts/render_next_agent_prompt.py",
        "run_id": run_id,
        "target_step": target_node.step,
        "target_node_id": target_node.id,
        "parent_step": parent_node.step if parent_node is not None else None,
        "parent_node_id": parent_node.id if parent_node is not None else None,
    }
    request_payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dry_run": True,
        "run_id": run_id,
        "target_step": target_node.step,
        "target_node_id": target_node.id,
        "parent_step": parent_node.step if parent_node is not None else None,
        "parent_node_id": parent_node.id if parent_node is not None else None,
        "system_message": system_message,
        "user_message": None,
    }
    write_llm_request_files(
        log_dir=out_dir,
        prefix=None,
        context=context,
        request_payload=request_payload,
        system_message=system_message,
        user_message=None,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = _resolve_run_dir(args.run)
        run_id = run_dir.name
        workspace_root = (
            args.workspace_root.resolve()
            if args.workspace_root is not None
            else _default_workspace_root(run_dir).resolve()
        )
        out_dir = (args.out_dir or _default_out_dir(run_id, args.step)).resolve()
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        cfg, resume_journal = load_resume_state(
            run_id=run_id,
            top_log_dir=run_dir.parent,
            top_workspace_dir=workspace_root,
            cli_overrides=list(args.overrides),
            force_check_submissions=False,
        )
        full_journal = reconstruct_journal_from_artifacts(run_dir)
        if not full_journal.nodes:
            raise ValueError(f"No artifact manifests found for run: {run_dir}")
        _restore_research_metadata(full_journal, resume_journal)
        target_node = _find_step(full_journal.nodes, args.step)
        parent_node = target_node.parent
        render_journal = _journal_before_step(full_journal, args.step)
        if parent_node is not None:
            parent_node = _find_step(render_journal.nodes, parent_node.step)

        agent = Agent(task_desc=load_task_desc(cfg), cfg=cfg, journal=render_journal)
        if not args.no_data_preview:
            agent.update_data_preview()

        def fake_plan_and_code(prompt: Any, retries: int = 3) -> tuple[str, str]:
            del retries
            if bool(getattr(agent.acfg.code, "web_search", False)):
                prompt = _add_code_web_search_instruction(prompt)
            _write_prompt_artifact(
                prompt=prompt,
                out_dir=out_dir,
                run_id=run_id,
                target_node=target_node,
                parent_node=parent_node,
                agent=agent,
            )
            raise PromptRendered

        agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

        try:
            agent.generate_node(parent_node, llm_log_dir=out_dir)
        except PromptRendered:
            pass

        print(out_dir)
        print(out_dir / "request.md")
        print(out_dir / "request.json")
        return 0
    except Exception as exc:
        print(f"Prompt render failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
