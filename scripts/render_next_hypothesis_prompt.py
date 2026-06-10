from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from aide.journal import Journal
from aide.research import (
    REPO_ROOT,
    _manual_library_dir,
    _next_hypothesis_number,
    build_research_prompt,
    collect_research_context,
    count_scored_working_nodes,
)
from aide.utils import serialize
from aide.utils.config import Config, load_cfg, load_task_desc
from aide.utils.path_portability import sanitize_persisted_payload, sanitize_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the prompt that would be used to generate the next single "
            "research hypothesis, without calling an LLM or persisting a hypothesis."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "autogluon"],
        default=None,
        help="Agent mode to render for. Defaults to config/env.",
    )
    parser.add_argument(
        "--aux",
        default=None,
        help=(
            "Auxiliary-data setting, e.g. false, true, merged, or a CSV filename. "
            "Defaults to config/env."
        ),
    )
    parser.add_argument(
        "--gpu",
        choices=["true", "false"],
        default=None,
        help="GPU setting for the rendered config snapshot. Defaults to config/env.",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        help="Optional journal.json whose scored nodes should be included in context.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "/tmp/aideml-next-hypothesis-<task>-<next-id>."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra AIDE config overrides, e.g. data_dir=... desc_file=...",
    )
    return parser.parse_args(argv)


def _load_journal(path: Path | None) -> Journal:
    if path is None:
        return Journal()
    return serialize.load_json(path, Journal)


def _cfg_overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(args.overrides)
    if args.mode is not None:
        overrides.append(f"agent.mode={args.mode}")
    if args.aux is not None:
        overrides.append(f"agent.aux={args.aux}")
    if args.gpu is not None:
        overrides.append(f"agent.gpu={args.gpu}")
    overrides.extend(
        [
            "agent.hypotheses=1",
            "research.materialize=false",
            "research.execute=false",
            "generate_report=false",
        ]
    )
    return overrides


def _task_slug(cfg: Config) -> str:
    return Path(cfg.data_dir).name


def _next_hypothesis_id(cfg: Config) -> str:
    source_dir = _manual_library_dir(cfg, repo_root=REPO_ROOT)
    return f"{_next_hypothesis_number(source_dir):06d}"


def _default_out_dir(cfg: Config, hypothesis_id: str) -> Path:
    return Path("/tmp") / f"aideml-next-hypothesis-{_task_slug(cfg)}-{hypothesis_id}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            sanitize_persisted_payload(payload),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_cfg(cli_args=_cfg_overrides(args))
        journal = _load_journal(args.journal)
        task_desc = load_task_desc(cfg)
        completed_steps = count_scored_working_nodes(journal)
        hypothesis_id = _next_hypothesis_id(cfg)
        context = collect_research_context(
            cfg=cfg,
            task_desc=task_desc,
            journal=journal,
            completed_steps=completed_steps,
        )
        context["hypothesis_count"] = 1
        prompt = build_research_prompt(context)

        out_dir = args.out_dir or _default_out_dir(cfg, hypothesis_id)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        request_payload = {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "dry_run": True,
            "would_create_hypothesis_id": hypothesis_id,
            "cfg_log_dir": str(cfg.log_dir),
            "cfg_workspace_dir": str(cfg.workspace_dir),
            "journal_path": str(args.journal) if args.journal is not None else None,
            "completed_steps": completed_steps,
            "context": context,
            "prompt": prompt,
        }
        (out_dir / "request.md").write_text(sanitize_text(prompt), encoding="utf-8")
        _write_json(out_dir / "request.json", request_payload)
        print(out_dir)
        print(out_dir / "request.md")
        print(out_dir / "request.json")
        return 0
    except Exception as exc:
        print(f"Prompt render failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
