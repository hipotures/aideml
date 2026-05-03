"""External Codex solution synthesis for long AIDE runs."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .journal import Journal, Node
from .research import (
    _checkpoint_label,
    _checkpoint_name,
    _checkpoint_status,
    _codex_profile_text,
    _json_default,
    _metric_value,
    _read_json,
    _write_json,
    build_data_overview,
)
from .utils import serialize
from .utils.config import Config
from .utils.response import extract_code, is_valid_python_script

SYNTHESIS_PROMPT_INTRO = (
    "You are a Kaggle grandmaster and senior machine learning engineer. Your "
    "job is to use live web search when useful, study the strongest successful "
    "AIDE solution scripts, and produce one coherent Python solution that "
    "combines the best compatible ideas. Return only the Python code."
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SynthesisNode:
    node: Node
    completed_steps: int
    checkpoint_dir: Path


def checkpoint_dir_for(cfg: Config, completed_steps: int) -> Path:
    return Path(cfg.log_dir) / "synthesis" / _checkpoint_name(completed_steps)


def _synthesis_dir(log_dir: Path | str) -> Path:
    return Path(log_dir) / "synthesis"


def _load_journal(path: Path) -> Journal | None:
    try:
        return serialize.load_json(path, Journal)
    except Exception:  # noqa: BLE001 - stale runs must not stop synthesis
        return None


def _candidate_run_ids(cfg: Config) -> list[str]:
    source_runs = list(cfg.synthesis.source_runs or [])
    if source_runs:
        return source_runs

    top_log_dir = Path(cfg.log_dir).resolve().parent
    if not top_log_dir.exists():
        return [cfg.exp_name]

    run_ids = sorted(path.parent.name for path in top_log_dir.glob("*/journal.json"))
    if cfg.exp_name not in run_ids:
        run_ids.append(cfg.exp_name)
    return run_ids


def _journal_for_run(
    *, cfg: Config, current_journal: Journal, run_id: str
) -> Journal | None:
    if run_id == cfg.exp_name:
        return current_journal
    return _load_journal(Path(cfg.log_dir).resolve().parent / run_id / "journal.json")


def _working_nodes_with_metrics(journal: Journal) -> list[Node]:
    return [
        node
        for node in journal.good_nodes
        if node.metric is not None and _metric_value(node) is not None
    ]


def collect_top_synthesis_solutions(
    *,
    cfg: Config,
    journal: Journal,
) -> list[dict[str, Any]]:
    selected: list[tuple[str, Node]] = []
    for run_id in _candidate_run_ids(cfg):
        source_journal = _journal_for_run(
            cfg=cfg,
            current_journal=journal,
            run_id=run_id,
        )
        if source_journal is None:
            continue
        selected.extend(
            (run_id, node) for node in _working_nodes_with_metrics(source_journal)
        )

    selected.sort(key=lambda item: item[1].metric, reverse=True)
    return [
        {
            "metric": _metric_value(node),
            "code": node.code,
        }
        for _run_id, node in selected[: cfg.synthesis.top_k]
    ]


def collect_synthesis_context(
    *,
    cfg: Config,
    task_desc: Any,
    journal: Journal,
    completed_steps: int,
) -> dict[str, Any]:
    best_node = journal.get_best_node()
    return {
        "run_id": cfg.exp_name,
        "checkpoint_step": completed_steps,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "task_desc": task_desc,
        "data_overview": build_data_overview(cfg),
        "metric_direction": (
            None
            if best_node is None or best_node.metric is None
            else ("maximize" if best_node.metric.maximize else "minimize")
        ),
        "best_working_solutions": collect_top_synthesis_solutions(
            cfg=cfg,
            journal=journal,
        ),
    }


def build_synthesis_prompt(context: dict[str, Any]) -> str:
    prompt_context = {
        key: context.get(key)
        for key in [
            "task_desc",
            "data_overview",
            "metric_direction",
            "best_working_solutions",
        ]
        if key in context and context.get(key) is not None
    }
    context_json = json.dumps(
        prompt_context, indent=2, ensure_ascii=False, default=_json_default
    )
    return (
        f"{SYNTHESIS_PROMPT_INTRO}\n\n"
        "# Synthesis task\n"
        "Create one complete, self-contained Python script. Combine compatible "
        "high-performing ideas from the provided successful solutions, but do "
        "not merely paste them together. You may use live web search to check "
        "competition-specific or closely related modeling tactics before "
        "writing the script.\n\n"
        "# Output contract\n"
        "Return only Python code. Do not include markdown fences, prose, JSON, "
        "explanations, titles, or comments outside the code. The code must read "
        "all inputs from ./input, print a hold-out or cross-validation metric, "
        "and save the final test predictions as ./working/submission.csv when "
        "test data exists. The script must be executable as a single file within "
        "the configured AIDE timeout.\n\n"
        "# Context field meanings\n"
        "best_working_solutions contains the highest-scoring scripts that ran "
        "successfully. Each solution has only its numeric validation metric and "
        "code. Use these examples to identify strong modeling and feature "
        "engineering patterns to merge into a better root solution.\n\n"
        "# Compact AIDE synthesis context\n"
        f"```json\n{context_json}\n```\n"
    )


def _codex_command(cfg: Config, checkpoint_dir: Path) -> list[str]:
    return [
        "codex",
        "--search",
        "--ask-for-approval",
        "never",
        "exec",
        "--ignore-user-config",
        "--sandbox",
        "read-only",
        "--cd",
        str(checkpoint_dir),
        "--model",
        cfg.synthesis.model,
        "-c",
        f'model_reasoning_effort="{cfg.synthesis.reasoning_effort}"',
        "--output-last-message",
        "response_raw.txt",
        "--json",
        "-",
    ]


def parse_synthesis_code(raw_response: str) -> str:
    code = extract_code(raw_response)
    if code and is_valid_python_script(code):
        return code

    stripped = raw_response.strip()
    if stripped and is_valid_python_script(stripped):
        return stripped
    raise ValueError("Codex synthesis response did not contain valid Python code.")


def run_synthesis_checkpoint(
    *,
    cfg: Config,
    context: dict[str, Any],
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    completed_steps = int(context["checkpoint_step"])
    checkpoint_dir = checkpoint_dir_for(cfg, completed_steps)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_synthesis_prompt(context)
    command = _codex_command(cfg, checkpoint_dir)
    started_at = time.monotonic()

    _write_json(
        checkpoint_dir / "status.json",
        {
            "status": "running",
            "run_id": cfg.exp_name,
            "checkpoint_step": completed_steps,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        },
    )
    _write_json(checkpoint_dir / "context.json", context)
    (checkpoint_dir / "request.md").write_text(prompt, encoding="utf-8")
    _write_json(
        checkpoint_dir / "request.json",
        {
            "run_id": cfg.exp_name,
            "checkpoint_step": completed_steps,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "command": command,
            "model": cfg.synthesis.model,
            "reasoning_effort": cfg.synthesis.reasoning_effort,
            "prompt": prompt,
        },
    )
    (checkpoint_dir / "codex_profile.toml").write_text(
        _codex_profile_text(cfg.synthesis.model, cfg.synthesis.reasoning_effort),
        encoding="utf-8",
    )

    exit_code: int | None = None
    stderr = ""
    stdout = ""
    error: str | None = None
    code: str | None = None
    try:
        completed = runner(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=cfg.synthesis.timeout,
            cwd=checkpoint_dir,
        )
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        error = f"Codex synthesis timed out after {cfg.synthesis.timeout} seconds."
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    except Exception as exc:  # noqa: BLE001 - external tool failure must not stop AIDE
        error = f"{exc.__class__.__name__}: {exc}"

    (checkpoint_dir / "codex_events.jsonl").write_text(stdout, encoding="utf-8")
    (checkpoint_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    raw_response_path = checkpoint_dir / "response_raw.txt"
    raw_response = (
        raw_response_path.read_text(encoding="utf-8")
        if raw_response_path.exists()
        else ""
    )

    if error is None and exit_code not in (None, 0):
        error = f"Codex exited with status {exit_code}."
    if error is None:
        try:
            code = parse_synthesis_code(raw_response)
            (checkpoint_dir / "response.py").write_text(code, encoding="utf-8")
        except ValueError as exc:
            error = str(exc)

    status = "ready" if error is None and code is not None else "failed"
    duration = time.monotonic() - started_at
    response_payload = {
        "status": status,
        "run_id": cfg.exp_name,
        "checkpoint_step": completed_steps,
        "exit_code": exit_code,
        "duration_seconds": duration,
        "raw_response": raw_response,
        "code": code,
        "stderr": stderr,
        "error": error,
    }
    _write_json(checkpoint_dir / "response.json", response_payload)
    _write_json(
        checkpoint_dir / "status.json",
        {
            "status": status,
            "run_id": cfg.exp_name,
            "checkpoint_step": completed_steps,
            "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": duration,
            "exit_code": exit_code,
            "error": error,
        },
    )
    return {
        "status": status,
        "checkpoint_dir": str(checkpoint_dir),
        "response": response_payload,
    }


def _latest_checkpoint_with_status(log_dir: Path | str) -> tuple[Path, str] | None:
    synthesis_dir = _synthesis_dir(log_dir)
    if not synthesis_dir.exists():
        return None

    candidates: list[tuple[Path, str]] = []
    for checkpoint in sorted(synthesis_dir.glob("checkpoint-*")):
        status = _checkpoint_status(checkpoint)
        if status is not None:
            candidates.append((checkpoint, status))
    return candidates[-1] if candidates else None


class SynthesisAdvisor:
    def __init__(
        self,
        *,
        cfg: Config,
        task_desc: Any,
        runner: Runner = subprocess.run,
    ):
        self.cfg = cfg
        self.task_desc = task_desc
        self.runner = runner
        self._active_checkpoint: int | None = None

    def _is_due(self, completed_steps: int) -> bool:
        return (
            self.cfg.synthesis.enabled
            and self.cfg.synthesis.every_scored_steps > 0
            and completed_steps > 0
            and completed_steps % self.cfg.synthesis.every_scored_steps == 0
        )

    def generate_node_if_due(
        self,
        *,
        journal: Journal,
        completed_steps: int,
    ) -> SynthesisNode | None:
        if not self._is_due(completed_steps):
            return None

        checkpoint_dir = checkpoint_dir_for(self.cfg, completed_steps)
        status = _checkpoint_status(checkpoint_dir)
        if status in {"injected", "failed"}:
            return None
        if status == "ready":
            code = (checkpoint_dir / "response.py").read_text(encoding="utf-8")
        elif status is None:
            self._active_checkpoint = completed_steps
            try:
                context = collect_synthesis_context(
                    cfg=self.cfg,
                    task_desc=self.task_desc,
                    journal=journal,
                    completed_steps=completed_steps,
                )
                result = run_synthesis_checkpoint(
                    cfg=self.cfg,
                    context=context,
                    runner=self.runner,
                )
            finally:
                self._active_checkpoint = None
            if result["status"] != "ready":
                return None
            code = result["response"]["code"]
        else:
            return None

        return SynthesisNode(
            node=Node(
                plan=f"External Codex synthesis checkpoint {completed_steps:06d}",
                code=code,
            ),
            completed_steps=completed_steps,
            checkpoint_dir=checkpoint_dir,
        )

    def mark_injected(self, synthesis_node: SynthesisNode, *, node: Node) -> None:
        status = _read_json(synthesis_node.checkpoint_dir / "status.json")
        if not isinstance(status, dict):
            status = {}
        status.update(
            {
                "status": "injected",
                "injected_at": dt.datetime.now().isoformat(timespec="seconds"),
                "injected_node_id": node.id,
                "injected_node_step": node.step,
            }
        )
        _write_json(synthesis_node.checkpoint_dir / "status.json", status)

    def status_text(self) -> str:
        if self._active_checkpoint is not None:
            return f"[cyan]Synthesis: ▶ {self._active_checkpoint:06d}"

        latest_checkpoint = _latest_checkpoint_with_status(self.cfg.log_dir)
        if latest_checkpoint is None:
            return "[dim]Synthesis: ○"

        checkpoint, status = latest_checkpoint
        label = _checkpoint_label(checkpoint)
        if status == "running":
            return f"[cyan]Synthesis: ▶ {label}"
        if status == "ready":
            return f"[yellow]Synthesis: … {label}"
        if status == "injected":
            return f"[green]Synthesis: ✓ {label}"
        if status == "failed":
            return f"[red]Synthesis: ✗ {label}"
        return f"[yellow]Synthesis: ? {label}"
