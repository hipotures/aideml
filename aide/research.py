"""External Codex research advisor for long AIDE runs."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config

RESEARCH_PROMPT_INTRO = (
    "You are a research scientist and Kaggle competition strategist. Your job "
    "is to investigate this machine learning problem using live web search, "
    "compare public techniques and adjacent competition patterns, and propose "
    "concise, testable hypotheses for the existing AIDE code-search agent. Do "
    "not write a full solution script. Return only structured JSON matching "
    "the provided schema."
)


RESEARCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "hypotheses": {
            "type": "array",
                "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "implementation_hint": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "risk": {"type": "string"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "title",
                    "rationale",
                    "implementation_hint",
                    "expected_effect",
                    "risk",
                    "sources",
                ],
            },
        },
    },
    "required": ["summary", "hypotheses"],
}


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(node: Node) -> float | None:
    return None if node.metric is None else node.metric.value


def count_scored_working_nodes(journal: Journal) -> int:
    return sum(1 for node in journal.good_nodes if _metric_value(node) is not None)


def _compact_prompt_text(value: Any, max_chars: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_data_overview(cfg: Config) -> str | None:
    for base_dir in [Path(cfg.workspace_dir), Path(cfg.data_dir)]:
        if base_dir.exists():
            try:
                return data_preview.generate(base_dir)
            except Exception:  # noqa: BLE001 - research should not stop the run
                continue
    return None


def _node_payload(node: Node) -> dict[str, Any]:
    return {
        "metric": _metric_value(node),
        "code": node.code,
    }


def _sort_best(nodes: list[Node]) -> list[Node]:
    return sorted(nodes, key=lambda n: n.metric, reverse=True)


def _sort_worst(nodes: list[Node]) -> list[Node]:
    return sorted(nodes, key=lambda n: n.metric)


def _checkpoint_name(completed_steps: int) -> str:
    return f"checkpoint-{completed_steps:06d}"


def _checkpoint_label(checkpoint: Path) -> str:
    return checkpoint.name.removeprefix("checkpoint-")


def checkpoint_dir_for(cfg: Config, completed_steps: int) -> Path:
    return Path(cfg.log_dir) / "research" / _checkpoint_name(completed_steps)


def collect_research_context(
    *,
    cfg: Config,
    task_desc: Any,
    journal: Journal,
    completed_steps: int,
) -> dict[str, Any]:
    good_nodes = [
        node for node in journal.good_nodes if _metric_value(node) is not None
    ]
    top_best = _sort_best(good_nodes)[: cfg.research.top_k_best]
    top_best_ids = {node.id for node in top_best}
    worst_candidates = [
        node for node in _sort_worst(good_nodes) if node.id not in top_best_ids
    ]
    top_worst = worst_candidates[: cfg.research.top_k_worst]
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
        "best_working_solutions": [_node_payload(node) for node in top_best],
        "worst_working_solutions": [_node_payload(node) for node in top_worst],
    }


def build_research_prompt(context: dict[str, Any]) -> str:
    prompt_context = {
        key: context.get(key)
        for key in [
            "task_desc",
            "data_overview",
            "metric_direction",
            "best_working_solutions",
            "worst_working_solutions",
        ]
        if key in context and context.get(key) is not None
    }
    context_json = json.dumps(
        prompt_context, indent=2, ensure_ascii=False, default=_json_default
    )
    return (
        f"{RESEARCH_PROMPT_INTRO}\n\n"
        "# Research task\n"
        "Use live web search to identify techniques, validation traps, feature "
        "engineering ideas, model families, and ensemble/calibration strategies "
        "that are relevant to this competition or closely related machine "
        "learning problems.\n\n"
        "# Output contract\n"
        "Return exactly 5 concise new solution ideas. Do not target a specific "
        "previous node or code block. Use the prior results only to avoid "
        "repeating approaches that have already been tried. Do not debug broken "
        "code.\n\n"
        "# Context field meanings\n"
        "best_working_solutions contains the highest-scoring code snippets that "
        "ran successfully. worst_working_solutions contains the lowest-scoring "
        "code snippets that still ran successfully. Each solution has a numeric "
        "validation metric and code. Use these examples only to understand what "
        "has already been tried and what performed well or poorly.\n\n"
        "# Required JSON output shape\n"
        "Return JSON with: summary; hypotheses[].title; hypotheses[].rationale; "
        "hypotheses[].implementation_hint; hypotheses[].expected_effect; "
        "hypotheses[].risk; hypotheses[].sources. The hypotheses array must "
        "contain exactly 5 items.\n\n"
        "# Compact AIDE run context\n"
        f"```json\n{context_json}\n```\n"
    )


def _codex_profile_text(model: str, reasoning_effort: str) -> str:
    return (
        f'model = "{model}"\n'
        f'model_reasoning_effort = "{reasoning_effort}"\n'
        'approval_policy = "never"\n'
        'sandbox_mode = "read-only"\n'
        "# This profile is archival. The actual invocation uses --ignore-user-config\n"
        "# plus explicit CLI overrides so no global MCP servers are loaded.\n"
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
        cfg.research.model,
        "-c",
        f'model_reasoning_effort="{cfg.research.reasoning_effort}"',
        "--output-schema",
        "schema.json",
        "--output-last-message",
        "response_raw.txt",
        "--json",
        "-",
    ]


def _parse_response(raw_response: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def run_research_checkpoint(
    *,
    cfg: Config,
    context: dict[str, Any],
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    completed_steps = int(context["checkpoint_step"])
    checkpoint_dir = checkpoint_dir_for(cfg, completed_steps)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_research_prompt(context)
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
            "model": cfg.research.model,
            "reasoning_effort": cfg.research.reasoning_effort,
            "prompt": prompt,
        },
    )
    _write_json(checkpoint_dir / "schema.json", RESEARCH_RESPONSE_SCHEMA)
    (checkpoint_dir / "codex_profile.toml").write_text(
        _codex_profile_text(cfg.research.model, cfg.research.reasoning_effort),
        encoding="utf-8",
    )

    exit_code: int | None = None
    stderr = ""
    stdout = ""
    error: str | None = None
    try:
        completed = runner(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=cfg.research.timeout,
            cwd=checkpoint_dir,
        )
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        error = f"Codex research timed out after {cfg.research.timeout} seconds."
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
    parsed_response = _parse_response(raw_response)
    status = "completed" if exit_code == 0 and parsed_response is not None else "failed"
    if error is not None:
        status = "failed"
    if error is None and exit_code not in (None, 0):
        error = f"Codex exited with status {exit_code}."
    if error is None and parsed_response is None:
        error = "Codex response was not valid JSON."

    duration = time.monotonic() - started_at
    response_payload = {
        "status": status,
        "run_id": cfg.exp_name,
        "checkpoint_step": completed_steps,
        "exit_code": exit_code,
        "duration_seconds": duration,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
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


def _checkpoint_status(path: Path) -> str | None:
    status_path = path / "status.json"
    if not status_path.exists():
        return None
    try:
        status = _read_json(status_path)
    except json.JSONDecodeError:
        return "unknown"
    return status.get("status") if isinstance(status, dict) else "unknown"


def _latest_checkpoint_with_status(log_dir: Path | str) -> tuple[Path, str] | None:
    research_dir = Path(log_dir) / "research"
    if not research_dir.exists():
        return None

    candidates: list[tuple[Path, str]] = []
    for checkpoint in sorted(research_dir.glob("checkpoint-*")):
        status = _checkpoint_status(checkpoint)
        if status is not None:
            candidates.append((checkpoint, status))
    return candidates[-1] if candidates else None


def load_latest_research_hints(log_dir: Path | str) -> dict[str, Any] | None:
    research_dir = Path(log_dir) / "research"
    if not research_dir.exists():
        return None
    completed: list[Path] = []
    for checkpoint in sorted(research_dir.glob("checkpoint-*")):
        if _checkpoint_status(checkpoint) == "completed":
            completed.append(checkpoint)
    if not completed:
        return None

    latest = completed[-1]
    response = _read_json(latest / "response.json")
    parsed = response.get("parsed_response", {})
    if not isinstance(parsed, dict):
        return None
    return {
        "checkpoint": latest.name,
        "summary": parsed.get("summary", ""),
        "hypotheses": parsed.get("hypotheses", []),
    }


def format_research_hints_for_prompt(hints: dict[str, Any]) -> str:
    checkpoint = str(hints.get("checkpoint", "")).removeprefix("checkpoint-")
    lines = [
        "Use these external Codex research hints only when relevant.",
        "Treat them as hypotheses to test, not as proven facts.",
    ]
    if checkpoint:
        lines.append(f"Research checkpoint: {checkpoint}")

    summary = _compact_prompt_text(hints.get("summary"), max_chars=800)
    if summary:
        lines.extend(["", f"Summary: {summary}"])

    hypotheses = hints.get("hypotheses", [])
    if isinstance(hypotheses, list) and hypotheses:
        lines.extend(["", "Prioritized hypotheses:"])
        for idx, hypothesis in enumerate(hypotheses[:8], start=1):
            if not isinstance(hypothesis, dict):
                lines.append(f"{idx}. {_compact_prompt_text(hypothesis)}")
                continue

            title = _compact_prompt_text(hypothesis.get("title"), max_chars=160)
            lines.append(f"{idx}. {title}")

            rationale = _compact_prompt_text(
                hypothesis.get("rationale"), max_chars=260
            )
            if rationale:
                lines.append(f"   Why: {rationale}")

            implementation = _compact_prompt_text(
                hypothesis.get("implementation_hint"), max_chars=360
            )
            if implementation:
                lines.append(f"   Try: {implementation}")

            expected = _compact_prompt_text(
                hypothesis.get("expected_effect"), max_chars=220
            )
            if expected:
                lines.append(f"   Expected: {expected}")

            risk = _compact_prompt_text(hypothesis.get("risk"), max_chars=220)
            if risk:
                lines.append(f"   Risk: {risk}")

    return "\n".join(lines)


class ResearchAdvisor:
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
        self._threads: list[threading.Thread] = []

    def maybe_start(self, *, journal: Journal, completed_steps: int) -> bool:
        if not self.cfg.research.enabled:
            return False
        if self.cfg.research.every_steps <= 0:
            return False
        if completed_steps <= 0 or completed_steps % self.cfg.research.every_steps != 0:
            return False

        checkpoint_dir = checkpoint_dir_for(self.cfg, completed_steps)
        if _checkpoint_status(checkpoint_dir) is not None:
            return False

        context = collect_research_context(
            cfg=self.cfg,
            task_desc=self.task_desc,
            journal=journal,
            completed_steps=completed_steps,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            checkpoint_dir / "status.json",
            {
                "status": "queued",
                "run_id": self.cfg.exp_name,
                "checkpoint_step": completed_steps,
                "queued_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
        )
        thread = threading.Thread(
            target=run_research_checkpoint,
            kwargs={"cfg": self.cfg, "context": context, "runner": self.runner},
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)
        return True

    def status_text(self) -> str:
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        latest_checkpoint = _latest_checkpoint_with_status(self.cfg.log_dir)
        latest_name = (
            _checkpoint_label(latest_checkpoint[0])
            if latest_checkpoint is not None
            else None
        )
        if self._threads:
            suffix = f" {latest_name}" if latest_name is not None else ""
            return f"[cyan]Research: ▶{suffix}"

        latest = load_latest_research_hints(self.cfg.log_dir)
        if latest is not None:
            return f"[green]Research: ✓ {latest['checkpoint'].removeprefix('checkpoint-')}"

        if latest_checkpoint is not None:
            checkpoint, status = latest_checkpoint
            if status in {"queued", "running"}:
                icon = "…" if status == "queued" else "▶"
                return f"[cyan]Research: {icon} {_checkpoint_label(checkpoint)}"
            if status == "failed":
                return f"[red]Research: ✗ {_checkpoint_label(checkpoint)}"
            return f"[yellow]Research: ? {_checkpoint_label(checkpoint)}"
        return "[dim]Research: ○"
