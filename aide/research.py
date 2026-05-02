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
                    "target": {"type": "string", "enum": ["root", "node"]},
                    "parent_node_id": {"type": ["string", "null"]},
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
                    "target",
                    "parent_node_id",
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


def _node_payload(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "step": node.step,
        "ctime": node.ctime,
        "stage": node.stage_name,
        "parent_id": node.parent.id if node.parent is not None else None,
        "is_buggy": node.is_buggy,
        "metric": _metric_value(node),
        "plan": node.plan,
        "analysis": node.analysis,
        "exec_time": node.exec_time,
        "exc_type": node.exc_type,
        "exc_info": node.exc_info,
        "terminal_output": node.term_out if node._term_out is not None else "",
        "code": node.code,
    }


def _ordered_unique(nodes: list[Node]) -> list[Node]:
    seen: set[str] = set()
    out: list[Node] = []
    for node in nodes:
        if node.id in seen:
            continue
        seen.add(node.id)
        out.append(node)
    return out


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
    buggy_nodes = list(journal.buggy_nodes)

    top_best = _sort_best(good_nodes)[: cfg.research.top_k_best]
    low_scoring_nodes = _sort_worst(good_nodes)
    top_worst = _ordered_unique(buggy_nodes + low_scoring_nodes)[
        : cfg.research.top_k_worst
    ]
    top_worst_ids = {node.id for node in top_worst}
    selected_nodes = _ordered_unique(
        [node for node in top_best if node.id not in top_worst_ids] + top_worst
    )

    best_node = journal.get_best_node()
    return {
        "run_id": cfg.exp_name,
        "checkpoint_step": completed_steps,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "task_desc": task_desc,
        "metric_direction": (
            None
            if best_node is None or best_node.metric is None
            else ("maximize" if best_node.metric.maximize else "minimize")
        ),
        "best_node_id": None if best_node is None else best_node.id,
        "selected_node_ids": [node.id for node in selected_nodes],
        "top_best_nodes": [_node_payload(node) for node in top_best],
        "top_worst_nodes": [_node_payload(node) for node in top_worst],
        "recent_nodes": [_node_payload(node) for node in journal.nodes[-10:]],
    }


def build_research_prompt(context: dict[str, Any]) -> str:
    context_json = json.dumps(
        context, indent=2, ensure_ascii=False, default=_json_default
    )
    schema_json = json.dumps(RESEARCH_RESPONSE_SCHEMA, indent=2, ensure_ascii=False)
    return (
        f"{RESEARCH_PROMPT_INTRO}\n\n"
        "# Research task\n"
        "Use live web search to identify techniques, validation traps, feature "
        "engineering ideas, model families, and ensemble/calibration strategies "
        "that are relevant to this competition or closely related machine "
        "learning problems.\n\n"
        "# Output contract\n"
        "Return concise hypotheses only. Each hypothesis must be actionable by "
        "the existing AIDE agent in one future node. Use target=root for a fresh "
        "approach, or target=node with parent_node_id when the idea should extend "
        "a specific existing node.\n\n"
        "# JSON schema\n"
        f"```json\n{schema_json}\n```\n\n"
        "# AIDE run context\n"
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
        "exec",
        "--ignore-user-config",
        "--search",
        "--ask-for-approval",
        "never",
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
            "selected_node_ids": context.get("selected_node_ids", []),
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
