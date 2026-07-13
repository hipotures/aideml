"""Backend for Codex app-server calls."""

import json
import tempfile
from pathlib import Path

from funcy import notnone, select_values

from .codex_app_server import CodexThreadFork, invoke_codex_app_server
from .utils import FunctionSpec, OutputType


def _prefixed(prefix: str | None, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _prompt_text(system_message: str | None, user_message: str | None) -> str:
    parts = []
    if system_message:
        parts.append(f"# System message\n\n{system_message}")
    if user_message:
        parts.append(f"# User message\n\n{user_message}")
    return "\n\n---\n\n".join(parts)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    model = filtered_kwargs["model"]
    reasoning_effort = filtered_kwargs.pop("reasoning_effort", None)
    web_search = bool(filtered_kwargs.pop("web_search", False))
    log_dir = filtered_kwargs.pop("llm_log_dir", None)
    log_prefix = filtered_kwargs.pop("llm_log_prefix", "")
    thread_id = filtered_kwargs.pop("codex_thread_id", None)
    fork_thread_id = filtered_kwargs.pop("codex_fork_thread_id", None)
    fork_turn_id = filtered_kwargs.pop("codex_fork_turn_id", None)
    fork_path = filtered_kwargs.pop("codex_fork_path", None)
    prompt = _prompt_text(system_message, user_message)

    temp_context = None
    temp_path: str | None = None
    if log_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="aide-codex-query-")
        temp_path = temp_context.__enter__()
    try:
        work_dir = Path(log_dir) if log_dir is not None else Path(temp_path)  # type: ignore[arg-type]
        work_dir.mkdir(parents=True, exist_ok=True)
        schema_path = work_dir / _prefixed(log_prefix, "schema.json")
        output_schema = func_spec.json_schema if func_spec is not None else None
        if output_schema is not None:
            schema_path.write_text(
                json.dumps(output_schema, indent=2),
                encoding="utf-8",
            )
        fork_from = (
            CodexThreadFork(
                thread_id=str(fork_thread_id),
                turn_id=str(fork_turn_id) if fork_turn_id else None,
                path=Path(fork_path) if fork_path else None,
            )
            if fork_thread_id
            else None
        )
        result = invoke_codex_app_server(
            prompt=prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
            work_dir=work_dir,
            timeout=filtered_kwargs.get("timeout"),
            output_schema=output_schema,
            log_dir=work_dir,
            log_prefix=log_prefix,
            thread_id=str(thread_id) if thread_id else None,
            fork_from=fork_from,
        )
        raw_output = result.text
    finally:
        if temp_context is not None:
            temp_context.__exit__(None, None, None)

    if func_spec is not None:
        output: OutputType = json.loads(raw_output)
    else:
        output = raw_output

    info = {
        "model": model,
        "backend": "codex",
        "provider_kind": "codex_app_server",
        "status": result.status,
        "thread_id": result.thread_id,
        "turn_id": result.turn_id,
        "usage": result.usage,
        "thread_action": "fork" if fork_from else ("resume" if thread_id else "start"),
        "source_thread_id": fork_from.thread_id if fork_from else thread_id,
        "source_turn_id": fork_from.turn_id if fork_from else None,
    }
    return (
        output,
        result.duration_seconds,
        result.input_tokens,
        result.output_tokens,
        info,
    )
