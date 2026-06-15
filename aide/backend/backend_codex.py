"""Backend for Codex CLI calls."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

from funcy import notnone, select_values

from aide.utils.path_portability import sanitize_persisted_payload

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


def _codex_command(
    *,
    model: str,
    reasoning_effort: str | None,
    web_search: bool,
    work_dir: Path,
    output_schema: bool,
    schema_path: Path,
    response_path: Path,
) -> list[str]:
    command = [
        "codex",
        "--ask-for-approval",
        "never",
        "exec",
        "--ignore-user-config",
        "--sandbox",
        "read-only",
        "--cd",
        str(work_dir),
        "--model",
        model,
    ]
    if web_search:
        command.insert(1, "--search")
    if reasoning_effort is not None:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    if output_schema:
        command.extend(["--output-schema", str(schema_path)])
    command.extend(["--output-last-message", str(response_path), "--json", "-"])
    return command


def _write_codex_profile(
    *,
    work_dir: Path,
    prefix: str | None,
    model: str,
    reasoning_effort: str | None,
    command: list[str],
) -> None:
    lines = [
        f'model = "{model}"',
        f'reasoning_effort = "{reasoning_effort or ""}"',
        'sandbox = "read-only"',
        'ask_for_approval = "never"',
        'output_mode = "json"',
        "",
        "[command]",
        "argv = [",
    ]
    profile_command = sanitize_persisted_payload(command)
    lines.extend(f'  {json.dumps(part)},' for part in profile_command)
    lines.extend(["]", ""])
    (work_dir / _prefixed(prefix, "codex_profile.toml")).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _codex_failure_message(stderr: str, stdout: str) -> str:
    stderr = stderr.strip()
    if stderr:
        return stderr

    messages: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event.get("message"), str):
            messages.append(event["message"])
        error = event.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            messages.append(error["message"])

    return messages[-1] if messages else ""


def _timeout_stream_text(value: str | bytes | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


def _format_timeout_seconds(timeout: object) -> str:
    if isinstance(timeout, float) and timeout.is_integer():
        return str(int(timeout))
    return str(timeout)


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
    prompt = _prompt_text(system_message, user_message)

    temp_context = None
    temp_path: str | None = None
    if log_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="aide-codex-query-")
        temp_path = temp_context.__enter__()
    try:
        work_dir = Path(log_dir) if log_dir is not None else Path(temp_path)  # type: ignore[arg-type]
        work_dir.mkdir(parents=True, exist_ok=True)
        schema_file = _prefixed(log_prefix, "schema.json")
        response_file = _prefixed(log_prefix, "response_raw.txt")
        schema_path = work_dir / schema_file
        response_path = work_dir / response_file
        if func_spec is not None:
            schema_path.write_text(
                json.dumps(func_spec.json_schema, indent=2),
                encoding="utf-8",
            )
        command = _codex_command(
            model=model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
            work_dir=work_dir,
            output_schema=func_spec is not None,
            schema_path=schema_path,
            response_path=response_path,
        )
        _write_codex_profile(
            work_dir=work_dir,
            prefix=log_prefix,
            model=model,
            reasoning_effort=reasoning_effort,
            command=command,
        )
        t0 = time.time()
        try:
            result = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=filtered_kwargs.get("timeout"),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            req_time = time.time() - t0
            stdout = _timeout_stream_text(exc.stdout)
            stderr = _timeout_stream_text(exc.stderr)
            (work_dir / _prefixed(log_prefix, "codex_events.jsonl")).write_text(
                stdout,
                encoding="utf-8",
            )
            (work_dir / _prefixed(log_prefix, "stderr.log")).write_text(
                stderr,
                encoding="utf-8",
            )
            timeout_seconds = _format_timeout_seconds(
                filtered_kwargs.get("timeout", exc.timeout)
            )
            raise RuntimeError(
                f"Codex CLI timed out after {timeout_seconds} seconds "
                f"({req_time:.1f}s elapsed)."
            ) from exc
        req_time = time.time() - t0
        (work_dir / _prefixed(log_prefix, "codex_events.jsonl")).write_text(
            result.stdout,
            encoding="utf-8",
        )
        (work_dir / _prefixed(log_prefix, "stderr.log")).write_text(
            result.stderr,
            encoding="utf-8",
        )
        if result.returncode != 0:
            message = _codex_failure_message(result.stderr, result.stdout)
            raise RuntimeError(
                f"Codex CLI failed with exit code {result.returncode}: "
                f"{message}"
            )
        raw_output = response_path.read_text(encoding="utf-8")
    finally:
        if temp_context is not None:
            temp_context.__exit__(None, None, None)

    if func_spec is not None:
        output: OutputType = json.loads(raw_output)
    else:
        output = raw_output

    return output, req_time, 0, 0, {"model": model, "backend": "codex"}
