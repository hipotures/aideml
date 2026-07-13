"""Codex app-server transport shared by AIDE Codex call sites."""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from aide.utils.path_portability import sanitize_persisted_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEX_HOME = REPO_ROOT / ".codex" / "aideml-app-server"
USAGE_WAIT_SECONDS = 10.0
_DELTA_METHODS = {
    "item/agentMessage/delta",
    "item/reasoning/summaryTextDelta",
    "item/reasoning/textDelta",
}


@dataclass
class CodexAppServerResult:
    text: str
    status: str
    thread_id: str
    turn_id: str
    duration_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    usage: dict[str, Any] | None = None
    notifications: list[dict[str, Any]] = field(default_factory=list)
    rpc_responses: list[dict[str, Any]] = field(default_factory=list)
    stderr: str = ""


@dataclass(frozen=True)
class CodexThreadFork:
    thread_id: str
    turn_id: str | None = None
    path: Path | None = None


def _prefixed(prefix: str | None, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{key} = {_toml_value(item)}" for key, item in value.items()
        ) + "}"
    return json.dumps(str(value))


def app_server_command(*, web_search: bool) -> list[str]:
    command = ["codex", "app-server"]
    for feature in (
        "apps",
        "browser_use",
        "computer_use",
        "multi_agent",
        "plugins",
        "shell_snapshot",
        "skill_mcp_dependency_install",
        "tool_suggest",
        "workspace_dependencies",
    ):
        command.extend(["--disable", feature])
    overrides = {
        "project_doc_max_bytes": 0,
        "project_doc_fallback_filenames": [],
        "web_search": "live" if web_search else "disabled",
        "mcp_servers": {},
        "features.skill_mcp_dependency_install": False,
    }
    for key, value in overrides.items():
        command.extend(["-c", f"{key}={_toml_value(value)}"])
    return command


def _sync_auth_file(source_home: Path, target_home: Path) -> None:
    target_home.mkdir(parents=True, exist_ok=True)
    try:
        target_home.chmod(0o700)
    except OSError:
        pass
    source = source_home / "auth.json"
    target = target_home / "auth.json"
    if not source.exists() or source.resolve() == target.resolve():
        return
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target_home,
            prefix=".auth.json.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
        shutil.copyfile(source, tmp_name)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, target)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def app_server_env(codex_home: Path = DEFAULT_CODEX_HOME) -> dict[str, str]:
    source_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    _sync_auth_file(source_home, codex_home)
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    env["CODEX_SQLITE_HOME"] = str(codex_home)
    return env


def _request(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": request_id,
        "method": method,
        "params": {key: value for key, value in params.items() if value is not None},
    }


def _send(proc: subprocess.Popen[str], message: dict[str, Any]) -> None:
    if proc.stdin is None:
        raise RuntimeError("Codex app-server stdin is closed.")
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _read_message(
    proc: subprocess.Popen[str], *, timeout_seconds: float
) -> dict[str, Any] | None:
    if proc.stdout is None:
        raise RuntimeError("Codex app-server stdout is closed.")
    ready, _, _ = select.select([proc.stdout], [], [], timeout_seconds)
    if not ready:
        return None
    line = proc.stdout.readline()
    if not line:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Codex app-server returned invalid JSONL: {line!r}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Codex app-server returned non-object JSON: {value!r}")
    return value


def _write_jsonl(handle: TextIO, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(sanitize_persisted_payload(payload), ensure_ascii=False) + "\n")
    handle.flush()


def _record_message(
    message: dict[str, Any],
    *,
    events_handle: TextIO,
    rpc_handle: TextIO,
    notifications: list[dict[str, Any]],
    rpc_responses: list[dict[str, Any]],
) -> None:
    if "id" in message:
        rpc_responses.append(message)
        _write_jsonl(rpc_handle, message)
        return
    notifications.append(message)
    if message.get("method") not in _DELTA_METHODS:
        _write_jsonl(events_handle, message)


def _raise_for_rpc_error(message: dict[str, Any], method: str) -> None:
    if "error" in message:
        raise RuntimeError(f"Codex app-server {method} failed: {message['error']!r}")


def _read_until_response(
    proc: subprocess.Popen[str],
    request_id: int,
    *,
    deadline: float,
    events_handle: TextIO,
    rpc_handle: TextIO,
    notifications: list[dict[str, Any]],
    rpc_responses: list[dict[str, Any]],
) -> dict[str, Any]:
    while time.monotonic() < deadline:
        message = _read_message(proc, timeout_seconds=1.0)
        if message is None:
            if proc.poll() is not None:
                break
            continue
        _record_message(
            message,
            events_handle=events_handle,
            rpc_handle=rpc_handle,
            notifications=notifications,
            rpc_responses=rpc_responses,
        )
        if message.get("id") == request_id:
            _raise_for_rpc_error(message, str(request_id))
            return message
    if proc.poll() is not None:
        raise RuntimeError(
            "Codex app-server exited with code "
            f"{proc.returncode} while waiting for response id={request_id}."
        )
    raise TimeoutError(
        f"Codex app-server timed out waiting for response id={request_id}."
    )


def _turn_status(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    turn = payload.get("turn")
    return str(turn.get("status")) if isinstance(turn, dict) and turn.get("status") else None


def _usage_counts(usage: dict[str, Any] | None) -> tuple[int, int]:
    token_usage = usage.get("tokenUsage") if isinstance(usage, dict) else None
    last = token_usage.get("last") if isinstance(token_usage, dict) else None
    if not isinstance(last, dict):
        return 0, 0
    return int(last.get("inputTokens") or 0), int(last.get("outputTokens") or 0)


def _extract_thread_id(response: dict[str, Any]) -> str:
    result = response.get("result")
    thread = result.get("thread") if isinstance(result, dict) else None
    thread_id = thread.get("id") if isinstance(thread, dict) else None
    if not isinstance(thread_id, str) or not thread_id:
        raise RuntimeError(
            f"Codex app-server did not return a thread id: {response!r}"
        )
    return thread_id


def _extract_turn_id(response: dict[str, Any]) -> str:
    result = response.get("result")
    turn = result.get("turn") if isinstance(result, dict) else None
    turn_id = turn.get("id") if isinstance(turn, dict) else None
    if not isinstance(turn_id, str) or not turn_id:
        raise RuntimeError(f"Codex app-server did not return a turn id: {response!r}")
    return turn_id


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=5)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _write_profile(
    path: Path,
    *,
    command: list[str],
    model: str,
    reasoning_effort: str | None,
    web_search: bool,
    work_dir: Path,
    output_schema: dict[str, Any] | None,
    thread_action: str,
    source_thread_id: str | None,
    source_turn_id: str | None,
) -> None:
    payload = {
        "transport": "codex_app_server",
        "model": model,
        "reasoning_effort": reasoning_effort or "",
        "sandbox": "read-only",
        "approval_policy": "never",
        "ephemeral": False,
        "web_search": "live" if web_search else "disabled",
        "work_dir": sanitize_persisted_payload(str(work_dir)),
        "output_schema": output_schema is not None,
        "thread_action": thread_action,
        "source_thread_id": source_thread_id or "",
        "source_turn_id": source_turn_id or "",
        "command": sanitize_persisted_payload(command),
    }
    lines = [
        f'transport = "{payload["transport"]}"',
        f'model = {json.dumps(payload["model"])}',
        f'reasoning_effort = {json.dumps(payload["reasoning_effort"])}',
        'sandbox = "read-only"',
        'approval_policy = "never"',
        "ephemeral = false",
        f'web_search = "{payload["web_search"]}"',
        f'work_dir = {json.dumps(payload["work_dir"])}',
        f'output_schema = {str(payload["output_schema"]).lower()}',
        f'thread_action = {json.dumps(payload["thread_action"])}',
        f'source_thread_id = {json.dumps(payload["source_thread_id"])}',
        f'source_turn_id = {json.dumps(payload["source_turn_id"])}',
        "",
        "[command]",
        "argv = [",
    ]
    lines.extend(f"  {json.dumps(part)}," for part in payload["command"])
    lines.extend(["]", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def invoke_codex_app_server(
    *,
    prompt: str,
    model: str,
    reasoning_effort: str | None,
    web_search: bool,
    work_dir: Path,
    timeout: float | int | None,
    output_schema: dict[str, Any] | None = None,
    log_dir: Path | None = None,
    log_prefix: str | None = None,
    thread_id: str | None = None,
    fork_from: CodexThreadFork | None = None,
) -> CodexAppServerResult:
    """Run one durable Codex thread/turn through a short-lived app-server."""
    work_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = log_dir or work_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    events_path = artifact_dir / _prefixed(log_prefix, "codex_events.jsonl")
    rpc_path = artifact_dir / _prefixed(log_prefix, "codex_rpc.jsonl")
    stderr_path = artifact_dir / _prefixed(log_prefix, "stderr.log")
    profile_path = artifact_dir / _prefixed(log_prefix, "codex_profile.toml")
    requested_thread_id = thread_id
    if requested_thread_id is not None and fork_from is not None:
        raise ValueError("thread_id and fork_from are mutually exclusive.")
    thread_action = (
        "fork" if fork_from is not None else ("resume" if requested_thread_id else "start")
    )
    source_thread_id = (
        fork_from.thread_id if fork_from is not None else requested_thread_id
    )
    source_turn_id = fork_from.turn_id if fork_from is not None else None
    command = app_server_command(web_search=web_search)
    _write_profile(
        profile_path,
        command=command,
        model=model,
        reasoning_effort=reasoning_effort,
        web_search=web_search,
        work_dir=work_dir,
        output_schema=output_schema,
        thread_action=thread_action,
        source_thread_id=source_thread_id,
        source_turn_id=source_turn_id,
    )

    timeout_seconds = float(timeout) if timeout is not None else 600.0
    started = time.monotonic()
    deadline = started + timeout_seconds
    notifications: list[dict[str, Any]] = []
    rpc_responses: list[dict[str, Any]] = []
    final_chunks: list[str] = []
    final_text: str | None = None
    usage: dict[str, Any] | None = None
    turn_completed: dict[str, Any] | None = None
    completion_deadline: float | None = None
    thread_id = ""
    turn_id = ""
    proc: subprocess.Popen[str] | None = None

    with (
        events_path.open("w", encoding="utf-8") as events_handle,
        rpc_path.open("w", encoding="utf-8") as rpc_handle,
        stderr_path.open("w+", encoding="utf-8") as stderr_handle,
    ):
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                env=app_server_env(),
                cwd=str(work_dir),
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            _send(
                proc,
                _request(
                    0,
                    "initialize",
                    {
                        "clientInfo": {
                            "name": "aideml",
                            "title": "AIDE ML",
                            "version": "0.1.0",
                        },
                        "capabilities": (
                            {"experimentalApi": True}
                            if fork_from is not None and fork_from.path is not None
                            else None
                        ),
                    },
                ),
            )
            _read_until_response(
                proc,
                0,
                deadline=deadline,
                events_handle=events_handle,
                rpc_handle=rpc_handle,
                notifications=notifications,
                rpc_responses=rpc_responses,
            )
            _send(proc, {"method": "initialized", "params": {}})

            thread_params: dict[str, Any] = {
                "model": model,
                "cwd": str(work_dir),
                "sandbox": "read-only",
                "approvalPolicy": "never",
                "config": (
                    {"model_reasoning_effort": reasoning_effort}
                    if reasoning_effort
                    else None
                ),
            }
            if fork_from is not None:
                thread_method = "thread/fork"
                thread_params["ephemeral"] = False
                thread_params |= {
                    "threadId": fork_from.thread_id,
                    "lastTurnId": fork_from.turn_id,
                    "path": str(fork_from.path) if fork_from.path is not None else None,
                }
            elif requested_thread_id is not None:
                thread_method = "thread/resume"
                thread_params |= {"threadId": requested_thread_id}
            else:
                thread_method = "thread/start"
                thread_params["ephemeral"] = False
            _send(proc, _request(1, thread_method, thread_params))
            thread_response = _read_until_response(
                proc,
                1,
                deadline=deadline,
                events_handle=events_handle,
                rpc_handle=rpc_handle,
                notifications=notifications,
                rpc_responses=rpc_responses,
            )
            thread_id = _extract_thread_id(thread_response)

            turn_params = {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
                "model": model,
                "effort": reasoning_effort,
                "cwd": str(work_dir),
                "outputSchema": output_schema,
            }
            _send(proc, _request(2, "turn/start", turn_params))
            turn_response = _read_until_response(
                proc,
                2,
                deadline=deadline,
                events_handle=events_handle,
                rpc_handle=rpc_handle,
                notifications=notifications,
                rpc_responses=rpc_responses,
            )
            turn_id = _extract_turn_id(turn_response)

            while time.monotonic() < deadline:
                if completion_deadline is not None and time.monotonic() >= completion_deadline:
                    break
                message = _read_message(proc, timeout_seconds=1.0)
                if message is None:
                    if proc.poll() is not None:
                        break
                    continue
                _record_message(
                    message,
                    events_handle=events_handle,
                    rpc_handle=rpc_handle,
                    notifications=notifications,
                    rpc_responses=rpc_responses,
                )
                if "id" in message:
                    _raise_for_rpc_error(message, str(message.get("id")))
                    continue
                method = message.get("method")
                params = message.get("params") if isinstance(message.get("params"), dict) else {}
                if method == "item/agentMessage/delta":
                    delta = params.get("delta")
                    if isinstance(delta, str):
                        final_chunks.append(delta)
                elif method == "item/completed":
                    item = params.get("item")
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "agentMessage"
                        and item.get("phase") == "final_answer"
                    ):
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            final_text = text
                elif method == "thread/tokenUsage/updated":
                    usage = params
                    if turn_completed is not None and (
                        final_text is not None or final_chunks
                    ):
                        break
                elif method == "turn/completed":
                    turn_completed = params
                    status = _turn_status(params)
                    if status not in (None, "completed"):
                        raise RuntimeError(
                            f"Codex app-server turn failed with status {status}: {params!r}"
                        )
                    completion_deadline = min(
                        deadline, time.monotonic() + USAGE_WAIT_SECONDS
                    )
                    if usage is not None and (final_text is not None or final_chunks):
                        break
                elif method == "thread/status/changed":
                    status = params.get("status")
                    if isinstance(status, dict) and status.get("type") == "systemError":
                        raise RuntimeError(
                            "Codex app-server thread entered systemError state."
                        )
                elif method == "error":
                    raise RuntimeError(
                        f"Codex app-server error notification: {params!r}"
                    )

            text = (final_text if final_text is not None else "".join(final_chunks)).strip()
            if turn_completed is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Codex app-server timed out before reporting turn/completed "
                        f"after {timeout_seconds:g} seconds."
                    )
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Codex app-server exited with code {proc.returncode} before reporting turn/completed."
                    )
                raise RuntimeError(
                    "Codex app-server returned output without reporting turn/completed."
                )
            if not text:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Codex app-server timed out after {timeout_seconds:g} seconds."
                    )
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Codex app-server exited with code {proc.returncode} before returning a final response."
                    )
                raise RuntimeError("Codex app-server did not return a final response.")
            status = _turn_status(turn_completed) or "completed"
            input_tokens, output_tokens = _usage_counts(usage)
        except BaseException:
            partial_text = (
                final_text if final_text is not None else "".join(final_chunks)
            ).strip()
            if partial_text:
                partial_path = artifact_dir / _prefixed(log_prefix, "response_raw.txt")
                partial_path.write_text(partial_text, encoding="utf-8")
            if proc is not None and proc.poll() is None and thread_id and turn_id:
                try:
                    _send(
                        proc,
                        _request(
                            3,
                            "turn/interrupt",
                            {"threadId": thread_id, "turnId": turn_id},
                        ),
                    )
                except Exception:
                    pass
            raise
        finally:
            if proc is not None:
                if proc.stdin is not None:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass
                _terminate_process(proc)
            stderr_handle.flush()
            stderr_handle.seek(0)
            stderr_text = stderr_handle.read()

    return CodexAppServerResult(
        text=text,
        status=status,
        thread_id=thread_id,
        turn_id=turn_id,
        duration_seconds=time.monotonic() - started,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage=usage,
        notifications=notifications,
        rpc_responses=rpc_responses,
        stderr=stderr_text,
    )
