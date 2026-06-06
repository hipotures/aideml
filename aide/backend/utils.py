import json
import threading
from dataclasses import dataclass
from pathlib import Path

import jsonschema
from dataclasses_json import DataClassJsonMixin
import backoff
import logging
from typing import Callable

from aide.utils.path_portability import sanitize_persisted_payload, sanitize_text, to_portable_path

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


logger = logging.getLogger("aide")
_llm_log_lock = threading.Lock()


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


def _format_log_value(value) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return str(value)


def _prefixed(prefix: str | None, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _json_default(value):
    if isinstance(value, Path):
        return to_portable_path(value)
    return str(value)


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(
            sanitize_persisted_payload(payload),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _raw_response_text(output: OutputType) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, indent=2, ensure_ascii=False, default=_json_default)


def request_markdown(
    system_message: str | None,
    user_message: str | None,
) -> str:
    parts = []
    if system_message:
        parts.append(f"# System message\n\n{system_message}")
    if user_message:
        parts.append(f"# User message\n\n{user_message}")
    return "\n\n---\n\n".join(parts).rstrip() + "\n"


def write_llm_request_files(
    *,
    log_dir: Path | str | None,
    prefix: str | None,
    context: dict,
    request_payload: dict,
    system_message: str | None,
    user_message: str | None,
) -> None:
    if log_dir is None:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    with _llm_log_lock:
        _write_json(path / _prefixed(prefix, "context.json"), context)
        _write_json(path / _prefixed(prefix, "request.json"), request_payload)
        (path / _prefixed(prefix, "request.md")).write_text(
            sanitize_text(request_markdown(system_message, user_message)),
            encoding="utf-8",
        )
        (path / _prefixed(prefix, "stderr.log")).touch()
        _write_json(
            path / _prefixed(prefix, "status.json"),
            {
                "status": "running",
                "provider": context.get("provider"),
                "model": context.get("model"),
                "phase": context.get("phase"),
                "sequence_id": context.get("sequence_id"),
            },
        )


def write_llm_response_files(
    *,
    log_dir: Path | str | None,
    prefix: str | None,
    context: dict,
    response_payload: dict,
    output: OutputType,
) -> None:
    if log_dir is None:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    with _llm_log_lock:
        (path / _prefixed(prefix, "response_raw.txt")).write_text(
            _raw_response_text(output),
            encoding="utf-8",
        )
        _write_json(path / _prefixed(prefix, "response.json"), response_payload)
        _write_json(
            path / _prefixed(prefix, "status.json"),
            {
                "status": "completed",
                "provider": context.get("provider"),
                "model": context.get("model"),
                "phase": context.get("phase"),
                "sequence_id": context.get("sequence_id"),
                "request_time_seconds": response_payload.get("request_time_seconds"),
            },
        )


def write_llm_error_files(
    *,
    log_dir: Path | str | None,
    prefix: str | None,
    context: dict,
    error_payload: dict,
) -> None:
    if log_dir is None:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    with _llm_log_lock:
        _write_json(path / _prefixed(prefix, "response.json"), error_payload)
        (path / _prefixed(prefix, "response_raw.txt")).write_text(
            str(error_payload.get("message", "")),
            encoding="utf-8",
        )
        _write_json(
            path / _prefixed(prefix, "status.json"),
            {
                "status": "failed",
                "provider": context.get("provider"),
                "model": context.get("model"),
                "phase": context.get("phase"),
                "sequence_id": context.get("sequence_id"),
                "error": error_payload,
            },
        )


def append_provider_event(
    *,
    log_dir: Path | str | None,
    prefix: str | None,
    event: dict,
) -> None:
    if log_dir is None:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, default=_json_default) + "\n"
    with _llm_log_lock:
        with open(path / _prefixed(prefix, "provider_events.jsonl"), "a", encoding="utf-8") as f:
            f.write(line)


def write_llm_response_code(
    *,
    log_dir: Path | str | None,
    prefix: str | None = None,
    code: str,
) -> None:
    if log_dir is None:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    with _llm_log_lock:
        (path / _prefixed(prefix, "response.py")).write_text(code, encoding="utf-8")


def log_llm_exchange(*args, **kwargs) -> None:
    """Deprecated no-op. LLM communication is logged as artifact files."""
    return None


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        """Convert to OpenAI's function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_anthropic_tool_dict(self):
        """Convert to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,  # Anthropic uses input_schema instead of parameters
        }

    @property
    def anthropic_tool_choice_dict(self):
        """Convert to Anthropic's tool choice format."""
        return {
            "type": "tool",  # Anthropic uses "tool" instead of "function"
            "name": self.name,
        }

    @property
    def as_openai_responses_tool_dict(self):
        """Convert to OpenAI Responses API tool format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_responses_tool_choice_dict(self):
        """Convert to OpenAI Responses API tool choice format."""
        return {
            "type": "function",
            "name": self.name,
        }
