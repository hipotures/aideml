import datetime as dt
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import jsonschema
from dataclasses_json import DataClassJsonMixin
import backoff
import logging
from typing import Callable

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


def log_llm_exchange(
    *,
    phase: str,
    provider: str,
    payload: dict,
    sequence_id: int | None = None,
) -> None:
    log_dir = os.getenv("AIDE_LOG_DIR")
    run_id = os.getenv("AIDE_RUN_ID")
    if not log_dir:
        return

    path = Path(log_dir) / "llm_communication.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    title_parts = [phase.upper(), timestamp]
    if run_id:
        title_parts.append(f"run={run_id}")
    if sequence_id is not None:
        title_parts.append(f"llm_call={sequence_id:04d}")

    lines = ["\n---\n", f"## {' | '.join(title_parts)}\n", f"provider: `{provider}`\n"]
    for key, value in payload.items():
        lines.append(f"\n### {key}\n")
        formatted_value = _format_log_value(value)
        if isinstance(value, (dict, list)) or formatted_value != value:
            lines.append(f"```json\n{formatted_value}\n```\n")
        else:
            lines.append(f"```text\n{formatted_value}\n```\n")

    with _llm_log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(lines))


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
