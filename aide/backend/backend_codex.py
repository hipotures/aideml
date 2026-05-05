"""Backend for Codex CLI calls."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

from funcy import notnone, select_values

from .utils import FunctionSpec, OutputType


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
    work_dir: Path,
    output_schema: bool,
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
    if reasoning_effort is not None:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    if output_schema:
        command.extend(["--output-schema", "schema.json"])
    command.extend(["--output-last-message", "response_raw.txt", "--json", "-"])
    return command


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    model = filtered_kwargs["model"]
    reasoning_effort = filtered_kwargs.pop("reasoning_effort", None)
    prompt = _prompt_text(system_message, user_message)

    with tempfile.TemporaryDirectory(prefix="aide-codex-query-") as tmp:
        work_dir = Path(tmp)
        if func_spec is not None:
            (work_dir / "schema.json").write_text(
                json.dumps(func_spec.json_schema, indent=2),
                encoding="utf-8",
            )
        command = _codex_command(
            model=model,
            reasoning_effort=reasoning_effort,
            work_dir=work_dir,
            output_schema=func_spec is not None,
        )
        t0 = time.time()
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=filtered_kwargs.get("timeout"),
            check=False,
        )
        req_time = time.time() - t0
        if result.returncode != 0:
            raise RuntimeError(
                f"Codex CLI failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        raw_output = (work_dir / "response_raw.txt").read_text(encoding="utf-8")

    if func_spec is not None:
        output: OutputType = json.loads(raw_output)
    else:
        output = raw_output

    return output, req_time, 0, 0, {"model": model, "backend": "codex"}
