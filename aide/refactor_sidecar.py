# refactor_sidecar.py
#
# Framework-agnostic helper for a non-invasive refactor pass after response.py is generated.

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict, Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_prompt_dir() -> Path:
    return _repo_root() / "assets" / "prompts" / "refactor_for_cache"


@dataclass
class RefactorConfig:
    enabled: bool = False
    model: str = ""
    prompt_path: Optional[Path] = None
    execution_contract_path: Optional[Path] = None
    runtime_api_path: Optional[Path] = None
    runtime_source_path: Optional[Path] = None
    max_input_chars: int = 240_000
    timeout_s: int = 180

    @staticmethod
    def from_env() -> "RefactorConfig":
        enabled = os.environ.get("AIDE_REFACTOR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

        def maybe_path(name: str) -> Optional[Path]:
            value = os.environ.get(name, "").strip()
            return Path(value) if value else None

        try:
            max_input_chars = int(os.environ.get("AIDE_REFACTOR_MAX_INPUT_CHARS", "240000"))
        except Exception:
            max_input_chars = 240_000

        try:
            timeout_s = int(os.environ.get("AIDE_REFACTOR_TIMEOUT_S", "180"))
        except Exception:
            timeout_s = 180

        return RefactorConfig(
            enabled=enabled,
            model=os.environ.get("AIDE_REFACTOR_MODEL", "").strip(),
            prompt_path=maybe_path("AIDE_REFACTOR_PROMPT_PATH")
            or _default_prompt_dir() / "refactor_for_cache.md",
            execution_contract_path=maybe_path("AIDE_REFACTOR_EXECUTION_CONTRACT_PATH")
            or _default_prompt_dir() / "execution_contract.md",
            runtime_api_path=maybe_path("AIDE_REFACTOR_RUNTIME_API_PATH")
            or _default_prompt_dir() / "aide_refactor_runtime_api.md",
            runtime_source_path=Path(__file__).with_name("aide_refactor_runtime.py"),
            max_input_chars=max_input_chars,
            timeout_s=timeout_s,
        )


def _print_refactor(event: str, **fields: Any) -> None:
    parts = ["AIDE_REFACTOR", f"event={event}"]
    for key, value in fields.items():
        if value is not None:
            text = str(value).replace("\n", " ").replace("\r", " ").replace("|", "/")
            parts.append(f"{key}={text}")
    print("|".join(parts), flush=True)


def _safe_write_text(path: Path, text: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return True
    except Exception:
        return False


def _safe_write_json(path: Path, data: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return True
    except Exception:
        return False


def _safe_copy_file(source: Path, destination: Path) -> bool:
    try:
        if not source.exists():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return True
    except Exception:
        return False


def extract_python_code(raw_response: str) -> tuple[str | None, str]:
    if not raw_response or not raw_response.strip():
        return None, "empty_response"

    pattern = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    blocks = pattern.findall(raw_response)

    if len(blocks) == 1:
        code = blocks[0].strip()
        return (code + "\n", "ok_fenced") if code else (None, "empty_code_block")

    if len(blocks) > 1:
        return None, "multiple_python_blocks"

    stripped = raw_response.strip()
    prefixes = ("import ", "from ", "def ", "class ", "#")
    if stripped.startswith(prefixes) and ("def " in stripped or "import " in stripped or "from " in stripped):
        return stripped + "\n", "ok_raw_code"

    return None, "no_python_code"


def _called_name(call: ast.Call) -> str | None:
    node = call.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    if not parts:
        return None
    return ".".join(reversed(parts))


def validate_refactored_code(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as ex:
        return False, f"syntax_error:{ex.__class__.__name__}"

    imports_runtime = False
    uses_stage = False
    finalizes = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports_runtime = imports_runtime or any(
                alias.name == "aide_refactor_runtime" for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            imports_runtime = imports_runtime or node.module == "aide_refactor_runtime"
        elif isinstance(node, ast.Call):
            called = _called_name(node)
            uses_stage = uses_stage or called in {
                "aide_stage",
                "aide_refactor_runtime.aide_stage",
            }
            finalizes = finalizes or called in {
                "finalize_aide_artifacts",
                "aide_refactor_runtime.finalize_aide_artifacts",
            }

    if not imports_runtime:
        return False, "missing_runtime_import"
    if not uses_stage:
        return False, "missing_aide_stage"
    if not finalizes:
        return False, "missing_finalize_aide_artifacts"
    return True, "ok"


def build_refactor_request(
    *,
    prompt_template: str,
    execution_contract: str,
    runtime_api: str,
    response_py: str,
) -> str:
    return (
        prompt_template
        .replace("{{EXECUTION_CONTRACT}}", execution_contract)
        .replace("{{AIDE_REFACTOR_RUNTIME_API}}", runtime_api)
        .replace("{{RESPONSE_PY}}", response_py)
    )


def maybe_refactor_response_py(
    *,
    response_py_path: Path,
    artifact_dir: Path,
    call_model: Callable[[str, str, int], str],
    config: Optional[RefactorConfig] = None,
) -> Dict[str, Any]:
    started = time.time()
    config = config or RefactorConfig.from_env()

    artifact_dir = Path(artifact_dir)
    code_path = artifact_dir / "response_refactored.py"
    meta_path = artifact_dir / "response_refactor_meta.json"
    runtime_artifact_path = artifact_dir / "aide_refactor_runtime.py"

    meta: Dict[str, Any] = {
        "enabled": config.enabled,
        "status": "not_started",
        "model": config.model,
        "started_at_unix": started,
        "finished_at_unix": None,
        "elapsed_s": None,
        "response_py_path": str(response_py_path),
        "output_path": str(code_path),
        "runtime_artifact_path": str(runtime_artifact_path),
        "error_type": None,
        "error": None,
        "fallback": "original_response_py_will_be_executed",
    }

    try:
        if not config.enabled:
            meta["status"] = "disabled"
            _print_refactor("disabled", reason="config_disabled")
            return meta

        if not config.model:
            meta["status"] = "skipped_missing_model"
            _print_refactor("disabled", reason="missing_model")
            return meta

        required = [
            ("prompt", config.prompt_path),
            ("execution_contract", config.execution_contract_path),
            ("runtime_api", config.runtime_api_path),
        ]
        for label, path in required:
            if not path or not Path(path).exists():
                meta["status"] = f"skipped_missing_{label}"
                _print_refactor("disabled", reason=f"missing_{label}")
                return meta

        if not config.runtime_source_path or not Path(config.runtime_source_path).exists():
            meta["status"] = "skipped_missing_runtime_source"
            _print_refactor("disabled", reason="missing_runtime_source")
            return meta

        response_py_path = Path(response_py_path)
        if not response_py_path.exists():
            meta["status"] = "failed_missing_response_py"
            meta["error"] = f"Missing response.py: {response_py_path}"
            _print_refactor("failed", reason="missing_response_py", action="continue_original")
            return meta

        _print_refactor("start", model=config.model)

        prompt_template = Path(config.prompt_path).read_text(encoding="utf-8")
        execution_contract = Path(config.execution_contract_path).read_text(encoding="utf-8")
        runtime_api = Path(config.runtime_api_path).read_text(encoding="utf-8")
        response_py = response_py_path.read_text(encoding="utf-8")

        request = build_refactor_request(
            prompt_template=prompt_template,
            execution_contract=execution_contract,
            runtime_api=runtime_api,
            response_py=response_py,
        )

        meta["input_code_chars"] = len(response_py)
        meta["input_request_chars"] = len(request)

        if len(request) > config.max_input_chars:
            meta["status"] = "skipped_input_too_large"
            meta["error"] = f"Refactor request has {len(request)} chars, limit is {config.max_input_chars}"
            _print_refactor("failed", reason="input_too_large", action="continue_original")
            return meta

        meta["runtime_artifact_saved"] = _safe_copy_file(
            Path(config.runtime_source_path),
            runtime_artifact_path,
        )

        raw = call_model(request, config.model, config.timeout_s)
        raw = raw or ""
        meta["raw_response_chars"] = len(raw)

        code, parse_status = extract_python_code(raw)
        meta["parse_status"] = parse_status

        if code is None:
            meta["status"] = "parse_failed"
            meta["error_type"] = parse_status
            _print_refactor("parse_failed", reason=parse_status, action="continue_original")
            return meta

        valid, validation_status = validate_refactored_code(code)
        meta["validation_status"] = validation_status
        if not valid:
            meta["status"] = "validation_failed"
            meta["error_type"] = validation_status
            _print_refactor("validation_failed", reason=validation_status, action="continue_original")
            return meta

        _safe_write_text(code_path, code)
        meta["extracted_code_chars"] = len(code)
        meta["status"] = "success"
        _print_refactor("code_saved", path=code_path)
        _print_refactor("success", elapsed_s=f"{time.time() - started:.3f}")
        return meta

    except Exception as ex:
        meta["status"] = "failed"
        meta["error_type"] = type(ex).__name__
        meta["error"] = str(ex)
        meta["traceback"] = traceback.format_exc()[-8000:]
        _print_refactor("failed", reason=type(ex).__name__, action="continue_original")
        return meta

    finally:
        meta["finished_at_unix"] = time.time()
        meta["elapsed_s"] = float(meta["finished_at_unix"] - started)
        _safe_write_json(meta_path, meta)
