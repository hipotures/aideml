from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


PATH_BASE_ENV = "AIDEML_PATH_BASE"
PATH_BASE_ALIASES_ENV = "AIDEML_PATH_BASE_ALIASES"

_ABS_PATH_RE = re.compile(
    r"(?<![\w$])(?:/home|/root|/workspace|/workspaces|/mnt|/data|/kaggle)"
    r"/[^\s\"'<>),;]+"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dotenv_value(name: str) -> str | None:
    paths = [Path(".env"), repo_root() / ".env"]
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip("'\"")
    return None


def _env_value(name: str) -> str | None:
    return os.getenv(name) or _dotenv_value(name)


def configured_path_base() -> Path:
    value = _env_value(PATH_BASE_ENV)
    return Path(value).expanduser() if value else Path.home()


def configured_path_base_aliases() -> list[Path]:
    values = [
        part.strip()
        for part in (_env_value(PATH_BASE_ALIASES_ENV) or "").split(os.pathsep)
        if part.strip()
    ]
    aliases = [Path(value).expanduser() for value in values]
    base = configured_path_base()
    if base not in aliases:
        aliases.append(base)
    return aliases


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _relative_to(path: Path, base: Path) -> Path | None:
    try:
        return path.relative_to(base)
    except ValueError:
        return None


def _clean_relative(path: Path) -> str:
    return path.as_posix().lstrip("./")


def _portable_for_absolute(path: Path, *, project_root: Path | None) -> str:
    root = project_root or repo_root()
    relative = _relative_to(path, root)
    if relative is not None:
        return _clean_relative(relative)

    base = configured_path_base()
    relative = _relative_to(path, base)
    if relative is not None:
        return f"${PATH_BASE_ENV}/{relative.as_posix()}"

    for alias in configured_path_base_aliases():
        relative = _relative_to(path, alias)
        if relative is not None:
            return f"${PATH_BASE_ENV}/{relative.as_posix()}"

    return path.as_posix()


def to_portable_path(
    value: str | Path,
    *,
    project_root: Path | None = None,
    base_dir: Path | None = None,
) -> str:
    path = _as_path(value)
    if not path.is_absolute():
        return _clean_relative(path)

    if base_dir is not None:
        relative = _relative_to(path, base_dir)
        if relative is not None:
            return _clean_relative(relative)

    return _portable_for_absolute(path, project_root=project_root)


def _expand_env_token(value: str) -> Path | None:
    if value.startswith(f"${PATH_BASE_ENV}/"):
        return configured_path_base() / value[len(PATH_BASE_ENV) + 2 :]
    if value.startswith(f"${{{PATH_BASE_ENV}}}/"):
        return configured_path_base() / value[len(PATH_BASE_ENV) + 4 :]
    expanded = os.path.expandvars(value)
    if expanded != value:
        return Path(expanded)
    return None


def resolve_portable_path(
    value: str | Path,
    *,
    project_root: Path | None = None,
    base_dir: Path | None = None,
) -> Path:
    text = str(value)
    expanded = _expand_env_token(text)
    if expanded is not None:
        return expanded

    path = Path(text).expanduser()
    if not path.is_absolute():
        if base_dir is not None:
            return base_dir / path
        return (project_root or repo_root()) / path

    base = configured_path_base()
    for alias in configured_path_base_aliases():
        relative = _relative_to(path, alias)
        if relative is not None:
            return base / relative
    return path


def sanitize_text(value: str, *, project_root: Path | None = None) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0).rstrip(".:")
        suffix = match.group(0)[len(raw) :]
        portable = to_portable_path(raw, project_root=project_root)
        return f"{portable}{suffix}"

    return _ABS_PATH_RE.sub(replace, value)


def sanitize_persisted_payload(
    value: Any,
    *,
    project_root: Path | None = None,
) -> Any:
    if isinstance(value, Path):
        return to_portable_path(value, project_root=project_root)
    if isinstance(value, str):
        if value.startswith("/") or value.startswith("~/"):
            return to_portable_path(value, project_root=project_root)
        return sanitize_text(value, project_root=project_root)
    if isinstance(value, list):
        return [
            sanitize_persisted_payload(item, project_root=project_root)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            sanitize_persisted_payload(item, project_root=project_root)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: sanitize_persisted_payload(item, project_root=project_root)
            for key, item in value.items()
        }
    return value
