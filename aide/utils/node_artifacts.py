import datetime as dt
import uuid
from pathlib import Path
from typing import Any


def legacy_node_artifact_dir_name(node: Any) -> str:
    return dt.datetime.fromtimestamp(node.ctime).strftime("%Y%m%dT%H%M%S")


def new_artifact_dir_name(
    *, ctime: float | None = None, step: int | str | None = None
) -> str:
    timestamp = dt.datetime.fromtimestamp(
        ctime if ctime is not None else dt.datetime.now().timestamp()
    ).strftime("%Y%m%dT%H%M%S")
    name = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    if step is None:
        return name
    step_text = str(step).strip()
    if not step_text:
        return name
    return f"{name}-{step_text}"


def node_artifact_dir_name(node: Any) -> str:
    explicit = getattr(node, "artifact_dir_name", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return legacy_node_artifact_dir_name(node)


def node_artifact_dir(log_dir: Path | str, node: Any) -> Path:
    return Path(log_dir) / "artifacts" / node_artifact_dir_name(node)


def node_artifact_submission_path(log_dir: Path | str, node: Any) -> Path:
    return node_artifact_dir(log_dir, node) / "submission.csv"
