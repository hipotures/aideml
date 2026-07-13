import copy
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
from typing import Type, TypeVar

import dataclasses_json
from ..journal import Journal


def _legacy_artifact_dir_name(node_dict: dict) -> str | None:
    ctime = node_dict.get("ctime")
    if isinstance(ctime, (int, float)) and not isinstance(ctime, bool):
        return dt.datetime.fromtimestamp(float(ctime)).strftime("%Y%m%dT%H%M%S")
    return None


def _relative_code_path(artifact_dir_name: str) -> str:
    return (Path("artifacts") / artifact_dir_name / "solution.py").as_posix()


def _is_inline_generated_node(node: object) -> bool:
    status = getattr(node, "status", None)
    code = getattr(node, "code", None)
    artifact_dir_name = getattr(node, "artifact_dir_name", None)
    code_path = getattr(node, "code_path", None)
    return (
        status == "generated"
        and isinstance(code, str)
        and bool(code)
        and not (isinstance(artifact_dir_name, str) and artifact_dir_name.strip())
        and not (isinstance(code_path, str) and code_path.strip())
    )


def _is_inline_generated_node_dict(node_dict: dict) -> bool:
    code = node_dict.get("code")
    code_path = node_dict.get("code_path")
    artifact_dir_name = node_dict.get("artifact_dir_name")
    return (
        node_dict.get("status") == "generated"
        and isinstance(code, str)
        and bool(code)
        and not (isinstance(artifact_dir_name, str) and artifact_dir_name.strip())
        and not (isinstance(code_path, str) and code_path.strip())
    )


def _is_inline_failed_node(node: object) -> bool:
    return (
        getattr(node, "status", None) == "failed"
        and isinstance(getattr(node, "code", None), str)
        and bool(getattr(node, "code", None))
        and not getattr(node, "artifact_dir_name", None)
        and not getattr(node, "code_path", None)
    )


def _is_inline_failed_node_dict(node_dict: dict) -> bool:
    return (
        node_dict.get("status") == "failed"
        and isinstance(node_dict.get("code"), str)
        and bool(node_dict["code"])
        and not node_dict.get("artifact_dir_name")
        and not node_dict.get("code_path")
    )


def dumps_json(
    obj: dataclasses_json.DataClassJsonMixin,
    *,
    base_dir: Path | None = None,
):
    """Serialize AIDE dataclasses (such as Journals) to JSON."""
    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        node2parent = {n.id: n.parent.id for n in obj.nodes if n.parent is not None}
        for n in obj.nodes:
            n.parent = None
            n.children = set()
            artifact_dir_name = (
                n.artifact_dir_name.strip()
                if isinstance(n.artifact_dir_name, str)
                else None
            )
            if _is_inline_generated_node(n) or _is_inline_failed_node(n):
                n.code_path = None
                n.artifact_dir_name = None
                continue
            if not artifact_dir_name:
                raise ValueError(
                    f"Cannot serialize node {n.id}: missing artifact_dir_name."
                )
            code_path = _relative_code_path(artifact_dir_name)
            if base_dir is not None and not (base_dir / code_path).exists():
                raise FileNotFoundError(
                    f"Cannot serialize node {n.id}: missing solution artifact "
                    f"{base_dir / code_path}"
                )
            n.code = ""
            n.code_path = code_path

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent  # type: ignore
        obj_dict["__version"] = "3"

    return json.dumps(obj_dict, separators=(",", ":"))


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dumps_json(obj, base_dir=path.parent)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_name = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def _hydrate_journal_code(obj_dict: dict, *, base_dir: Path) -> None:
    version = str(obj_dict.get("__version") or "2")
    nodes = obj_dict.get("nodes")
    if not isinstance(nodes, list):
        return

    for node_dict in nodes:
        if not isinstance(node_dict, dict):
            continue
        node_id = str(node_dict.get("id") or "<unknown>")
        code_path_value = node_dict.get("code_path")
        code_path = (
            code_path_value.strip()
            if isinstance(code_path_value, str) and code_path_value.strip()
            else None
        )
        artifact_dir_name = node_dict.get("artifact_dir_name")
        if not code_path:
            if _is_inline_generated_node_dict(node_dict) or _is_inline_failed_node_dict(
                node_dict
            ):
                node_dict["code_path"] = None
                node_dict["artifact_dir_name"] = None
                continue
            if version != "2":
                raise ValueError(f"Journal node {node_id} is missing code_path.")
            if isinstance(artifact_dir_name, str) and artifact_dir_name.strip():
                code_path = _relative_code_path(artifact_dir_name.strip())
            else:
                legacy_name = _legacy_artifact_dir_name(node_dict)
                if legacy_name is None:
                    raise ValueError(
                        f"Legacy journal node {node_id} cannot infer code_path."
                    )
                artifact_dir_name = legacy_name
                code_path = _relative_code_path(legacy_name)

        solution_path = base_dir / code_path
        if not solution_path.exists():
            if node_dict.get("status") == "failed":
                message = str(
                    node_dict.get("analysis")
                    or node_dict.get("plan")
                    or "Failed node artifact was removed."
                )
                node_dict["code"] = f"raise RuntimeError({message!r})\n"
                node_dict["code_path"] = None
                node_dict["artifact_dir_name"] = None
                continue
            raise FileNotFoundError(
                f"Journal node {node_id} solution artifact is missing: {solution_path}"
            )
        if not isinstance(artifact_dir_name, str) or not artifact_dir_name.strip():
            parts = Path(code_path).parts
            if len(parts) >= 3 and parts[0] == "artifacts":
                artifact_dir_name = parts[1]
        if not isinstance(artifact_dir_name, str) or not artifact_dir_name.strip():
            raise ValueError(f"Journal node {node_id} is missing artifact_dir_name.")
        node_dict["code"] = solution_path.read_text(encoding="utf-8")
        node_dict["code_path"] = code_path
        if (
            not isinstance(node_dict.get("artifact_dir_name"), str)
            or not node_dict["artifact_dir_name"].strip()
        ):
            node_dict["artifact_dir_name"] = artifact_dir_name


def loads_json(s: str, cls: Type[G], *, base_dir: Path | None = None) -> G:
    """Deserialize JSON to AIDE dataclasses."""
    obj_dict = json.loads(s)
    if cls is Journal:
        _hydrate_journal_code(obj_dict, base_dir=base_dir or Path.cwd())
    obj = cls.from_dict(obj_dict)

    if isinstance(obj, Journal):
        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            id2nodes[child_id].parent = id2nodes[parent_id]
            id2nodes[child_id].__post_init__()
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls, base_dir=path.parent)
