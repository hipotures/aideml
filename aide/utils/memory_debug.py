"""Optional runtime memory diagnostics for long AIDE runs."""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

import psutil


DEFAULT_DEBUG_LOG_PATH = Path("/tmp/aideml/debug.log")


def _safe_call(func: Callable[[], Any], default: Any = None) -> Any:
    try:
        return func()
    except Exception:  # noqa: BLE001 - debug logging must not stop the run
        return default


def _memory_payload(process: Any) -> dict[str, int | None]:
    info = _safe_call(process.memory_info)
    full = _safe_call(process.memory_full_info)
    return {
        "rss_bytes": int(getattr(info, "rss", 0) or 0) if info is not None else None,
        "vms_bytes": int(getattr(info, "vms", 0) or 0) if info is not None else None,
        "shared_bytes": (
            int(getattr(info, "shared", 0) or 0) if info is not None else None
        ),
        "pss_bytes": (
            int(getattr(full, "pss", 0) or 0)
            if full is not None and hasattr(full, "pss")
            else None
        ),
        "uss_bytes": (
            int(getattr(full, "uss", 0) or 0)
            if full is not None and hasattr(full, "uss")
            else None
        ),
    }


def _process_payload(process: Any, *, include_cmdline: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pid": getattr(process, "pid", None),
        **_memory_payload(process),
        "thread_count": _safe_call(getattr(process, "num_threads", None)),
    }
    ppid = _safe_call(getattr(process, "ppid", None))
    if ppid is not None:
        payload["ppid"] = ppid
    name = _safe_call(getattr(process, "name", None))
    if name:
        payload["name"] = name
    status = _safe_call(getattr(process, "status", None))
    if status:
        payload["status"] = status
    if include_cmdline:
        cmdline = _safe_call(getattr(process, "cmdline", None), [])
        payload["cmdline"] = cmdline
    return payload


def _parent_payload(process: Any) -> dict[str, Any]:
    payload = _process_payload(process)
    payload["fd_count"] = _safe_call(getattr(process, "num_fds", None))
    return payload


def _system_payload(system_memory: Any) -> dict[str, Any]:
    return {
        "total_bytes": int(getattr(system_memory, "total", 0) or 0),
        "available_bytes": int(getattr(system_memory, "available", 0) or 0),
        "used_bytes": int(getattr(system_memory, "used", 0) or 0),
        "percent": float(getattr(system_memory, "percent", 0.0) or 0.0),
    }


class MemoryDebugLogger:
    """Append JSON memory snapshots when `--debug` is enabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        path: Path = DEFAULT_DEBUG_LOG_PATH,
        run_id: str | None = None,
        root_process_factory: Callable[[], Any] | None = None,
        system_memory_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.path = path
        self.run_id = run_id
        self.root_process_factory = root_process_factory or (
            lambda: psutil.Process(os.getpid())
        )
        self.system_memory_factory = system_memory_factory or psutil.virtual_memory
        self._lock = threading.Lock()

    def log(
        self,
        event: str,
        *,
        phase: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        root_process = _safe_call(self.root_process_factory)
        system_memory = _safe_call(self.system_memory_factory)
        children = []
        if root_process is not None:
            children = [
                _process_payload(child, include_cmdline=True)
                for child in _safe_call(
                    lambda: root_process.children(recursive=True),
                    [],
                )
            ]

        record: dict[str, Any] = {
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            "phase": phase,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "parent": _parent_payload(root_process) if root_process is not None else None,
            "children": children,
            "system": (
                _system_payload(system_memory) if system_memory is not None else None
            ),
            "extra": extra or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
