from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class WebTreeLine:
    prefix: str
    label: str
    kind: str = "ok"
    desktop_prefix: str = ""


@dataclass(frozen=True)
class WebRunDatum:
    label: str
    value: str


@dataclass(frozen=True)
class WebRunSection:
    title: str
    items: list[WebRunDatum] = field(default_factory=list)


@dataclass(frozen=True)
class WebDashboardSnapshot:
    run_id: str = ""
    refresh_seconds: float = 2.0
    tree_title: str = "Solution tree"
    tree_lines: list[WebTreeLine] = field(default_factory=list)
    run_data: list[WebRunDatum] = field(default_factory=list)
    run_sections: list[WebRunSection] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    status: str = "starting"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "WebDashboardSnapshot":
        return cls(
            run_id=str(payload.get("run_id") or ""),
            refresh_seconds=float(payload.get("refresh_seconds") or 2.0),
            tree_title=str(payload.get("tree_title") or "Solution tree"),
            tree_lines=[
                WebTreeLine(
                    prefix=str(line.get("prefix") or ""),
                    label=str(line.get("label") or ""),
                    kind=str(line.get("kind") or "ok"),
                    desktop_prefix=str(line.get("desktop_prefix") or ""),
                )
                for line in payload.get("tree_lines") or []
                if isinstance(line, dict)
            ],
            run_data=[
                WebRunDatum(
                    label=str(item.get("label") or ""),
                    value=str(item.get("value") or ""),
                )
                for item in payload.get("run_data") or []
                if isinstance(item, dict)
            ],
            run_sections=[
                WebRunSection(
                    title=str(section.get("title") or ""),
                    items=[
                        WebRunDatum(
                            label=str(item.get("label") or ""),
                            value=str(item.get("value") or ""),
                        )
                        for item in section.get("items") or []
                        if isinstance(item, dict)
                    ],
                )
                for section in payload.get("run_sections") or []
                if isinstance(section, dict)
            ],
            log_lines=[str(line) for line in payload.get("log_lines") or []],
            status=str(payload.get("status") or "starting"),
        )


class WebDashboardState:
    def __init__(self, *, snapshot_path: Path | str | None = None) -> None:
        self._lock = threading.RLock()
        self._snapshot = WebDashboardSnapshot()
        self._snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        self._snapshot_mtime_ns: int | None = None

    def update(self, snapshot: WebDashboardSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot
            self._write_snapshot_file(snapshot)

    def get_snapshot(self) -> WebDashboardSnapshot:
        with self._lock:
            self._read_snapshot_file()
            return self._snapshot

    def _write_snapshot_file(self, snapshot: WebDashboardSnapshot) -> None:
        if self._snapshot_path is None:
            return
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._snapshot_path.with_suffix(self._snapshot_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, self._snapshot_path)
        try:
            self._snapshot_mtime_ns = self._snapshot_path.stat().st_mtime_ns
        except OSError:
            self._snapshot_mtime_ns = None

    def _read_snapshot_file(self) -> None:
        if self._snapshot_path is None:
            return
        try:
            mtime_ns = self._snapshot_path.stat().st_mtime_ns
        except OSError:
            return
        if self._snapshot_mtime_ns == mtime_ns:
            return
        try:
            payload = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self._snapshot = WebDashboardSnapshot.from_dict(payload)
        self._snapshot_mtime_ns = mtime_ns
