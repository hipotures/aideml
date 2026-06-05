from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class WebTreeLine:
    prefix: str
    label: str
    kind: str = "ok"


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


class WebDashboardState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = WebDashboardSnapshot()

    def update(self, snapshot: WebDashboardSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def get_snapshot(self) -> WebDashboardSnapshot:
        with self._lock:
            return self._snapshot
