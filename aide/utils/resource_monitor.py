"""Process-tree resource sampling helpers."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_percent: float
    ram_bytes: int
    peak_ram_bytes: int
    process_count: int


def downsample_max(values: list[float], width: int) -> list[float]:
    if width <= 0 or not values:
        return []
    if len(values) <= width:
        return list(values)

    step = len(values) / width
    sampled: list[float] = []
    for index in range(width):
        start = int(index * step)
        end = max(start + 1, int((index + 1) * step))
        sampled.append(max(values[start:end]))
    return sampled


class ResourceHistory:
    def __init__(self, *, window_seconds: int = 30 * 60, interval_seconds: int = 1):
        maxlen = max(1, int(window_seconds / max(interval_seconds, 1)))
        self.snapshots: deque[ResourceSnapshot] = deque(maxlen=maxlen)

    def add(self, snapshot: ResourceSnapshot) -> None:
        self.snapshots.append(snapshot)

    @property
    def latest(self) -> ResourceSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def cpu_percent_values(self) -> list[float]:
        return [snapshot.cpu_percent for snapshot in self.snapshots]

    @property
    def ram_gib_values(self) -> list[float]:
        return [snapshot.ram_bytes / 1024**3 for snapshot in self.snapshots]

    @property
    def peak_ram_gib_values(self) -> list[float]:
        return [snapshot.peak_ram_bytes / 1024**3 for snapshot in self.snapshots]

    @property
    def process_count_values(self) -> list[float]:
        return [float(snapshot.process_count) for snapshot in self.snapshots]


class ProcessResourceMonitor:
    def __init__(self, root_process: Any):
        self.root_process = root_process
        self.peak_ram_bytes = 0
        self._last_wall: float | None = None
        self._last_cpu_seconds: float | None = None
        self._clock = time.monotonic

    @classmethod
    def from_pid(cls, pid: int) -> "ProcessResourceMonitor":
        return cls(psutil.Process(pid))

    @classmethod
    def from_process(cls, process: Any) -> "ProcessResourceMonitor":
        return cls(process)

    def _processes(self) -> list[Any]:
        try:
            return [self.root_process, *self.root_process.children(recursive=True)]
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            return []

    def sample(self) -> ResourceSnapshot:
        now = self._clock()
        ram_bytes = 0
        cpu_seconds = 0.0
        process_count = 0

        for process in self._processes():
            try:
                memory = process.memory_info()
                times = process.cpu_times()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            ram_bytes += int(memory.rss)
            cpu_seconds += float(times.user) + float(times.system)
            process_count += 1

        self.peak_ram_bytes = max(self.peak_ram_bytes, ram_bytes)
        cpu_percent = 0.0
        if self._last_wall is not None and self._last_cpu_seconds is not None:
            elapsed = max(now - self._last_wall, 0.0)
            if elapsed > 0:
                cpu_percent = max(
                    0.0,
                    ((cpu_seconds - self._last_cpu_seconds) / elapsed) * 100.0,
                )

        self._last_wall = now
        self._last_cpu_seconds = cpu_seconds
        return ResourceSnapshot(
            cpu_percent=cpu_percent,
            ram_bytes=ram_bytes,
            peak_ram_bytes=self.peak_ram_bytes,
            process_count=process_count,
        )
