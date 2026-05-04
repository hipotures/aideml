"""Process-tree resource sampling helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_percent: float
    ram_bytes: int
    peak_ram_bytes: int
    process_count: int


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
