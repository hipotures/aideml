"""Process-tree resource sampling helpers."""

from __future__ import annotations

import csv
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from io import StringIO
from typing import Any

import psutil


DEFAULT_RESOURCE_HISTORY_WINDOW_SECONDS = 5 * 60


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_percent: float
    ram_bytes: int
    peak_ram_bytes: int
    process_count: int
    gpu_percent: float | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_memory_total_bytes: int | None = None
    gpu_power_draw_watts: float | None = None
    gpu_power_limit_watts: float | None = None
    gpu_temperature_celsius: float | None = None


@dataclass(frozen=True)
class NvidiaGpuStats:
    gpu_percent: float | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_memory_total_bytes: int | None = None
    gpu_power_draw_watts: float | None = None
    gpu_power_limit_watts: float | None = None
    gpu_temperature_celsius: float | None = None


def _parse_float(value: str) -> float | None:
    cleaned = value.strip().replace("[", "").replace("]", "")
    if not cleaned or cleaned.upper() == "N/A":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _mib_to_bytes(value: float | None) -> int | None:
    if value is None:
        return None
    return int(value * 1024**2)


def _parse_nvidia_smi_gpu_stats(output: str) -> NvidiaGpuStats | None:
    rows: list[NvidiaGpuStats] = []
    reader = csv.reader(StringIO(output))
    for row in reader:
        if len(row) != 8:
            continue
        gpu_percent = _parse_float(row[2])
        memory_used_mib = _parse_float(row[3])
        memory_total_mib = _parse_float(row[4])
        power_draw_watts = _parse_float(row[5])
        power_limit_watts = _parse_float(row[6])
        temperature_celsius = _parse_float(row[7])
        rows.append(
            NvidiaGpuStats(
                gpu_percent=gpu_percent,
                gpu_memory_used_bytes=_mib_to_bytes(memory_used_mib),
                gpu_memory_total_bytes=_mib_to_bytes(memory_total_mib),
                gpu_power_draw_watts=power_draw_watts,
                gpu_power_limit_watts=power_limit_watts,
                gpu_temperature_celsius=temperature_celsius,
            )
        )
    if not rows:
        return None
    return max(
        rows,
        key=lambda stats: (
            stats.gpu_percent if stats.gpu_percent is not None else -1.0,
            stats.gpu_memory_used_bytes if stats.gpu_memory_used_bytes is not None else -1,
        ),
    )


def sample_nvidia_gpu_stats() -> NvidiaGpuStats | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _parse_nvidia_smi_gpu_stats(result.stdout)


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
    def __init__(
        self,
        *,
        window_seconds: int = DEFAULT_RESOURCE_HISTORY_WINDOW_SECONDS,
        interval_seconds: int = 1,
    ):
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

    @property
    def gpu_percent_values(self) -> list[float]:
        return [
            snapshot.gpu_percent
            for snapshot in self.snapshots
            if snapshot.gpu_percent is not None
        ]

    @property
    def gpu_memory_used_gib_values(self) -> list[float]:
        return [
            snapshot.gpu_memory_used_bytes / 1024**3
            for snapshot in self.snapshots
            if snapshot.gpu_memory_used_bytes is not None
        ]

    @property
    def gpu_power_draw_watts_values(self) -> list[float]:
        return [
            snapshot.gpu_power_draw_watts
            for snapshot in self.snapshots
            if snapshot.gpu_power_draw_watts is not None
        ]

    @property
    def gpu_temperature_celsius_values(self) -> list[float]:
        return [
            snapshot.gpu_temperature_celsius
            for snapshot in self.snapshots
            if snapshot.gpu_temperature_celsius is not None
        ]


class ProcessResourceMonitor:
    def __init__(self, root_process: Any, *, gpu_sampler=sample_nvidia_gpu_stats):
        self.root_process = root_process
        self.gpu_sampler = gpu_sampler
        self.peak_ram_bytes = 0
        self._last_wall: float | None = None
        self._last_cpu_seconds: float | None = None
        self._clock = time.monotonic

    @classmethod
    def from_pid(cls, pid: int, *, gpu_sampler=sample_nvidia_gpu_stats) -> "ProcessResourceMonitor":
        return cls(psutil.Process(pid), gpu_sampler=gpu_sampler)

    @classmethod
    def from_process(
        cls,
        process: Any,
        *,
        gpu_sampler=sample_nvidia_gpu_stats,
    ) -> "ProcessResourceMonitor":
        return cls(process, gpu_sampler=gpu_sampler)

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
        gpu_stats = self.gpu_sampler()
        return ResourceSnapshot(
            cpu_percent=cpu_percent,
            ram_bytes=ram_bytes,
            peak_ram_bytes=self.peak_ram_bytes,
            process_count=process_count,
            gpu_percent=gpu_stats.gpu_percent if gpu_stats is not None else None,
            gpu_memory_used_bytes=(
                gpu_stats.gpu_memory_used_bytes if gpu_stats is not None else None
            ),
            gpu_memory_total_bytes=(
                gpu_stats.gpu_memory_total_bytes if gpu_stats is not None else None
            ),
            gpu_power_draw_watts=(
                gpu_stats.gpu_power_draw_watts if gpu_stats is not None else None
            ),
            gpu_power_limit_watts=(
                gpu_stats.gpu_power_limit_watts if gpu_stats is not None else None
            ),
            gpu_temperature_celsius=(
                gpu_stats.gpu_temperature_celsius if gpu_stats is not None else None
            ),
        )
