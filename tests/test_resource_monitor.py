from aide.utils.resource_monitor import (
    ProcessResourceMonitor,
    ResourceHistory,
    ResourceSnapshot,
    _parse_nvidia_smi_gpu_stats,
    downsample_max,
)


class FakeMemory:
    def __init__(self, rss):
        self.rss = rss


class FakeTimes:
    def __init__(self, user, system):
        self.user = user
        self.system = system


class FakeProcess:
    def __init__(self, *, rss, user, system, children=None):
        self.rss = rss
        self.user = user
        self.system = system
        self._children = children or []

    def children(self, recursive=True):
        return list(self._children)

    def memory_info(self):
        return FakeMemory(self.rss)

    def cpu_times(self):
        return FakeTimes(self.user, self.system)


def test_process_resource_monitor_sums_process_tree_and_tracks_peak():
    child = FakeProcess(rss=2 * 1024**3, user=4.0, system=1.0)
    root = FakeProcess(rss=1 * 1024**3, user=10.0, system=2.0, children=[child])
    monitor = ProcessResourceMonitor.from_process(root, gpu_sampler=lambda: None)
    monitor._clock = lambda: 100.0
    first = monitor.sample()

    assert first.cpu_percent == 0.0
    assert first.ram_bytes == 3 * 1024**3
    assert first.peak_ram_bytes == 3 * 1024**3
    assert first.process_count == 2

    root.user = 13.0
    root.system = 3.0
    child.user = 5.0
    child.system = 1.0
    child.rss = 4 * 1024**3
    monitor._clock = lambda: 101.0
    second = monitor.sample()

    assert second.cpu_percent == 500.0
    assert second.ram_bytes == 5 * 1024**3
    assert second.peak_ram_bytes == 5 * 1024**3
    assert second.process_count == 2


def test_resource_history_keeps_sliding_window_and_latest_values():
    history = ResourceHistory(window_seconds=5, interval_seconds=2)

    history.add(
        ResourceSnapshot(
            cpu_percent=100.0,
            ram_bytes=1 * 1024**3,
            peak_ram_bytes=1 * 1024**3,
            process_count=1,
        )
    )
    history.add(
        ResourceSnapshot(
            cpu_percent=200.0,
            ram_bytes=2 * 1024**3,
            peak_ram_bytes=2 * 1024**3,
            process_count=2,
        )
    )
    history.add(
        ResourceSnapshot(
            cpu_percent=300.0,
            ram_bytes=3 * 1024**3,
            peak_ram_bytes=4 * 1024**3,
            process_count=3,
        )
    )

    assert history.latest is not None
    assert history.latest.cpu_percent == 300.0
    assert history.cpu_percent_values == [200.0, 300.0]
    assert history.ram_gib_values == [2.0, 3.0]
    assert history.peak_ram_gib_values == [2.0, 4.0]
    assert history.process_count_values == [2.0, 3.0]


def test_resource_history_exposes_gpu_values_when_available():
    history = ResourceHistory(window_seconds=5, interval_seconds=1)

    history.add(
        ResourceSnapshot(
            cpu_percent=100.0,
            ram_bytes=1 * 1024**3,
            peak_ram_bytes=1 * 1024**3,
            process_count=1,
            gpu_percent=37.0,
            gpu_memory_used_bytes=7 * 1024**3,
            gpu_memory_total_bytes=24 * 1024**3,
            gpu_power_draw_watts=122.5,
            gpu_power_limit_watts=450.0,
            gpu_temperature_celsius=61.0,
        )
    )

    assert history.gpu_percent_values == [37.0]
    assert history.gpu_memory_used_gib_values == [7.0]
    assert history.gpu_power_draw_watts_values == [122.5]
    assert history.gpu_temperature_celsius_values == [61.0]


def test_parse_nvidia_smi_gpu_stats_uses_busiest_nvidia_gpu():
    output = (
        "0, NVIDIA GeForce RTX 4090, 12, 2048, 24564, 80.25, 450.00, 45\n"
        "1, NVIDIA RTX A6000, 76, 32768, 49140, 241.80, 300.00, 67\n"
    )

    stats = _parse_nvidia_smi_gpu_stats(output)

    assert stats is not None
    assert stats.gpu_percent == 76.0
    assert stats.gpu_memory_used_bytes == 32768 * 1024**2
    assert stats.gpu_memory_total_bytes == 49140 * 1024**2
    assert stats.gpu_power_draw_watts == 241.8
    assert stats.gpu_power_limit_watts == 300.0
    assert stats.gpu_temperature_celsius == 67.0


def test_downsample_max_preserves_spikes_when_compressing_series():
    values = [1.0, 100.0, 2.0, 3.0, 80.0, 4.0]

    assert downsample_max(values, width=3) == [100.0, 3.0, 80.0]
