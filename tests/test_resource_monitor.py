from aide.utils.resource_monitor import ProcessResourceMonitor


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
    monitor = ProcessResourceMonitor.from_process(root)
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
