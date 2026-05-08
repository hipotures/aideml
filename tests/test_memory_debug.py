import json

from aide.utils.memory_debug import MemoryDebugLogger


class FakeMemoryInfo:
    rss = 100
    vms = 200
    shared = 30


class FakeFullMemoryInfo(FakeMemoryInfo):
    pss = 70
    uss = 50


class FakeSystemMemory:
    total = 1000
    available = 700
    used = 300
    percent = 30.0


class FakeChildProcess:
    pid = 222
    ppid_value = 111

    def ppid(self):
        return self.ppid_value

    def name(self):
        return "child"

    def status(self):
        return "running"

    def cmdline(self):
        return ["python", "runfile.py"]

    def memory_info(self):
        return FakeMemoryInfo()

    def memory_full_info(self):
        return FakeFullMemoryInfo()

    def num_threads(self):
        return 3


class FakeRootProcess:
    pid = 111

    def memory_info(self):
        return FakeMemoryInfo()

    def memory_full_info(self):
        return FakeFullMemoryInfo()

    def num_fds(self):
        return 9

    def num_threads(self):
        return 4

    def children(self, recursive=True):
        return [FakeChildProcess()]


def test_memory_debug_logger_writes_parent_system_and_child_stats(tmp_path):
    log_path = tmp_path / "debug.log"
    logger = MemoryDebugLogger(
        enabled=True,
        path=log_path,
        run_id="2-test-run",
        root_process_factory=FakeRootProcess,
        system_memory_factory=FakeSystemMemory,
    )

    logger.log("before_generate", phase="generate", extra={"step": 12})

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["event"] == "before_generate"
    assert payload["phase"] == "generate"
    assert payload["run_id"] == "2-test-run"
    assert payload["extra"] == {"step": 12}
    assert payload["parent"]["pid"] == 111
    assert payload["parent"]["rss_bytes"] == 100
    assert payload["parent"]["vms_bytes"] == 200
    assert payload["parent"]["pss_bytes"] == 70
    assert payload["parent"]["uss_bytes"] == 50
    assert payload["parent"]["fd_count"] == 9
    assert payload["parent"]["thread_count"] == 4
    assert payload["system"]["available_bytes"] == 700
    assert payload["children"][0]["pid"] == 222
    assert payload["children"][0]["pss_bytes"] == 70


def test_disabled_memory_debug_logger_does_not_create_file(tmp_path):
    log_path = tmp_path / "debug.log"
    logger = MemoryDebugLogger(enabled=False, path=log_path)

    logger.log("ignored")

    assert not log_path.exists()
