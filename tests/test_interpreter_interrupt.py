import os
import resource

import pytest

from aide.interpreter import ExecutionInterrupted, Interpreter
from aide.utils.resource_monitor import ResourceSnapshot


class FakeCodeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class FakeEventQueue:
    def __init__(self, events):
        self.events = list(events)

    def get(self, timeout=None):
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


class FakeResultQueue:
    def __init__(self, results):
        self.results = list(results)

    def empty(self):
        return not self.results

    def get(self, timeout=None):
        return self.results.pop(0)


class FakeProcess:
    pid = 12345

    def is_alive(self):
        return True


def _interpreter_with_queues(tmp_path, events):
    interpreter = Interpreter(tmp_path, timeout=60)
    interpreter.process = FakeProcess()
    interpreter.code_inq = FakeCodeQueue()
    interpreter.event_outq = FakeEventQueue(events)
    interpreter.result_outq = FakeResultQueue(["<|EOF|>"])
    return interpreter


def test_first_keyboard_interrupt_waits_for_execution_to_finish(tmp_path):
    interpreter = _interpreter_with_queues(
        tmp_path,
        [
            ("state:ready",),
            KeyboardInterrupt(),
            ("state:finished", None, None, None),
        ],
    )
    interrupts = []

    result = interpreter.run(
        "print('ok')",
        reset_session=False,
        interrupt_callback=interrupts.append,
    )

    assert interrupts == [1]
    assert result.exc_type is None
    assert result.term_out == [
        "Execution time: a moment seconds (time limit is a minute)."
    ]


def test_resource_callback_receives_execution_process_snapshot(tmp_path, monkeypatch):
    interpreter = _interpreter_with_queues(
        tmp_path,
        [
            ("state:ready",),
            ("state:finished", None, None, None),
        ],
    )
    snapshot = ResourceSnapshot(
        cpu_percent=120.0,
        ram_bytes=1024,
        peak_ram_bytes=2048,
        process_count=2,
    )

    class FakeMonitor:
        def sample(self):
            return snapshot

    monkeypatch.setattr(
        "aide.interpreter.ProcessResourceMonitor.from_pid",
        lambda pid: FakeMonitor(),
    )
    snapshots = []

    result = interpreter.run(
        "print('ok')",
        reset_session=False,
        resource_callback=snapshots.append,
    )

    assert result.exc_type is None
    assert snapshots == [snapshot]


def test_second_keyboard_interrupt_cleans_up_and_exits_cleanly(tmp_path, monkeypatch):
    interpreter = _interpreter_with_queues(
        tmp_path,
        [
            ("state:ready",),
            KeyboardInterrupt(),
            KeyboardInterrupt(),
        ],
    )
    interrupts = []
    cleaned = []
    monkeypatch.setattr(interpreter, "cleanup_session", lambda: cleaned.append(True))

    with pytest.raises(ExecutionInterrupted):
        interpreter.run(
            "print('ok')",
            reset_session=False,
            interrupt_callback=interrupts.append,
        )

    assert interrupts == [1, 2]
    assert cleaned == [True]


def test_child_process_runs_in_separate_process_group(tmp_path):
    interpreter = Interpreter(tmp_path, timeout=10)

    result = interpreter.run("import os\nprint(os.getpgrp())")
    interpreter.cleanup_session()

    child_process_group = int(result.term_out[0].strip())

    assert result.exc_type is None
    assert child_process_group != os.getpgrp()


def test_child_process_has_configured_memory_limit(tmp_path):
    interpreter = Interpreter(tmp_path, timeout=10, memory_limit_gb=80)

    result = interpreter.run(
        "import resource\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_AS)\n"
        "print(soft)\n"
        "print(hard)\n"
    )
    interpreter.cleanup_session()

    expected = 80 * 1024**3
    assert result.exc_type is None
    limits = [
        int(line.strip())
        for line in result.term_out
        if line.strip().isdigit()
    ]
    assert limits == [expected, expected]


def test_parent_process_memory_limit_is_not_changed_by_child_limit(tmp_path):
    before = resource.getrlimit(resource.RLIMIT_AS)
    interpreter = Interpreter(tmp_path, timeout=10, memory_limit_gb=80)

    result = interpreter.run("print('ok')")
    interpreter.cleanup_session()

    assert result.exc_type is None
    assert resource.getrlimit(resource.RLIMIT_AS) == before
