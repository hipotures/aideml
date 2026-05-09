import os
import queue
import resource
import signal

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
        if not self.results:
            raise queue.Empty
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


def test_interpreter_executes_main_guard_like_script(tmp_path):
    interpreter = Interpreter(tmp_path, timeout=10, memory_limit_gb=None)
    try:
        result = interpreter.run(
            "def main():\n"
            "    print('main ran')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
    finally:
        interpreter.cleanup_session()

    assert result.exc_type is None
    assert "main ran" in "".join(result.term_out)


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


def test_missing_eof_marker_returns_execution_error(tmp_path):
    interpreter = _interpreter_with_queues(
        tmp_path,
        [
            ("state:ready",),
            ("state:finished", None, None, None),
        ],
    )
    interpreter.result_outq = FakeResultQueue([])
    interpreter.output_collection_timeout = 0.0

    result = interpreter.run("print('ok')", reset_session=False)

    assert result.exc_type == "RuntimeError"
    assert result.exc_info == {"args": ["REPL output stream ended before EOF marker"]}
    assert "REPL output stream ended before EOF marker" in "".join(result.term_out)


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


def test_cleanup_session_joins_after_sigkill_before_closing(tmp_path, monkeypatch):
    interpreter = Interpreter(tmp_path, timeout=60)
    sent_signals = []

    class StubbornProcess:
        pid = 12345
        exitcode = None
        closed = False

        def terminate(self):
            pass

        def join(self, timeout=None):
            if signal.SIGKILL in sent_signals:
                self.exitcode = -9

        def kill(self):
            pass

        def close(self):
            if self.exitcode is None:
                raise ValueError("Cannot close a process while it is still running.")
            self.closed = True

    process = StubbornProcess()
    interpreter.process = process
    monkeypatch.setattr("aide.interpreter.os.getpgid", lambda pid: 12345)
    monkeypatch.setattr(
        "aide.interpreter.os.killpg",
        lambda pgid, sig: sent_signals.append(sig),
    )

    interpreter.cleanup_session()

    assert sent_signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.closed is True
    assert interpreter.process is None


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


def test_child_process_streams_output_to_artifact_log(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifact"
    monkeypatch.setenv("AIDE_NODE_ARTIFACT_DIR", str(artifact_dir))
    interpreter = Interpreter(tmp_path, timeout=10)

    result = interpreter.run("print('legacy progress', flush=True)")
    interpreter.cleanup_session()

    assert result.exc_type is None
    log_path = artifact_dir / "process_stdout.log"
    assert log_path.exists()
    assert "legacy progress" in log_path.read_text(encoding="utf-8")
