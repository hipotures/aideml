import pytest

from aide.interpreter import ExecutionInterrupted, Interpreter


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
