import json
import subprocess

import pytest

from aide.backend import backend_codex
from aide.backend.utils import FunctionSpec


def test_codex_backend_uses_cli_with_reasoning_effort(tmp_path, monkeypatch):
    seen = {}

    class FakeTmp:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["stdin"] = kwargs["input"]
        response_path = cmd[cmd.index("--output-last-message") + 1]
        assert response_path == str(tmp_path / "response_raw.txt")
        (tmp_path / "response_raw.txt").write_text("answer")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(backend_codex.tempfile, "TemporaryDirectory", lambda prefix: FakeTmp())
    monkeypatch.setattr(backend_codex.subprocess, "run", fake_run)

    output, *_ = backend_codex.query(
        system_message="system",
        user_message="user",
        model="gpt-5.5",
        reasoning_effort="low",
    )

    assert output == "answer"
    assert seen["cmd"][:6] == [
        "codex",
        "--ask-for-approval",
        "never",
        "exec",
        "--ignore-user-config",
        "--sandbox",
    ]
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == "gpt-5.5"
    assert 'model_reasoning_effort="low"' in seen["cmd"]
    assert "system" in seen["stdin"]
    assert "user" in seen["stdin"]


def test_codex_backend_parses_schema_response(tmp_path, monkeypatch):
    class FakeTmp:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_run(cmd, **kwargs):
        assert "--output-schema" in cmd
        assert cmd[cmd.index("--output-schema") + 1] == str(tmp_path / "schema.json")
        assert cmd[cmd.index("--output-last-message") + 1] == str(
            tmp_path / "response_raw.txt"
        )
        (tmp_path / "response_raw.txt").write_text('{"is_bug": false, "metric": 0.9}')
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(backend_codex.tempfile, "TemporaryDirectory", lambda prefix: FakeTmp())
    monkeypatch.setattr(backend_codex.subprocess, "run", fake_run)
    spec = FunctionSpec(
        name="submit_review",
        description="review",
        json_schema={
            "type": "object",
            "properties": {
                "is_bug": {"type": "boolean"},
                "metric": {"type": "number"},
            },
            "required": ["is_bug", "metric"],
        },
    )

    output, *_ = backend_codex.query(
        system_message="system",
        user_message=None,
        func_spec=spec,
        model="gpt-5.5",
    )

    assert output == {"is_bug": False, "metric": 0.9}
    assert json.loads((tmp_path / "schema.json").read_text()) == spec.json_schema


def test_codex_backend_reports_json_event_error_when_stderr_is_empty(
    tmp_path, monkeypatch
):
    class FakeTmp:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_run(cmd, **kwargs):
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "abc"}),
                json.dumps(
                    {
                        "type": "error",
                        "message": "You've hit your usage limit for GPT-5.3-Codex-Spark.",
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.failed",
                        "error": {
                            "message": "Switch to another model now, or try again later."
                        },
                    }
                ),
            ]
        )
        return subprocess.CompletedProcess(cmd, 1, stdout=stdout, stderr="")

    monkeypatch.setattr(backend_codex.tempfile, "TemporaryDirectory", lambda prefix: FakeTmp())
    monkeypatch.setattr(backend_codex.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        backend_codex.query(
            system_message="system",
            user_message=None,
            model="gpt-5.3-codex-spark",
        )

    message = str(exc_info.value)
    assert "Codex CLI failed with exit code 1" in message
    assert "Switch to another model now" in message
    assert "codex_events.jsonl" not in message


def test_codex_backend_reports_timeout_and_preserves_partial_logs(
    tmp_path, monkeypatch
):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=7,
            output='{"event":"started"}\n',
            stderr="still running\n",
        )

    monkeypatch.setattr(backend_codex.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        backend_codex.query(
            system_message="system",
            user_message=None,
            model="gpt-5.5",
            timeout=7,
            llm_log_dir=tmp_path,
        )

    message = str(exc_info.value)
    assert "Codex CLI timed out after 7 seconds" in message
    assert (tmp_path / "codex_events.jsonl").read_text() == '{"event":"started"}\n'
    assert (tmp_path / "stderr.log").read_text() == "still running\n"
