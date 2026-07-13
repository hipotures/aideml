import json

import pytest

from aide.backend import backend_codex
from aide.backend.codex_app_server import CodexAppServerResult
from aide.backend.utils import FunctionSpec


def _result(text="answer"):
    return CodexAppServerResult(
        text=text,
        status="completed",
        thread_id="thread-1",
        turn_id="turn-1",
        duration_seconds=1.25,
        input_tokens=12,
        output_tokens=4,
        usage={"tokenUsage": {"last": {"inputTokens": 12, "outputTokens": 4}}},
    )


def test_codex_backend_uses_app_server_with_reasoning_effort(tmp_path, monkeypatch):
    seen = {}

    def fake_invoke(**kwargs):
        seen.update(kwargs)
        return _result()

    monkeypatch.setattr(backend_codex, "invoke_codex_app_server", fake_invoke)

    output, req_time, input_tokens, output_tokens, info = backend_codex.query(
        system_message="system",
        user_message="user",
        model="gpt-5.5",
        reasoning_effort="low",
        llm_log_dir=tmp_path,
    )

    assert output == "answer"
    assert seen["model"] == "gpt-5.5"
    assert seen["reasoning_effort"] == "low"
    assert seen["work_dir"] == tmp_path
    assert "system" in seen["prompt"]
    assert "user" in seen["prompt"]
    assert req_time == 1.25
    assert (input_tokens, output_tokens) == (12, 4)
    assert info["provider_kind"] == "codex_app_server"
    assert info["thread_id"] == "thread-1"
    assert info["turn_id"] == "turn-1"


def test_codex_backend_enables_search(tmp_path, monkeypatch):
    seen = {}

    def fake_invoke(**kwargs):
        seen.update(kwargs)
        return _result()

    monkeypatch.setattr(backend_codex, "invoke_codex_app_server", fake_invoke)

    backend_codex.query(
        system_message="system",
        user_message=None,
        model="gpt-5.5",
        web_search=True,
        llm_log_dir=tmp_path,
    )

    assert seen["web_search"] is True


def test_codex_backend_forwards_fork_metadata(tmp_path, monkeypatch):
    seen = {}

    def fake_invoke(**kwargs):
        seen.update(kwargs)
        return _result()

    monkeypatch.setattr(backend_codex, "invoke_codex_app_server", fake_invoke)

    *_, info = backend_codex.query(
        system_message="system",
        user_message=None,
        model="gpt-5.5",
        llm_log_dir=tmp_path,
        codex_fork_thread_id="thread-parent",
        codex_fork_turn_id="turn-parent",
    )

    assert seen["thread_id"] is None
    assert seen["fork_from"].thread_id == "thread-parent"
    assert seen["fork_from"].turn_id == "turn-parent"
    assert info["thread_action"] == "fork"


def test_codex_backend_passes_schema_and_parses_response(tmp_path, monkeypatch):
    seen = {}

    def fake_invoke(**kwargs):
        seen.update(kwargs)
        return _result('{"is_bug": false, "metric": 0.9}')

    monkeypatch.setattr(backend_codex, "invoke_codex_app_server", fake_invoke)
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
        llm_log_dir=tmp_path,
    )

    assert output == {"is_bug": False, "metric": 0.9}
    assert seen["output_schema"] == spec.json_schema
    assert json.loads((tmp_path / "schema.json").read_text()) == spec.json_schema


def test_codex_backend_propagates_app_server_error(tmp_path, monkeypatch):
    def fake_invoke(**kwargs):
        raise RuntimeError("Codex app-server turn failed: usage limit")

    monkeypatch.setattr(backend_codex, "invoke_codex_app_server", fake_invoke)

    with pytest.raises(RuntimeError, match="usage limit"):
        backend_codex.query(
            system_message="system",
            user_message=None,
            model="gpt-5.5",
            llm_log_dir=tmp_path,
        )


def test_codex_backend_propagates_timeout(tmp_path, monkeypatch):
    def fake_invoke(**kwargs):
        (tmp_path / "codex_events.jsonl").write_text('{"method":"turn/started"}\n')
        (tmp_path / "stderr.log").write_text("still running\n")
        raise TimeoutError("Codex app-server timed out after 7 seconds.")

    monkeypatch.setattr(backend_codex, "invoke_codex_app_server", fake_invoke)

    with pytest.raises(TimeoutError, match="timed out after 7 seconds"):
        backend_codex.query(
            system_message="system",
            user_message=None,
            model="gpt-5.5",
            timeout=7,
            llm_log_dir=tmp_path,
        )

    assert (tmp_path / "codex_events.jsonl").read_text() == (
        '{"method":"turn/started"}\n'
    )
    assert (tmp_path / "stderr.log").read_text() == "still running\n"
