import json

import pytest

from aide.backend import codex_app_server


class _RecordingStdin:
    def __init__(self):
        self.text = ""

    def write(self, value):
        self.text += value

    def flush(self):
        return None

    def close(self):
        return None


class _FakeProcess:
    def __init__(self):
        self.stdin = _RecordingStdin()
        self.stdout = object()
        self.returncode = None
        self.pid = 12345

    def poll(self):
        return self.returncode


def _install_protocol(monkeypatch, messages):
    process = _FakeProcess()
    seen = {}

    def fake_popen(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return process

    queue = list(messages)
    monkeypatch.setattr(codex_app_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        codex_app_server,
        "_read_message",
        lambda _proc, *, timeout_seconds: queue.pop(0) if queue else None,
    )
    monkeypatch.setattr(codex_app_server, "_terminate_process", lambda _proc: None)
    monkeypatch.setattr(codex_app_server, "app_server_env", lambda: {"CODEX_HOME": "/tmp/codex"})
    return process, seen


def test_app_server_protocol_returns_ids_usage_and_live_logs(tmp_path, monkeypatch):
    process, seen = _install_protocol(
        monkeypatch,
        [
            {"id": 0, "result": {"userAgent": "test"}},
            {"id": 1, "result": {"thread": {"id": "thread-1"}}},
            {"id": 2, "result": {"turn": {"id": "turn-1"}}},
            {
                "method": "item/agentMessage/delta",
                "params": {"delta": "partial"},
            },
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "item": {
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": "final answer",
                    },
                },
            },
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "tokenUsage": {
                        "last": {"inputTokens": 20, "outputTokens": 7}
                    },
                },
            },
        ],
    )

    schema = {"type": "object"}
    result = codex_app_server.invoke_codex_app_server(
        prompt="hello",
        model="gpt-5.5",
        reasoning_effort="high",
        web_search=True,
        work_dir=tmp_path,
        timeout=5,
        output_schema=schema,
        log_dir=tmp_path,
    )

    assert result.text == "final answer"
    assert result.thread_id == "thread-1"
    assert result.turn_id == "turn-1"
    assert (result.input_tokens, result.output_tokens) == (20, 7)
    assert seen["command"][:2] == ["codex", "app-server"]
    assert "exec" not in seen["command"]
    assert seen["kwargs"]["start_new_session"] is True

    requests = [json.loads(line) for line in process.stdin.text.splitlines()]
    assert [request.get("method") for request in requests[:4]] == [
        "initialize",
        "initialized",
        "thread/start",
        "turn/start",
    ]
    assert requests[2]["params"]["ephemeral"] is False
    assert requests[2]["params"]["sandbox"] == "read-only"
    assert requests[2]["params"]["approvalPolicy"] == "never"
    assert requests[3]["params"]["outputSchema"] == schema
    assert requests[3]["params"]["effort"] == "high"

    events = (tmp_path / "codex_events.jsonl").read_text()
    assert "item/completed" in events
    assert "thread/tokenUsage/updated" in events
    assert "item/agentMessage/delta" not in events
    rpc = (tmp_path / "codex_rpc.jsonl").read_text()
    assert '"id": 0' in rpc
    assert '"id": 2' in rpc
    profile = (tmp_path / "codex_profile.toml").read_text()
    assert 'transport = "codex_app_server"' in profile
    assert "ephemeral = false" in profile


def test_app_server_rpc_error_is_not_retried_with_codex_exec(tmp_path, monkeypatch):
    process, seen = _install_protocol(
        monkeypatch,
        [{"id": 0, "error": {"message": "initialization failed"}}],
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        codex_app_server.invoke_codex_app_server(
            prompt="hello",
            model="gpt-5.5",
            reasoning_effort="low",
            web_search=False,
            work_dir=tmp_path,
            timeout=5,
            log_dir=tmp_path,
        )

    assert seen["command"][:2] == ["codex", "app-server"]
    assert "exec" not in seen["command"]
    assert len(process.stdin.text.splitlines()) == 1
    assert "initialization failed" in (tmp_path / "codex_rpc.jsonl").read_text()


def test_app_server_preserves_partial_response_after_turn_failure(tmp_path, monkeypatch):
    _install_protocol(
        monkeypatch,
        [
            {"id": 0, "result": {"userAgent": "test"}},
            {"id": 1, "result": {"thread": {"id": "thread-1"}}},
            {"id": 2, "result": {"turn": {"id": "turn-1"}}},
            {
                "method": "item/agentMessage/delta",
                "params": {"delta": "useful partial answer"},
            },
            {
                "method": "error",
                "params": {"message": "turn failed"},
            },
        ],
    )

    with pytest.raises(RuntimeError, match="turn failed"):
        codex_app_server.invoke_codex_app_server(
            prompt="hello",
            model="gpt-5.5",
            reasoning_effort="low",
            web_search=False,
            work_dir=tmp_path,
            timeout=5,
            log_dir=tmp_path,
        )

    assert (tmp_path / "response_raw.txt").read_text() == "useful partial answer"


def test_app_server_env_uses_isolated_home_and_only_copies_auth(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "auth.json").write_text('{"token":"secret"}')
    (source / "config.toml").write_text('model = "should-not-copy"')
    monkeypatch.setenv("CODEX_HOME", str(source))

    env = codex_app_server.app_server_env(target)

    assert env["CODEX_HOME"] == str(target)
    assert env["CODEX_SQLITE_HOME"] == str(target)
    assert (target / "auth.json").read_text() == '{"token":"secret"}'
    assert not (target / "config.toml").exists()


def test_app_server_command_disables_external_capabilities():
    command = codex_app_server.app_server_command(web_search=False)

    assert command[:2] == ["codex", "app-server"]
    assert "exec" not in command
    assert "plugins" in command
    assert "multi_agent" in command
    assert "mcp_servers={}" in command
    assert 'web_search="disabled"' in command
