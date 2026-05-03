from pathlib import Path

from aide.backend import query
from aide.backend.utils import log_llm_exchange


def test_log_llm_exchange_writes_pretty_json_to_run_log(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("AIDE_RUN_ID", "2-test-run")

    log_llm_exchange(
        phase="request",
        provider="openai",
        sequence_id=7,
        payload={
            "model": "qwen35b",
            "messages": [{"role": "user", "content": '{"a":1}'}],
            "raw_json": '{"a":1}',
            "response": {"nested": {"value": 1}},
        },
    )

    output = (tmp_path / "llm_communication.md").read_text()

    assert "REQUEST" in output
    assert "run=2-test-run" in output
    assert "llm_call=0007" in output
    assert '"nested": {' in output
    assert '{\n  "a": 1\n}' in output


def test_backend_query_logs_request_and_response(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("AIDE_RUN_ID", "2-test-run")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setitem(query.__globals__, "_llm_call_counter", 0)

    def fake_query_func(**kwargs):
        return (
            {"ok": True},
            1.25,
            10,
            5,
            {"model": kwargs["model"]},
        )

    monkeypatch.setattr("aide.backend.determine_provider", lambda model: "openai")
    monkeypatch.setitem(
        query.__globals__["provider_to_query_func"], "openai", fake_query_func
    )

    result = query(
        system_message={"Instructions": {"Use JSON": ["yes"]}},
        user_message='{"input": true}',
        model="qwen35b",
        temperature=0.2,
    )

    output = Path(tmp_path / "llm_communication.md").read_text()

    assert result == {"ok": True}
    assert "REQUEST" in output
    assert "RESPONSE" in output
    assert "llm_call=" in output
    assert "# Instructions" in output
    assert '{\n  "input": true\n}' in output
    assert '"ok": true' in output


def test_backend_query_continues_llm_call_counter_from_existing_log(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AIDE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("AIDE_RUN_ID", "2-test-run")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setitem(query.__globals__, "_llm_call_counter", 0)
    (tmp_path / "llm_communication.md").write_text(
        "## REQUEST | 2026-05-03T02:00:00 | run=2-test-run | llm_call=0010\n",
        encoding="utf-8",
    )

    def fake_query_func(**kwargs):
        return ("ok", 0.1, 1, 1, {"model": kwargs["model"]})

    monkeypatch.setattr("aide.backend.determine_provider", lambda model: "openai")
    monkeypatch.setitem(
        query.__globals__["provider_to_query_func"], "openai", fake_query_func
    )

    query(system_message="system", user_message=None, model="qwen35b")

    output = Path(tmp_path / "llm_communication.md").read_text()
    assert "llm_call=0011" in output
