from pathlib import Path
from types import SimpleNamespace

from aide.backend import determine_provider, query
from aide.backend import backend_openai
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


def test_gpt_models_use_codex_provider_not_openai_api():
    assert determine_provider("gpt-5.5") == "codex"
    assert determine_provider("gpt-5.4-mini") == "codex"
    assert determine_provider("o4-mini") == "codex"


def test_openai_responses_api_receives_reasoning_effort(monkeypatch):
    captured = {}

    def fake_backoff_create(create_fn, exceptions, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output=[],
            output_text="ok",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            model="gpt-5.4-mini",
        )

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(backend_openai, "_setup_openai_client", lambda: None)
    monkeypatch.setattr(backend_openai, "_client", SimpleNamespace(responses=SimpleNamespace(create=object())))
    monkeypatch.setattr(backend_openai, "backoff_create", fake_backoff_create)

    output, *_ = backend_openai.query(
        system_message="system",
        user_message=None,
        model="gpt-5.4-mini",
        reasoning_effort="low",
    )

    assert output == "ok"
    assert captured["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in captured


def test_local_openai_compatible_chat_api_does_not_receive_reasoning_effort(
    monkeypatch,
):
    captured = {}

    def fake_backoff_create(create_fn, exceptions, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            model="gemma-4-31B",
        )

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8081/v1")
    monkeypatch.setattr(backend_openai, "_setup_openai_client", lambda: None)
    monkeypatch.setattr(backend_openai, "_setup_custom_client", lambda: None)
    monkeypatch.setattr(
        backend_openai,
        "_custom_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=object()))),
    )
    monkeypatch.setattr(backend_openai, "backoff_create", fake_backoff_create)

    output, *_ = backend_openai.query(
        system_message="system",
        user_message=None,
        model="gemma-4-31B",
        reasoning_effort="low",
    )

    assert output == "ok"
    assert "reasoning" not in captured
    assert "reasoning_effort" not in captured
