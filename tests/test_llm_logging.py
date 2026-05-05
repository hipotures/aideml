from types import SimpleNamespace

from aide.backend import determine_provider, query
from aide.backend import backend_codex, backend_openai


def test_backend_query_logs_request_and_response_to_artifact_files(
    tmp_path, monkeypatch
):
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
        llm_log_dir=tmp_path,
        llm_log_context={"phase": "generate", "node_id": "node-1"},
    )

    assert result == {"ok": True}
    assert not (tmp_path / "llm_communication.md").exists()
    request_md = (tmp_path / "request.md").read_text()
    assert "# Instructions" in request_md
    assert '"input": true' in request_md
    assert '"ok": true' in (tmp_path / "response.json").read_text()
    assert (tmp_path / "status.json").read_text().count("completed") == 1
    assert (tmp_path / "stderr.log").exists()
    assert (tmp_path / "provider_events.jsonl").exists()


def test_backend_query_uses_prefixed_review_files(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    def fake_query_func(**kwargs):
        return ("ok", 0.1, 1, 1, {"model": kwargs["model"]})

    monkeypatch.setattr("aide.backend.determine_provider", lambda model: "openai")
    monkeypatch.setitem(
        query.__globals__["provider_to_query_func"], "openai", fake_query_func
    )

    query(
        system_message="system",
        user_message=None,
        model="qwen35b",
        llm_log_dir=tmp_path,
        llm_log_prefix="review",
        llm_log_context={"phase": "review"},
    )

    assert (tmp_path / "review_request.md").exists()
    assert (tmp_path / "review_response_raw.txt").read_text() == "ok"
    assert (tmp_path / "review_status.json").exists()


def test_codex_backend_writes_codex_artifact_files(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        output_path = tmp_path / "response_raw.txt"
        output_path.write_text("answer", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout='{"event":"done"}\n', stderr="")

    monkeypatch.setattr(backend_codex.subprocess, "run", fake_run)

    output, *_ = backend_codex.query(
        system_message="system",
        user_message=None,
        model="gpt-5.5",
        reasoning_effort="low",
        llm_log_dir=tmp_path,
    )

    assert output == "answer"
    assert (tmp_path / "codex_events.jsonl").read_text() == '{"event":"done"}\n'
    assert (tmp_path / "stderr.log").read_text() == ""
    assert 'model = "gpt-5.5"' in (tmp_path / "codex_profile.toml").read_text()


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
    monkeypatch.setattr(
        backend_openai,
        "_client",
        SimpleNamespace(responses=SimpleNamespace(create=object())),
    )
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
