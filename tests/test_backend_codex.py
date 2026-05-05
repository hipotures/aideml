import json
import subprocess

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
