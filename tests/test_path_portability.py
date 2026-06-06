from pathlib import Path

from aide.utils.path_portability import (
    resolve_portable_path,
    sanitize_persisted_payload,
    sanitize_text,
    to_portable_path,
)


def test_to_portable_path_prefers_project_relative(tmp_path):
    project = tmp_path / "project"
    path = project / "logs" / "run-a" / "artifacts" / "node" / "submission.csv"

    assert to_portable_path(path, project_root=project) == (
        "logs/run-a/artifacts/node/submission.csv"
    )


def test_resolve_legacy_home_alias_rebases_to_current_base(monkeypatch):
    monkeypatch.setenv("AIDEML_PATH_BASE", "/home/user")
    monkeypatch.setenv("AIDEML_PATH_BASE_ALIASES", "/home/xai")

    assert resolve_portable_path("/home/xai/DEV/aideml/logs/run-a") == Path(
        "/home/user/DEV/aideml/logs/run-a"
    )


def test_sanitize_payload_replaces_absolute_paths(monkeypatch):
    monkeypatch.setenv("AIDEML_PATH_BASE", "/home/user")
    monkeypatch.setenv("AIDEML_PATH_BASE_ALIASES", "/home/xai")

    payload = {
        "trace": "File /home/xai/DEV/aideml/aide/agent.py:10 failed",
        "path": "/home/xai/.cache/uv/build",
    }

    assert sanitize_persisted_payload(payload) == {
        "trace": "File aide/agent.py:10 failed",
        "path": "$AIDEML_PATH_BASE/.cache/uv/build",
    }
    assert sanitize_text("see /home/xai/DEV/aideml/logs/run") == (
        "see logs/run"
    )
