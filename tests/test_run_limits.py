import json

import pytest

from aide.run import (
    codex_rate_limit_stop_message,
    read_codex_primary_used_percent,
    validate_run_limits,
)
from aide.utils.config import _load_cfg, prep_cfg


def _write_events(path, values):
    lines = ["not-json"]
    lines.extend(
        json.dumps(
            {
                "method": "account/rateLimits/updated",
                "params": {"rateLimits": {"primary": {"usedPercent": value}}},
            }
        )
        for value in values
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_rate_limit_uses_latest_reported_primary_percentage(tmp_path):
    events_path = tmp_path / "codex_events.jsonl"
    _write_events(events_path, [19, 21])

    assert read_codex_primary_used_percent(events_path) == 21
    message = codex_rate_limit_stop_message(
        events_path=events_path,
        threshold=20,
        step=143,
    )

    assert message is not None
    assert "step 143" in message
    assert "usedPercent=21" in message
    assert "codex.limits.usedPercent=20" in message


@pytest.mark.parametrize("reported", [19, 20])
def test_rate_limit_does_not_stop_at_or_below_threshold(tmp_path, reported):
    events_path = tmp_path / "codex_events.jsonl"
    _write_events(events_path, [reported])

    assert (
        codex_rate_limit_stop_message(
            events_path=events_path,
            threshold=20,
            step=1,
        )
        is None
    )


def test_rate_limit_is_disabled_without_threshold_or_events(tmp_path):
    missing_path = tmp_path / "missing.jsonl"

    assert read_codex_primary_used_percent(missing_path) is None
    assert (
        codex_rate_limit_stop_message(
            events_path=missing_path,
            threshold=None,
            step=1,
        )
        is None
    )


@pytest.mark.parametrize("value", [-1, 101])
def test_rate_limit_config_rejects_invalid_threshold(tmp_path, value):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "goal"
    cfg.codex.limits.usedPercent = value
    cfg = prep_cfg(cfg)

    with pytest.raises(ValueError, match="codex.limits.usedPercent"):
        validate_run_limits(cfg)


def test_rate_limit_config_accepts_cli_override(tmp_path):
    cfg = _load_cfg(
        use_cli_args=True,
        cli_args=[
            f"data_dir={tmp_path}",
            "goal=goal",
            "codex.limits.usedPercent=20",
        ],
    )
    cfg = prep_cfg(cfg)

    validate_run_limits(cfg)
    assert cfg.codex.limits.usedPercent == 20
