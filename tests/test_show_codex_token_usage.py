import json

from scripts.show_codex_token_usage import collect_token_usage, render_table


def _write_response(path, *, thread_id, turn_id, input_tokens, action="resume"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "info": {
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "thread_action": action,
                    "usage": {
                        "tokenUsage": {
                            "last": {
                                "inputTokens": input_tokens,
                                "cachedInputTokens": 10,
                                "outputTokens": 5,
                                "totalTokens": input_tokens + 5,
                            },
                            "total": {"totalTokens": input_tokens + 100},
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_collect_token_usage_sorts_steps_and_skips_missing_usage(tmp_path):
    artifacts = tmp_path / "run" / "artifacts"
    _write_response(
        artifacts / "20260713T000000-new-147" / "response.json",
        thread_id="thread-a",
        turn_id="turn-147",
        input_tokens=147,
    )
    _write_response(
        artifacts / "20260713T000000-new-143" / "response.json",
        thread_id="thread-a",
        turn_id="turn-143",
        input_tokens=143,
    )
    _write_response(
        artifacts / "20260713T000000-new-143" / "review_response.json",
        thread_id="thread-feedback",
        turn_id="turn-feedback-143",
        input_tokens=100,
        action="start",
    )
    _write_response(
        artifacts / "20260713T000000-old-51" / "response.json",
        thread_id="thread-old",
        turn_id="turn-51",
        input_tokens=51,
        action="",
    )
    incomplete = artifacts / "20260713T000000-new-144" / "response.json"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text('{"status":"failed"}\n', encoding="utf-8")

    rows = collect_token_usage(tmp_path / "run")

    assert [row.step for row in rows] == [143, 143, 147]
    assert [row.agent for row in rows] == ["code", "feedback", "code"]
    assert [row.thread_id for row in rows] == [
        "thread-a",
        "thread-feedback",
        "thread-a",
    ]
    assert rows[0].input_tokens == 143
    assert rows[0].cached_input_tokens == 10

    full_rows = collect_token_usage(tmp_path / "run", full_view=True)
    assert [row.step for row in full_rows] == [51, 143, 143, 147]
    assert full_rows[0].action == ""

    table = render_table(tmp_path / "run", rows)
    assert table.rows[0].style == table.rows[1].style
    assert table.rows[1].style != table.rows[2].style
