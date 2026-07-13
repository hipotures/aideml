import json

from scripts.show_codex_token_usage import collect_token_usage


def _write_response(path, *, thread_id, turn_id, input_tokens):
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "info": {
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "thread_action": "resume",
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
    incomplete = artifacts / "20260713T000000-new-144" / "response.json"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text('{"status":"failed"}\n', encoding="utf-8")

    rows = collect_token_usage(tmp_path / "run")

    assert [row.step for row in rows] == [143, 147]
    assert [row.thread_id for row in rows] == ["thread-a", "thread-a"]
    assert rows[0].input_tokens == 143
    assert rows[0].cached_input_tokens == 10
