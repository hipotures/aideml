from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_USAGE_REFRESH_SECONDS = 10.0


@dataclass(frozen=True)
class TokenUsageRow:
    step: int
    agent: str
    thread_id: str
    turn_id: str
    action: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    turn_total_tokens: int
    thread_total_tokens: int


def _integer(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _row_from_response(
    step: int,
    agent: str,
    response_path: Path,
) -> TokenUsageRow | None:
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    info = response.get("info")
    if not isinstance(info, dict):
        return None
    thread_id = info.get("thread_id")
    turn_id = info.get("turn_id")
    usage = info.get("usage")
    token_usage = usage.get("tokenUsage") if isinstance(usage, dict) else None
    last = token_usage.get("last") if isinstance(token_usage, dict) else None
    total = token_usage.get("total") if isinstance(token_usage, dict) else None
    if not all(isinstance(value, dict) for value in (last, total)) or not all(
        isinstance(value, str) and value for value in (thread_id, turn_id)
    ):
        return None

    input_tokens = _integer(last.get("inputTokens"))
    cached_input_tokens = _integer(last.get("cachedInputTokens"))
    output_tokens = _integer(last.get("outputTokens"))
    turn_total_tokens = _integer(last.get("totalTokens"))
    thread_total_tokens = _integer(total.get("totalTokens"))
    values = (
        input_tokens,
        cached_input_tokens,
        output_tokens,
        turn_total_tokens,
        thread_total_tokens,
    )
    if any(value is None for value in values):
        return None
    return TokenUsageRow(
        step=step,
        agent=agent,
        thread_id=thread_id,
        turn_id=turn_id,
        action=str(info.get("thread_action") or ""),
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        turn_total_tokens=turn_total_tokens,
        thread_total_tokens=thread_total_tokens,
    )


def collect_token_usage(
    run_dir: Path,
    *,
    full_view: bool = False,
) -> list[TokenUsageRow]:
    rows_by_key: dict[tuple[int, str], TokenUsageRow] = {}
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"AIDE artifacts directory not found: {artifacts_dir}")
    for artifact_dir in sorted(artifacts_dir.iterdir()):
        if not artifact_dir.is_dir():
            continue
        try:
            step = int(artifact_dir.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        for agent, response_name in (
            ("code", "response.json"),
            ("feedback", "review_response.json"),
        ):
            response_path = artifact_dir / response_name
            if not response_path.exists():
                continue
            row = _row_from_response(step, agent, response_path)
            if row is not None and (full_view or row.action):
                rows_by_key[(step, agent)] = row
    agent_order = {"code": 0, "feedback": 1}
    return sorted(
        rows_by_key.values(), key=lambda row: (row.step, agent_order[row.agent])
    )


class TokenUsageCache:
    def __init__(self) -> None:
        self.rows: list[TokenUsageRow] = []
        self.updated_at = 0.0
        self._refresh_deadline = 0.0

    def refresh(
        self,
        run_dir: Path,
        *,
        monotonic_now: float | None = None,
        epoch_now: float | None = None,
    ) -> tuple[list[TokenUsageRow], float]:
        now = time.monotonic() if monotonic_now is None else monotonic_now
        if now < self._refresh_deadline:
            return self.rows, self.updated_at
        try:
            self.rows = collect_token_usage(run_dir)
        except FileNotFoundError:
            self.rows = []
        self.updated_at = time.time() if epoch_now is None else epoch_now
        self._refresh_deadline = now + TOKEN_USAGE_REFRESH_SECONDS
        return self.rows, self.updated_at
