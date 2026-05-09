from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from .journal import Journal, Node

logger = logging.getLogger("aide")

TELEGRAM_API_BASE_URL = "https://api.telegram.org"
DEFAULT_TELEGRAM_LOG_PATH = Path("/tmp/aideml/telegram.log")


def _best_scored_node(journal: Journal) -> Node | None:
    candidates = [
        node
        for node in journal.good_nodes
        if node.metric is not None and node.metric.value is not None
    ]
    return max(candidates, key=lambda node: node.metric, default=None)


def _is_scored_node(node: Node) -> bool:
    return (
        not node.is_buggy
        and node.status != "failed"
        and node.metric is not None
        and node.metric.value is not None
    )


def _format_score(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value):.5f}"


def _best_score_message(
    *,
    node: Node,
    previous_best: Node | None,
    experiment_id: str,
) -> str:
    previous_value = (
        previous_best.metric.value
        if previous_best is not None and previous_best.metric is not None
        else None
    )
    return "\n".join(
        [
            f"Best score: {_format_score(node.metric.value)}",
            f"Old best score: {_format_score(previous_value)}",
            f"Step: {node.step if node.step is not None else '?'}",
            f"Id: {experiment_id}",
        ]
    )


def _telegram_credentials() -> tuple[str, str] | None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return None
    return token, chat_id


def _telegram_log_path() -> Path:
    return Path(os.getenv("AIDE_TELEGRAM_LOG_PATH", str(DEFAULT_TELEGRAM_LOG_PATH)))


def _log_telegram_status(status: str, detail: str = "") -> None:
    try:
        path = _telegram_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            line = f"{timestamp} {status}"
            if detail:
                line += f" {detail}"
            f.write(line + "\n")
    except OSError:
        return


def send_telegram_message(
    text: str,
    *,
    http_post: Callable[..., Any] = requests.post,
    attempts: int = 3,
) -> bool:
    credentials = _telegram_credentials()
    if credentials is None:
        _log_telegram_status("skipped", "missing_credentials")
        return False

    token, chat_id = credentials
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            response = http_post(
                f"{TELEGRAM_API_BASE_URL}/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10,
            )
            response.raise_for_status()
            _log_telegram_status("sent", f"attempt={attempt}")
            return True
        except Exception as exc:  # noqa: BLE001 - notification failure is recoverable
            detail = f"attempt={attempt} error={exc.__class__.__name__}: {exc}"
            _log_telegram_status("failed", detail)
            logger.warning("Telegram notification failed: %s", detail)
            if attempt < attempts:
                time.sleep(min(2, attempt))

    return False


def _send_message(
    send_message: Callable[[str], Any],
    text: str,
) -> None:
    try:
        send_message(text)
    except Exception as exc:  # noqa: BLE001 - notifications must not stop runs
        detail = f"error={exc.__class__.__name__}: {exc}"
        _log_telegram_status("failed", detail)
        logger.warning(
            "Telegram notification callback failed: %s",
            detail,
        )


def send_telegram_test_message(
    *,
    send_message: Callable[[str], None] = send_telegram_message,
) -> None:
    _send_message(send_message, "AIDE Telegram OK")


def notify_new_best_score(
    *,
    node: Node,
    previous_best: Node | None,
    experiment_id: str,
    send_message: Callable[[str], None] = send_telegram_message,
) -> None:
    if not _is_scored_node(node):
        return
    if previous_best is not None and not node.metric > previous_best.metric:
        return
    _send_message(
        send_message,
        _best_score_message(
            node=node,
            previous_best=previous_best,
            experiment_id=experiment_id,
        ),
    )


def append_node_with_best_score_notification(
    *,
    journal: Journal,
    node: Node,
    experiment_id: str,
    send_message: Callable[[str], None] = send_telegram_message,
) -> None:
    previous_best = _best_scored_node(journal)
    journal.append(node)
    notify_new_best_score(
        node=node,
        previous_best=previous_best,
        experiment_id=experiment_id,
        send_message=send_message,
    )
