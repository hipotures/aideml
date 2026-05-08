from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import requests
from dotenv import load_dotenv

from .journal import Journal, Node

logger = logging.getLogger("aide")

TELEGRAM_API_BASE_URL = "https://api.telegram.org"


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


def send_telegram_message(
    text: str,
    *,
    http_post: Callable[..., Any] = requests.post,
) -> None:
    credentials = _telegram_credentials()
    if credentials is None:
        return

    token, chat_id = credentials
    try:
        response = http_post(
            f"{TELEGRAM_API_BASE_URL}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - notification failure is recoverable
        logger.warning(
            "Telegram notification failed: %s",
            exc.__class__.__name__,
        )


def send_telegram_test_message(
    *,
    send_message: Callable[[str], None] = send_telegram_message,
) -> None:
    send_message("AIDE Telegram OK")


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
    send_message(
        _best_score_message(
            node=node,
            previous_best=previous_best,
            experiment_id=experiment_id,
        )
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
