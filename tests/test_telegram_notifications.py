import aide.run as run_module
import pytest

from aide.journal import Journal, Node
from aide.telegram_notifications import (
    append_node_with_best_score_notification,
    send_telegram_message,
    send_telegram_test_message,
)
from aide.utils.metric import MetricValue, WorstMetricValue


def _node(score: float | None) -> Node:
    node = Node(code="print('ok')", plan="ok")
    node.metric = (
        MetricValue(score, maximize=True)
        if score is not None
        else WorstMetricValue()
    )
    node.is_buggy = score is None
    return node


def test_append_node_notifies_new_best_score(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
    journal = Journal()
    journal.append(_node(0.95101))
    sent_messages: list[str] = []

    append_node_with_best_score_notification(
        journal=journal,
        node=_node(0.95108),
        experiment_id="2-sarcastic-skilled-echidna",
        send_message=sent_messages.append,
    )

    assert sent_messages == [
        "\n".join(
            [
                "Best score: 0.95108",
                "Old best score: 0.95101",
                "Step: 1",
                "Id: 2-sarcastic-skilled-echidna",
            ]
        )
    ]


def test_append_node_skips_notification_when_score_does_not_improve(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
    journal = Journal()
    journal.append(_node(0.95108))
    sent_messages: list[str] = []

    append_node_with_best_score_notification(
        journal=journal,
        node=_node(0.95101),
        experiment_id="2-sarcastic-skilled-echidna",
        send_message=sent_messages.append,
    )

    assert sent_messages == []


def test_send_telegram_message_noops_without_token(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
    monkeypatch.setenv("AIDE_TELEGRAM_LOG_PATH", str(tmp_path / "telegram.log"))
    calls = []

    sent = send_telegram_message(
        "hello",
        http_post=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert sent is False
    assert calls == []


def test_send_telegram_message_posts_to_telegram_bot_api(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@channel_name")
    calls = []

    class Response:
        @staticmethod
        def raise_for_status():
            return None

    def http_post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    sent = send_telegram_message("hello", http_post=http_post)

    assert sent is True
    assert calls == [
        (
            ("https://api.telegram.org/botfake-token/sendMessage",),
            {
                "json": {"chat_id": "@channel_name", "text": "hello"},
                "timeout": 10,
            },
        )
    ]


def test_send_telegram_message_retries_and_logs_status(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
    log_path = tmp_path / "telegram.log"
    monkeypatch.setenv("AIDE_TELEGRAM_LOG_PATH", str(log_path))
    monkeypatch.setattr("aide.telegram_notifications.time.sleep", lambda _seconds: None)
    calls = []

    class Response:
        @staticmethod
        def raise_for_status():
            return None

    def http_post(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise RuntimeError("network down")
        return Response()

    sent = send_telegram_message("hello", http_post=http_post, attempts=2)

    assert sent is True
    assert len(calls) == 2
    log_text = log_path.read_text(encoding="utf-8")
    assert "failed attempt=1 error=RuntimeError: network down" in log_text
    assert "sent attempt=2" in log_text


def test_send_telegram_message_logs_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
    log_path = tmp_path / "telegram.log"
    monkeypatch.setenv("AIDE_TELEGRAM_LOG_PATH", str(log_path))

    sent = send_telegram_message("hello")

    assert sent is False
    assert "skipped missing_credentials" in log_path.read_text(encoding="utf-8")


def test_send_telegram_test_message_uses_minimal_connection_text():
    sent_messages: list[str] = []

    send_telegram_test_message(send_message=sent_messages.append)

    assert sent_messages == ["AIDE Telegram OK"]


def test_telegram_test_message_flag_sends_then_continues_to_load_run_config(monkeypatch):
    class LoadedConfig(Exception):
        pass

    calls: list[str] = []
    monkeypatch.setattr(run_module, "send_telegram_test_message", lambda: calls.append("sent"))

    def load_cfg(*args, **kwargs):
        raise LoadedConfig

    monkeypatch.setattr(run_module, "load_cfg", load_cfg)

    with pytest.raises(LoadedConfig):
        run_module.run(["--telegram-test-message"])

    assert calls == ["sent"]
