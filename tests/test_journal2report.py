from aide.journal import Journal, Node
from aide.journal2report import journal2report
from aide.utils.config import StageConfig
from aide.utils.metric import MetricValue


def test_journal2report_uses_bounded_summary_without_full_code(monkeypatch):
    journal = Journal()
    node = Node(code="print('large code')\n" * 10_000, plan="train model")
    node.analysis = "reported useful validation findings"
    node.metric = MetricValue(0.91234, maximize=True)
    node.is_buggy = False
    journal.append(node)

    captured = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return "# Report"

    monkeypatch.setattr("aide.journal2report.query", fake_query)

    report = journal2report(
        journal,
        {"goal": "predict target"},
        StageConfig(model="gpt-5.4-mini", temp=0.0, reasoning_effort="low"),
    )

    user_message = captured["user_message"]
    assert report == "# Report"
    assert "Design: train model" in user_message
    assert "Results: reported useful validation findings" in user_message
    assert "Validation Metric: 0.91234" in user_message
    assert "print('large code')" not in user_message
