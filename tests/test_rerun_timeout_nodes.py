from aide.journal import Journal, Node
from aide.utils import serialize
from scripts.rerun_timeout_nodes import reset_timeout_nodes


def _timeout_node(step: int, artifact_dir: str) -> Node:
    node = Node(
        code="print('timeout')\n",
        plan=f"timeout {step}",
        step=step,
        artifact_dir_name=artifact_dir,
        status="bug",
    )
    node.exc_type = "TimeoutError"
    node.analysis = "Execution timed out"
    return node


def test_reset_timeout_nodes_queues_selected_steps_and_keeps_backup(tmp_path):
    log_dir = tmp_path / "logs" / "manual"
    artifacts = log_dir / "artifacts"
    artifacts.mkdir(parents=True)
    nodes = [
        _timeout_node(20, "20260807T100000-timeout-20"),
        _timeout_node(22, "20260807T100001-timeout-22"),
    ]
    for node in nodes:
        artifact_dir = artifacts / node.artifact_dir_name
        artifact_dir.mkdir()
        (artifact_dir / "solution.py").write_text(node.code, encoding="utf-8")
    journal_path = log_dir / "journal.json"
    serialize.dump_json(Journal(nodes=nodes), journal_path)

    selected, backup = reset_timeout_nodes(log_dir, steps={22})

    assert selected == [22]
    assert backup is not None and backup.exists()
    restored = serialize.load_json(journal_path, Journal)
    assert [node.status for node in restored.nodes] == ["bug", "generated"]
    assert restored.nodes[1].exc_type is None
    assert restored.nodes[1].analysis.startswith("Generated only")


def test_reset_timeout_nodes_dry_run_does_not_change_journal(tmp_path):
    log_dir = tmp_path / "logs" / "manual"
    artifacts = log_dir / "artifacts" / "20260807T100000-timeout-20"
    artifacts.mkdir(parents=True)
    node = _timeout_node(20, artifacts.name)
    (artifacts / "solution.py").write_text(node.code, encoding="utf-8")
    journal_path = log_dir / "journal.json"
    serialize.dump_json(Journal(nodes=[node]), journal_path)

    selected, backup = reset_timeout_nodes(log_dir, dry_run=True)

    assert selected == [20]
    assert backup is None
    restored = serialize.load_json(journal_path, Journal)
    assert restored.nodes[0].status == "bug"
