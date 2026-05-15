import json
from pathlib import Path

from aide.journal import Journal, Node
from aide.utils import serialize
from aide.utils.ai_run_export import export_run_for_ai
from aide.utils.metric import MetricValue


def _write_run(tmp_path: Path) -> Path:
    log_dir = tmp_path / "logs" / "run-a"
    log_dir.mkdir(parents=True)
    root = Node(
        code="print('root')\n",
        plan="root plan",
        id="node-root",
        ctime=1770000000.0,
        metric=MetricValue(0.9, maximize=True),
        is_buggy=False,
        analysis="root analysis",
    )
    child = Node(
        code="print('child')\n",
        plan="child plan",
        id="node-child",
        ctime=1770000060.0,
        parent=root,
        metric=MetricValue(0.91, maximize=True),
        is_buggy=False,
        analysis="child analysis",
    )
    bug = Node(
        code="raise RuntimeError('bad')\n",
        plan="bug plan",
        id="node-bug",
        ctime=1770000120.0,
        parent=root,
        status="bug",
        is_buggy=True,
        analysis="bug analysis",
        exc_type="RuntimeError",
    )
    journal = Journal()
    journal.append(root)
    journal.append(child)
    journal.append(bug)
    serialize.dump_json(journal, log_dir / "journal.json")
    return log_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_export_preserves_tree_and_full_code(tmp_path):
    log_dir = _write_run(tmp_path)

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")

    meta = json.loads(result.meta_path.read_text())
    nodes = _read_jsonl(result.nodes_path)

    assert meta["run"] == "run-a"
    assert meta["node_count"] == 3
    assert meta["scored_node_count"] == 2
    assert [node["step"] for node in nodes] == [0, 1, 2]
    assert nodes[0]["node_id"] == "node-root"
    assert nodes[0]["parent_id"] is None
    assert nodes[0]["children_ids"] == ["node-child", "node-bug"]
    assert nodes[1]["parent_id"] == "node-root"
    assert nodes[1]["depth"] == 1
    assert nodes[2]["is_buggy"] is True
    assert nodes[2]["error"]["exc_type"] == "RuntimeError"
    assert nodes[0]["code"] == "print('root')\n"
    assert nodes[1]["local_cv_score"] == 0.91
