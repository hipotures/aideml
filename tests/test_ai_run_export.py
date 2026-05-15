import json
from pathlib import Path

from aide.journal import Journal, Node
from aide.utils import serialize
from aide.utils.ai_run_export import export_run_for_ai
from aide.utils.artifact_manifest import artifact_timestamp_from_ctime
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


def test_export_handles_invalid_metric_value(tmp_path):
    log_dir = tmp_path / "logs" / "run-invalid-metric"
    log_dir.mkdir(parents=True)
    bug = Node(
        code="raise RuntimeError('bad')\n",
        plan="bug plan",
        id="node-bug",
        ctime=1770000000.0,
        status="failed",
        metric=MetricValue(None, maximize=True),
        is_buggy=True,
        analysis="bug analysis",
        exc_type="RuntimeError",
    )
    journal = Journal()
    journal.append(bug)
    serialize.dump_json(journal, log_dir / "journal.json")

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")

    meta = json.loads(result.meta_path.read_text())
    nodes = _read_jsonl(result.nodes_path)

    assert meta["scored_node_count"] == 0
    assert meta["best_local"] is None
    assert nodes[0]["local_cv_score"] is None
    assert nodes[0]["metric_maximize"] is True


def test_export_best_local_uses_minimizing_metric_semantics(tmp_path):
    log_dir = tmp_path / "logs" / "run-minimize"
    log_dir.mkdir(parents=True)
    worse = Node(
        code="print('worse')\n",
        plan="worse plan",
        id="node-worse",
        ctime=1770000000.0,
        metric=MetricValue(0.2, maximize=False),
        is_buggy=False,
        analysis="worse analysis",
    )
    better = Node(
        code="print('better')\n",
        plan="better plan",
        id="node-better",
        ctime=1770000060.0,
        metric=MetricValue(0.1, maximize=False),
        is_buggy=False,
        analysis="better analysis",
    )
    journal = Journal()
    journal.append(worse)
    journal.append(better)
    serialize.dump_json(journal, log_dir / "journal.json")

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")

    meta = json.loads(result.meta_path.read_text())

    assert meta["best_local"]["node_id"] == "node-better"
    assert meta["best_local"]["local_cv_score"] == 0.1


def test_export_best_local_keeps_zero_score_as_valid(tmp_path):
    log_dir = tmp_path / "logs" / "run-zero"
    log_dir.mkdir(parents=True)
    worse = Node(
        code="print('worse')\n",
        plan="worse plan",
        id="node-worse",
        ctime=1770000000.0,
        metric=MetricValue(-1.0, maximize=True),
        is_buggy=False,
        analysis="worse analysis",
    )
    better = Node(
        code="print('better')\n",
        plan="better plan",
        id="node-better",
        ctime=1770000060.0,
        metric=MetricValue(0.0, maximize=True),
        is_buggy=False,
        analysis="better analysis",
    )
    journal = Journal()
    journal.append(worse)
    journal.append(better)
    serialize.dump_json(journal, log_dir / "journal.json")

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")

    meta = json.loads(result.meta_path.read_text())

    assert meta["scored_node_count"] == 2
    assert meta["best_local"]["node_id"] == "node-better"
    assert meta["best_local"]["local_cv_score"] == 0.0


def test_export_includes_submission_hash_and_public_score_by_node_id(tmp_path):
    log_dir = _write_run(tmp_path)
    timestamp = artifact_timestamp_from_ctime(1770000000.0)
    artifact_dir = log_dir / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True)
    submission = artifact_dir / "submission.csv"
    submission.write_text("id,PitNextLap\n1,0.8\n")
    registry = log_dir.parent / "submission_registry.json"
    registry.write_text(
        json.dumps(
            {
                "submissions": [
                    {
                        "competition": "playground-series-s6e5",
                        "run": "run-a",
                        "step": 0,
                        "node_id": "node-root",
                        "timestamp": timestamp,
                        "sha256": "placeholder",
                        "remote_status": "COMPLETE",
                        "public_score": "0.91234",
                    }
                ]
            }
        )
    )

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")
    nodes = _read_jsonl(result.nodes_path)
    meta = json.loads(result.meta_path.read_text())

    assert nodes[0]["submission_sha256"] is not None
    assert nodes[0]["kaggle_public_score"] == 0.91234
    assert meta["best_public"]["node_id"] == "node-root"
    assert meta["best_public"]["kaggle_public_score"] == 0.91234


def test_export_maps_public_score_by_sha_prefix(tmp_path):
    log_dir = _write_run(tmp_path)
    timestamp = artifact_timestamp_from_ctime(1770000000.0)
    artifact_dir = log_dir / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True)
    submission = artifact_dir / "submission.csv"
    submission.write_text("id,PitNextLap\n1,0.8\n")
    from aide.utils.ai_run_export import _sha256_file

    full_sha = _sha256_file(submission)
    registry = log_dir.parent / "submission_registry.json"
    registry.write_text(
        json.dumps(
            {
                "submissions": [
                    {
                        "competition": "playground-series-s6e5",
                        "run": "other-seeded-run",
                        "step": 0,
                        "timestamp": "20260510T021544",
                        "sha256": full_sha[:10],
                        "remote_status": "COMPLETE",
                        "public_score": "0.92345",
                    }
                ]
            }
        )
    )

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")
    nodes = _read_jsonl(result.nodes_path)

    assert nodes[0]["submission_sha256"] == full_sha
    assert nodes[0]["kaggle_public_score"] == 0.92345


def test_export_marks_exact_code_and_submission_duplicates_without_pruning(tmp_path):
    log_dir = _write_run(tmp_path)
    root_timestamp = artifact_timestamp_from_ctime(1770000000.0)
    child_timestamp = artifact_timestamp_from_ctime(1770000060.0)
    root_artifact = log_dir / "artifacts" / root_timestamp
    child_artifact = log_dir / "artifacts" / child_timestamp
    root_artifact.mkdir(parents=True)
    child_artifact.mkdir(parents=True)
    body = "id,PitNextLap\n1,0.8\n2,0.2\n"
    (root_artifact / "submission.csv").write_text(body)
    (child_artifact / "submission.csv").write_text(body)

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")
    nodes = _read_jsonl(result.nodes_path)

    assert len(nodes) == 3
    assert nodes[0]["duplicate"]["exact_code_role"] == "canonical"
    assert nodes[1]["duplicate"]["exact_code_role"] == "canonical"
    assert nodes[0]["duplicate"]["exact_submission_role"] == "duplicate"
    assert (
        nodes[0]["duplicate"]["exact_submission_canonical_node_id"] == "node-child"
    )
    assert nodes[1]["duplicate"]["exact_submission_role"] == "canonical"


def test_export_public_scores_follow_minimizing_metric_semantics(tmp_path):
    log_dir = tmp_path / "logs" / "run-min-public"
    log_dir.mkdir(parents=True)
    first = Node(
        code="print('first')\n",
        plan="first plan",
        id="node-first",
        ctime=1770000000.0,
        metric=MetricValue(0.3, maximize=False),
        is_buggy=False,
        analysis="first analysis",
    )
    second = Node(
        code="print('second')\n",
        plan="second plan",
        id="node-second",
        ctime=1770000060.0,
        metric=MetricValue(0.2, maximize=False),
        is_buggy=False,
        analysis="second analysis",
    )
    journal = Journal()
    journal.append(first)
    journal.append(second)
    serialize.dump_json(journal, log_dir / "journal.json")

    first_timestamp = artifact_timestamp_from_ctime(first.ctime)
    second_timestamp = artifact_timestamp_from_ctime(second.ctime)
    (log_dir / "artifacts" / first_timestamp).mkdir(parents=True)
    (log_dir / "artifacts" / first_timestamp / "submission.csv").write_text(
        "id,PitNextLap\n1,0.8\n"
    )
    (log_dir / "artifacts" / second_timestamp).mkdir(parents=True)
    (log_dir / "artifacts" / second_timestamp / "submission.csv").write_text(
        "id,PitNextLap\n1,0.7\n"
    )
    registry = log_dir.parent / "submission_registry.json"
    registry.write_text(
        json.dumps(
            {
                "submissions": [
                    {
                        "competition": "playground-series-s6e5",
                        "run": "run-min-public",
                        "step": "0",
                        "timestamp": first_timestamp,
                        "remote_status": "COMPLETE",
                        "public_score": "0.4",
                    },
                    {
                        "competition": "playground-series-s6e5",
                        "run": "run-min-public",
                        "step": "0",
                        "timestamp": first_timestamp,
                        "remote_status": "COMPLETE",
                        "public_score": "0.3",
                    },
                    {
                        "competition": "playground-series-s6e5",
                        "run": "run-min-public",
                        "step": 1,
                        "timestamp": second_timestamp,
                        "remote_status": "COMPLETE",
                        "public_score": "0.2",
                    },
                ]
            }
        )
    )

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")
    nodes = _read_jsonl(result.nodes_path)
    meta = json.loads(result.meta_path.read_text())

    assert nodes[0]["kaggle_public_score"] == 0.3
    assert nodes[1]["kaggle_public_score"] == 0.2
    assert meta["best_public"]["node_id"] == "node-second"
    assert meta["best_public"]["kaggle_public_score"] == 0.2
