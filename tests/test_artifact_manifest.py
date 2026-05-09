import json
from dataclasses import dataclass
from pathlib import Path

from aide.journal import Node
from aide.utils.artifact_manifest import build_node_artifact_manifest
from aide.utils.artifact_manifest import reconstruct_journal_from_artifacts
from aide.utils.metric import MetricValue


@dataclass
class DummyConfig:
    log_dir: Path
    workspace_dir: Path
    exp_name: str = "run-a"


def test_build_node_artifact_manifest_for_successful_node(tmp_path):
    artifact_dir = tmp_path / "logs" / "run-a" / "artifacts" / "20260506T120000"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "solution.py").write_text(
        "AIDE_AG_CONFIG = {'profile': 'fast_boost', "
        "'included_model_types': ['XGB', 'GBM'], 'presets': 'medium_quality', "
        "'time_limit': 300, 'use_gpu': False}\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    (artifact_dir / "submission.csv").write_text("id,target\n1,0.9\n", encoding="utf-8")
    cfg = DummyConfig(log_dir=tmp_path / "logs" / "run-a", workspace_dir=tmp_path)
    node = Node(code="print('ok')", plan="try useful features", ctime=1778061600.0)
    node.step = 7
    node.metric = MetricValue(0.95098, maximize=True)
    node.is_buggy = False
    node.exec_time = 12.5
    node.analysis = "AutoGluon preprocess wrapper completed."
    node.validity_warning = "Possible leakage in race-lap aggregate features."
    node.submission_validation = {"status": "ok"}

    manifest = build_node_artifact_manifest(
        cfg=cfg,
        node=node,
        artifact_dir=artifact_dir,
    )

    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "source_node"
    assert manifest["run"] == "run-a"
    assert manifest["timestamp"] == "20260506T120000"
    assert manifest["status"] == "ok"
    assert manifest["local_score"] == 0.95098
    assert manifest["metric_maximize"] is True
    assert manifest["profile"] == "fast_boost"
    assert manifest["included_model_types"] == ["XGB", "GBM"]
    assert manifest["time_limit"] == 300
    assert manifest["node"]["id"] == node.id
    assert manifest["node"]["step"] == 7
    assert (
        manifest["node"]["validity_warning"]
        == "Possible leakage in race-lap aggregate features."
    )
    assert manifest["node"]["metric"] == {"value": 0.95098, "maximize": True}
    assert manifest["execution"]["exec_time"] == 12.5
    assert manifest["files"]["submission"]["path"] == "submission.csv"
    assert len(manifest["files"]["submission"]["sha256"]) == 64


def test_build_node_artifact_manifest_for_bug_node(tmp_path):
    artifact_dir = tmp_path / "logs" / "run-a" / "artifacts" / "20260506T120000"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "solution.py").write_text("raise RuntimeError('bug')\n", encoding="utf-8")
    (artifact_dir / "error.txt").write_text("RuntimeError: bug\n", encoding="utf-8")
    cfg = DummyConfig(log_dir=tmp_path / "logs" / "run-a", workspace_dir=tmp_path)
    node = Node(code="raise RuntimeError('bug')", plan="buggy plan", ctime=1778061600.0)
    node.step = 8
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = True
    node.exec_time = 1.0
    node.exc_type = "RuntimeError"
    node.exc_info = {"args": ["bug"]}
    node.exc_stack = [("runfile.py", 1, "<module>", "raise RuntimeError('bug')")]
    node.analysis = "bug"

    manifest = build_node_artifact_manifest(
        cfg=cfg,
        node=node,
        artifact_dir=artifact_dir,
    )

    assert manifest["status"] == "bug"
    assert manifest["local_score"] is None
    assert manifest["is_buggy"] is True
    assert manifest["files"]["submission"] is None
    assert manifest["files"]["error"]["path"] == "error.txt"
    assert manifest["execution"]["exc_type"] == "RuntimeError"
    assert manifest["node"]["metric"] == {"value": None, "maximize": True}


def test_manifest_json_is_round_trippable(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    cfg = DummyConfig(log_dir=tmp_path / "logs" / "run-a", workspace_dir=tmp_path)
    node = Node(code="print('ok')", plan="ok", ctime=1778061600.0)
    node.step = 1
    node.metric = MetricValue(0.1, maximize=True)
    node.is_buggy = False

    manifest = build_node_artifact_manifest(cfg=cfg, node=node, artifact_dir=artifact_dir)

    assert json.loads(json.dumps(manifest))["node"]["id"] == node.id


def test_reconstruct_journal_from_artifact_manifests(tmp_path):
    log_dir = tmp_path / "logs" / "run-a"
    parent_artifact = log_dir / "artifacts" / "20260506T120000"
    child_artifact = log_dir / "artifacts" / "20260506T121000"
    parent_artifact.mkdir(parents=True)
    child_artifact.mkdir(parents=True)
    (parent_artifact / "solution.py").write_text("print('parent')\n", encoding="utf-8")
    (child_artifact / "solution.py").write_text("print('child')\n", encoding="utf-8")
    cfg = DummyConfig(log_dir=log_dir, workspace_dir=tmp_path)
    parent = Node(code="print('parent')", plan="parent plan", ctime=1778061600.0)
    parent.step = 0
    parent.metric = MetricValue(0.9, maximize=True)
    parent.is_buggy = False
    parent.analysis = "parent analysis"
    parent.validity_warning = "parent warning"
    child = Node(
        code="print('child')",
        plan="child plan",
        parent=parent,
        ctime=1778062200.0,
    )
    child.step = 1
    child.metric = MetricValue(None, maximize=True)
    child.is_buggy = True
    child.status = "bug"
    child.exec_time = 2.0
    child.exc_type = "ValueError"
    child.analysis = "child analysis"

    parent_manifest = build_node_artifact_manifest(
        cfg=cfg,
        node=parent,
        artifact_dir=parent_artifact,
    )
    child_manifest = build_node_artifact_manifest(
        cfg=cfg,
        node=child,
        artifact_dir=child_artifact,
    )
    (parent_artifact / "aide_result.json").write_text(
        json.dumps(parent_manifest),
        encoding="utf-8",
    )
    (child_artifact / "aide_result.json").write_text(
        json.dumps(child_manifest),
        encoding="utf-8",
    )

    journal = reconstruct_journal_from_artifacts(log_dir)

    assert len(journal.nodes) == 2
    restored_parent = journal.nodes[0]
    restored_child = journal.nodes[1]
    assert restored_parent.id == parent.id
    assert restored_parent.code == "print('parent')\n"
    assert restored_parent.plan == "parent plan"
    assert restored_parent.metric.value == 0.9
    assert restored_parent.is_buggy is False
    assert restored_parent.validity_warning == "parent warning"
    assert restored_child.id == child.id
    assert restored_child.parent is restored_parent
    assert restored_child in restored_parent.children
    assert restored_child.code == "print('child')\n"
    assert restored_child.is_buggy is True
    assert restored_child.status == "bug"
    assert restored_child.exc_type == "ValueError"
