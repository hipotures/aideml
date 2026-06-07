import json
from dataclasses import dataclass
from pathlib import Path

from aide.journal import Journal, Node
from aide.run import load_resume_state
from aide.utils import serialize
from aide.utils.artifact_manifest import (
    RESULT_MANIFEST_NAME,
    build_node_artifact_manifest,
    sha256_file,
)
from aide.utils.metric import MetricValue
from scripts.seed_run_from_nodes import (
    seed_run_from_nodes,
    select_seed_sources,
)


@dataclass
class DummyConfig:
    log_dir: Path
    workspace_dir: Path
    exp_name: str = "test-exp"


def _append_source_node(
    *,
    journal: Journal,
    log_dir: Path,
    workspace_dir: Path,
    timestamp: str,
    code: str,
    score: float,
    parent: Node | None = None,
) -> tuple[Node, Path]:
    artifact_dir = log_dir / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "solution.py").write_text(code, encoding="utf-8")
    (artifact_dir / "submission.csv").write_text("id,target\n1,0.9\n", encoding="utf-8")
    (artifact_dir / "test_predictions.csv.gz").write_bytes(b"prediction-marker")
    (artifact_dir / "notes.txt").write_text("copy me\n", encoding="utf-8")

    node = Node(
        code=code,
        plan=f"source plan {score}",
        parent=parent,
        artifact_dir_name=timestamp,
    )
    node.status = "ok"
    node.metric = MetricValue(score, maximize=True)
    node.is_buggy = False
    node.exec_time = 12.0
    node.analysis = "source analysis"
    node.submission_validation = {"status": "ok"}
    journal.append(node)

    manifest = build_node_artifact_manifest(
        cfg=DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir),
        node=node,
        artifact_dir=artifact_dir,
    )
    (artifact_dir / RESULT_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return node, artifact_dir


def _write_source_run(tmp_path: Path) -> tuple[Path, Path, str, list[Node]]:
    logs_dir = tmp_path / "logs"
    workspaces_dir = tmp_path / "workspaces"
    run_id = "1-source-run"
    log_dir = logs_dir / run_id
    workspace_dir = workspaces_dir / run_id
    log_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    journal = Journal()
    low, _ = _append_source_node(
        journal=journal,
        log_dir=log_dir,
        workspace_dir=workspace_dir,
        timestamp="20260601T010000-a1111111",
        code="print('low')\n",
        score=0.80,
    )
    high, _ = _append_source_node(
        journal=journal,
        log_dir=log_dir,
        workspace_dir=workspace_dir,
        timestamp="20260601T020000-b2222222",
        code="print('high')\n",
        score=0.95,
    )
    child, _ = _append_source_node(
        journal=journal,
        log_dir=log_dir,
        workspace_dir=workspace_dir,
        timestamp="20260601T030000-c3333333",
        code="print('child')\n",
        score=0.90,
        parent=low,
    )
    serialize.dump_json(journal, log_dir / "journal.json")
    return logs_dir, workspaces_dir, run_id, [low, high, child]


def _cli_overrides(tmp_path: Path) -> list[str]:
    return [
        f"data_dir={tmp_path}",
        "goal=test",
        f"log_dir={tmp_path / 'logs'}",
        f"workspace_dir={tmp_path / 'workspaces'}",
        "generate_report=False",
    ]


def test_seed_run_from_nodes_top_n_creates_resumable_root_nodes(tmp_path):
    logs_dir, workspaces_dir, source_run, _nodes = _write_source_run(tmp_path)

    result = seed_run_from_nodes(
        source_run=source_run,
        top_n=2,
        run_id="2-seeded-run",
        cli_overrides=_cli_overrides(tmp_path),
        prepare_workspace=False,
    )

    assert result.run_id == "2-seeded-run"
    assert len(result.seeded) == 2
    loaded_cfg, journal = load_resume_state(
        run_id=result.run_id,
        top_log_dir=logs_dir,
        top_workspace_dir=workspaces_dir,
        cli_overrides=[],
    )
    assert loaded_cfg.agent.search.num_drafts == 2
    assert [node.metric.value for node in journal.nodes] == [0.95, 0.90]
    assert [node.parent for node in journal.nodes] == [None, None]
    assert [node.step for node in journal.nodes] == [0, 1]

    first_artifact = result.seeded[0].artifact_dir
    assert (first_artifact / "notes.txt").read_text(encoding="utf-8") == "copy me\n"
    assert (first_artifact / "test_predictions.csv.gz").read_bytes() == b"prediction-marker"

    manifest = json.loads((first_artifact / RESULT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["run"] == "2-seeded-run"
    assert manifest["node"]["id"] == journal.nodes[0].id
    assert manifest["node"]["step"] == 0
    assert manifest["node"]["parent_id"] is None
    assert manifest["source"]["source_run"] == source_run
    assert manifest["source"]["source_step"] == 1
    assert manifest["files"]["test_predictions"]["path"] == "test_predictions.csv.gz"


def test_seed_run_from_nodes_code_only_creates_unscored_seed(tmp_path):
    logs_dir, workspaces_dir, source_run, _nodes = _write_source_run(tmp_path)

    result = seed_run_from_nodes(
        source_run=source_run,
        steps=(1,),
        run_id="2-code-only-seed",
        cli_overrides=_cli_overrides(tmp_path),
        prepare_workspace=False,
        code_only=True,
    )

    loaded_cfg, journal = load_resume_state(
        run_id=result.run_id,
        top_log_dir=logs_dir,
        top_workspace_dir=workspaces_dir,
        cli_overrides=[],
    )
    node = journal.nodes[0]
    assert loaded_cfg.exp_name == "2-code-only-seed"
    assert node.status == "generated"
    assert node.metric is None
    assert node.exec_time is None

    artifact = result.seeded[0].artifact_dir
    assert (artifact / "solution.py").exists()
    assert not (artifact / "submission.csv").exists()
    assert not (artifact / "notes.txt").exists()

    manifest = json.loads((artifact / RESULT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["local_score"] is None
    assert manifest["source"]["code_only"] is True


def test_select_seed_sources_by_steps_node_ids_and_shas(tmp_path):
    logs_dir, _workspaces_dir, source_run, nodes = _write_source_run(tmp_path)

    by_step = select_seed_sources(
        logs_dir=logs_dir,
        source_run=source_run,
        steps=(0, 2),
    )
    assert [source.source_step for source in by_step] == [0, 2]

    by_node_id = select_seed_sources(
        logs_dir=logs_dir,
        source_run=source_run,
        node_ids=(nodes[1].id[:12],),
    )
    assert by_node_id[0].node_payload["id"] == nodes[1].id

    source_artifact = logs_dir / source_run / "artifacts" / nodes[2].artifact_dir_name
    by_sha = select_seed_sources(
        logs_dir=logs_dir,
        source_run=source_run,
        shas=(sha256_file(source_artifact / "solution.py")[:12],),
    )
    assert by_sha[0].artifact_dir == source_artifact


def test_seed_run_from_nodes_requires_existing_artifact_manifest(tmp_path):
    logs_dir, _workspaces_dir, source_run, nodes = _write_source_run(tmp_path)
    manifest_path = (
        logs_dir
        / source_run
        / "artifacts"
        / nodes[0].artifact_dir_name
        / RESULT_MANIFEST_NAME
    )
    manifest_path.unlink()

    try:
        select_seed_sources(logs_dir=logs_dir, source_run=source_run, steps=(0,))
    except FileNotFoundError as exc:
        assert "artifact manifest" in str(exc)
    else:
        raise AssertionError("Expected missing manifest to fail source selection.")
