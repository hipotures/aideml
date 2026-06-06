import json
from pathlib import Path

from scripts.migrate_research_hypothesis_library import (
    apply_root_code_export,
    apply_structure_migration,
    plan_root_code_export,
    plan_structure_migration,
)


def _write_old_hypothesis(root: Path, task: str, hypothesis_id: str) -> None:
    path = (
        root
        / "research_hypotheses"
        / task
        / "hypotheses"
        / f"hypothesis-{hypothesis_id}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "agent_modes": ["legacy", "autogluon"],
                "title": f"Hypothesis {hypothesis_id}",
                "summary": "Summary.",
                "rationale": "Rationale.",
                "implementation_hint": "Implementation.",
                "expected_effect": "Effect.",
                "risk": "Risk.",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )


def _attach_solution_artifact(run_dir: Path, node: dict, code: str) -> None:
    artifact_dir_name = node.get("artifact_dir_name") or f"{node['id']}-artifact"
    node["artifact_dir_name"] = artifact_dir_name
    artifact_dir = run_dir / "artifacts" / artifact_dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "solution.py").write_text(code, encoding="utf-8")
    node["code"] = ""
    node["code_path"] = f"artifacts/{artifact_dir_name}/solution.py"


def test_structure_migration_moves_hypotheses_to_flat_id_dirs(tmp_path):
    task = "playground-series-s6e5"
    _write_old_hypothesis(tmp_path, task, "000001")
    _write_old_hypothesis(tmp_path, task, "000002")

    plan = plan_structure_migration(root=tmp_path, task=task)

    assert not plan.conflicts
    assert [move.hypothesis_id for move in plan.moves] == ["000001", "000002"]

    apply_structure_migration(plan, dry_run=False)

    assert (
        tmp_path
        / "research_hypotheses"
        / task
        / "000001"
        / "hypothesis-000001.json"
    ).exists()
    assert not (
        tmp_path
        / "research_hypotheses"
        / task
        / "hypotheses"
        / "hypothesis-000001.json"
    ).exists()


def test_root_code_export_writes_versions_and_active_manifest(tmp_path):
    task = "playground-series-s6e5"
    _write_old_hypothesis(tmp_path, task, "000001")
    apply_structure_migration(
        plan_structure_migration(root=tmp_path, task=task),
        dry_run=False,
    )
    run_dir = tmp_path / "logs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        "agent:\n  mode: autogluon_preprocess\n",
        encoding="utf-8",
    )
    nodes = [
        {
            "id": "node-ok",
            "parent": None,
            "research_mode": "hypothesis",
            "research_hypotheses_offered": ["000001"],
            "code": "print('ok')\n",
            "is_buggy": False,
            "metric": {"value": 0.91, "maximize": True},
            "ctime": 1000.0,
        },
        {
            "id": "branch",
            "parent": None,
            "research_mode": "hypothesis",
            "research_hypotheses_offered": ["000001"],
            "code": "print('branch')\n",
            "is_buggy": False,
        },
    ]
    for node in nodes:
        _attach_solution_artifact(run_dir, node, str(node["code"]))
    journal = {
        "nodes": nodes,
        "node2parent": {"branch": "node-ok"},
    }
    journal_path = run_dir / "journal.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    plan = plan_root_code_export(
        root=tmp_path,
        task=task,
        journal_path=journal_path,
    )

    assert not plan.conflicts
    assert [entry.destination.name for entry in plan.entries] == ["autogluon-001.py"]
    assert plan.entries[0].node_id == "node-ok"

    hypothesis_dir = tmp_path / "research_hypotheses" / task / "000001"
    (hypothesis_dir / "autogluon-002.py").write_text("print('manual newer')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "versions": {
                    "autogluon": [
                        {
                            "file": "autogluon-002.py",
                            "buggy": False,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    apply_root_code_export(plan, dry_run=False)

    assert (hypothesis_dir / "autogluon-001.py").read_text() == "print('ok')\n"
    assert (hypothesis_dir / "autogluon-002.py").read_text() == "print('manual newer')\n"
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    assert manifest["active"]["autogluon"] == "autogluon-001.py"
    assert [entry["file"] for entry in manifest["versions"]["autogluon"]] == [
        "autogluon-002.py",
        "autogluon-001.py",
    ]
    assert manifest["versions"]["autogluon"][1]["buggy"] is False


def test_root_code_export_reports_duplicate_journal_roots_as_conflict(tmp_path):
    task = "playground-series-s6e5"
    _write_old_hypothesis(tmp_path, task, "000001")
    apply_structure_migration(
        plan_structure_migration(root=tmp_path, task=task),
        dry_run=False,
    )
    run_dir = tmp_path / "logs" / "run-1"
    run_dir.mkdir(parents=True)
    nodes = [
        {
            "id": "root-1",
            "parent": None,
            "research_mode": "hypothesis",
            "research_hypotheses_offered": ["000001"],
            "code": "print('one')\n",
            "is_buggy": False,
        },
        {
            "id": "root-2",
            "parent": None,
            "research_mode": "hypothesis",
            "research_hypotheses_offered": ["000001"],
            "code": "print('two')\n",
            "is_buggy": False,
        },
    ]
    for node in nodes:
        _attach_solution_artifact(run_dir, node, str(node["code"]))
    journal = {
        "nodes": nodes,
        "node2parent": {},
    }
    journal_path = run_dir / "journal.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    plan = plan_root_code_export(
        root=tmp_path,
        task=task,
        journal_path=journal_path,
        agent_mode="legacy",
    )

    assert plan.entries == ()
    assert plan.conflicts == (
        "Multiple journal ROOT nodes for hypothesis 000001: root-1 and root-2",
    )
