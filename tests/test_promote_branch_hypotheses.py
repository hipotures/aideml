import json
from pathlib import Path

from scripts.promote_branch_hypotheses import (
    apply_promotion_plan,
    main,
    plan_branch_hypothesis_promotion,
    plan_branch_hypothesis_promotion_from_logs,
    plan_node_promotion,
)


def _write_run(
    tmp_path: Path,
    *,
    run_id: str = "run-1",
    mode: str = "autogluon_preprocess",
    branch_scores: tuple[float, float, float] = (0.95, 0.94, 0.93),
) -> Path:
    run_dir = tmp_path / "logs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(f"agent:\n  mode: {mode}\n", encoding="utf-8")
    branch_a_score, branch_b_score, branch_c_score = branch_scores
    journal = {
        "nodes": [
            {
                "id": "root-a",
                "parent": None,
                "research_mode": "hypothesis",
                "research_hypotheses_offered": ["000101"],
                "plan": "Root A source plan.",
                "analysis": "Root A source analysis.",
                "validity_warning": "",
                "code": "print('root a')\n",
                "is_buggy": False,
                "metric": {"value": 0.90, "maximize": True},
                "artifact_dir_name": "root-a-artifact",
                "ctime": 1000.0,
            },
            {
                "id": "branch-a",
                "parent": None,
                "research_mode": "hypothesis",
                "research_hypotheses_offered": ["000201"],
                "plan": "Branch A source plan.",
                "analysis": "Branch A source analysis.",
                "validity_warning": "",
                "code": "print('branch a')\n",
                "is_buggy": False,
                "metric": {"value": branch_a_score, "maximize": True},
                "artifact_dir_name": "branch-a-artifact",
                "ctime": 1001.0,
            },
            {
                "id": "branch-b",
                "parent": None,
                "research_mode": "hypothesis",
                "research_hypotheses_offered": ["000202"],
                "plan": "Branch B source plan.",
                "analysis": "Branch B source analysis.",
                "validity_warning": "",
                "code": "print('branch b')\n",
                "is_buggy": False,
                "metric": {"value": branch_b_score, "maximize": True},
                "artifact_dir_name": "branch-b-artifact",
                "ctime": 1002.0,
            },
            {
                "id": "branch-c",
                "parent": None,
                "research_mode": "hypothesis",
                "research_hypotheses_offered": ["000203"],
                "plan": "Branch C source plan.",
                "analysis": "Branch C source analysis.",
                "validity_warning": "",
                "code": "print('branch c')\n",
                "is_buggy": False,
                "metric": {"value": branch_c_score, "maximize": True},
                "ctime": 1003.0,
            },
        ],
        "node2parent": {
            "branch-a": "root-a",
            "branch-b": "branch-a",
            "branch-c": "root-a",
        },
    }
    journal_path = run_dir / "journal.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    return journal_path


def test_promotes_top_branch_nodes_to_new_root_hypotheses_with_manifest(tmp_path):
    task = "playground-series-s6e5"
    journal_path = _write_run(tmp_path)

    plan = plan_branch_hypothesis_promotion(
        root=tmp_path,
        task=task,
        journal_path=journal_path,
        top_n=2,
        agent_mode=None,
    )

    assert not plan.conflicts
    assert [entry.source_node_id for entry in plan.created] == ["branch-a", "branch-b"]
    assert [entry.hypothesis_id for entry in plan.created] == ["000001", "000002"]
    assert not plan.existing

    apply_promotion_plan(plan, dry_run=False)

    first_dir = tmp_path / "research_hypotheses" / task / "000001"
    hypothesis = json.loads((first_dir / "hypothesis-000001.json").read_text())
    manifest = json.loads((first_dir / "code_manifest.json").read_text())

    assert hypothesis["origin"]["source_run_id"] == "run-1"
    assert hypothesis["origin"]["source_node_id"] == "branch-a"
    assert hypothesis["origin"]["source_branch_path"] == ["000101", "000201"]
    assert (first_dir / "autogluon-001.py").read_text() == "print('branch a')\n"
    assert manifest["active"]["autogluon"] == "autogluon-001.py"
    entry = manifest["versions"]["autogluon"][0]
    assert entry["score"] == 0.95
    assert entry["aux"] is False
    assert entry["source_run_id"] == "run-1"
    assert entry["source_node_id"] == "branch-a"
    assert entry["source_artifact_dir"] == "logs/run-1/artifacts/branch-a-artifact"


def test_promotion_is_idempotent_and_extends_top_n_without_duplicates(tmp_path):
    task = "playground-series-s6e5"
    journal_path = _write_run(tmp_path)

    first_plan = plan_branch_hypothesis_promotion(
        root=tmp_path,
        task=task,
        journal_path=journal_path,
        top_n=2,
        agent_mode=None,
    )
    apply_promotion_plan(first_plan, dry_run=False)

    second_plan = plan_branch_hypothesis_promotion(
        root=tmp_path,
        task=task,
        journal_path=journal_path,
        top_n=3,
        agent_mode=None,
    )

    assert [entry.hypothesis_id for entry in second_plan.existing] == [
        "000001",
        "000002",
    ]
    assert [entry.source_node_id for entry in second_plan.created] == ["branch-c"]
    assert second_plan.created[0].hypothesis_id == "000003"


def test_promotion_can_override_agent_mode(tmp_path):
    task = "playground-series-s6e5"
    journal_path = _write_run(tmp_path, mode="legacy")

    plan = plan_branch_hypothesis_promotion(
        root=tmp_path,
        task=task,
        journal_path=journal_path,
        top_n=1,
        agent_mode="legacy",
    )
    apply_promotion_plan(plan, dry_run=False)

    hypothesis_dir = tmp_path / "research_hypotheses" / task / "000001"
    assert (hypothesis_dir / "legacy-001.py").exists()
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    assert manifest["active"]["legacy"] == "legacy-001.py"


def test_promotes_classic_node_by_step_to_root_hypothesis(tmp_path):
    task = "playground-series-s6e5"
    run_dir = tmp_path / "logs" / "classic-run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("agent:\n  mode: legacy\n", encoding="utf-8")
    journal = {
        "nodes": [
            {
                "id": "node-root",
                "parent": None,
                "step": 0,
                "code": "print('root')\n",
                "is_buggy": False,
                "metric": {"value": 0.90, "maximize": True},
                "ctime": 1000.0,
            },
            {
                "id": "node-113",
                "parent": "node-root",
                "step": 113,
                "plan": "Keep the RealMLP core and add nested fold-bagged blend configurations.",
                "analysis": "Nested meta-OOF blend selection completed successfully.",
                "validity_warning": "Nested blend warning.",
                "code": "print('classic top')\n",
                "is_buggy": False,
                "metric": {"value": 0.95467, "maximize": True},
                "artifact_dir_name": "classic-artifact",
                "ctime": 1001.0,
            },
        ],
        "node2parent": {"node-113": "node-root"},
    }
    journal_path = run_dir / "journal.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    plan = plan_node_promotion(
        root=tmp_path,
        task=task,
        journal_path=journal_path,
        step=113,
        node_id=None,
        agent_mode=None,
    )

    assert not plan.conflicts
    assert [entry.source_node_id for entry in plan.created] == ["node-113"]
    assert plan.created[0].source_step == 113
    assert plan.created[0].source_kind == "promoted_classic_node"

    apply_promotion_plan(plan, dry_run=False)

    hypothesis_dir = tmp_path / "research_hypotheses" / task / "000001"
    hypothesis = json.loads((hypothesis_dir / "hypothesis-000001.json").read_text())
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())

    assert hypothesis["origin"]["kind"] == "promoted_classic_node"
    assert hypothesis["title"] == "Nested meta-OOF blend selection completed successfully."
    assert hypothesis["summary"] == (
        "Nested meta-OOF blend selection completed successfully."
    )
    assert hypothesis["rationale"] == (
        "Keep the RealMLP core and add nested fold-bagged blend configurations."
    )
    assert hypothesis["implementation_hint"] == hypothesis["rationale"]
    assert "source classic node" not in hypothesis["rationale"]
    assert hypothesis["risk"] == "Nested blend warning."
    assert hypothesis["origin"]["source_step"] == 113
    assert hypothesis["origin"]["source_node_id"] == "node-113"
    assert (hypothesis_dir / "legacy-001.py").read_text() == "print('classic top')\n"
    assert manifest["versions"]["legacy"][0]["source_step"] == 113


def test_cli_promotes_classic_node_by_run_and_step(tmp_path):
    task = "playground-series-s6e5"
    run_dir = tmp_path / "logs" / "classic-run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("agent:\n  mode: legacy\n", encoding="utf-8")
    journal = {
        "nodes": [
            {
                "id": "node-113",
                "parent": None,
                "step": 113,
                "plan": "Classic CLI source plan.",
                "analysis": "Classic CLI source analysis.",
                "validity_warning": "",
                "code": "print('classic cli')\n",
                "is_buggy": False,
                "metric": {"value": 0.95467, "maximize": True},
                "ctime": 1001.0,
            },
        ],
        "node2parent": {},
    }
    (run_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "--task",
            task,
            "--run",
            "classic-run",
            "--step",
            "113",
        ]
    )

    assert exit_code == 0
    hypothesis_dir = tmp_path / "research_hypotheses" / task / "000001"
    assert (hypothesis_dir / "legacy-001.py").read_text() == "print('classic cli')\n"


def test_promotion_without_journal_uses_all_run_journals(tmp_path):
    task = "playground-series-s6e5"
    _write_run(tmp_path, run_id="run-1", branch_scores=(0.91, 0.90, 0.89))
    _write_run(tmp_path, run_id="run-2", branch_scores=(0.97, 0.96, 0.88))

    plan = plan_branch_hypothesis_promotion_from_logs(
        root=tmp_path,
        task=task,
        logs_dir=tmp_path / "logs",
        top_n=3,
        agent_mode=None,
    )

    assert [entry.source_run_id for entry in plan.created] == [
        "run-2",
        "run-2",
        "run-1",
    ]
    assert [entry.source_node_id for entry in plan.created] == [
        "branch-a",
        "branch-b",
        "branch-a",
    ]
    assert [entry.source_score for entry in plan.created] == [0.97, 0.96, 0.91]
