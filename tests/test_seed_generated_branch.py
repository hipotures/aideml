import json
from pathlib import Path

from aide.journal import Journal, Node
from aide.run import load_resume_state
from aide.utils import serialize
from scripts.seed_generated_branch import (
    FORCED_CHILD_QUEUE_FILE,
    queue_for_existing_run,
    seed_generated_branch,
)


def _write_hypothesis(
    root: Path,
    task: str,
    hypothesis_id: str,
    *,
    code: str | None = None,
    enabled: bool = True,
    score: float | None = None,
) -> None:
    hypothesis_dir = root / "research_hypotheses" / task / hypothesis_id
    hypothesis_dir.mkdir(parents=True)
    (hypothesis_dir / f"hypothesis-{hypothesis_id}.json").write_text(
        json.dumps(
            {
                "enabled": enabled,
                "agent_modes": ["legacy"],
                "title": f"Hypothesis {hypothesis_id}",
                "summary": "summary",
                "rationale": "rationale",
                "implementation_hint": "implementation",
                "expected_effect": "effect",
                "risk": "risk",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    if code is not None:
        (hypothesis_dir / "legacy-001.py").write_text(code, encoding="utf-8")
    if score is not None:
        artifact_dir = root / "logs" / "source-run" / "artifacts" / hypothesis_id
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "aide_result.json").write_text(
            json.dumps(
                {
                    "node": {
                        "plan": f"Source plan for {hypothesis_id}.",
                        "analysis": f"Source analysis for {hypothesis_id}.",
                    },
                    "run_stats": {"cv_score": score},
                }
            ),
            encoding="utf-8",
        )
        (hypothesis_dir / "code_manifest.json").write_text(
            json.dumps(
                {
                    "active": {"legacy": "legacy-001.py"},
                    "versions": {
                        "legacy": [
                            {
                                "file": "legacy-001.py",
                                "buggy": False,
                                "status": "ok",
                                "score": score,
                                "source_artifact_dir": str(artifact_dir),
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )


def test_seed_generated_branch_writes_root_and_child_queue_without_generating(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")

    _write_hypothesis(
        repo_root,
        task,
        "001172",
        code="print('root old code')\n",
        score=0.95405,
    )
    _write_hypothesis(repo_root, task, "000806", enabled=False)
    _write_hypothesis(repo_root, task, "000530", enabled=False)

    result = seed_generated_branch(
        task=task,
        agent_mode="legacy",
        root_hypothesis="001172",
        root_code_file="legacy-001.py",
        children=("000806", "000530"),
        run_id="2-generated-branch-test",
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=tmp_path / "logs",
        workspaces_dir=tmp_path / "workspaces",
        repo_root=repo_root,
        prepare_workspace=False,
    )

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)

    assert [node.research_hypotheses_offered[0] for node in journal.nodes] == [
        "001172",
    ]
    assert [node.status for node in journal.nodes] == [
        "ok",
    ]
    assert journal.nodes[0].parent is None
    assert journal.nodes[0].code == "print('root old code')\n"
    assert journal.nodes[0].metric.value == 0.95405

    queue = json.loads((result.log_dir / FORCED_CHILD_QUEUE_FILE).read_text())
    assert queue == {
        "root_hypothesis": "001172",
        "children": ["000806", "000530"],
    }


def test_seed_generated_branch_allows_root_only_without_child_queue(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")

    _write_hypothesis(
        repo_root,
        task,
        "001263",
        code="print('root only code')\n",
        score=0.95467,
    )

    result = seed_generated_branch(
        task=task,
        agent_mode="legacy",
        root_hypothesis="001263",
        root_code_file="legacy-001.py",
        children=(),
        run_id="2-root-only-generated-branch-test",
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=tmp_path / "logs",
        workspaces_dir=tmp_path / "workspaces",
        repo_root=repo_root,
        prepare_workspace=False,
    )

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)

    assert result.children == ()
    assert [node.research_hypotheses_offered[0] for node in journal.nodes] == [
        "001263",
    ]
    assert journal.nodes[0].status == "ok"
    assert journal.nodes[0].code == "print('root only code')\n"
    assert journal.nodes[0].metric.value == 0.95467
    assert not (result.log_dir / FORCED_CHILD_QUEUE_FILE).exists()


def test_seed_generated_scored_root_is_not_submission_validated_on_resume(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")

    _write_hypothesis(
        repo_root,
        task,
        "001263",
        code="print('root only code')\n",
        score=0.95467,
    )

    result = seed_generated_branch(
        task=task,
        agent_mode="legacy",
        root_hypothesis="001263",
        root_code_file="legacy-001.py",
        children=(),
        run_id="2-root-only-generated-branch-test",
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=tmp_path / "logs",
        workspaces_dir=tmp_path / "workspaces",
        repo_root=repo_root,
        prepare_workspace=False,
    )
    (result.workspace_dir / "input" / "sample_submission.csv").write_text(
        "id,PitNextLap\n1,0.0\n",
        encoding="utf-8",
    )

    _cfg, loaded = load_resume_state(
        run_id=result.run_id,
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )
    persisted = serialize.load_json(result.log_dir / "journal.json", Journal)

    assert loaded.nodes[0].status == "ok"
    assert loaded.nodes[0].is_buggy is False
    assert loaded.nodes[0].exc_type is None
    assert loaded.nodes[0].metric.value == 0.95467
    assert persisted.nodes[0].status == "ok"
    assert persisted.nodes[0].is_buggy is False
    assert persisted.nodes[0].exc_type is None
    assert persisted.nodes[0].metric.value == 0.95467


def test_seed_generated_branch_uses_source_node_prompt_summary(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")
    artifact_dir = tmp_path / "logs" / "2-source-run" / "artifacts" / "source-artifact"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "aide_result.json").write_text(
        json.dumps(
            {
                "node": {
                    "plan": "Keep the RealMLP core and add nested fold-bagged blend configurations.",
                    "analysis": "Nested meta-OOF blend selection completed successfully.",
                    "validity_warning": "Nested blend warning.",
                },
                "run_stats": {
                    "cv_score": 0.95467,
                    "realmlp_n_ens": 40,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_hypothesis(
        repo_root,
        task,
        "001263",
        code="print('root only code')\n",
        score=0.95467,
    )
    manifest_path = (
        repo_root
        / "research_hypotheses"
        / task
        / "001263"
        / "code_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["versions"]["legacy"][0]["source_artifact_dir"] = str(artifact_dir)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = seed_generated_branch(
        task=task,
        agent_mode="legacy",
        root_hypothesis="001263",
        root_code_file="legacy-001.py",
        children=(),
        run_id="2-root-only-generated-branch-test",
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=tmp_path / "logs",
        workspaces_dir=tmp_path / "workspaces",
        repo_root=repo_root,
        prepare_workspace=False,
    )

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)
    summary = journal.generate_summary()

    assert journal.nodes[0].plan == (
        "Keep the RealMLP core and add nested fold-bagged blend configurations."
    )
    assert journal.nodes[0].analysis == (
        "Nested meta-OOF blend selection completed successfully."
    )
    assert journal.nodes[0].validity_warning == "Nested blend warning."
    assert journal.nodes[0].run_stats["source_node_run_stats"]["realmlp_n_ens"] == 40
    assert "Seeded scored ROOT hypothesis" not in summary
    assert "execution skipped" not in summary
    assert "Keep the RealMLP core" in summary


def test_seed_generated_branch_queue_accepts_disabled_child_without_code(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")

    _write_hypothesis(repo_root, task, "001172", code="print('root old code')\n")
    _write_hypothesis(repo_root, task, "000806", enabled=False)

    result = seed_generated_branch(
        task=task,
        agent_mode="legacy",
        root_hypothesis="001172",
        root_code_file="legacy-001.py",
        children=("000806",),
        run_id="2-generated-branch-test",
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=tmp_path / "logs",
        workspaces_dir=tmp_path / "workspaces",
        repo_root=repo_root,
        prepare_workspace=False,
    )

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)
    assert len(journal.nodes) == 1
    queue = json.loads((result.log_dir / FORCED_CHILD_QUEUE_FILE).read_text())
    assert queue["children"] == ["000806"]


def test_seed_generated_branch_accepts_disabled_root_with_code(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")

    _write_hypothesis(
        repo_root,
        task,
        "001214",
        code="print('disabled root code')\n",
        enabled=False,
        score=0.95450,
    )
    _write_hypothesis(repo_root, task, "001253", enabled=False)

    result = seed_generated_branch(
        task=task,
        agent_mode="legacy",
        root_hypothesis="001214",
        root_code_file="legacy-001.py",
        children=("001253",),
        run_id="2-disabled-root-generated-branch-test",
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=tmp_path / "logs",
        workspaces_dir=tmp_path / "workspaces",
        repo_root=repo_root,
        prepare_workspace=False,
    )

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)

    assert journal.nodes[0].research_hypotheses_offered == ["001214"]
    assert journal.nodes[0].status == "ok"
    assert journal.nodes[0].code == "print('disabled root code')\n"
    assert journal.nodes[0].metric.value == 0.95450


def test_queue_for_existing_run_appends_children_without_creating_run(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")
    logs_dir = tmp_path / "logs"
    run_id = "2-existing-family"
    log_dir = logs_dir / run_id
    log_dir.mkdir(parents=True)

    _write_hypothesis(repo_root, task, "001214", code="print('root')\n", enabled=False)
    _write_hypothesis(repo_root, task, "001253", enabled=False)
    _write_hypothesis(repo_root, task, "001254", enabled=False)
    _write_hypothesis(repo_root, task, "001255", enabled=False)

    root = Journal()
    root_node = Node(code="print('root')\n", plan="root")
    root_node.research_mode = "hypothesis"
    root_node.research_hypotheses_offered = ["001214"]
    root.append(root_node)
    serialize.dump_json(root, log_dir / "journal.json")
    (log_dir / FORCED_CHILD_QUEUE_FILE).write_text(
        json.dumps({"root_hypothesis": "001214", "children": ["001253"]}),
        encoding="utf-8",
    )

    result = queue_for_existing_run(
        task=task,
        agent_mode="legacy",
        root_hypothesis="001214",
        children=("001253", "001254", "001255"),
        run_id=run_id,
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=logs_dir,
        repo_root=repo_root,
    )

    assert result.run_id == run_id
    assert result.log_dir == log_dir.resolve()
    queue = json.loads((log_dir / FORCED_CHILD_QUEUE_FILE).read_text())
    assert queue == {
        "root_hypothesis": "001214",
        "children": ["001253", "001254", "001255"],
    }


def test_queue_for_existing_run_requires_root_in_journal(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")
    logs_dir = tmp_path / "logs"
    run_id = "2-existing-family"
    log_dir = logs_dir / run_id
    log_dir.mkdir(parents=True)

    _write_hypothesis(repo_root, task, "001214", code="print('root')\n", enabled=False)
    _write_hypothesis(repo_root, task, "001253", enabled=False)

    journal = Journal()
    other_node = Node(code="print('other')\n", plan="other")
    other_node.research_mode = "hypothesis"
    other_node.research_hypotheses_offered = ["000001"]
    journal.append(other_node)
    serialize.dump_json(journal, log_dir / "journal.json")

    try:
        queue_for_existing_run(
            task=task,
            agent_mode="legacy",
            root_hypothesis="001214",
            children=("001253",),
            run_id=run_id,
            data_dir=data_dir,
            desc_file=desc_file,
            logs_dir=logs_dir,
            repo_root=repo_root,
        )
    except ValueError as exc:
        assert "not found in existing run journal" in str(exc)
    else:
        raise AssertionError("queue_for_existing_run should reject missing root")
