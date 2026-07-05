import os
import time
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aide.agent import Agent
from aide.journal import Journal, Node
from aide.research import effective_agent_gpu_enabled
from aide.run import (
    allocate_node_artifact_slot,
    _completed_work_units,
    CodeAheadJob,
    code_ahead_has_capacity,
    code_ahead_pending_count,
    ensure_node_artifact_slot,
    enforce_journal_submission_contract,
    find_latest_run_id,
    generate_code_ahead_node,
    load_resume_state,
    mark_node_generated_only,
    maybe_seed_scored_hypothesis_roots,
    next_generated_only_node,
    parse_runtime_args,
    parse_resume_args,
    ParallelRootFailureState,
    record_generated_only_node,
    recover_generated_only_root_artifacts,
    save_parallel_generate_only_run,
    rebind_code_ahead_node,
    should_parallel_generate_only_roots,
    should_seed_scored_hypothesis_roots_for_run,
    should_cleanup_workspace_on_exit,
    should_stop_after_generate_only_roots,
    should_code_ahead_run,
    should_wait_for_code_ahead,
    validate_code_ahead,
    validate_hypothesis_root_generate_workers,
)
from aide.utils.config import _load_cfg, prep_cfg, save_run
from aide.utils.node_artifacts import node_artifact_dir, node_artifact_submission_path
from aide.utils.metric import MetricValue
from aide.utils import serialize


@pytest.fixture(autouse=True)
def _isolate_repo_dotenv(monkeypatch, tmp_path):
    # These unit tests assert default config behavior; repo .env is production input.
    monkeypatch.chdir(tmp_path)


def _write_run(tmp_path: Path, run_id: str, *, steps: int, mtime: float) -> None:
    log_dir = tmp_path / "logs" / run_id
    workspace_dir = tmp_path / "workspaces" / run_id
    (workspace_dir / "input").mkdir(parents=True)
    (workspace_dir / "working").mkdir(parents=True)

    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = run_id.split("-", 1)[1]
    cfg.agent.steps = steps
    cfg = prep_cfg(cfg)
    cfg.exp_name = run_id
    cfg.log_dir = log_dir
    cfg.workspace_dir = workspace_dir

    journal = Journal()
    node = Node(code="print('ok')", plan="ok")
    node.metric = MetricValue(0.9, maximize=True)
    node.is_buggy = False
    node._term_out = ["score 0.9"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ok"
    journal.append(node)
    save_run(cfg, journal, current_node=node)
    os.utime(log_dir / "journal.json", (mtime, mtime))


def test_parse_resume_args_accepts_run_id_and_keeps_omegaconf_overrides():
    resume, remaining = parse_resume_args(
        [
            "--resume",
            "2-judicious-unbreakable-hoatzin",
            "agent.steps=200",
            "generate_report=False",
        ]
    )

    assert resume.requested is True
    assert resume.run_id == "2-judicious-unbreakable-hoatzin"
    assert resume.use_latest is False
    assert remaining == ["agent.steps=200", "generate_report=False"]


def test_parse_resume_args_without_run_id_uses_latest_and_keeps_next_override():
    resume, remaining = parse_resume_args(["--resume", "agent.steps=200"])

    assert resume.requested is True
    assert resume.run_id is None
    assert resume.use_latest is True
    assert remaining == ["agent.steps=200"]


def test_parse_runtime_args_extracts_submission_flags_from_omegaconf_overrides():
    resume, runtime, remaining = parse_runtime_args(
        [
            "--resume",
            "2-example-run",
            "--show-invalid-submission-branches",
            "--force-check-submissions",
            "--telegram-test-message",
            "--debug",
            "agent.steps=200",
        ]
    )

    assert resume.run_id == "2-example-run"
    assert runtime.show_invalid_submission_branches is True
    assert runtime.force_check_submissions is True
    assert runtime.telegram_test_message is True
    assert runtime.debug is True
    assert remaining == ["agent.steps=200"]


def test_parse_runtime_args_extracts_web_dashboard_flags():
    resume, runtime, remaining = parse_runtime_args(
        [
            "--resume",
            "2-example-run",
            "--web",
            "--web-host=127.0.0.1",
            "--web-port",
            "9001",
            "agent.steps=200",
        ]
    )

    assert resume.run_id == "2-example-run"
    assert runtime.web_enabled is True
    assert runtime.web_host == "127.0.0.1"
    assert runtime.web_port == 9001
    assert remaining == ["agent.steps=200"]


def test_parse_runtime_args_rejects_invalid_web_port():
    with pytest.raises(ValueError, match="web-port"):
        parse_runtime_args(["--web-port", "not-a-port"])


def test_parse_runtime_args_extracts_skip_execution_flag():
    resume, runtime, remaining = parse_runtime_args(
        [
            "--resume",
            "2-example-run",
            "--skip-execution",
            "agent.steps=200",
        ]
    )

    assert resume.run_id == "2-example-run"
    assert runtime.skip_execution is True
    assert runtime.generate_only_requested is False
    assert remaining == ["agent.steps=200"]


def test_parse_runtime_args_extracts_generate_only_without_ids():
    resume, runtime, remaining = parse_runtime_args(
        [
            "--resume",
            "2-example-run",
            "--generate-only",
            "agent.steps=200",
        ]
    )

    assert resume.run_id == "2-example-run"
    assert runtime.skip_execution is True
    assert runtime.generate_only_requested is True
    assert runtime.generate_only_hypothesis_ids == ()
    assert remaining == ["agent.steps=200"]


def test_parse_runtime_args_extracts_generate_only_hypothesis_ids():
    resume, runtime, remaining = parse_runtime_args(
        [
            "--resume",
            "2-example-run",
            "--generate-only",
            "001162",
            "001170",
            "agent.steps=200",
        ]
    )

    assert resume.run_id == "2-example-run"
    assert runtime.skip_execution is True
    assert runtime.generate_only_requested is True
    assert runtime.generate_only_hypothesis_ids == ("001162", "001170")
    assert remaining == ["agent.steps=200"]


def test_generate_only_disables_scored_root_seeding():
    _resume, runtime, _remaining = parse_runtime_args(["--generate-only"])

    assert should_seed_scored_hypothesis_roots_for_run(runtime) is False


def test_skip_execution_does_not_disable_scored_root_seeding():
    _resume, runtime, _remaining = parse_runtime_args(["--skip-execution"])

    assert should_seed_scored_hypothesis_roots_for_run(runtime) is True


@pytest.mark.parametrize("workers", [0, -1, 9, "four"])
def test_validate_hypothesis_root_generate_workers_rejects_invalid_values(workers):
    cfg = _load_cfg(use_cli_args=False)
    cfg.research.hypothesis_root_generate_workers = workers

    with pytest.raises(ValueError, match="hypothesis_root_generate_workers"):
        validate_hypothesis_root_generate_workers(cfg)


def test_validate_code_ahead_accepts_default():
    cfg = _load_cfg(use_cli_args=False)

    assert validate_code_ahead(cfg) == 0


@pytest.mark.parametrize("value", [-1, 9, True, 1.5, "2"])
def test_validate_code_ahead_rejects_invalid_values(value):
    cfg = _load_cfg(use_cli_args=False)
    cfg.agent.search.code_ahead = value

    with pytest.raises(ValueError, match="agent.search.code_ahead"):
        validate_code_ahead(cfg)


def test_should_code_ahead_run_only_for_execute_legacy_or_autogluon():
    assert should_code_ahead_run(
        code_ahead=1,
        skip_execution=False,
        research_mode="llm",
        synthesis_enabled=False,
        agent_mode="legacy",
    )
    assert should_code_ahead_run(
        code_ahead=1,
        skip_execution=False,
        research_mode="llm",
        synthesis_enabled=False,
        agent_mode="autogluon_preprocess",
    )
    assert not should_code_ahead_run(
        code_ahead=0,
        skip_execution=False,
        research_mode="llm",
        synthesis_enabled=False,
        agent_mode="legacy",
    )
    assert not should_code_ahead_run(
        code_ahead=1,
        skip_execution=True,
        research_mode="llm",
        synthesis_enabled=False,
        agent_mode="legacy",
    )
    assert not should_code_ahead_run(
        code_ahead=1,
        skip_execution=False,
        research_mode="hypothesis",
        synthesis_enabled=False,
        agent_mode="legacy",
    )
    assert not should_code_ahead_run(
        code_ahead=1,
        skip_execution=False,
        research_mode="llm",
        synthesis_enabled=True,
        agent_mode="legacy",
    )


def test_forced_generate_only_hypothesis_ids_disable_parallel_workers():
    assert not should_parallel_generate_only_roots(
        skip_execution=True,
        materialize_enabled=True,
        research_mode="hypothesis",
        hypothesis_root_generate_workers=3,
        forced_hypothesis_ids=("000014", "000015", "000016"),
    )


def test_generate_only_parallel_workers_require_no_forced_ids():
    assert should_parallel_generate_only_roots(
        skip_execution=True,
        materialize_enabled=True,
        research_mode="hypothesis",
        hypothesis_root_generate_workers=3,
        forced_hypothesis_ids=(),
    )


def test_seed_scored_hypothesis_roots_appends_missing_on_resume(monkeypatch):
    cfg = _load_cfg(use_cli_args=False)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.research.seed_scored_roots = True
    journal = Journal()
    seeded = Node(code="print('seeded')", plan="seeded")
    seeded.research_mode = "hypothesis"
    seeded.research_hypotheses_offered = ["000019"]

    monkeypatch.setattr(
        "aide.run.scored_hypothesis_root_nodes",
        lambda _cfg: [seeded],
    )

    count = maybe_seed_scored_hypothesis_roots(
        cfg,
        journal,
        is_resume=True,
    )

    assert count == 1
    assert journal.nodes == [seeded]


def test_seed_scored_hypothesis_roots_appends_only_for_new_hypothesis_runs(
    monkeypatch,
):
    cfg = _load_cfg(use_cli_args=False)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.research.seed_scored_roots = True
    journal = Journal()
    seeded = Node(code="print('seeded')", plan="seeded")

    monkeypatch.setattr(
        "aide.run.scored_hypothesis_root_nodes",
        lambda _cfg: [seeded],
    )

    count = maybe_seed_scored_hypothesis_roots(
        cfg,
        journal,
        is_resume=False,
    )

    assert count == 1
    assert journal.nodes == [seeded]
    assert seeded.step == 0


def test_seed_scored_hypothesis_roots_skips_existing_resume_root(monkeypatch):
    cfg = _load_cfg(use_cli_args=False)
    cfg.research.enabled = False
    cfg.research.mode = "llm"
    cfg.research.seed_scored_roots = False
    journal = Journal()
    existing = Node(code="print('existing')", plan="existing")
    existing.research_mode = "hypothesis"
    existing.research_hypotheses_offered = ["000019"]
    journal.append(existing)
    seeded = Node(code="print('seeded')", plan="seeded")
    seeded.research_mode = "hypothesis"
    seeded.research_hypotheses_offered = ["000019"]

    monkeypatch.setattr(
        "aide.run.scored_hypothesis_root_nodes",
        lambda _cfg: [seeded],
    )

    count = maybe_seed_scored_hypothesis_roots(
        cfg,
        journal,
        is_resume=True,
    )

    assert count == 0
    assert journal.nodes == [existing]


def test_seed_scored_hypothesis_roots_recovers_manifest_roots_for_old_resume(
    monkeypatch,
):
    cfg = _load_cfg(use_cli_args=False)
    cfg.research.enabled = False
    cfg.research.mode = "llm"
    cfg.research.seed_scored_roots = False
    journal = Journal()
    existing = Node(code="print('existing')", plan="existing")
    existing.research_mode = "hypothesis"
    existing.research_hypotheses_offered = ["000011"]
    journal.append(existing)
    seeded = Node(code="print('seeded')", plan="seeded")
    seeded.research_mode = "hypothesis"
    seeded.research_hypotheses_offered = ["000019"]

    monkeypatch.setattr(
        "aide.run.scored_hypothesis_root_nodes",
        lambda _cfg: [seeded],
    )

    count = maybe_seed_scored_hypothesis_roots(
        cfg,
        journal,
        is_resume=True,
    )

    assert count == 1
    assert journal.nodes == [existing, seeded]


def test_seed_scored_hypothesis_roots_refreshes_broken_seeded_root(monkeypatch):
    cfg = _load_cfg(use_cli_args=False)
    cfg.research.enabled = False
    cfg.research.mode = "llm"
    cfg.research.seed_scored_roots = False
    journal = Journal()
    existing = Node(code="print('broken')", plan="broken")
    existing.research_mode = "hypothesis"
    existing.research_hypotheses_offered = ["000019"]
    existing.research_runtime_config = {"gpu": bool(cfg.agent.gpu)}
    existing.status = "ok"
    existing.is_buggy = True
    existing.metric = None
    existing.run_stats = {"seeded_from_manifest": True}
    journal.append(existing)
    seeded = Node(code="print('seeded')", plan="seeded")
    seeded.research_mode = "hypothesis"
    seeded.research_hypotheses_offered = ["000019"]
    seeded.research_runtime_config = {"gpu": bool(cfg.agent.gpu)}
    seeded.status = "ok"
    seeded.is_buggy = False
    seeded.metric = MetricValue(0.966515, maximize=True)
    seeded.run_stats = {"seeded_from_manifest": True}
    seeded._term_out = ["score=0.966515"]
    seeded.exec_time = 670.84
    seeded.analysis = "seeded ok"

    monkeypatch.setattr(
        "aide.run.scored_hypothesis_root_nodes",
        lambda _cfg: [seeded],
    )

    count = maybe_seed_scored_hypothesis_roots(
        cfg,
        journal,
        is_resume=True,
    )

    assert count == 1
    assert len(journal.nodes) == 1
    assert existing.code == "print('seeded')"
    assert existing.is_buggy is False
    assert existing.metric.value == 0.966515
    assert existing in journal.good_nodes


def test_seed_scored_hypothesis_roots_persists_recovered_process_stdout(
    tmp_path,
    monkeypatch,
):
    cfg = _load_cfg(use_cli_args=False)
    cfg.log_dir = tmp_path / "logs" / "2-current-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-current-run"
    (cfg.workspace_dir / "input").mkdir(parents=True)
    cfg.research.enabled = False
    cfg.research.mode = "llm"
    cfg.research.seed_scored_roots = False
    journal = Journal()
    existing = Node(code="print('existing')", plan="existing")
    existing.research_mode = "hypothesis"
    existing.research_hypotheses_offered = ["000011"]
    journal.append(existing)
    seeded = Node(code="print('seeded')", plan="seeded")
    seeded.research_mode = "hypothesis"
    seeded.research_hypotheses_offered = ["000019"]
    seeded.status = "ok"
    seeded.is_buggy = False
    seeded.metric = MetricValue(0.966515, maximize=True)
    seeded._term_out = [
        "Fold 1 balanced_accuracy=0.967258\n",
        "OOF balanced_accuracy=0.966515\n",
    ]
    seeded.run_stats = {
        "seeded_from_manifest": True,
        "source_process_stdout_recovered": True,
    }

    monkeypatch.setattr(
        "aide.run.scored_hypothesis_root_nodes",
        lambda _cfg: [seeded],
    )

    count = maybe_seed_scored_hypothesis_roots(
        cfg,
        journal,
        is_resume=True,
    )

    stdout_path = (
        cfg.log_dir / "artifacts" / seeded.artifact_dir_name / "process_stdout.log"
    )
    assert count == 1
    assert stdout_path.read_text(encoding="utf-8") == (
        "Fold 1 balanced_accuracy=0.967258\n"
        "OOF balanced_accuracy=0.966515\n"
    )


def test_seed_scored_hypothesis_roots_refreshes_existing_ok_node_for_recovered_log(
    tmp_path,
    monkeypatch,
):
    cfg = _load_cfg(use_cli_args=False)
    cfg.log_dir = tmp_path / "logs" / "2-current-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-current-run"
    (cfg.workspace_dir / "input").mkdir(parents=True)
    cfg.research.enabled = False
    cfg.research.mode = "llm"
    cfg.research.seed_scored_roots = False
    journal = Journal()
    existing = Node(code="print('seeded')", plan="Seeded scored ROOT hypothesis 000019")
    existing.research_mode = "hypothesis"
    existing.research_hypotheses_offered = ["000019"]
    existing.status = "ok"
    existing.is_buggy = False
    existing.metric = MetricValue(0.966515, maximize=True)
    existing._term_out = ["Seeded from code_manifest.json; score=0.96652.\n"]
    existing.run_stats = {"seeded_from_manifest": True}
    journal.append(existing)
    ensure_node_artifact_slot(cfg, existing)

    seeded = Node(code="print('seeded')", plan="Seeded scored ROOT hypothesis 000019")
    seeded.research_mode = "hypothesis"
    seeded.research_hypotheses_offered = ["000019"]
    seeded.status = "ok"
    seeded.is_buggy = False
    seeded.metric = MetricValue(0.966515, maximize=True)
    seeded._term_out = ["OOF balanced_accuracy=0.966515\n"]
    seeded.run_stats = {
        "seeded_from_manifest": True,
        "source_process_stdout_recovered": True,
    }

    monkeypatch.setattr(
        "aide.run.scored_hypothesis_root_nodes",
        lambda _cfg: [seeded],
    )

    count = maybe_seed_scored_hypothesis_roots(
        cfg,
        journal,
        is_resume=True,
    )

    stdout_path = (
        cfg.log_dir / "artifacts" / existing.artifact_dir_name / "process_stdout.log"
    )
    assert count == 1
    assert existing._term_out == ["OOF balanced_accuracy=0.966515\n"]
    assert existing.run_stats["source_process_stdout_recovered"] is True
    assert stdout_path.read_text(encoding="utf-8") == "OOF balanced_accuracy=0.966515\n"


def test_seed_scored_hypothesis_roots_keeps_manifest_runtime(monkeypatch):
    cfg = _load_cfg(use_cli_args=False)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.research.seed_scored_roots = True
    journal = Journal()
    seeded = Node(code="print('seeded')", plan="seeded")
    seeded.status = "ok"
    seeded.exec_time = 399.0
    seeded.run_stats = {"seeded_from_manifest": True}

    monkeypatch.setattr(
        "aide.run.scored_hypothesis_root_nodes",
        lambda _cfg: [seeded],
    )

    count = maybe_seed_scored_hypothesis_roots(
        cfg,
        journal,
        is_resume=False,
    )

    assert count == 1
    assert journal.nodes == [seeded]
    assert seeded.exec_time == 399.0


def test_submission_contract_skips_seeded_scored_roots_without_artifacts(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.log_dir = tmp_path / "logs"
    (cfg.workspace_dir / "input").mkdir(parents=True)
    (cfg.workspace_dir / "input" / "sample_submission.csv").write_text(
        "id,PitNextLap\n1,0.5\n",
        encoding="utf-8",
    )

    journal = Journal()
    node = Node(
        code="print('seeded')",
        plan="Seeded scored ROOT hypothesis 001172 from legacy legacy-001.py.",
    )
    node.metric = MetricValue(0.95405, maximize=True)
    node.status = "ok"
    node.is_buggy = False
    node.exec_time = 0.0
    node._term_out = [
        "Seeded from code_manifest.json; score=0.95405; file=legacy-001.py."
    ]
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["001172"]
    journal.append(node)

    changed = enforce_journal_submission_contract(cfg, journal)

    assert changed == 0
    assert node.metric.value == 0.95405
    assert node.is_buggy is False
    assert node.status == "ok"


def test_generated_only_nodes_are_pending_until_evaluated():
    journal = Journal()
    executed = Node(code="print('ok')", plan="ok")
    executed.metric = MetricValue(0.9, maximize=True)
    executed.is_buggy = False
    pending = Node(code="print('later')", plan="later")
    journal.append(executed)
    journal.append(pending)

    mark_node_generated_only(pending)

    assert pending.status == "generated"
    assert pending.is_buggy is False
    assert pending.metric is None
    assert next_generated_only_node(journal) is pending

    pending.status = "ok"
    assert next_generated_only_node(journal) is None


def test_generated_only_selection_respects_forced_root_scope():
    journal = Journal()
    outside_root = Node(code="print('outside')", plan="outside")
    outside_root.research_mode = "hypothesis"
    outside_root.research_hypotheses_offered = ["000111"]
    forced_root = Node(code="print('forced')", plan="forced")
    forced_root.research_mode = "hypothesis"
    forced_root.research_hypotheses_offered = ["000222"]
    journal.append(outside_root)
    journal.append(forced_root)

    mark_node_generated_only(outside_root)
    mark_node_generated_only(forced_root)

    assert next_generated_only_node(journal, forced_root="000222") is forced_root


def test_generated_only_selection_respects_forced_hypothesis_exact_id():
    journal = Journal()
    forced_root = Node(code="print('forced')", plan="forced")
    forced_root.research_mode = "hypothesis"
    forced_root.research_hypotheses_offered = ["000941"]
    child = Node(code="print('child')", plan="child", parent=forced_root)
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["000365"]
    journal.append(forced_root)
    journal.append(child)

    mark_node_generated_only(forced_root)
    mark_node_generated_only(child)

    assert (
        next_generated_only_node(journal, forced_hypothesis="000941") is forced_root
    )


def test_record_generated_only_node_marks_saves_and_appends():
    journal = Journal()
    node = Node(code="print('generated')", plan="generated")

    class AgentStub:
        saved_node: Node | None = None
        activate: bool | None = None

        def save_hypothesis_root_code_for_node(
            self,
            node: Node,
            *,
            activate: bool = True,
        ) -> None:
            self.saved_node = node
            self.activate = activate

    agent = AgentStub()

    record_generated_only_node(
        agent=agent,
        journal=journal,
        node=node,
        experiment_id="test-run",
    )

    assert node.status == "generated"
    assert journal.nodes == [node]
    assert agent.saved_node is node
    assert agent.activate is False


def test_record_generated_only_node_restores_active_parent_when_missing():
    parent = Node(code="print('parent')", plan="parent")
    node = Node(code="print('generated')", plan="generated")
    journal = Journal(nodes=[parent])

    class AgentStub:
        active_parent_node = parent
        saved_node: Node | None = None
        activate: bool | None = None

        def save_hypothesis_root_code_for_node(
            self,
            node: Node,
            *,
            activate: bool = True,
        ) -> None:
            self.saved_node = node
            self.activate = activate

    agent = AgentStub()

    record_generated_only_node(
        agent=agent,
        journal=journal,
        node=node,
        experiment_id="test-run",
    )

    assert node.parent is parent
    assert node in parent.children
    assert journal.nodes == [parent, node]


def test_save_parallel_generate_only_run_persists_journal(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "2-generated-only-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-generated-only-run"
    cfg.exp_name = "2-generated-only-run"
    cfg = prep_cfg(cfg)
    cfg.log_dir = tmp_path / "logs" / "2-generated-only-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-generated-only-run"

    journal = Journal()
    node = Node(code="print('generated')", plan="generated")
    mark_node_generated_only(node)
    journal.append(node)

    message = save_parallel_generate_only_run(
        cfg=cfg,
        journal=journal,
        current_node=node,
    )

    assert message == (
        "Skip-execution mode finished generating root candidates; no code was executed."
    )
    loaded = serialize.load_json(cfg.log_dir / "journal.json", Journal)
    assert len(loaded.nodes) == 1
    assert loaded.nodes[0].status == "generated"
    assert loaded.nodes[0].code == "print('generated')"


def test_skip_execution_can_generate_branch_children():
    parent = Node(code="print('parent')", plan="parent")

    generate_only_runtime = parse_runtime_args(["--generate-only"])[1]
    skip_execution_runtime = parse_runtime_args(["--skip-execution"])[1]

    assert should_stop_after_generate_only_roots(
        generate_only_runtime,
        parent,
    )
    assert not should_stop_after_generate_only_roots(
        skip_execution_runtime,
        parent,
    )


def test_load_json_accepts_inline_generated_node_without_artifact(tmp_path):
    log_dir = tmp_path / "logs" / "2-inline-generated"
    log_dir.mkdir(parents=True)
    journal_path = log_dir / "journal.json"
    journal_path.write_text(
        json.dumps(
            {
                "__version": "3",
                "nodes": [
                    {
                        "id": "generated-root",
                        "code": "print('generated')",
                        "code_path": None,
                        "artifact_dir_name": None,
                        "plan": "generated",
                        "step": 1,
                        "ctime": 1777750547.0,
                        "parent": None,
                        "status": "generated",
                        "_term_out": [],
                        "metric": None,
                        "is_buggy": False,
                        "research_mode": "hypothesis",
                        "research_hypotheses_offered": ["000002"],
                    }
                ],
                "node2parent": {},
            }
        ),
        encoding="utf-8",
    )

    loaded = serialize.load_json(journal_path, Journal)
    serialized = json.loads(serialize.dumps_json(loaded, base_dir=log_dir))

    assert loaded.nodes[0].status == "generated"
    assert loaded.nodes[0].code == "print('generated')"
    assert loaded.nodes[0].code_path is None
    assert loaded.nodes[0].artifact_dir_name is None
    assert serialized["nodes"][0]["code"] == "print('generated')"
    assert serialized["nodes"][0]["code_path"] is None
    assert serialized["nodes"][0]["artifact_dir_name"] is None


def test_generated_only_journal_prevents_workspace_cleanup():
    journal = Journal()
    node = Node(code="print('generated')", plan="generated")
    mark_node_generated_only(node)
    journal.append(node)

    assert should_cleanup_workspace_on_exit(is_resume=False, journal=journal) is False


def test_next_generated_only_node_requires_matching_gpu_runtime(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "2-generated-only-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-generated-only-run"
    cfg.exp_name = "2-generated-only-run"
    cfg.agent.gpu = True
    cfg = prep_cfg(cfg)
    cfg.agent.gpu = True

    journal = Journal()
    cpu_node = Node(code="print('cpu')", plan="generated")
    cpu_node.research_mode = "hypothesis"
    cpu_node.research_hypotheses_offered = ["000011"]
    cpu_node.research_runtime_config = {"gpu": False}
    mark_node_generated_only(cpu_node)
    journal.append(cpu_node)

    assert next_generated_only_node(journal, cfg=cfg) is None

    gpu_node = Node(code="print('gpu')", plan="generated")
    gpu_node.research_mode = "hypothesis"
    gpu_node.research_hypotheses_offered = ["000012"]
    gpu_node.research_runtime_config = {"gpu": True}
    mark_node_generated_only(gpu_node)
    journal.append(gpu_node)

    assert next_generated_only_node(journal, cfg=cfg) is gpu_node


def test_next_generated_only_node_requires_matching_agent_mode_runtime(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "2-generated-only-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-generated-only-run"
    cfg.exp_name = "2-generated-only-run"
    cfg.agent.mode = "autogluon_preprocess"
    cfg = prep_cfg(cfg)

    journal = Journal()
    runtime_gpu = effective_agent_gpu_enabled(cfg)
    legacy_node = Node(code="print('legacy')", plan="generated")
    legacy_node.research_runtime_config = {
        "agent_mode": "legacy",
        "gpu": runtime_gpu,
    }
    mark_node_generated_only(legacy_node)
    journal.append(legacy_node)

    assert next_generated_only_node(journal, cfg=cfg) is None

    autogluon_node = Node(code="print('autogluon')", plan="generated")
    autogluon_node.research_runtime_config = {
        "agent_mode": "autogluon_preprocess",
        "gpu": runtime_gpu,
    }
    mark_node_generated_only(autogluon_node)
    journal.append(autogluon_node)

    assert next_generated_only_node(journal, cfg=cfg) is autogluon_node


def test_next_generated_only_node_reads_agent_mode_from_artifact_context(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "2-generated-only-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-generated-only-run"
    cfg.exp_name = "2-generated-only-run"
    cfg.agent.mode = "autogluon_preprocess"
    cfg = prep_cfg(cfg)

    artifact_dir = Path(cfg.log_dir) / "artifacts" / "legacy-artifact"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "context.json").write_text(
        json.dumps(
            {
                "agent_mode": "legacy",
                "agent_gpu": effective_agent_gpu_enabled(cfg),
            }
        ),
        encoding="utf-8",
    )
    legacy_node = Node(
        code="print('legacy')",
        plan="generated",
        artifact_dir_name="legacy-artifact",
    )
    mark_node_generated_only(legacy_node)
    journal = Journal()
    journal.append(legacy_node)

    assert next_generated_only_node(journal, cfg=cfg) is None


def test_artifacts_prevent_workspace_cleanup_with_empty_journal(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    (log_dir / "artifacts" / "20260101T000000-abcdef12-0").mkdir(parents=True)

    assert (
        should_cleanup_workspace_on_exit(
            is_resume=False,
            journal=Journal(),
            log_dir=log_dir,
        )
        is False
    )


def test_recover_generated_only_root_artifacts_materializes_completed_orphans(
    tmp_path,
    monkeypatch,
):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path / "playground-series-s6e5")
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "2-existing-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    cfg.exp_name = "2-existing-run"
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.mode = "legacy"
    cfg = prep_cfg(cfg)
    cfg.log_dir.mkdir(parents=True)
    cfg.workspace_dir.mkdir(parents=True)

    run_research_dir = cfg.log_dir / "research_hypotheses"
    run_research_dir.mkdir(parents=True)
    (run_research_dir / "offers.jsonl").write_text(
        json.dumps(
            {
                "checkpoint_step": 0,
                "offered": ["000123"],
                "source_hash": "sha256:test",
                "created_at": "2026-05-23T00:00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifact_dir = cfg.log_dir / "artifacts" / "20260523T233540-abcdef12"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "context.json").write_text(
        json.dumps(
            {
                "phase": "generate",
                "parent_node_id": None,
                "agent_mode": "legacy",
                "node_ctime": 1779575740.0,
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "status.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    (artifact_dir / "request.md").write_text(
        "# Hypothesis under verification\n\n"
        "Hypothesis ID: 000123\n"
        "Implement this exact hypothesis.\n",
        encoding="utf-8",
    )
    (artifact_dir / "response_raw.txt").write_text(
        "Hypothesis ID 000123 recovered plan.\n\n```python\nprint('recovered')\n```\n",
        encoding="utf-8",
    )
    (artifact_dir / "response.py").write_text(
        "print('recovered')\n",
        encoding="utf-8",
    )

    journal = Journal()
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    monkeypatch.setattr(
        agent,
        "save_hypothesis_root_code_for_node",
        lambda _node, *, activate=True: None,
    )

    recovered = recover_generated_only_root_artifacts(
        cfg=cfg,
        journal=journal,
        agent=agent,
    )

    assert recovered == 1
    assert len(journal.nodes) == 1
    node = journal.nodes[0]
    assert node.status == "generated"
    assert node.code == "print('recovered')\n"
    assert node.plan == "Hypothesis ID 000123 recovered plan."
    assert node.artifact_dir_name == "20260523T233540-abcdef12"
    assert node.research_hypotheses_offered == ["000123"]
    assert node.research_source_hash == "sha256:test"
    usage = json.loads((run_research_dir / "usage.json").read_text())
    assert usage["000123"]["prompt_node_ids"] == [node.id]


def test_generated_only_nodes_are_not_scored_good_candidates():
    journal = Journal()
    executed = Node(code="print('ok')", plan="ok")
    executed.metric = MetricValue(0.9, maximize=True)
    executed.is_buggy = False
    pending = Node(code="print('later')", plan="later")
    journal.append(executed)
    journal.append(pending)

    mark_node_generated_only(pending)

    assert journal.good_nodes == [executed]
    assert journal.get_best_node() is executed
    assert journal.get_best_node(only_good=False) is executed


def test_completed_work_units_excludes_execute_mode_generated_drafts():
    journal = Journal()
    executed = Node(code="print('ok')", plan="ok")
    executed.metric = MetricValue(0.9, maximize=True)
    executed.is_buggy = False
    journal.append(executed)
    for index in range(2):
        pending = Node(code=f"print({index})", plan="pending")
        journal.append(pending)
        mark_node_generated_only(pending)

    assert _completed_work_units(journal, generated_only_evaluations=0) == 1
    assert _completed_work_units(journal, generated_only_evaluations=2) == 3


def test_code_ahead_pending_count_includes_generated_nodes_and_in_flight(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "2-code-ahead-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-code-ahead-run"
    cfg.exp_name = "2-code-ahead-run"
    cfg = prep_cfg(cfg)

    journal = Journal()
    executed = Node(code="print('ok')", plan="ok")
    executed.metric = MetricValue(0.9, maximize=True)
    executed.is_buggy = False
    journal.append(executed)
    pending = Node(code="print('pending')", plan="pending")
    journal.append(pending)
    mark_node_generated_only(pending)
    failed = Node(code="raise RuntimeError()", plan="failed")
    failed.status = "failed"
    journal.append(failed)

    assert code_ahead_pending_count(journal, cfg=cfg, in_flight_count=1) == 2


def test_code_ahead_capacity_counts_pending_and_in_flight_against_limit():
    assert code_ahead_has_capacity(
        code_ahead=3,
        total_steps=10,
        completed_work_units=2,
        pending_count=2,
    )
    assert not code_ahead_has_capacity(
        code_ahead=3,
        total_steps=10,
        completed_work_units=2,
        pending_count=3,
    )
    assert not code_ahead_has_capacity(
        code_ahead=3,
        total_steps=5,
        completed_work_units=3,
        pending_count=2,
    )


def test_code_ahead_wait_decision_does_not_block_ready_generated_draft():
    assert not should_wait_for_code_ahead(
        code_ahead_enabled=True,
        has_pending_generated_node=True,
        has_in_flight_generation=True,
    )


def test_code_ahead_wait_decision_blocks_when_generation_is_only_next_work():
    assert should_wait_for_code_ahead(
        code_ahead_enabled=True,
        has_pending_generated_node=False,
        has_in_flight_generation=True,
    )


def test_generate_code_ahead_node_uses_isolated_parent_then_rebinds(
    tmp_path,
    monkeypatch,
):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "2-code-ahead-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-code-ahead-run"
    cfg.exp_name = "2-code-ahead-run"
    cfg = prep_cfg(cfg)

    journal = Journal()
    parent = Node(code="print('parent')", plan="parent")
    parent.metric = MetricValue(0.9, maximize=True)
    parent.is_buggy = False
    journal.append(parent)
    base_agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    base_agent.data_preview = "preview"

    def fake_generate_node(self, parent_node, *, node_ctime=None, llm_log_dir=None):
        assert parent_node is not parent
        return Node(
            code="print('child')",
            plan="child",
            parent=parent_node,
            ctime=node_ctime,
        )

    monkeypatch.setattr(Agent, "generate_node", fake_generate_node)

    job = CodeAheadJob(
        parent_node=parent,
        node_ctime=123.0,
        artifact_dir_name="draft-artifact",
        artifact_dir=tmp_path / "draft-artifact",
        active_step=1,
        launched_index=1,
    )

    result = generate_code_ahead_node(
        base_agent=base_agent,
        journal=journal,
        job=job,
    )

    assert parent.children == set()
    node = rebind_code_ahead_node(result)
    assert node.parent is parent
    assert node in parent.children


def test_node_artifact_dir_uses_explicit_name(tmp_path):
    node = Node(
        code="print('x')",
        plan="x",
        ctime=1_779_492_701.0,
        artifact_dir_name="20260523T220603-a1b2c3d4",
    )

    assert node_artifact_dir(tmp_path, node) == (
        tmp_path / "artifacts" / "20260523T220603-a1b2c3d4"
    )
    assert node_artifact_submission_path(tmp_path, node) == (
        tmp_path / "artifacts" / "20260523T220603-a1b2c3d4" / "submission.csv"
    )


def test_node_artifact_dir_falls_back_to_legacy_ctime_timestamp(tmp_path):
    node = Node(code="print('x')", plan="x", ctime=1_779_492_701.0)

    assert node_artifact_dir(tmp_path, node).name == "20260523T013141"


def test_allocate_node_artifact_slot_sets_unique_explicit_name(tmp_path):
    first_ctime, first_dir_name, first_dir = allocate_node_artifact_slot(tmp_path)
    second_ctime, second_dir_name, second_dir = allocate_node_artifact_slot(tmp_path)

    assert first_ctime <= second_ctime
    assert first_dir_name != second_dir_name
    assert first_dir.name == first_dir_name
    assert second_dir.name == second_dir_name
    assert first_dir.exists()
    assert second_dir.exists()
    assert (first_dir / "aide_solution_helpers.py").exists()
    assert (second_dir / "aide_solution_helpers.py").exists()
    assert len(first_dir_name.split("-")[-1]) == 8


def test_allocate_node_artifact_slot_appends_step_to_explicit_name(tmp_path):
    _ctime, dir_name, artifact_dir = allocate_node_artifact_slot(tmp_path, step=113)

    parts = dir_name.split("-")
    assert len(parts) == 3
    assert len(parts[1]) == 8
    assert parts[2] == "113"
    assert artifact_dir.name == dir_name
    assert artifact_dir.exists()
    assert (artifact_dir / "aide_solution_helpers.py").exists()


def test_allocate_node_artifact_slot_links_workspace_input(tmp_path):
    workspace_dir = tmp_path / "workspaces" / "run"
    (workspace_dir / "input").mkdir(parents=True)

    _ctime, _dir_name, artifact_dir = allocate_node_artifact_slot(
        tmp_path / "logs" / "run",
        workspace_dir=workspace_dir,
    )

    input_link = artifact_dir / "input"
    assert input_link.is_symlink()
    assert input_link.resolve() == (workspace_dir / "input").resolve()


def test_ensure_node_artifact_slot_assigns_hash_name_to_legacy_node(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "run"
    cfg = prep_cfg(cfg)
    node = Node(code="print('x')", plan="x", ctime=1_779_492_701.0)

    artifact_dir = ensure_node_artifact_slot(cfg, node)

    assert node.artifact_dir_name is not None
    assert node.artifact_dir_name.startswith("20260523T013141-")
    assert len(node.artifact_dir_name.split("-")[-1]) == 8
    assert artifact_dir == cfg.log_dir / "artifacts" / node.artifact_dir_name
    assert artifact_dir.exists()

    same_dir = ensure_node_artifact_slot(cfg, node)

    assert same_dir == artifact_dir


def test_ensure_node_artifact_slot_copies_solution_helper(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "run"
    cfg = prep_cfg(cfg)
    node = Node(
        code="from aide_solution_helpers import load_competition_data\n",
        plan="loaded root code",
        ctime=1_779_492_701.0,
    )

    artifact_dir = ensure_node_artifact_slot(cfg, node)

    assert (artifact_dir / "aide_solution_helpers.py").exists()


def test_ensure_node_artifact_slot_appends_node_step_to_explicit_name(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "run"
    cfg = prep_cfg(cfg)
    node = Node(code="print('x')", plan="x", ctime=1_779_492_701.0, step=113)

    artifact_dir = ensure_node_artifact_slot(cfg, node)

    assert node.artifact_dir_name is not None
    assert node.artifact_dir_name.startswith("20260523T013141-")
    parts = node.artifact_dir_name.split("-")
    assert len(parts) == 3
    assert len(parts[1]) == 8
    assert parts[2] == "113"
    assert artifact_dir == cfg.log_dir / "artifacts" / node.artifact_dir_name
    assert artifact_dir.exists()


def test_generation_retry_policy_stops_refill_after_three_failures():
    state = ParallelRootFailureState()

    assert state.record_failure("000405", RuntimeError("network")) is False
    assert state.record_failure("000405", RuntimeError("network")) is False
    assert state.record_failure("000405", RuntimeError("network")) is True
    assert state.stop_refill is True


def test_parse_runtime_args_extracts_seed_options():
    resume, runtime, remaining = parse_runtime_args(
        [
            "--seed-from-sha",
            "abcdef12",
            "--seed-source-run=2-source-run",
            "agent.steps=200",
        ]
    )

    assert resume.requested is False
    assert runtime.seed_sha_prefix == "abcdef12"
    assert runtime.seed_source_run == "2-source-run"
    assert remaining == ["agent.steps=200"]


def test_parse_runtime_args_rejects_resume_with_seed():
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_runtime_args(["--resume", "--seed-from-sha=abcdef12"])


def test_find_latest_run_id_uses_newest_journal_mtime(tmp_path):
    _write_run(tmp_path, "1-old-run", steps=10, mtime=time.time() - 100)
    _write_run(tmp_path, "2-new-run", steps=20, mtime=time.time())

    assert find_latest_run_id(tmp_path / "logs") == "2-new-run"


def test_load_resume_state_uses_existing_paths_and_cli_overrides(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())

    cfg, journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.steps=25", "generate_report=False"],
    )

    assert cfg.exp_name == "2-existing-run"
    assert Path(cfg.log_dir) == tmp_path / "logs" / "2-existing-run"
    assert Path(cfg.workspace_dir) == tmp_path / "workspaces" / "2-existing-run"
    assert cfg.agent.steps == 25
    assert cfg.generate_report is False
    assert len(journal.nodes) == 1
    assert journal.nodes[0].metric.value == 0.9


def test_load_resume_state_restores_solution_helper_to_workspace(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    helper_path = (
        tmp_path / "workspaces" / "2-existing-run" / "aide_solution_helpers.py"
    )
    helper_path.unlink(missing_ok=True)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert helper_path.exists()
    assert Path(cfg.workspace_dir) == helper_path.parent


def test_load_resume_state_applies_env_model_over_saved_config(tmp_path, monkeypatch):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    monkeypatch.setenv("AIDE_AGENT_CODE_MODEL", "gpt-5.3-codex-spark:medium")

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.agent.code.model == "gpt-5.3-codex-spark"
    assert cfg.agent.code.reasoning_effort == "medium"


def test_load_resume_state_preserves_saved_agent_mode_over_env(tmp_path, monkeypatch):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.mode = "autogluon_preprocess"
    OmegaConf.save(cfg_data, config_path)
    monkeypatch.setenv("AIDE_AGENT_MODE", "legacy")

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.agent.mode == "autogluon_preprocess"


def test_load_resume_state_cli_agent_mode_override_wins_over_saved_mode(
    tmp_path,
    monkeypatch,
):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    monkeypatch.setenv("AIDE_AGENT_MODE", "legacy")

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.mode=autogluon_preprocess"],
    )

    assert cfg.agent.mode == "autogluon_preprocess"


def test_load_resume_state_applies_dotenv_model_over_saved_config(tmp_path, monkeypatch):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    monkeypatch.delenv("AIDE_AGENT_CODE_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "AIDE_AGENT_CODE_MODEL=gpt-5.3-codex-spark:medium\n",
        encoding="utf-8",
    )

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.agent.code.model == "gpt-5.3-codex-spark"
    assert cfg.agent.code.reasoning_effort == "medium"


def test_load_resume_state_cli_model_override_wins_over_env(tmp_path, monkeypatch):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    monkeypatch.setenv("AIDE_AGENT_CODE_MODEL", "gpt-5.3-codex-spark:medium")

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.code.model=gpt-5.4"],
    )

    assert cfg.agent.code.model == "gpt-5.4"
    assert cfg.agent.code.reasoning_effort is None


def test_load_resume_state_migrates_legacy_memory_prompt_defaults(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.memory_recent_steps = 100
    cfg_data.agent.memory_full_recent_steps = 20
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.agent.memory_recent_steps == 50
    assert cfg.agent.memory_full_recent_steps == 10


def test_load_resume_state_preserves_custom_memory_prompt_values(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.memory_recent_steps = 80
    cfg_data.agent.memory_full_recent_steps = 0
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.agent.memory_recent_steps == 80
    assert cfg.agent.memory_full_recent_steps == 0


def test_load_resume_state_materializes_aux_file_override(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    data_dir = tmp_path / "data"
    source_dir = data_dir / "original_sdss17"
    source_dir.mkdir(parents=True)
    (source_dir / "star_classification.csv").write_text(
        "alpha,class\n1.0,STAR\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.data_dir = data_dir
    OmegaConf.save(cfg_data, config_path)
    workspace_input = tmp_path / "workspaces" / "2-existing-run" / "input"
    workspace_source_dir = workspace_input / "original_sdss17"
    workspace_source_dir.mkdir()
    (workspace_source_dir / "star_classification.csv").write_text(
        "alpha,class\n1.0,STAR\n",
        encoding="utf-8",
    )

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.aux=star_classification.csv"],
    )

    assert cfg.agent.aux == "star_classification.csv"
    assert (workspace_input / "star_classification.csv").read_text(
        encoding="utf-8"
    ) == "alpha,class\n1.0,STAR\n"
    assert not workspace_source_dir.exists()


def test_load_resume_state_rebases_remote_repo_data_dir_for_aux_file(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    data_dir = tmp_path / "aide" / "example_tasks" / "playground-series-s6e6"
    source_dir = data_dir / "original_sdss17"
    source_dir.mkdir(parents=True)
    (source_dir / "star_classification.csv").write_text(
        "alpha,class\n1.0,STAR\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.data_dir = Path(
        "/home/user/DEV/aideml/aide/example_tasks/playground-series-s6e6"
    )
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.aux=star_classification.csv"],
    )

    assert Path(cfg.data_dir) == data_dir
    assert (
        tmp_path
        / "workspaces"
        / "2-existing-run"
        / "input"
        / "star_classification.csv"
    ).read_text(encoding="utf-8") == "alpha,class\n1.0,STAR\n"


def test_save_run_serializes_repo_paths_relative(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = repo_root / "aide" / "example_tasks" / "house_prices"
    cfg.desc_file = repo_root / "aide" / "example_tasks" / "house_prices.md"
    cfg.goal = None
    cfg.log_dir = tmp_path / "logs" / "2-existing-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    cfg.exp_name = "2-existing-run"
    cfg = prep_cfg(cfg)
    cfg.exp_name = "2-existing-run"
    cfg.log_dir = tmp_path / "logs" / "2-existing-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    journal = Journal()
    node = Node(code="print('ok')", plan="ok")
    node._term_out = ["ok"]
    journal.append(node)

    save_run(cfg, journal)

    saved = (tmp_path / "logs" / "2-existing-run" / "config.yaml").read_text(
        encoding="utf-8"
    )
    assert str(repo_root / "aide" / "example_tasks") not in saved
    assert "aide/example_tasks/house_prices" in saved
    assert "aide/example_tasks/house_prices.md" in saved


def test_load_resume_state_clears_saved_forced_root_without_cli_override(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.search.forced_root = "000405"
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.agent.search.forced_root is None


def test_load_resume_state_preserves_env_forced_root_override(tmp_path, monkeypatch):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    monkeypatch.setenv("AIDE_AGENT_SEARCH_FORCED_ROOT", "000405")

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.agent.search.forced_root == "000405"


def test_load_resume_state_preserves_unquoted_forced_root_cli_id(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.search.forced_root=000405"],
    )

    assert cfg.agent.search.forced_root == "000405"


def test_load_resume_state_accepts_short_forced_root_cli_alias(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["forced_root=000405"],
    )

    assert cfg.agent.search.forced_root == "000405"


def test_resume_profile_override_clears_old_autogluon_model_list(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.autogluon.included_model_types = ["XGB", "GBM", "CAT"]
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.autogluon.profile=fast_boost"],
    )

    assert cfg.agent.autogluon.profile == "fast_boost"
    assert cfg.agent.autogluon.included_model_types is None


def test_resume_explicit_model_list_override_wins_over_profile(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.autogluon.included_model_types = ["XGB", "GBM", "CAT"]
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[
            "agent.autogluon.profile=fast_boost",
            "agent.autogluon.included_model_types=[CAT]",
        ],
    )

    assert cfg.agent.autogluon.profile == "fast_boost"
    assert cfg.agent.autogluon.included_model_types == ["CAT"]


def test_load_resume_state_persists_submission_contract_revalidation(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    log_dir = tmp_path / "logs" / "2-existing-run"
    workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    (workspace_dir / "input" / "sample_submission.csv").write_text(
        "id,PitNextLap\n1,0.0\n2,0.0\n"
    )

    journal = serialize.load_json(log_dir / "journal.json", Journal)
    artifact_dir = node_artifact_dir(log_dir, journal.nodes[0])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "submission.csv").write_text("id,PitNextLap\n1,0.8\n1,0.9\n")

    _cfg, loaded = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )
    persisted = serialize.load_json(log_dir / "journal.json", Journal)

    assert loaded.nodes[0].is_buggy is True
    assert loaded.nodes[0].metric.value is None
    assert persisted.nodes[0].is_buggy is True
    assert persisted.nodes[0].metric.value is None
    assert persisted.nodes[0].exc_type == "SubmissionValidationError"


def test_load_resume_state_does_not_submission_validate_generated_nodes(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    log_dir = tmp_path / "logs" / "2-existing-run"
    workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    (workspace_dir / "input" / "sample_submission.csv").write_text(
        "id,PitNextLap\n1,0.0\n"
    )

    journal = Journal()
    generated = Node(
        code="print('generated')",
        plan="generated",
        artifact_dir_name="generated-artifact",
    )
    journal.append(generated)
    mark_node_generated_only(generated)
    generated.is_buggy = True
    generated.exc_type = "SubmissionValidationError"
    generated.exc_info = {"args": ["missing artifact submission.csv"]}
    generated._term_out = [
        "SubmissionValidationError: missing artifact submission.csv\n"
    ]
    artifact_dir = log_dir / "artifacts" / generated.artifact_dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "solution.py").write_text(generated.code, encoding="utf-8")
    serialize.dump_json(journal, log_dir / "journal.json")

    _cfg, loaded = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )
    persisted = serialize.load_json(log_dir / "journal.json", Journal)

    assert loaded.nodes[0].status == "generated"
    assert loaded.nodes[0].is_buggy is False
    assert loaded.nodes[0].exc_type is None
    assert loaded.nodes[0]._term_out == []
    assert persisted.nodes[0].status == "generated"
    assert persisted.nodes[0].is_buggy is False
    assert persisted.nodes[0].exc_type is None


def test_load_resume_state_force_revalidates_cached_submission(tmp_path, monkeypatch):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    log_dir = tmp_path / "logs" / "2-existing-run"
    workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    (workspace_dir / "input" / "sample_submission.csv").write_text(
        "id,PitNextLap\n1,0.0\n"
    )

    journal = serialize.load_json(log_dir / "journal.json", Journal)
    artifact_dir = node_artifact_dir(log_dir, journal.nodes[0])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    submission_path = artifact_dir / "submission.csv"
    submission_path.write_text("id,PitNextLap\n1,0.8\n")
    journal.nodes[0].submission_validation = {
        "status": "ok",
        "sample_signature": {"size": 1, "mtime_ns": 1},
        "submission_signature": {"size": 1, "mtime_ns": 1},
    }
    serialize.dump_json(journal, log_dir / "journal.json")
    calls = []

    def fake_validate(_submission_path, _sample_path):
        calls.append((_submission_path, _sample_path))
        return None

    monkeypatch.setattr("aide.run.validate_submission_file", fake_validate)

    load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
        force_check_submissions=True,
    )

    assert calls


def test_load_resume_state_rejects_missing_workspace(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    OmegaConf.load(tmp_path / "logs" / "2-existing-run" / "config.yaml")
    missing_workspace = tmp_path / "workspaces" / "2-existing-run"
    missing_workspace.rename(tmp_path / "workspaces" / "moved-run")

    with pytest.raises(FileNotFoundError, match="workspace"):
        load_resume_state(
            run_id="2-existing-run",
            top_log_dir=tmp_path / "logs",
            top_workspace_dir=tmp_path / "workspaces",
            cli_overrides=[],
        )


def test_load_resume_state_defaults_research_for_older_configs(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    del cfg_data["research"]
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.research.enabled is False
    assert cfg.research.root_hypothesis_model == "gpt-5.4-mini"
    assert cfg.research.reasoning_effort == "low"
    assert cfg.research.previous_summary_count == 5
    assert cfg.synthesis.enabled is False
    assert cfg.synthesis.model == "gpt-5.4-mini"
    assert cfg.synthesis.reasoning_effort == "low"
    assert cfg.synthesis.every_scored_steps == 15
    assert cfg.exec.memory_limit_gb == 80.0
    assert cfg.agent.mode == "legacy"
    assert cfg.agent.autogluon.profile == "s6e6_boost_gpu"
    assert cfg.agent.autogluon.time_limit == 600
    assert cfg.agent.autogluon.included_model_types is None


def test_load_resume_state_migrates_legacy_research_model_key(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.research.model = "gpt-legacy-root"
    del cfg_data.research.root_hypothesis_model
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.research.root_hypothesis_model == "gpt-legacy-root"
    assert "model" not in cfg.research


def test_load_resume_state_normalizes_autogluon_mode_alias(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.mode=autogluon"],
    )

    assert cfg.agent.mode == "autogluon_preprocess"


def test_load_resume_state_ignores_deprecated_seeded_base_limit(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.search.seeded_base_max_children = 3
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert not hasattr(cfg.agent.search, "seeded_base_max_children")


def test_load_resume_state_ignores_deprecated_forced_parent(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.search.forced_parent = "34"
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert not hasattr(cfg.agent.search, "forced_parent")
