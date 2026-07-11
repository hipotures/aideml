import json
from dataclasses import dataclass
from pathlib import Path

from aide.journal import Journal, Node
from aide.utils.artifact_manifest import (
    RESULT_MANIFEST_NAME,
    SEEDED_BASE_PLAN_PREFIX,
    build_node_artifact_manifest,
    sha256_file,
)
from aide.utils.metric import MetricValue
from aide.utils.seed_artifact import (
    autogluon_seed_settings_changed,
    find_seed_artifact,
    seed_journal_from_artifact,
    source_is_autogluon,
)
from aide.utils.tree_export import cfg_to_tree_struct


@dataclass
class DummyConfig:
    log_dir: Path
    workspace_dir: Path
    exp_name: str = "test-exp"


def _write_source_artifact(
    top_log_dir: Path,
    run_id: str,
    timestamp: str,
    *,
    code: str,
    score: float,
    step: int,
    submission: str = "id,target\n1,0.9\n",
    log_text: str | None = None,
) -> Path:
    log_dir = top_log_dir / run_id
    artifact_dir = log_dir / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "solution.py").write_text(code, encoding="utf-8")
    (artifact_dir / "submission.csv").write_text(submission, encoding="utf-8")
    if log_text is not None:
        (artifact_dir / "autogluon_stdout.log").write_text(log_text, encoding="utf-8")
    (artifact_dir / "notes.txt").write_text("copied marker\n", encoding="utf-8")
    cfg = DummyConfig(log_dir=log_dir, workspace_dir=top_log_dir / "work")
    node = Node(code=code, plan="original plan", ctime=1778061600.0)
    node.step = step
    node.metric = MetricValue(score, maximize=True)
    node.is_buggy = False
    node.exec_time = 12.0
    node.analysis = "original analysis"
    node.validity_warning = "possible warning"
    node.submission_validation = {"status": "ok"}
    manifest = build_node_artifact_manifest(
        cfg=cfg,
        node=node,
        artifact_dir=artifact_dir,
    )
    (artifact_dir / RESULT_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return artifact_dir


def test_find_seed_artifact_matches_submission_hash_and_picks_best_duplicate(tmp_path):
    top_log_dir = tmp_path / "logs"
    first = _write_source_artifact(
        top_log_dir,
        "1-source-run",
        "20260506T120000",
        code="print('first')\n",
        score=0.90,
        step=1,
        submission="id,target\n1,0.1\n",
    )
    second = _write_source_artifact(
        top_log_dir,
        "2-source-run",
        "20260506T121000",
        code="print('second')\n",
        score=0.91,
        step=2,
        submission="id,target\n1,0.1\n",
    )
    prefix = sha256_file(first / "submission.csv")[:10]

    source = find_seed_artifact(top_log_dir, prefix)

    assert source.artifact_dir == second
    assert source.matched_kind == "submission"
    assert source.matched_sha256 == sha256_file(second / "submission.csv")


def test_find_seed_artifact_can_filter_by_source_run(tmp_path):
    top_log_dir = tmp_path / "logs"
    first = _write_source_artifact(
        top_log_dir,
        "1-source-run",
        "20260506T120000",
        code="print('first')\n",
        score=0.90,
        step=1,
        submission="id,target\n1,0.1\n",
    )
    _write_source_artifact(
        top_log_dir,
        "2-source-run",
        "20260506T121000",
        code="print('second')\n",
        score=0.91,
        step=2,
        submission="id,target\n1,0.1\n",
    )
    prefix = sha256_file(first / "submission.csv")[:10]

    source = find_seed_artifact(top_log_dir, prefix, source_run="1-source-run")

    assert source.artifact_dir == first


def test_seed_journal_from_artifact_copies_artifact_and_rewrites_manifest(tmp_path):
    top_log_dir = tmp_path / "logs"
    source_artifact = _write_source_artifact(
        top_log_dir,
        "1-source-run",
        "20260506T120000",
        code="print('seed')\n",
        score=0.95,
        step=11,
        log_text="seed execution log\n",
    )
    source = find_seed_artifact(top_log_dir, sha256_file(source_artifact / "submission.csv")[:12])
    cfg = DummyConfig(
        log_dir=top_log_dir / "2-new-run",
        workspace_dir=tmp_path / "workspaces" / "2-new-run",
    )

    journal, node, artifact_dir = seed_journal_from_artifact(
        cfg,
        source,
        ctime=1778065200.0,
    )

    assert len(journal.nodes) == 1
    assert journal.nodes[0] is node
    assert node.step == 0
    assert node.metric.value == 0.95
    assert node.analysis == "original analysis"
    assert node.term_out == "seed execution log\n"
    assert node.validity_warning == "possible warning"
    assert node.plan.startswith(SEEDED_BASE_PLAN_PREFIX)
    assert (artifact_dir / "notes.txt").read_text(encoding="utf-8") == "copied marker\n"
    assert (cfg.workspace_dir / "working" / "submission.csv").exists()

    manifest = json.loads((artifact_dir / RESULT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["run"] == "2-new-run"
    assert manifest["node"]["id"] == node.id
    assert manifest["node"]["step"] == 0
    assert manifest["node"]["origin"] == "seeded_base"
    assert manifest["source"]["source_run"] == "1-source-run"
    assert manifest["source"]["source_step"] == 11
    assert manifest["source"]["source_match_kind"] == "submission"
    assert manifest["files"]["solution"]["sha256"] == sha256_file(artifact_dir / "solution.py")


def test_seed_journal_from_artifact_code_only_copies_solution_without_score(tmp_path):
    top_log_dir = tmp_path / "logs"
    source_artifact = _write_source_artifact(
        top_log_dir,
        "1-source-run",
        "20260506T120000",
        code="print('seed')\n",
        score=0.95,
        step=11,
        log_text="seed execution log\n",
    )
    source = find_seed_artifact(top_log_dir, sha256_file(source_artifact / "submission.csv")[:12])
    cfg = DummyConfig(
        log_dir=top_log_dir / "2-new-run",
        workspace_dir=tmp_path / "workspaces" / "2-new-run",
    )

    journal, node, artifact_dir = seed_journal_from_artifact(
        cfg,
        source,
        ctime=1778065200.0,
        code_only=True,
    )

    assert journal.nodes == [node]
    assert node.status == "generated"
    assert node.metric is None
    assert node.exec_time is None
    assert node.term_out == ""
    assert node.submission_validation is None
    assert (artifact_dir / "solution.py").read_text(encoding="utf-8") == "print('seed')\n"
    assert not (artifact_dir / "submission.csv").exists()
    assert not (artifact_dir / "notes.txt").exists()
    assert not (cfg.workspace_dir / "working" / "submission.csv").exists()

    manifest = json.loads((artifact_dir / RESULT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "generated"
    assert manifest["local_score"] is None
    assert manifest["sha256"] is None
    assert manifest["files"]["submission"] is None
    assert manifest["node"]["metric"]["value"] is None
    assert manifest["execution"]["exec_time"] is None
    assert manifest["source"]["code_only"] is True


def test_seed_journal_from_artifact_code_override_replaces_pending_solution(tmp_path):
    top_log_dir = tmp_path / "logs"
    source_artifact = _write_source_artifact(
        top_log_dir,
        "1-source-run",
        "20260506T120000",
        code="print('old profile')\n",
        score=0.95,
        step=11,
    )
    source = find_seed_artifact(
        top_log_dir,
        sha256_file(source_artifact / "submission.csv")[:12],
    )
    cfg = DummyConfig(
        log_dir=top_log_dir / "2-new-run",
        workspace_dir=tmp_path / "workspaces" / "2-new-run",
    )

    journal, node, artifact_dir = seed_journal_from_artifact(
        cfg,
        source,
        code_only=True,
        code_override="print('new profile')\n",
    )

    assert journal.nodes == [node]
    assert node.status == "generated"
    assert node.code == "print('new profile')\n"
    assert (artifact_dir / "solution.py").read_text(encoding="utf-8") == node.code


def test_autogluon_seed_settings_changed_compares_effective_profile_settings(tmp_path):
    top_log_dir = tmp_path / "logs"
    source_artifact = _write_source_artifact(
        top_log_dir,
        "1-source-run",
        "20260506T120000",
        code=(
            "AIDE_AG_CONFIG = {'presets': 'medium_quality', "
            "'validation_strategy': 'holdout', "
            "'fit_args': {'auto_stack': False}}\n"
        ),
        score=0.95,
        step=11,
    )
    source = find_seed_artifact(
        top_log_dir,
        sha256_file(source_artifact / "submission.csv")[:12],
    )

    same_code = (
        "AIDE_AG_CONFIG = {'presets': 'medium_quality', "
        "'validation_strategy': 'holdout', "
        "'fit_args': {'auto_stack': False}}\n"
    )
    cv3_code = (
        "AIDE_AG_CONFIG = {'presets': 'high', "
        "'validation_strategy': 'autogluon', "
        "'fit_args': {'auto_stack': False, 'num_bag_folds': 3}}\n"
    )
    same_profile_changed_code = (
        "AIDE_AG_CONFIG = {'profile': 'shared', 'presets': 'high', "
        "'validation_strategy': 'autogluon', "
        "'fit_args': {'auto_stack': False, 'num_bag_folds': 3}}\n"
    )

    assert autogluon_seed_settings_changed(source, same_code) is False
    assert autogluon_seed_settings_changed(source, cv3_code) is True
    source.manifest["profile"] = "shared"
    assert autogluon_seed_settings_changed(source, same_profile_changed_code) is True


def test_seeded_single_node_can_be_exported_to_tree_struct(tmp_path):
    top_log_dir = tmp_path / "logs"
    source_artifact = _write_source_artifact(
        top_log_dir,
        "1-source-run",
        "20260506T120000",
        code="print('seed')\n",
        score=0.95,
        step=11,
    )
    source = find_seed_artifact(top_log_dir, sha256_file(source_artifact / "submission.csv")[:12])
    cfg = DummyConfig(
        log_dir=top_log_dir / "2-new-run",
        workspace_dir=tmp_path / "workspaces" / "2-new-run",
    )
    journal, _, _ = seed_journal_from_artifact(cfg, source, ctime=1778065200.0)

    tree = cfg_to_tree_struct(cfg, journal)

    assert tree["layout"] == [[0.5, 1.0]]
    assert tree["term_out"] == [""]


def test_empty_journal_can_be_exported_to_tree_struct(tmp_path):
    cfg = DummyConfig(
        log_dir=tmp_path / "logs" / "empty-run",
        workspace_dir=tmp_path / "workspaces" / "empty-run",
    )

    tree = cfg_to_tree_struct(cfg, Journal())

    assert tree["edges"] == []
    assert tree["layout"] == []
    assert tree["plan"] == []
    assert tree["code"] == []
    assert tree["term_out"] == []
    assert tree["analysis"] == []
    assert tree["metrics"] == []


def test_seed_source_is_autogluon_when_solution_has_ag_config(tmp_path):
    top_log_dir = tmp_path / "logs"
    source_artifact = _write_source_artifact(
        top_log_dir,
        "1-source-run",
        "20260506T120000",
        code="AIDE_AG_CONFIG = {'profile': 'fast_boost'}\nprint('ag')\n",
        score=0.95,
        step=11,
    )
    source = find_seed_artifact(top_log_dir, sha256_file(source_artifact / "solution.py")[:12])

    assert source.matched_kind == "solution"
    assert source_is_autogluon(source) is True


def test_find_seed_artifact_accepts_profile_eval_manifest(tmp_path):
    top_log_dir = tmp_path / "logs"
    artifact_dir = _write_source_artifact(
        top_log_dir,
        "1-source-run",
        "20260506T120000",
        code="AIDE_AG_CONFIG = {'profile': 'best_boost_2h'}\nprint('ag')\n",
        score=0.95224,
        step=-1,
        submission="id,target\n1,0.1\n",
    )
    manifest_path = artifact_dir / RESULT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["kind"] = "profile_eval"
    manifest["profile"] = "best_boost_2h"
    manifest["node"]["origin"] = "profile_eval"
    manifest["node"]["id"] = None
    manifest["node"]["step"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    source = find_seed_artifact(
        top_log_dir,
        sha256_file(artifact_dir / "submission.csv")[:10],
        source_run="1-source-run",
    )

    assert source.artifact_dir == artifact_dir
    assert source.manifest["kind"] == "profile_eval"
    assert source.source_step is None
    assert source_is_autogluon(source) is True
