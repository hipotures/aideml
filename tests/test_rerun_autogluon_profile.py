import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

LAB_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_submission_lab.py"
LAB_SPEC = importlib.util.spec_from_file_location("kaggle_submission_lab", LAB_PATH)
kaggle_submission_lab = importlib.util.module_from_spec(LAB_SPEC)
assert LAB_SPEC and LAB_SPEC.loader
sys.modules[LAB_SPEC.name] = kaggle_submission_lab
LAB_SPEC.loader.exec_module(kaggle_submission_lab)

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rerun_autogluon_profile.py"
SPEC = importlib.util.spec_from_file_location("rerun_autogluon_profile", MODULE_PATH)
rerun_autogluon_profile = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = rerun_autogluon_profile
SPEC.loader.exec_module(rerun_autogluon_profile)


def _ctime(timestamp: str) -> float:
    return dt.datetime.strptime(timestamp, "%Y%m%dT%H%M%S").timestamp()


def test_create_profile_eval_artifact_without_modifying_journal(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    run_dir = logs_dir / "run-a"
    artifact_dir = run_dir / "artifacts" / "20260504T100000"
    artifact_dir.mkdir(parents=True)
    input_dir = tmp_path / "workspaces" / "run-a" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n")
    source_code = (
        "AIDE_AG_CONFIG = {'included_model_types': ['XGB', 'GBM'], 'time_limit': 300}\n"
        "def preprocess(df):\n"
        "    return df\n"
    )
    (artifact_dir / "solution.py").write_text(source_code)
    (artifact_dir / "submission.csv").write_text("id,target\n1,0.8\n")
    journal_path = run_dir / "journal.json"
    journal_path.write_text(
        json.dumps(
            {
                "__version": "test",
                "node2parent": {},
                "nodes": [
                    {
                        "step": 1,
                        "id": "node-source",
                        "ctime": _ctime("20260504T100000"),
                        "metric": {"value": 0.95, "maximize": True},
                        "is_buggy": False,
                    }
                ],
            }
        )
    )
    original_journal_text = journal_path.read_text()
    source_record = {
        "kind": "source_node",
        "competition": "playground-series-s6e5",
        "run": "run-a",
        "step": 1,
        "node_id": "node-source",
        "timestamp": "20260504T100000",
        "artifact_dir": str(artifact_dir),
        "solution_path": str(artifact_dir / "solution.py"),
        "local_score": 0.95,
        "sha256": kaggle_submission_lab.sha256_file(artifact_dir / "submission.csv"),
    }

    class FakeResult:
        term_out = [
            'AIDE_RESULT_JSON: {"is_bug": false, "lower_is_better": false, '
            '"metric": 0.951, "eval_metric": "balanced_accuracy", "summary": "ok"}\n'
        ]
        exec_time = 42.0
        exc_type = None
        exc_info = None
        exc_stack = None

    def fake_execute(code, *, workspace_dir, artifact_dir, timeout, memory_limit_gb):
        assert "'included_model_types': ['XGB', 'GBM', 'CAT']" in code
        assert "'time_limit': 600" in code
        assert "'presets': 'best_quality'" in code
        assert "'use_gpu': False" in code
        assert "'XGB': [{" in code
        assert "'device': 'cpu'" in code
        assert "'ag_args': {'priority': 999}" in code
        assert "'ag_args_fit': {'num_gpus': 0}" in code
        assert "'fit_args'" not in code.split("RESULT_MARKER", 1)[0]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "submission.csv").write_text("id,target\n1,0.9\n")
        (artifact_dir / "autogluon_stdout.log").write_text("training log\n")
        return FakeResult()

    monkeypatch.setattr(rerun_autogluon_profile, "execute_code", fake_execute)
    monkeypatch.setattr(
        rerun_autogluon_profile,
        "timestamp_now",
        lambda: "20260504T120000",
    )

    record = rerun_autogluon_profile.run_profile_eval(
        source_record,
        logs_dir=logs_dir,
        profile="full_boost",
        presets="best_quality",
        time_limit=600,
        fit_args={},
        competition="playground-series-s6e5",
        timeout=1200,
        memory_limit_gb=80.0,
    )

    eval_artifact = run_dir / "artifacts" / "20260504T120000"
    assert journal_path.read_text() == original_journal_text
    assert (eval_artifact / "solution.py").exists()
    assert (eval_artifact / "submission.csv").read_text() == "id,target\n1,0.9\n"
    eval_meta = json.loads((eval_artifact / "submission_eval.json").read_text())
    assert eval_meta["source_sha256"] == source_record["sha256"]
    assert eval_meta["profile"] == "full_boost"
    assert eval_meta["autogluon_presets"] == "best_quality"
    assert eval_meta["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert eval_meta["eval_metric"] == "balanced_accuracy"
    manifest = json.loads((eval_artifact / "aide_result.json").read_text())
    assert manifest["kind"] == "profile_eval"
    assert manifest["run"] == "run-a"
    assert manifest["timestamp"] == "20260504T120000"
    assert manifest["status"] == "ok"
    assert manifest["local_score"] == 0.951
    assert manifest["eval_metric"] == "balanced_accuracy"
    assert manifest["node"]["metric"]["name"] == "balanced_accuracy"
    assert manifest["autogluon"]["eval_metric"] == "balanced_accuracy"
    assert manifest["profile"] == "full_boost"
    assert manifest["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert manifest["time_limit"] == 600
    assert manifest["execution"]["exec_time"] == 42.0
    assert manifest["source"]["source_sha256"] == source_record["sha256"]
    assert manifest["files"]["submission"]["path"] == "submission.csv"
    assert record["kind"] == "profile_eval"
    assert record["eval_metric"] == "balanced_accuracy"
    assert record["sha256"] == kaggle_submission_lab.sha256_file(
        eval_artifact / "submission.csv"
    )

    console = rerun_autogluon_profile.Console(
        record=True,
        width=200,
        color_system=None,
    )
    rerun_autogluon_profile.render_profile_eval_results(console, [record])
    output = console.export_text()
    assert "20260504T120000" in output
    assert "1" in output
    assert record["sha256"][:10] in output
    assert source_record["sha256"][:10] in output


def test_run_profile_eval_can_execute_whole_solution_without_rebuilding_wrapper(
    tmp_path, monkeypatch
):
    logs_dir = tmp_path / "logs"
    run_dir = logs_dir / "run-a"
    source_artifact = run_dir / "artifacts" / "20260504T100000"
    source_artifact.mkdir(parents=True)
    input_dir = tmp_path / "workspaces" / "run-a" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n")
    whole_solution = (
        "CUSTOM_AUTOGLUON_BEHAVIOR = True\n"
        "print('AIDE_RESULT_JSON: {\"is_bug\": false, \"lower_is_better\": false, "
        "\"metric\": 0.952, \"eval_metric\": \"balanced_accuracy\"}')\n"
    )
    solution_path = source_artifact / "solution.py"
    solution_path.write_text(whole_solution)
    (source_artifact / "submission.csv").write_text("id,target\n1,0.8\n")
    source_record = {
        "kind": "profile_eval",
        "competition": "playground-series-s6e5",
        "run": "run-a",
        "step": None,
        "node_id": None,
        "timestamp": "20260504T100000",
        "artifact_dir": str(source_artifact),
        "solution_path": str(solution_path),
        "local_score": 0.95,
        "sha256": kaggle_submission_lab.sha256_file(source_artifact / "submission.csv"),
        "profile": "best_boost_gpu_1h",
    }

    class FakeResult:
        term_out = [
            'AIDE_RESULT_JSON: {"is_bug": false, "lower_is_better": false, '
            '"metric": 0.952, "eval_metric": "balanced_accuracy"}\n'
        ]
        exec_time = 42.0
        exc_type = None
        exc_info = None
        exc_stack = None

    def fake_execute(code, *, workspace_dir, artifact_dir, timeout, memory_limit_gb):
        assert code == whole_solution
        assert "build_autogluon_wrapper" not in code
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "submission.csv").write_text("id,target\n1,0.9\n")
        return FakeResult()

    monkeypatch.setattr(rerun_autogluon_profile, "execute_code", fake_execute)
    monkeypatch.setattr(
        rerun_autogluon_profile,
        "timestamp_now",
        lambda: "20260504T120000",
    )

    record = rerun_autogluon_profile.run_profile_eval(
        source_record,
        logs_dir=logs_dir,
        profile="full_boost",
        competition="playground-series-s6e5",
        timeout=1200,
        memory_limit_gb=80.0,
        whole_solution_path=solution_path,
    )

    eval_artifact = run_dir / "artifacts" / "20260504T120000"
    assert (eval_artifact / "solution.py").read_text() == whole_solution
    assert record["profile"] == "best_boost_gpu_1h"
    assert record["source_sha256"] == source_record["sha256"]
    assert record["source_solution_path"] == str(solution_path)

    eval_meta = json.loads((eval_artifact / "submission_eval.json").read_text())
    assert eval_meta["source_solution_path"] == str(solution_path)

    manifest = json.loads((eval_artifact / "aide_result.json").read_text())
    assert manifest["source_solution_path"] == str(solution_path)
    assert manifest["source"]["source_solution_path"] == str(solution_path)


def test_run_profile_eval_recovers_valid_submission_after_timeout(
    tmp_path, monkeypatch
):
    logs_dir = tmp_path / "logs"
    run_dir = logs_dir / "run-a"
    source_artifact = run_dir / "artifacts" / "20260504T100000"
    source_artifact.mkdir(parents=True)
    input_dir = tmp_path / "workspaces" / "run-a" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n")
    solution_path = source_artifact / "solution.py"
    solution_path.write_text(
        "AIDE_AG_CONFIG = {'time_limit': 7200, 'presets': 'best'}\n"
        "print('legacy solution')\n"
    )
    (source_artifact / "submission.csv").write_text("id,target\n1,0.8\n")
    source_record = {
        "kind": "profile_eval",
        "competition": "playground-series-s6e5",
        "run": "run-a",
        "timestamp": "20260504T100000",
        "artifact_dir": str(source_artifact),
        "solution_path": str(solution_path),
        "submission_path": str(source_artifact / "submission.csv"),
        "sha256": kaggle_submission_lab.sha256_file(source_artifact / "submission.csv"),
        "local_score": 0.95123,
        "eval_metric": "balanced_accuracy",
        "metric_maximize": True,
        "profile": "best_boost_2h",
    }

    class FakeTimeoutResult:
        term_out = ["partial output\n"]
        exec_time = 18000.0
        exc_type = "TimeoutError"
        exc_info = {}
        exc_stack = []

    def fake_execute(code, *, workspace_dir, artifact_dir, timeout, memory_limit_gb):
        del code, artifact_dir, timeout, memory_limit_gb
        working = workspace_dir / "working"
        working.mkdir(parents=True, exist_ok=True)
        (working / "submission.csv").write_text("id,target\n1,0.9\n")
        return FakeTimeoutResult()

    monkeypatch.setattr(rerun_autogluon_profile, "execute_code", fake_execute)
    monkeypatch.setattr(
        rerun_autogluon_profile,
        "timestamp_now",
        lambda: "20260504T120000",
    )

    record = rerun_autogluon_profile.run_profile_eval(
        source_record,
        logs_dir=logs_dir,
        profile="full_boost",
        competition="playground-series-s6e5",
        timeout=18000,
        memory_limit_gb=80.0,
        whole_solution_path=solution_path,
    )

    eval_artifact = run_dir / "artifacts" / "20260504T120000"
    assert record["status"] == "ok"
    assert record["is_buggy"] is False
    assert record["local_score"] == 0.95123
    assert record["eval_metric"] == "balanced_accuracy"
    assert record["sha256"] == kaggle_submission_lab.sha256_file(
        eval_artifact / "submission.csv"
    )
    assert not (eval_artifact / "error.txt").exists()

    eval_meta = json.loads((eval_artifact / "submission_eval.json").read_text())
    assert eval_meta["recovered_submission"] is True
    assert eval_meta["status"] == "ok"

    manifest = json.loads((eval_artifact / "aide_result.json").read_text())
    assert manifest["recovered_submission"] is True
    assert manifest["execution"]["exc_type"] == "TimeoutError"
    assert manifest["node"]["status"] == "ok"


def test_instrument_whole_solution_autogluon_logging_patches_legacy_wrapper():
    legacy_code = '''
from autogluon.tabular import TabularPredictor
import time
import shutil

def _save_prediction_artifact(frame, working_dir, filename):
    working_path = working_dir / filename
    working_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(working_path, index=False, compression="gzip")
    artifact_path = working_path
    if artifact_path.resolve() != working_path.resolve():
        shutil.copy2(working_path, artifact_path)
    return working_path

def main() -> None:
    predictor = TabularPredictor(label="target")
    print("AIDE AutoGluon: starting validation and prediction", flush=True)
    predictor.fit(**fit_kwargs)
    print("AIDE AutoGluon: finished fit", flush=True)
'''

    instrumented = rerun_autogluon_profile.instrument_whole_solution_autogluon_logging(
        legacy_code
    )

    assert "def _aide_wrap_predictor_progress_logging(predictor):" in instrumented
    assert "\n    _aide_wrap_predictor_progress_logging(predictor)\n" in instrumented
    assert "AIDE AutoGluon: {method_name} start" in instrumented
    assert '("predict", "predict_proba", "predict_proba_oof", "evaluate")' in instrumented
    assert "AIDE AutoGluon: writing prediction artifact" in instrumented
    assert (
        rerun_autogluon_profile.instrument_whole_solution_autogluon_logging(
            instrumented
        )
        == instrumented
    )


def test_s6e6_autogluon_defaults_are_competition_scoped():
    source_record = {"solution_path": "unused.py"}

    s6e6_cfg = rerun_autogluon_profile.build_profile_config(
        source_record=source_record,
        profile="best_boost_2h",
        competition="playground-series-s6e6",
        presets=None,
        time_limit=None,
        fit_args=None,
    )
    s6e6_settings = rerun_autogluon_profile.resolve_autogluon_settings(s6e6_cfg)

    assert s6e6_settings["eval_metric"] == "balanced_accuracy"
    assert s6e6_settings["class_balance"] == "balanced"

    other_cfg = rerun_autogluon_profile.build_profile_config(
        source_record=source_record,
        profile="best_boost_2h",
        competition="playground-series-s6e5",
        presets=None,
        time_limit=None,
        fit_args=None,
    )
    other_settings = rerun_autogluon_profile.resolve_autogluon_settings(other_cfg)

    assert "eval_metric" not in other_settings or other_settings["eval_metric"] == "auto"
    assert "class_balance" not in other_settings


def test_parse_fit_args_json_requires_object():
    assert rerun_autogluon_profile.parse_fit_args_json("{}") == {}
    try:
        rerun_autogluon_profile.parse_fit_args_json("[]")
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_find_existing_eval_ignores_failed_eval_without_submission(tmp_path):
    records = [
        {
            "kind": "profile_eval",
            "status": "error",
            "source_sha256": "source-sha",
            "profile": "full_best_30m_gpu",
            "autogluon_presets": "best_quality",
            "time_limit": 1800,
            "sha256": None,
            "submission_path": str(tmp_path / "missing.csv"),
        }
    ]

    existing = rerun_autogluon_profile._find_existing_eval(
        records,
        source_sha256="source-sha",
        profile="full_best_30m_gpu",
        presets=None,
        time_limit=None,
    )

    assert existing is None


def test_main_aborts_same_profile_eval_rerun_without_force_when_noninteractive(
    tmp_path, monkeypatch, capsys
):
    source_submission = tmp_path / "source.csv"
    source_submission.write_text("id,target\n1,0.8\n")
    source_sha = kaggle_submission_lab.sha256_file(source_submission)
    source_record = {
        "kind": "source_node",
        "status": "ok",
        "run": "run-a",
        "step": 1,
        "timestamp": "20260504T100000",
        "sha256": source_sha,
        "submission_path": str(source_submission),
    }
    existing_eval = {
        "kind": "profile_eval",
        "status": "ok",
        "profile": "full_best_30m_gpu",
        "source_sha256": source_sha,
        "sha256": "existing-eval-sha",
        "submission_path": str(source_submission),
    }
    calls = []

    index_path = tmp_path / "submission_index.json"
    index_path.write_text(json.dumps({"records": [source_record, existing_eval]}))

    def fail_refresh_index(**_kwargs):
        raise AssertionError("rerun should not refresh index unless requested")

    monkeypatch.setattr(rerun_autogluon_profile.lab, "refresh_index", fail_refresh_index)
    monkeypatch.setattr(
        rerun_autogluon_profile.lab,
        "filter_records_by_sha256",
        lambda _records, _filters: [source_record],
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    def fake_run_profile_eval(record, **kwargs):
        calls.append((record, kwargs))
        return {
            "kind": "profile_eval",
            "status": "ok",
            "local_score": 0.9,
            "profile": kwargs["profile"],
            "run": "run-a",
            "timestamp": "20260504T120000",
            "source_step": 1,
            "source_sha256": source_sha,
            "sha256": "new-eval-sha",
            "artifact_dir": str(tmp_path / "artifact"),
        }

    monkeypatch.setattr(
        rerun_autogluon_profile,
        "run_profile_eval",
        fake_run_profile_eval,
    )

    exit_code = rerun_autogluon_profile.main(
        [
            "--execute",
            "--profile",
            "full_best_30m_gpu",
            "--index",
            str(index_path),
            "--sha256",
            source_sha[:10],
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert calls == []
    assert "already has a successful profile evaluation" in output
    assert source_sha[:10] in output
    assert "existing-eval-sha"[:10] in output
    assert "--force" in output


def test_main_reruns_same_profile_eval_with_force(tmp_path, monkeypatch):
    source_submission = tmp_path / "source.csv"
    source_submission.write_text("id,target\n1,0.8\n")
    source_sha = kaggle_submission_lab.sha256_file(source_submission)
    source_record = {
        "kind": "source_node",
        "status": "ok",
        "run": "run-a",
        "step": 1,
        "timestamp": "20260504T100000",
        "sha256": source_sha,
        "submission_path": str(source_submission),
    }
    existing_eval = {
        "kind": "profile_eval",
        "status": "ok",
        "profile": "full_best_30m_gpu",
        "source_sha256": source_sha,
        "sha256": "existing-eval-sha",
        "submission_path": str(source_submission),
    }
    calls = []

    index_path = tmp_path / "submission_index.json"
    index_path.write_text(json.dumps({"records": [source_record, existing_eval]}))

    def fail_refresh_index(**_kwargs):
        raise AssertionError("rerun should not refresh index unless requested")

    monkeypatch.setattr(rerun_autogluon_profile.lab, "refresh_index", fail_refresh_index)
    monkeypatch.setattr(
        rerun_autogluon_profile.lab,
        "filter_records_by_sha256",
        lambda _records, _filters: [source_record],
    )

    def fake_run_profile_eval(record, **kwargs):
        calls.append((record, kwargs))
        return {
            "kind": "profile_eval",
            "status": "ok",
            "local_score": 0.9,
            "profile": kwargs["profile"],
            "run": "run-a",
            "timestamp": "20260504T120000",
            "source_step": 1,
            "source_sha256": source_sha,
            "sha256": "new-eval-sha",
            "artifact_dir": str(tmp_path / "artifact"),
        }

    monkeypatch.setattr(
        rerun_autogluon_profile,
        "run_profile_eval",
        fake_run_profile_eval,
    )

    exit_code = rerun_autogluon_profile.main(
        [
            "--execute",
            "--force",
            "--profile",
            "full_best_30m_gpu",
            "--index",
            str(index_path),
            "--sha256",
            source_sha[:10],
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1


def test_main_allows_same_source_with_different_profile_without_force(
    tmp_path, monkeypatch, capsys
):
    source_submission = tmp_path / "source.csv"
    source_submission.write_text("id,target\n1,0.8\n")
    source_sha = kaggle_submission_lab.sha256_file(source_submission)
    source_record = {
        "kind": "source_node",
        "status": "ok",
        "run": "run-a",
        "step": 1,
        "timestamp": "20260504T100000",
        "sha256": source_sha,
        "submission_path": str(source_submission),
    }
    existing_eval = {
        "kind": "profile_eval",
        "status": "ok",
        "profile": "full_best_30m_gpu",
        "source_sha256": source_sha,
        "sha256": "existing-eval-sha",
        "submission_path": str(source_submission),
    }
    calls = []

    index_path = tmp_path / "submission_index.json"
    index_path.write_text(json.dumps({"records": [source_record, existing_eval]}))

    monkeypatch.setattr(
        rerun_autogluon_profile.lab,
        "filter_records_by_sha256",
        lambda _records, _filters: [source_record],
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    def fake_run_profile_eval(record, **kwargs):
        calls.append((record, kwargs))
        return {
            "kind": "profile_eval",
            "status": "ok",
            "local_score": 0.9,
            "profile": kwargs["profile"],
            "run": "run-a",
            "timestamp": "20260504T120000",
            "source_step": 1,
            "source_sha256": source_sha,
            "sha256": "new-eval-sha",
            "artifact_dir": str(tmp_path / "artifact"),
        }

    monkeypatch.setattr(
        rerun_autogluon_profile,
        "run_profile_eval",
        fake_run_profile_eval,
    )

    exit_code = rerun_autogluon_profile.main(
        [
            "--execute",
            "--profile",
            "best_boost_gpu_1h",
            "--index",
            str(index_path),
            "--sha256",
            source_sha[:10],
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert len(calls) == 1
    assert "already has a successful profile evaluation" not in output


def test_parse_args_treats_sha_as_sha256_alias():
    args = rerun_autogluon_profile.parse_args(
        ["--sha", "abc123", "--sha256", "def456"]
    )

    assert args.sha256 == ["abc123", "def456"]


def test_parse_args_help_lists_sha_alias(capsys):
    with pytest.raises(SystemExit):
        rerun_autogluon_profile.parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert "--sha " in help_text


def test_main_reports_unknown_profile_without_traceback(tmp_path, capsys):
    index_path = tmp_path / "submission_index.json"
    index_path.write_text(json.dumps({"records": []}))

    exit_code = rerun_autogluon_profile.main(
        [
            "--execute",
            "--profile",
            "best_boost_gpu_30msss",
            "--index",
            str(index_path),
            "--sha256",
            "5469962e4f",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "Unknown AutoGluon profile: best_boost_gpu_30msss" in output
    assert "Did you mean: best_boost_gpu_30m" in output
    assert "Available profiles:" in output
    assert "Traceback" not in output


def test_resolve_process_timeout_defaults_to_profile_time_limit_plus_margin():
    assert rerun_autogluon_profile.resolve_process_timeout(None, 1800) == 12600
    assert rerun_autogluon_profile.resolve_process_timeout(None, 60) == 10860
    assert rerun_autogluon_profile.resolve_process_timeout(300, 1800) == 300


def test_execute_code_preserves_process_stdout_and_autogluon_log_file(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    code = (
        "from pathlib import Path\n"
        "import os\n"
        "artifact = Path(os.environ['AIDE_NODE_ARTIFACT_DIR'])\n"
        "artifact.mkdir(parents=True, exist_ok=True)\n"
        "(artifact / 'autogluon_stdout.log').write_text('training log\\n')\n"
        "print('v1')\n"
        "print('AIDE_RESULT_JSON: {\"is_bug\": false, \"metric\": 0.9, \"lower_is_better\": false}')\n"
    )

    result = rerun_autogluon_profile.execute_code(
        code,
        workspace_dir=tmp_path / "workspace",
        artifact_dir=artifact_dir,
        timeout=60,
        memory_limit_gb=None,
        console=rerun_autogluon_profile.Console(record=True),
        progress_time_limit=1,
    )

    assert result.exc_type is None
    assert "AIDE_RESULT_JSON:" in result.term_out[0]
    assert (artifact_dir / "autogluon_stdout.log").read_text() == "training log\n"
    process_stdout = (artifact_dir / "process_stdout.log").read_text()
    assert "v1" in process_stdout
    assert "AIDE_RESULT_JSON:" in process_stdout


def test_main_leaves_timeout_unset_for_profile_default(tmp_path, monkeypatch):
    source_submission = tmp_path / "source.csv"
    source_submission.write_text("id,target\n1,0.8\n")
    source_sha = kaggle_submission_lab.sha256_file(source_submission)
    source_record = {
        "kind": "source_node",
        "status": "ok",
        "run": "run-a",
        "step": 1,
        "timestamp": "20260504T100000",
        "sha256": source_sha,
        "submission_path": str(source_submission),
    }
    index_path = tmp_path / "submission_index.json"
    index_path.write_text(json.dumps({"records": [source_record]}))
    calls = []

    monkeypatch.setattr(
        rerun_autogluon_profile.lab,
        "filter_records_by_sha256",
        lambda _records, _filters: [source_record],
    )

    def fake_run_profile_eval(record, **kwargs):
        calls.append((record, kwargs))
        return {
            "kind": "profile_eval",
            "status": "ok",
            "local_score": 0.9,
            "profile": kwargs["profile"],
            "run": "run-a",
            "timestamp": "20260504T120000",
            "source_step": 1,
            "source_sha256": source_sha,
            "sha256": "new-eval-sha",
            "artifact_dir": str(tmp_path / "artifact"),
        }

    monkeypatch.setattr(
        rerun_autogluon_profile,
        "run_profile_eval",
        fake_run_profile_eval,
    )

    exit_code = rerun_autogluon_profile.main(
        [
            "--execute",
            "--profile",
            "full_best_30m",
            "--index",
            str(index_path),
            "--sha256",
            source_sha[:10],
        ]
    )

    assert exit_code == 0
    assert calls[0][1]["timeout"] is None


def test_main_accepts_solution_path_without_sha256(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    source_artifact = logs_dir / "run-a" / "artifacts" / "20260504T100000"
    source_artifact.mkdir(parents=True)
    solution_path = source_artifact / "solution.py"
    solution_path.write_text("print('custom whole solution')\n")
    submission_path = source_artifact / "submission.csv"
    submission_path.write_text("id,target\n1,0.8\n")
    (source_artifact / "submission_eval.json").write_text(
        json.dumps(
            {
                "local_score": 0.81234,
                "eval_metric": "balanced_accuracy",
                "metric_maximize": True,
            }
        )
    )
    calls = []

    def fake_run_profile_eval(record, **kwargs):
        calls.append((record, kwargs))
        return {
            "kind": "profile_eval",
            "status": "ok",
            "local_score": 0.9,
            "profile": "source_profile",
            "run": "run-a",
            "timestamp": "20260504T120000",
            "source_step": None,
            "source_sha256": record["sha256"],
            "sha256": "new-eval-sha",
            "artifact_dir": str(tmp_path / "artifact"),
        }

    monkeypatch.setattr(
        rerun_autogluon_profile,
        "run_profile_eval",
        fake_run_profile_eval,
    )

    exit_code = rerun_autogluon_profile.main(
        [
            "--execute",
            "--logs-dir",
            str(logs_dir),
            "--solution-path",
            str(solution_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    record, kwargs = calls[0]
    assert record["run"] == "run-a"
    assert record["kind"] == "profile_eval"
    assert record["solution_path"] == str(solution_path)
    assert record["sha256"] == kaggle_submission_lab.sha256_file(submission_path)
    assert record["local_score"] == 0.81234
    assert record["eval_metric"] == "balanced_accuracy"
    assert record["source_solution_sha256"] == kaggle_submission_lab.sha256_file(
        solution_path
    )
    assert kwargs["whole_solution_path"] == solution_path
