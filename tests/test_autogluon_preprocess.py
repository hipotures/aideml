import os
import time
from pathlib import Path

import pandas as pd
import pytest

from aide.interpreter import RedirectQueue
from aide.agent import Agent
from aide.autogluon_preprocess import (
    AGENT_MODE,
    BASELINE_PLAN_PREFIX,
    build_autogluon_wrapper,
    extract_preprocess_source,
    parse_result_marker,
    resolve_autogluon_settings,
    resolve_autogluon_included_model_types,
    sanitize_preprocess_prompt_text,
    validate_preprocess_source,
)
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.config import _load_cfg, prep_cfg
from aide.utils.metric import MetricValue


def _cfg(tmp_path: Path):
    os.environ.setdefault("AIDE_PROJECT_NAME", "test-project")
    os.environ.setdefault("AIDE_PROJECT_METRIC", "balanced_accuracy")
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "ag-preprocess-test"
    cfg.agent.mode = AGENT_MODE
    cfg.agent.search.num_drafts = 0
    cfg = prep_cfg(cfg)
    input_dir = Path(cfg.workspace_dir) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "sample_submission.csv").write_text("id,PitNextLap\n10,0.0\n")
    return cfg


def test_prep_cfg_reads_project_paths_from_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "project-data"
    data_dir.mkdir()
    desc_file = tmp_path / "project.md"
    desc_file.write_text("project goal\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "AIDE_PROJECT_DATA_DIR=project-data\n"
        "AIDE_PROJECT_DESC_FILE=project.md\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = _load_cfg(use_cli_args=False)
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "env-path-test"

    cfg = prep_cfg(cfg)

    assert Path(cfg.data_dir) == data_dir.resolve()
    assert Path(cfg.desc_file) == desc_file.resolve()


def _assert_cpu_boost_hyperparameters(settings):
    assert settings["use_gpu"] is False
    assert settings["included_model_types"][0] == "XGB"
    assert settings["hyperparameters"]["XGB"][0] == {
        "device": "cpu",
        "tree_method": "hist",
        "ag_args": {"priority": 999},
        "ag_args_fit": {"num_gpus": 0},
    }
    assert settings["hyperparameters"]["GBM"][0]["ag_args_fit"] == {"num_gpus": 0}
    if "CAT" in settings["included_model_types"]:
        assert settings["hyperparameters"]["CAT"][0]["ag_args_fit"] == {"num_gpus": 0}


def test_extract_preprocess_source_from_markdown_code_block():
    source = extract_preprocess_source(
        "plan\n```python\n"
        "def preprocess(df):\n"
        "    df = df.copy()\n"
        "    return df\n"
        "```"
    )

    assert source.startswith("def preprocess(df):")
    assert "return df" in source


def test_extract_preprocess_source_from_raw_code_does_not_format(monkeypatch):
    import aide.utils.response as response

    def fail_format(*args, **kwargs):
        raise AssertionError("raw preprocess extraction should not call Black")

    monkeypatch.setattr(response.black, "format_str", fail_format)

    source = extract_preprocess_source(
        "import pandas as pd\n\n"
        "def preprocess(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    out = df.copy()\n"
        "    return out\n"
    )

    assert source.startswith("def preprocess")
    assert "return out" in source


def test_validate_preprocess_source_rejects_target_reference():
    with pytest.raises(ValueError, match="forbidden column"):
        validate_preprocess_source(
            "def preprocess(df):\n"
            "    df['target_copy'] = df['PitNextLap']\n"
            "    return df\n",
            target_col="PitNextLap",
        )


def test_validate_preprocess_source_rejects_split_marker_reference():
    with pytest.raises(ValueError, match="__is_train__"):
        validate_preprocess_source(
            "def preprocess(df):\n"
            "    df['split_feature'] = df['__is_train__'].astype(int)\n"
            "    return df\n",
            target_col="PitNextLap",
        )


def test_validate_preprocess_source_rejects_row_id_reference():
    with pytest.raises(ValueError, match="__aide_row_id__"):
        validate_preprocess_source(
            "def preprocess(df):\n"
            "    df['row_feature'] = df['__aide_row_id__']\n"
            "    return df\n",
            target_col="PitNextLap",
        )


def test_validate_preprocess_source_rejects_parent_composition_signature():
    with pytest.raises(ValueError, match="second argument must be named `aux`"):
        validate_preprocess_source(
            "def preprocess(df, _base_preprocess=globals().get('preprocess')):\n"
            "    return _base_preprocess(df) if _base_preprocess else df\n",
            target_col="PitNextLap",
        )


def test_validate_preprocess_source_accepts_optional_aux_argument():
    validate_preprocess_source(
        "def preprocess(df, aux):\n"
        "    out = df.copy()\n"
        "    out['aux_rows'] = len(aux)\n"
        "    return out\n",
        target_col="PitNextLap",
    )


def test_validate_preprocess_source_rejects_misnamed_aux_argument():
    with pytest.raises(ValueError, match="second argument must be named `aux`"):
        validate_preprocess_source(
            "def preprocess(df, external):\n"
            "    return df\n",
            target_col="PitNextLap",
        )


def test_sanitize_preprocess_prompt_text_removes_unavailable_columns():
    text = (
        "Goal: predict `PitNextLap`.\n"
        "The identifier column is `id`.\n"
        "TyreLife (float64) has useful signal.\n"
        "__is_train__ is hidden.\n"
    )

    sanitized = sanitize_preprocess_prompt_text(
        text,
        unavailable_columns=["id", "PitNextLap", "__is_train__"],
    )

    assert "TyreLife" in sanitized
    assert "PitNextLap" not in sanitized
    assert "`id`" not in sanitized
    assert "__is_train__" not in sanitized


def test_sanitize_preprocess_prompt_text_removes_dangling_target_sentence_fragment():
    text = (
        "`PitNextLap` should contain probabilities for the positive class, not hard class\n"
        "labels.\n"
        "TyreLife (float64) has useful signal.\n"
    )

    sanitized = sanitize_preprocess_prompt_text(
        text,
        unavailable_columns=["id", "PitNextLap", "__is_train__"],
    )

    assert "TyreLife" in sanitized
    assert "labels." not in sanitized


def test_build_autogluon_wrapper_compiles_and_preserves_preprocess(tmp_path):
    cfg = _cfg(tmp_path)

    code = build_autogluon_wrapper(
        "def preprocess(df):\n"
        "    df = df.copy()\n"
        "    return df\n",
        cfg,
    )

    compile(code, "<generated_autogluon_wrapper>", "exec")
    assert "TabularPredictor" in code
    assert "def preprocess(df):" in code
    assert "AIDE_RESULT_JSON:" in code
    assert "'time_limit': 600" in code
    assert "'preprocess_timeout': 180" in code
    assert "train_features = train_df.drop(columns=[target_col, id_col]" in code
    assert "_make_combined_frame(train_features, test_features)" in code
    assert "df[HELPER_ROW_ID]" not in code
    assert "FORBIDDEN_ROW_ID in after.columns" in code
    assert '"verbosity": 2' in code
    assert "import sys" in code
    assert "import signal" in code
    assert "import inspect" in code
    assert 'os.environ.get("AIDE_NODE_ARTIFACT_DIR"' in code
    assert "def _preprocess_timeout" in code
    assert "AIDE AutoGluon preprocess exceeded the dedicated timeout" in code
    assert "with _preprocess_timeout" in code
    assert "def _save_submission" in code
    assert 'artifact_submission_path = artifact_dir / "submission.csv"' in code
    assert "shutil.copy2(submission_path, artifact_submission_path)" in code
    assert "class _TeeWriter" in code
    assert "def fileno(self):" in code
    assert "logging.StreamHandler(stderr_writer)" in code
    assert "logger.handlers = [log_handler]" in code
    assert "logger.propagate = False" in code
    assert "def _force_autogluon_cpu_resources" in code
    assert "ResourceManager.get_gpu_count = staticmethod(lambda: 0)" in code
    assert "ResourceManager.get_gpu_count_torch = staticmethod(lambda cuda_only=False: 0)" in code
    assert "_force_autogluon_cpu_resources()" in code
    assert '"autogluon"' in code
    assert 'print("AIDE AutoGluon: starting preprocess", flush=True)' in code
    assert "AIDE AutoGluon: finished preprocess rows=" in code
    assert 'print("AIDE AutoGluon: starting fit", flush=True)' in code
    assert 'if __name__ == "__main__"' not in code
    assert code.rstrip().endswith("main()")


def test_autogluon_wrapper_passes_aux_when_preprocess_accepts_it(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.aux = "external.csv"

    code = build_autogluon_wrapper(
        "def preprocess(df, aux):\n"
        "    out = df.copy()\n"
        "    out['aux_rows'] = len(aux)\n"
        "    return out\n",
        cfg,
    )
    namespace: dict[str, object] = {}
    exec(code.rsplit("\nmain()", 1)[0], namespace)

    result = namespace["_run_preprocess"](
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [10, 11, 12]}),
    )

    assert "'aux_file': 'external.csv'" in code
    assert "def _read_aux_csv" in code
    assert "AIDE AutoGluon: loaded aux file " in code
    assert "passed_to_preprocess=" in code
    assert result["aux_rows"].to_list() == [3, 3]


def test_autogluon_wrapper_keeps_single_argument_preprocess_with_aux_file(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.aux = "external.csv"

    code = build_autogluon_wrapper(
        "def preprocess(df):\n"
        "    out = df.copy()\n"
        "    out['used_aux'] = 0\n"
        "    return out\n",
        cfg,
    )
    namespace: dict[str, object] = {}
    exec(code.rsplit("\nmain()", 1)[0], namespace)

    result = namespace["_run_preprocess"](
        pd.DataFrame({"x": [1]}),
        pd.DataFrame({"x": [10, 11, 12]}),
    )

    assert result["used_aux"].to_list() == [0]


def test_build_autogluon_wrapper_emits_run_stats_collection(tmp_path):
    cfg = _cfg(tmp_path)
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)

    assert "import time" in code
    assert "preprocess_started_at = time.time()" in code
    assert "preprocess_time = time.time() - preprocess_started_at" in code
    assert "feature_count = int(len(preprocessed.columns))" in code
    assert "training_started_at = time.time()" in code
    assert "training_time = time.time() - training_started_at" in code
    assert "leaderboard = predictor.leaderboard(silent=True)" in code
    assert '"run_stats": run_stats' in code


def test_autogluon_wrapper_reads_metric_from_project_env(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AIDE_PROJECT_NAME=playground-series-s6e6\n"
        "AIDE_PROJECT_METRIC=balanced_accuracy\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)

    compile(code, "<generated_autogluon_wrapper>", "exec")
    assert "'project_name': 'playground-series-s6e6'" in code
    assert "'eval_metric': 'balanced_accuracy'" in code
    assert "def _predict_values" in code
    assert "pred = predictor.predict(data, model=model)" in code
    assert "test_pred = _predict_values(" in code
    assert "eval_metric=eval_metric" in code


def test_autogluon_wrapper_requires_project_env_metric(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIDE_PROJECT_NAME", "test-project")
    monkeypatch.delenv("AIDE_PROJECT_METRIC", raising=False)

    with pytest.raises(ValueError, match="AIDE_PROJECT_METRIC"):
        build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)


def test_autogluon_wrapper_can_enable_balanced_sample_weights(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profiles["balanced_test"] = {
        "included_model_types": ["XGB", "GBM", "CAT"],
        "class_balance": "balanced",
    }
    cfg.agent.autogluon.profile = "balanced_test"

    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)

    compile(code, "<generated_autogluon_wrapper>", "exec")
    assert "'class_balance': 'balanced'" in code
    assert "CLASS_WEIGHT_COL = \"__aide_class_weight__\"" in code
    assert "def _balanced_sample_weight" in code
    assert "train_model[CLASS_WEIGHT_COL] = _balanced_sample_weight(y_train)" in code
    assert 'predictor_kwargs["sample_weight"] = CLASS_WEIGHT_COL' in code
    assert 'predictor_kwargs["weight_evaluation"] = False' in code
    assert 'valid_data.drop(columns=[target_col, CLASS_WEIGHT_COL], errors="ignore")' in code


def test_autogluon_wrapper_row_count_error_explains_row_preserving_fix(tmp_path):
    cfg = _cfg(tmp_path)

    code = build_autogluon_wrapper("def preprocess(df):\n    return df.iloc[:-1]\n", cfg)

    assert "preprocess changed row count" in code
    assert "outlier flag" in code
    assert "clipped/winsorized value" in code


def test_autogluon_preprocess_prompt_mentions_row_preserving_outlier_fix(tmp_path):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    contract = agent._prompt_autogluon_preprocess_guideline[
        "AutoGluon preprocess mode contract"
    ]

    assert any("outlier filtering" in item for item in contract)


def test_autogluon_preprocess_prompt_describes_aux_argument_and_file(tmp_path):
    cfg = _cfg(tmp_path)
    Path(cfg.data_dir, "external.csv").write_text("x,target\n5,1\n", encoding="utf-8")
    Path(cfg.data_dir, "external.txt").write_text("external description\n", encoding="utf-8")
    cfg.agent.aux = "external.csv"
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    contract = agent._prompt_autogluon_preprocess_guideline[
        "AutoGluon preprocess mode contract"
    ]
    prompt: dict[str, object] = {}
    agent._add_autogluon_context(prompt)

    contract_text = "\n".join(contract)
    assert "def preprocess(df: pd.DataFrame, aux: pd.DataFrame)" in contract_text
    assert "./input/external.csv" in contract_text
    assert "Do not read files" in contract_text
    assert "ignore `aux`" in contract_text
    assert "Auxiliary data description for external.csv" in prompt
    assert prompt["Auxiliary data description for external.csv"] == "external description"
    assert any("clipped/winsorized value" in item for item in contract)


def test_generated_preprocess_timeout_raises_clear_error(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.preprocess_timeout = 1
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    namespace = {}
    exec(code.replace("\nmain()\n", "\n"), namespace)

    with pytest.raises(TimeoutError, match="preprocess exceeded the dedicated timeout"):
        with namespace["_preprocess_timeout"](1):
            time.sleep(2)


def test_build_autogluon_wrapper_does_not_emit_hypothesis_claim(tmp_path):
    cfg = _cfg(tmp_path)

    code = build_autogluon_wrapper(
        "def preprocess(df):\n    return df\n",
        cfg,
        research_hypothesis_id="000123",
    )

    assert "research_hypotheses_llm_claimed_used" not in code
    assert "research_usage_note" not in code


def test_generated_quiet_model_output_supports_redirect_queue_streams(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    namespace = {}
    exec(code.replace("\nmain()\n", "\n"), namespace)

    class DummyQueue:
        def put(self, _msg, timeout=None):
            return None

    redirect = RedirectQueue(DummyQueue())
    monkeypatch.setattr(namespace["sys"], "stdout", redirect)
    monkeypatch.setattr(namespace["sys"], "stderr", redirect)

    with namespace["_quiet_model_output"](tmp_path):
        print("runtime log", file=namespace["sys"].stdout)

    assert "runtime log" in (tmp_path / "autogluon_stdout.log").read_text()


def test_generated_save_submission_copies_to_artifact_dir(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    namespace = {}
    exec(code.replace("\nmain()\n", "\n"), namespace)

    working_dir = tmp_path / "workspace" / "working"
    artifact_dir = tmp_path / "logs" / "run" / "artifacts" / "20260504T171209"
    working_dir.mkdir(parents=True)
    submission = namespace["pd"].DataFrame({"id": [1, 2], "PitNextLap": [0.1, 0.9]})
    monkeypatch.setenv("AIDE_NODE_ARTIFACT_DIR", str(artifact_dir))

    namespace["_save_submission"](submission, working_dir)

    assert (working_dir / "submission.csv").read_text() == (
        artifact_dir / "submission.csv"
    ).read_text()


def test_generated_make_submission_maps_predictions_by_id_and_sorts(tmp_path):
    cfg = _cfg(tmp_path)
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    namespace = {}
    exec(code.replace("\nmain()\n", "\n"), namespace)

    sample_submission = namespace["pd"].DataFrame(
        {"id": [3, 1, 2], "PitNextLap": [0.0, 0.0, 0.0]}
    )
    test_ids = namespace["pd"].Series([1, 2, 3])
    test_pred = namespace["pd"].Series([0.1, 0.2, 0.3])

    submission = namespace["_make_submission"](
        sample_submission,
        id_col="id",
        target_col="PitNextLap",
        test_ids=test_ids,
        test_pred=test_pred,
    )

    assert submission["id"].to_list() == [1, 2, 3]
    assert submission["PitNextLap"].to_list() == [0.1, 0.2, 0.3]


def test_autogluon_fast_boost_profile_excludes_catboost(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "fast_boost"
    cfg.agent.autogluon.included_model_types = None

    assert resolve_autogluon_included_model_types(cfg) == ["XGB", "GBM"]
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)

    assert "'included_model_types': ['XGB', 'GBM']" in code


def test_autogluon_full_boost_profile_includes_catboost(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "full_boost"
    cfg.agent.autogluon.included_model_types = None

    assert resolve_autogluon_included_model_types(cfg) == ["XGB", "GBM", "CAT"]
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)

    assert "'included_model_types': ['XGB', 'GBM', 'CAT']" in code


def test_autogluon_included_model_types_overrides_profile(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "fast_boost"
    cfg.agent.autogluon.included_model_types = ["CAT"]

    assert resolve_autogluon_included_model_types(cfg) == ["CAT"]


def test_autogluon_default_full_boost_keeps_legacy_fit_settings(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "full_boost"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_strategy"] == "holdout"
    _assert_cpu_boost_hyperparameters(settings)
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    assert "if AIDE_AG_CONFIG.get(\"use_gpu\") is not None:" in code
    assert "fit_kwargs[\"num_gpus\"] = 1 if AIDE_AG_CONFIG[\"use_gpu\"] else 0" in code
    assert "fit_args = dict(AIDE_AG_CONFIG.get(\"fit_args\", {}) or {})" in code
    assert "fit_kwargs.update(fit_args)" in code
    assert "'device': 'cpu'" in code
    assert "'ag_args': {'priority': 999}" in code


def test_autogluon_best_profile_forces_cpu_xgb_settings(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "full_best_30m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "best_quality"
    assert settings["time_limit"] == 1800
    _assert_cpu_boost_hyperparameters(settings)
    assert "validation_strategy" not in settings
    assert "fit_args" not in settings
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    assert "'presets': 'best_quality'" in code
    assert "'time_limit': 1800" in code
    assert "'use_gpu': False" in code
    assert "'device': 'cpu'" in code
    assert "'validation_strategy'" not in code.split("RESULT_MARKER", 1)[0]
    assert "'validation_fraction'" not in code.split("RESULT_MARKER", 1)[0]
    assert "'fit_args'" not in code.split("RESULT_MARKER", 1)[0]
    assert 'if valid_data is not None:' in code


def test_autogluon_best_boost_cpu_profiles_only_add_save_space(tmp_path):
    expected_limits = {
        "best_boost_1h": 3600,
        "best_boost_2h": 7200,
    }
    for profile, time_limit in expected_limits.items():
        cfg = _cfg(tmp_path)
        cfg.agent.autogluon.profile = profile
        cfg.agent.autogluon.included_model_types = None

        settings = resolve_autogluon_settings(cfg)

        assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
        assert settings["presets"] == "best"
        assert settings["time_limit"] == time_limit
        _assert_cpu_boost_hyperparameters(settings)
        if "fit_args" in cfg.agent.autogluon.profiles[profile]:
            assert settings["fit_args"] == {"save_space": True}
        else:
            assert "fit_args" not in settings
        assert "eval_metric" not in settings or settings["eval_metric"] == "auto"
        assert "class_balance" not in settings
        assert "validation_strategy" not in settings


def test_autogluon_best_boost_gpu_1h_matches_gpu_30m_with_longer_limit(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "best_boost_gpu_1h"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "best"
    assert settings["time_limit"] == 3600
    assert settings["use_gpu"] is True
    assert settings["fit_args"] == {}
    assert settings["hyperparameters"]["CAT"][0]["gpu_ram_part"] == 0.8
    assert settings["hyperparameters"]["XGB"][0]["device"] == "cuda"
    assert settings["hyperparameters"]["XGB"][0]["ag_args"] == {"priority": 999}
    assert settings["hyperparameters"]["GBM"][0]["device"] == "cuda"
    assert settings["hyperparameters"]["GBM"][0]["ag_args_fit"] == {"num_gpus": 1}
    assert settings["validation_strategy"] == "autogluon"


def test_autogluon_best_xgb_1h_profiles_are_xgb_only(tmp_path):
    expected_devices = {
        "best_xgb_gpu_1h": ("cuda", True, 1),
        "best_xgb_cpu_1h": ("cpu", False, 0),
    }
    for profile, (device, use_gpu, num_gpus) in expected_devices.items():
        cfg = _cfg(tmp_path)
        cfg.agent.autogluon.profile = profile
        cfg.agent.autogluon.included_model_types = None

        settings = resolve_autogluon_settings(cfg)

        assert settings["included_model_types"] == ["XGB"]
        assert settings["presets"] == "best"
        assert settings["time_limit"] == 3600
        assert settings["use_gpu"] is use_gpu
        assert settings["class_balance"] == "balanced"
        assert settings["validation_strategy"] == "autogluon"
        assert settings["fit_args"] == {}
        assert set(settings["hyperparameters"]) == {"XGB"}
        assert settings["hyperparameters"]["XGB"][0]["device"] == device
        assert settings["hyperparameters"]["XGB"][0]["tree_method"] == "hist"
        assert settings["hyperparameters"]["XGB"][0]["ag_args"] == {"priority": 999}
        assert settings["hyperparameters"]["XGB"][0]["ag_args_fit"] == {"num_gpus": num_gpus}


def test_autogluon_xgb_medium_gpu_balanced_10m_profile(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "xgb_medium_gpu_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["preprocess_timeout"] == 180
    assert settings["validation_strategy"] == "holdout"
    assert settings["class_balance"] == "balanced"
    assert settings["use_gpu"] is True
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }
    assert settings["hyperparameters"] == {
        "XGB": [
            {
                "device": "cuda",
                "tree_method": "hist",
                "ag_args": {"priority": 999},
                "ag_args_fit": {"num_gpus": 1},
            }
        ]
    }


def test_autogluon_gpu_profile_backfills_xgb_priority_for_saved_configs(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "stale_gpu"
    cfg.agent.autogluon.included_model_types = None
    cfg.agent.autogluon.profiles["stale_gpu"] = {
        "included_model_types": ["XGB", "GBM", "CAT"],
        "presets": "medium_quality",
        "time_limit": 600,
        "use_gpu": True,
        "hyperparameters": {
            "GBM": [{"ag_args_fit": {"num_gpus": 0}}],
            "CAT": [{"task_type": "GPU", "devices": "0", "ag_args_fit": {"num_gpus": 1}}],
            "XGB": [{"device": "cuda", "tree_method": "hist", "ag_args_fit": {"num_gpus": 1}}],
        },
    }

    settings = resolve_autogluon_settings(cfg)

    assert settings["hyperparameters"]["XGB"][0]["ag_args"] == {"priority": 999}
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    assert "'ag_args': {'priority': 999}" in code


def test_autogluon_profiles_are_not_restored_from_python_schema(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "ag-preprocess-test"
    cfg.agent.mode = AGENT_MODE
    cfg.agent.autogluon.profile = "full_best_30m"
    cfg.agent.autogluon.profiles = {}
    cfg = prep_cfg(cfg)

    with pytest.raises(ValueError, match="Unknown AutoGluon profile 'full_best_30m'"):
        resolve_autogluon_settings(cfg)


def test_autogluon_gpu_named_best_profile_uses_per_model_gpu_settings(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "full_best_30m_gpu"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "best_quality"
    assert settings["time_limit"] == 1800
    assert settings["use_gpu"] is True
    assert "validation_strategy" not in settings
    assert "fit_args" not in settings
    assert settings["hyperparameters"] == {
        "GBM": [{"device": "cuda", "ag_args_fit": {"num_gpus": 1}}],
        "CAT": [
            {
                "task_type": "GPU",
                "devices": "0",
                "gpu_ram_part": 0.8,
                "ag_args_fit": {"num_gpus": 1},
            }
        ],
        "XGB": [
            {
                "device": "cuda",
                "tree_method": "hist",
                "ag_args": {"priority": 999},
                "ag_args_fit": {"num_gpus": 1},
            }
        ],
    }
    assert "n_jobs" not in settings["hyperparameters"]["XGB"][0]
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    assert "'GBM': [{'ag_args_fit': {'num_gpus': 1}, 'device': 'cuda'}]" in code
    assert "'use_gpu': True" in code
    assert "fit_kwargs[\"hyperparameters\"] = AIDE_AG_CONFIG[\"hyperparameters\"]" in code


def test_autogluon_gpu_profiles_use_cuda_gbm(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.included_model_types = None

    for profile, profile_settings in cfg.agent.autogluon.profiles.items():
        if not getattr(profile_settings, "use_gpu", False):
            continue
        if "GBM" not in list(getattr(profile_settings, "included_model_types", []) or []):
            continue

        cfg.agent.autogluon.profile = profile
        settings = resolve_autogluon_settings(cfg)

        assert settings["hyperparameters"]["GBM"][0]["device"] == "cuda"
        assert settings["hyperparameters"]["GBM"][0]["ag_args_fit"] == {"num_gpus": 1}


def test_autogluon_unknown_profile_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "slow_magic"
    cfg.agent.autogluon.included_model_types = None

    with pytest.raises(ValueError, match="Unknown AutoGluon profile"):
        resolve_autogluon_included_model_types(cfg)


def test_agent_autogluon_draft_wraps_preprocess_response(tmp_path):
    cfg = _cfg(tmp_path)
    agent = Agent(
        task_desc="Predict `PitNextLap`. The identifier column is `id`.",
        cfg=cfg,
        journal=Journal(),
    )
    agent.data_preview = (
        "PitNextLap (float64) has 2 unique values.\n"
        "id (int64) has range 1 - 2.\n"
        "TyreLife (float64) has useful signal.\n"
    )
    captured = {}

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return (
            "add a simple numeric ratio",
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['TyreLife_x2'] = df.get('TyreLife', 0) * 2\n"
            "    return df\n",
        )

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent._draft()

    assert "AutoGluon preprocess mode contract" in captured["prompt"]["Instructions"]
    prompt_text = str(captured["prompt"])
    assert "TyreLife" in prompt_text
    assert "PitNextLap" not in prompt_text
    assert "`id`" not in prompt_text
    assert "__is_train__" not in prompt_text
    assert "__aide_row_id__" not in prompt_text
    assert "must replace the previous preprocess function" in prompt_text
    assert "Do not call `globals().get(\"preprocess\")`" in prompt_text
    assert "Mechanical simplifications are allowed only" in prompt_text
    assert "Do not optimize by changing algorithms" in prompt_text
    assert "dedicated timeout of 180 seconds" in prompt_text
    assert "Avoid expensive Python callbacks" in prompt_text
    assert "rolling.apply" in prompt_text
    assert "TabularPredictor" in node.code
    assert "TyreLife_x2" in node.code


def test_agent_autogluon_first_node_is_raw_baseline_without_llm(tmp_path):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fail_plan_and_code(_prompt):
        raise AssertionError("baseline should not call the code LLM")

    agent.plan_and_code_query = fail_plan_and_code  # type: ignore[method-assign]

    node = agent.generate_node(None)

    assert node.parent is None
    assert node.plan.startswith(BASELINE_PLAN_PREFIX)
    assert "def preprocess(df: pd.DataFrame) -> pd.DataFrame:" in node.code
    assert "return df.copy()" in node.code
    assert "TabularPredictor" in node.code


def test_agent_autogluon_baseline_is_selected_for_expansion(tmp_path):
    cfg = _cfg(tmp_path)
    journal = Journal()
    baseline = Node(
        code=build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg),
        plan=f"{BASELINE_PLAN_PREFIX}: raw features",
    )
    baseline.metric = MetricValue(0.95, maximize=True)
    baseline.is_buggy = False
    journal.append(baseline)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    assert agent.search_policy() is baseline


def test_agent_generation_logs_code_response_to_preallocated_artifact(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    journal = Journal()
    parent = Node(
        code=build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg),
        plan=f"{BASELINE_PLAN_PREFIX}: raw features",
    )
    parent.metric = MetricValue(0.95, maximize=True)
    parent.is_buggy = False
    journal.append(parent)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    query_func = agent.plan_and_code_query.__globals__["query"]

    def fake_query_func(**kwargs):
        return (
            "Add a simple feature.\n"
            "```python\n"
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['TyreLife_x2'] = df.get('TyreLife', 0) * 2\n"
            "    return df\n"
            "```",
            0.1,
            1,
            1,
            {"model": kwargs["model"]},
        )

    monkeypatch.setitem(
        query_func.__globals__,
        "determine_provider",
        lambda _model: "openai",
    )
    monkeypatch.setitem(
        query_func.__globals__["provider_to_query_func"],
        "openai",
        fake_query_func,
    )

    artifact_dir = tmp_path / "artifact"
    node_ctime = 1778000000.0
    node = agent.generate_node(
        parent,
        node_ctime=node_ctime,
        llm_log_dir=artifact_dir,
    )

    assert node.ctime == node_ctime
    assert "TyreLife_x2" in node.code
    assert (artifact_dir / "request.md").exists()
    assert "TyreLife_x2" in (artifact_dir / "response.py").read_text()
    assert not (artifact_dir / "llm_communication.md").exists()


def test_agent_autogluon_improve_prompt_uses_previous_preprocess(tmp_path):
    cfg = _cfg(tmp_path)
    parent = Node(
        plan="base",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['base_feature'] = 1\n"
            "    return df\n",
            cfg,
        ),
    )
    parent.metric = MetricValue(0.9, maximize=True)
    parent.is_buggy = False
    journal = Journal()
    journal.append(parent)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    agent.data_preview = "TyreLife (float64) has range 1.00 - 77.00\n"
    captured = {}

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return (
            "improve feature",
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['base_feature'] = 2\n"
            "    return df\n",
        )

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent._improve(parent)

    assert "base_feature" in captured["prompt"]["Previous preprocess function"]
    assert "Data Overview" in captured["prompt"]
    assert "TyreLife" in captured["prompt"]["Data Overview"]
    assert node.parent is parent
    assert "base_feature" in node.code


def test_parse_result_marker_short_circuits_feedback_review(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    exec_result = ExecutionResult(
        term_out=[
            'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ag ok", '
            '"metric": 0.91, "lower_is_better": false}\n'
        ],
        exec_time=1.0,
        exc_type=None,
    )

    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: pytest.fail("feedback LLM should not be called"),
    )

    agent.parse_exec_result(node, exec_result)

    assert node.is_buggy is False
    assert node.metric.value == 0.91
    assert node.analysis == "ag ok"


def test_parse_result_marker_preserves_run_stats(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    exec_result = ExecutionResult(
        term_out=[
            'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ag ok", '
            '"metric": 0.91, "lower_is_better": false, '
            '"run_stats": {"feature_count": 42, "preprocess_time": 1.2, '
            '"training_time": 3.4, "models": [{"model": "XGBoost", '
            '"score_val": 0.91}]}}\n'
        ],
        exec_time=5.0,
        exc_type=None,
    )

    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: pytest.fail("feedback LLM should not be called"),
    )

    agent.parse_exec_result(node, exec_result)

    assert node.run_stats == {
        "feature_count": 42,
        "preprocess_time": 1.2,
        "training_time": 3.4,
        "models": [{"model": "XGBoost", "score_val": 0.91}],
    }


def test_parse_result_marker_uses_latest_valid_marker():
    parsed = parse_result_marker(
        'AIDE_RESULT_JSON: {"metric": 0.1}\n'
        'AIDE_RESULT_JSON: {"metric": 0.2, "summary": "latest"}\n'
    )

    assert parsed == {"metric": 0.2, "summary": "latest"}
