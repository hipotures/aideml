import os
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

import aide.autogluon_preprocess as autogluon_preprocess
from aide.interpreter import RedirectQueue
from aide.agent import Agent
from aide.autogluon_preprocess import (
    AGENT_MODE,
    BASELINE_PLAN_PREFIX,
    build_autogluon_wrapper,
    extract_preprocess_source,
    parse_result_marker,
    preprocess_task_prompt_text,
    resolve_autogluon_settings,
    resolve_autogluon_included_model_types,
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
    (input_dir / "sample_submission.csv").write_text("id,class\n10,GALAXY\n")
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

    cfg = _load_cfg(use_cli_args=False, load_env=True)
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "env-path-test"

    cfg = prep_cfg(cfg, load_env=True)

    assert Path(cfg.data_dir) == data_dir.resolve()
    assert Path(cfg.desc_file) == desc_file.resolve()


def test_prep_cfg_reads_refactor_settings_from_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "project-data"
    data_dir.mkdir()
    desc_file = tmp_path / "project.md"
    desc_file.write_text("project goal\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "AIDE_PROJECT_DATA_DIR=project-data\n"
        "AIDE_PROJECT_DESC_FILE=project.md\n"
        "AIDE_REFACTOR_ENABLED=true\n"
        "AIDE_REFACTOR_MODEL=gpt-refactor\n"
        "AIDE_REFACTOR_TIMEOUT_S=301\n"
        "AIDE_REFACTOR_MAX_INPUT_CHARS=12345\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = _load_cfg(use_cli_args=False, load_env=True)
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "env-refactor-test"

    cfg = prep_cfg(cfg, load_env=True)

    assert cfg.refactor.enabled is True
    assert cfg.refactor.model == "gpt-refactor"
    assert cfg.refactor.timeout == 301
    assert cfg.refactor.max_input_chars == 12345


def test_prep_cfg_reads_agent_and_research_settings_from_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "project-data"
    data_dir.mkdir()
    desc_file = tmp_path / "project.md"
    desc_file.write_text("project goal\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "AIDE_PROJECT_DATA_DIR=project-data\n"
        "AIDE_PROJECT_DESC_FILE=project.md\n"
        "AIDE_AGENT_GPU=true\n"
        "AIDE_AGENT_STEPS=7\n"
        "AIDE_AGENT_HYPOTHESES=3\n"
        "AIDE_AGENT_CODE_TIMEOUT=901\n"
        "AIDE_RESEARCH_MATERIALIZE=false\n"
        "AIDE_RESEARCH_EXECUTE=false\n"
        "AIDE_GENERATE_REPORT=false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = _load_cfg(use_cli_args=False, load_env=True)
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "env-agent-test"

    cfg = prep_cfg(cfg, load_env=True)

    assert cfg.agent.gpu is True
    assert cfg.agent.steps == 7
    assert cfg.agent.hypotheses == 3
    assert cfg.agent.code.timeout == 901
    assert cfg.research.enabled is True
    assert cfg.research.mode == "hypothesis"
    assert cfg.research.materialize is False
    assert cfg.research.execute is False
    assert cfg.generate_report is False


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


def test_validate_preprocess_source_rejects_parent_composition_signature():
    with pytest.raises(ValueError, match="second argument must be named `aux`"):
        validate_preprocess_source(
            "def preprocess(df, _base_preprocess=globals().get('preprocess')):\n"
            "    return _base_preprocess(df) if _base_preprocess else df\n",
        )


def test_validate_preprocess_source_accepts_optional_aux_argument():
    validate_preprocess_source(
        "def preprocess(df, aux):\n"
        "    out = df.copy()\n"
        "    out['aux_rows'] = len(aux)\n"
        "    return out\n",
    )


def test_validate_preprocess_source_rejects_misnamed_aux_argument():
    with pytest.raises(ValueError, match="second argument must be named `aux`"):
        validate_preprocess_source(
            "def preprocess(df, external):\n"
            "    return df\n",
        )


def test_preprocess_task_prompt_text_removes_full_solution_only_instructions():
    text = (
        "## Goal\n"
        "Predict the target for each object in the test set.\n\n"
        "For each row in `test.csv`, predict the `target` label. The target column in\n"
        "`train.csv` is `target`; the identifier column is `row_id`.\n\n"
        "## Evaluation\n"
        "Submissions are evaluated using balanced accuracy. Higher is better.\n\n"
        "Competition-specific modeling hint: if using CatBoost for this multiclass task,\n"
        "include `auto_class_weights=\"Balanced\"` unless explicitly testing a different\n"
        "class-weighting strategy; this has empirically improved local CV and public\n"
        "leaderboard score for this competition.\n"
        "Analogous balanced-class settings should be used for other multiclass tree\n"
        "models unless explicitly testing a different class-weighting strategy: for\n"
        "LightGBM use `class_weight=\"balanced\"`, and for XGBoost pass fold-specific\n"
        "`sample_weight=compute_sample_weight(class_weight=\"balanced\", y=y_train)` to\n"
        "`.fit()`.\n\n"
        "The submission file must contain a header and exactly these columns:\n\n"
        "```csv\n"
        "row_id,target\n"
        "123,A\n"
        "```\n\n"
        "`target` must contain one of `A`, `B`, or `C`.\n"
        "\n"
        "## Data description\n"
        "- **train.csv** - training data with the multiclass target column `target`\n"
        "- **test.csv** - test data without the target column\n"
        "\n"
        "Additional auxiliary data description for `external.csv`:\n\n"
        "External reference data.\n\n"
        "Common columns with the competition data:\n"
        "feature_a, feature_b, target.\n\n"
        "Columns present in this original dataset but not in the competition files:\n"
        "extra_a.\n\n"
        "Competition columns not present in this original dataset:\n"
        "row_id, metadata_a.\n\n"
        "Generated code should decide whether and how to use this file. Any merge,\n"
        "filtering, cleaning of sentinel magnitudes, or column mapping must be done\n"
        "explicitly by the generated solution code.\n"
    )

    rendered = preprocess_task_prompt_text(text)

    assert "Predict the target" in rendered
    assert "`target` label" in rendered
    assert "identifier column is `row_id`" in rendered
    assert "fixed wrapper evaluates feature changes using balanced accuracy" in rendered
    assert "`target` must contain one of" in rendered
    assert "auto_class_weights" not in rendered
    assert "class_weight" not in rendered
    assert "sample_weight" not in rendered
    assert "The submission file must contain" not in rendered
    assert "row_id,target" not in rendered
    assert "Data description" in rendered
    assert "Additional auxiliary data description" not in rendered
    assert "External reference data" not in rendered
    assert "Competition columns not present" not in rendered


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
    assert "'preprocess_timeout': 600" in code
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
    assert "def _selected_model_metadata" in code
    assert '"selected_model": _json_safe_scalar(selected_model)' in code
    assert '"run_stats": run_stats' in code


def test_fair_cpu_scheduling_preserves_required_model_order_and_priority():
    settings = {
        "included_model_types": ["XGB", "GBM", "CAT"],
        "use_gpu": False,
        "fair_model_scheduling": True,
    }

    autogluon_preprocess._force_cpu_boost_hyperparameters(settings)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["hyperparameters"]["XGB"][0]["device"] == "cpu"
    assert "priority" not in settings["hyperparameters"]["XGB"][0]["ag_args"]


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
    assert "def _inverse_frequency_sample_weight" in code
    assert "_inverse_frequency_sample_weight(\n            train_data[target_col]" in code
    assert code.index("train_data, valid_data = train_test_split(") < code.index(
        "train_weights, class_weight_mapping = _inverse_frequency_sample_weight("
    )
    assert 'predictor_kwargs["sample_weight"] = CLASS_WEIGHT_COL' in code
    assert 'predictor_kwargs["weight_evaluation"] = False' in code
    assert 'valid_data.drop(columns=[target_col, CLASS_WEIGHT_COL], errors="ignore")' in code


def _class_balancing_helpers():
    namespace = {"np": np, "pd": pd}
    exec(autogluon_preprocess._CLASS_BALANCING_HELPER_SOURCE, namespace)
    return namespace


def test_inverse_frequency_weights_support_alpha_normalization_and_index_alignment():
    helper = _class_balancing_helpers()["_inverse_frequency_sample_weight"]
    labels = pd.Series(["major", "major", "major", "minor"], index=[8, 3, 5, 1])

    weights, mapping = helper(labels, alpha=0.5)

    assert weights.index.equals(labels.index)
    assert weights.mean() == pytest.approx(1.0)
    assert mapping["minor"] / mapping["major"] == pytest.approx(3**0.5)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()


def test_inverse_frequency_alpha_one_matches_training_partition_only_formula():
    helper = _class_balancing_helpers()["_inverse_frequency_sample_weight"]
    full_labels = pd.Series(["major"] * 8 + ["minor"] * 4)
    training_labels = full_labels.iloc[[0, 1, 2, 3, 4, 8, 9]]

    weights, mapping = helper(training_labels, alpha=1.0)

    assert mapping == pytest.approx({"major": 0.7, "minor": 1.75})
    assert weights.mean() == pytest.approx(1.0)
    assert len(weights) == len(training_labels)


@pytest.mark.parametrize("alpha", [-0.1, float("nan"), float("inf"), "invalid"])
def test_inverse_frequency_rejects_invalid_alpha(alpha):
    helper = _class_balancing_helpers()["_inverse_frequency_sample_weight"]
    with pytest.raises(ValueError, match="finite number >= 0"):
        helper(pd.Series(["a", "b"]), alpha=alpha)


def test_inverse_frequency_rejects_empty_and_missing_labels():
    helper = _class_balancing_helpers()["_inverse_frequency_sample_weight"]
    with pytest.raises(ValueError, match="empty labels"):
        helper(pd.Series(dtype="object"), alpha=1.0)
    with pytest.raises(ValueError, match="missing labels"):
        helper(pd.Series(["a", None]), alpha=1.0)


def test_class_balance_config_supports_none_inverse_frequency_and_legacy_balanced():
    helper = _class_balancing_helpers()["_class_balance_config"]

    assert helper({"method": "none"}) == {"method": "none"}
    assert helper({"method": "inverse_frequency", "alpha": 0.5}) == {
        "method": "inverse_frequency",
        "alpha": 0.5,
    }
    assert helper("balanced") == {"method": "inverse_frequency", "alpha": 1.0}


def test_custom_balancing_rejects_bagged_and_internal_validation():
    helper = _class_balancing_helpers()["_validate_custom_balancing_context"]
    config = {"method": "inverse_frequency", "alpha": 1.0}

    with pytest.raises(ValueError, match="not supported with bagging"):
        helper(config, bagged_mode=True, validation_strategy="holdout")
    with pytest.raises(ValueError, match="requires validation_strategy='holdout'"):
        helper(config, bagged_mode=False, validation_strategy="autogluon")
    helper(config, bagged_mode=False, validation_strategy="holdout")


def test_class_balance_cpu_profiles_match_reference_except_class_balance():
    cfg = _load_cfg(use_cli_args=False)
    profiles = (
        (
            "s6e7_class_balance_stage_a_none_cpu_capped180_fairone_seed1729_10m",
            {"method": "none"},
        ),
        (
            "s6e7_class_balance_stage_a_inverse_frequency_alpha1_cpu_capped180_fairone_seed1729_10m",
            {"method": "inverse_frequency", "alpha": 1.0},
        ),
        (
            "s6e7_class_balance_stage_b_inverse_frequency_alpha075_cpu_capped180_fairone_seed1729_10m",
            {"method": "inverse_frequency", "alpha": 0.75},
        ),
        (
            "s6e7_class_balance_stage_b_inverse_frequency_alpha050_cpu_capped180_fairone_seed1729_10m",
            {"method": "inverse_frequency", "alpha": 0.5},
        ),
    )
    resolved = []
    for profile_name, expected_class_balance in profiles:
        cfg.agent.autogluon.profile = profile_name
        settings = resolve_autogluon_settings(cfg)
        class_balance = settings.pop("class_balance")
        resolved.append(settings)
        assert class_balance == expected_class_balance

    assert all(settings == resolved[0] for settings in resolved[1:])
    settings = resolved[0]
    assert settings["use_gpu"] is False
    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }
    for model_type, model_configs in settings["hyperparameters"].items():
        assert model_type in {"XGB", "GBM", "CAT"}
        assert len(model_configs) == 1
        assert model_configs[0]["ag_args"] == {"priority": 100}
        assert model_configs[0]["ag_args_fit"]["num_gpus"] == 0
        assert model_configs[0]["ag_args_fit"]["max_time_limit"] == 180
        assert "task_type" not in model_configs[0]
        assert "devices" not in model_configs[0]
        assert "gpu_ram_part" not in model_configs[0]
    assert settings["hyperparameters"]["XGB"][0]["device"] == "cpu"
    assert settings["hyperparameters"]["XGB"][0]["tree_method"] == "hist"
    allocated_model_seconds = sum(
        settings["hyperparameters"][model_type][0]["ag_args_fit"][
            "max_time_limit"
        ]
        for model_type in ("XGB", "GBM", "CAT")
    )
    assert allocated_model_seconds == 540
    assert settings["time_limit"] - allocated_model_seconds == 60


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
    submission = namespace["pd"].DataFrame({"id": [1, 2], "class": ["STAR", "QSO"]})
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
        {"id": [3, 1, 2], "class": ["GALAXY", "GALAXY", "GALAXY"]}
    )
    test_ids = namespace["pd"].Series([1, 2, 3])
    test_pred = namespace["pd"].Series(["STAR", "QSO", "GALAXY"])

    submission = namespace["_make_submission"](
        sample_submission,
        id_col="id",
        target_col="class",
        test_ids=test_ids,
        test_pred=test_pred,
    )

    assert submission["id"].to_list() == [1, 2, 3]
    assert submission["class"].to_list() == ["STAR", "QSO", "GALAXY"]


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

    assert "'profile': 'full_boost'" in code
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
    assert settings["save_prediction_artifacts"] is True
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


def test_autogluon_can_disable_prediction_artifact_export(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.save_prediction_artifacts = False

    settings = resolve_autogluon_settings(cfg)
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)

    assert settings["save_prediction_artifacts"] is False
    assert "'save_prediction_artifacts': False" in code
    assert 'if AIDE_AG_CONFIG.get("save_prediction_artifacts", True):' in code
    assert "prediction artifact export disabled" in code
    assert "_clear_prediction_artifacts(working_dir)" in code


def test_high_cv3_profile_disables_prediction_artifact_export(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "boost_gpu_ens_high_cv3"

    settings = resolve_autogluon_settings(cfg)
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)

    assert settings["save_prediction_artifacts"] is False
    assert "'save_prediction_artifacts': False" in code


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


def test_autogluon_s6e7_fast_alignment_profiles_are_medium_screening_profiles(tmp_path):
    cfg = _cfg(tmp_path)
    profiles = dict(cfg.agent.autogluon.profiles)
    candidate_names = sorted(
        name
        for name in profiles
        if name.startswith("s6e7_align_") or name.startswith("s6e7_fast_medium_")
    )

    assert "s6e7_fast_medium_holdout20_nobalance_10m" in candidate_names
    assert "s6e7_fast_medium_noensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_holdout10_noensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_noensemble_balanced_seed7_10m" in candidate_names
    assert "s6e7_fast_medium_noensemble_balanced_seed123_10m" in candidate_names
    assert "s6e7_fast_medium_xgb_seed123_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_xgb_seed777_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_xgbgbm_seed123_ensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_xgb_seed123_nobalance_10m" in candidate_names
    assert "s6e7_fast_medium_xgb_seed123_holdout15_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_xgb_seed123_holdout22_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_xgb_seed123_holdout25_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_xgb_seed123_holdout30_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_gbmcat_noensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_gbmcat_seed123_noensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_gbmcat_seed123_nobalance_10m" in candidate_names
    assert "s6e7_fast_medium_gbmcat_seed123_holdout22_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_gbmcat_seed123_holdout25_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_gbmcat_seed777_noensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_gbm_seed123_noensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_cat_seed123_noensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_cat_seed42_noensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_gbmcat_seed123_ensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_xgbcat_seed123_ensemble_balanced_10m" in candidate_names
    assert "s6e7_fast_medium_holdout30_noensemble_balanced_10m" in candidate_names

    for profile in candidate_names:
        cfg = _cfg(tmp_path)
        cfg.agent.autogluon.profile = profile
        cfg.agent.autogluon.included_model_types = None

        settings = resolve_autogluon_settings(cfg)

        assert settings["presets"] in {"medium", "medium_quality"}
        assert settings["time_limit"] <= 600
        assert settings["preprocess_timeout"] <= 600


def test_autogluon_s6e7_fast_medium_profile_variants(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_holdout20_nobalance_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert "class_balance" not in settings
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": True,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_noensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_strategy"] == "holdout"
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_holdout10_noensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.1
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 42
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_noensemble_balanced_seed7_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 7
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_noensemble_balanced_seed123_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_xgb_seed123_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_xgb_seed777_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 777
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = (
        "s6e7_fast_medium_xgbgbm_seed123_ensemble_balanced_10m"
    )
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": True,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_xgb_seed123_nobalance_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert "class_balance" not in settings
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_xgb_seed123_holdout15_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.15
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_xgb_seed123_holdout22_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.22
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_xgb_seed123_holdout25_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.25
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_xgb_seed123_holdout30_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.3
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_gbmcat_noensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 42
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_gbmcat_seed123_noensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_gbmcat_seed123_nobalance_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert "class_balance" not in settings
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = (
        "s6e7_fast_medium_gbmcat_seed123_holdout22_balanced_10m"
    )
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.22
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = (
        "s6e7_fast_medium_gbmcat_seed123_holdout25_balanced_10m"
    )
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.25
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_gbmcat_seed777_noensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 777
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_gbm_seed123_noensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["GBM"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_cat_seed123_noensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_cat_seed42_noensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 42
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_gbmcat_seed123_ensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": True,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_xgbcat_seed123_ensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.2
    assert settings["validation_strategy"] == "holdout"
    assert settings["seed"] == 123
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": True,
        "auto_stack": False,
    }

    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "s6e7_fast_medium_holdout30_noensemble_balanced_10m"
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert settings["presets"] == "medium_quality"
    assert settings["time_limit"] == 600
    assert settings["validation_fraction"] == 0.3
    assert settings["validation_strategy"] == "holdout"
    assert settings["class_balance"] == "balanced"
    assert settings["fit_args"] == {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
    }


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
    assert settings["preprocess_timeout"] == 600
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


def test_s6e7_calibration_profiles_keep_equal_model_priorities(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.included_model_types = None

    profiles = [
        profile
        for profile in cfg.agent.autogluon.profiles
        if profile.startswith("s6e7_calibration_")
    ]

    assert profiles
    for profile in profiles:
        cfg.agent.autogluon.profile = profile
        settings = resolve_autogluon_settings(cfg)
        priorities = {
            model_type: settings["hyperparameters"][model_type][0]["ag_args"][
                "priority"
            ]
            for model_type in ("XGB", "GBM", "CAT")
        }
        assert priorities == {"XGB": 100, "GBM": 100, "CAT": 100}


def test_capped_cpu_calibration_profile_caps_all_required_families_equally(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = (
        "s6e7_calibration_reference_holdout20_unweighted_cpu_"
        "capped180_fairone_seed1729_10m"
    )
    cfg.agent.autogluon.included_model_types = None

    settings = resolve_autogluon_settings(cfg)

    assert settings["use_gpu"] is False
    for model_type in ("XGB", "GBM", "CAT"):
        model_settings = settings["hyperparameters"][model_type][0]
        assert model_settings["ag_args"]["priority"] == 100
        assert model_settings["ag_args_fit"]["max_time_limit"] == 180
        assert model_settings["ag_args_fit"]["num_gpus"] == 0


def test_autogluon_settings_default_lightgbm_gpu_categorical_fallback(tmp_path):
    cfg = _cfg(tmp_path)

    settings = resolve_autogluon_settings(cfg)

    assert settings["lightgbm_gpu_categorical_fallback"] == {
        "action": "fallback_to_cpu",
        "max_categorical_cardinality": 512,
    }

    cfg.agent.autogluon.lightgbm_gpu_categorical_fallback = {
        "action": "fallback2cpu",
        "max_categorical_cardinality": 1024,
    }
    alias_settings = resolve_autogluon_settings(cfg)

    assert alias_settings["lightgbm_gpu_categorical_fallback"] == {
        "action": "fallback_to_cpu",
        "max_categorical_cardinality": 1024,
    }


def test_autogluon_wrapper_falls_back_lightgbm_gpu_to_cpu_for_high_cardinality_categories(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "full_boost_gpu"
    cfg.agent.autogluon.included_model_types = None
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    namespace = {}
    exec(code.replace("\nmain()\n", "\n"), namespace)

    train_frame = pd.DataFrame(
        {
            "cat": [f"level_{idx}" for idx in range(513)],
            "small_cat": ["a", "b", "a"] * 171,
            "num": range(513),
        }
    )
    test_frame = pd.DataFrame({"cat": ["new"], "small_cat": ["a"], "num": [1]})

    ag_config, train_out, test_out, stats = namespace["_apply_lightgbm_gpu_categorical_fallback"](
        namespace["AIDE_AG_CONFIG"],
        train_frame,
        test_frame,
    )

    assert stats["triggered"] is True
    assert stats["action"] == "fallback_to_cpu"
    assert stats["columns"] == {"cat": 513}
    assert train_out is train_frame
    assert test_out is test_frame
    assert ag_config["hyperparameters"]["GBM"][0]["device"] == "cpu"
    assert ag_config["hyperparameters"]["GBM"][0]["ag_args_fit"]["num_gpus"] == 0
    assert ag_config["hyperparameters"]["XGB"][0]["device"] == "cuda"
    assert ag_config["hyperparameters"]["CAT"][0]["task_type"] == "GPU"


def test_autogluon_wrapper_can_drop_high_cardinality_categories(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "full_boost_gpu"
    cfg.agent.autogluon.included_model_types = None
    cfg.agent.autogluon.lightgbm_gpu_categorical_fallback = {
        "action": "drop",
        "max_categorical_cardinality": 2,
    }
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    namespace = {}
    exec(code.replace("\nmain()\n", "\n"), namespace)

    train_frame = pd.DataFrame({"cat": ["a", "b", "c"], "small": ["x", "x", "y"]})
    test_frame = pd.DataFrame({"cat": ["d"], "small": ["x"]})

    _ag_config, train_out, test_out, stats = namespace["_apply_lightgbm_gpu_categorical_fallback"](
        namespace["AIDE_AG_CONFIG"],
        train_frame,
        test_frame,
    )

    assert stats["action"] == "drop_columns"
    assert stats["triggered"] is True
    assert stats["columns"] == {"cat": 3}
    assert stats["dropped_columns"] == ["cat"]
    assert train_out.columns.to_list() == ["small"]
    assert test_out.columns.to_list() == ["small"]


def test_autogluon_wrapper_lightgbm_categorical_fallback_is_noop_for_cpu_profile(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "full_boost"
    cfg.agent.autogluon.included_model_types = None
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    namespace = {}
    exec(code.replace("\nmain()\n", "\n"), namespace)

    train_frame = pd.DataFrame({"cat": [f"level_{idx}" for idx in range(513)]})
    test_frame = pd.DataFrame({"cat": ["new"]})

    ag_config, train_out, test_out, stats = namespace["_apply_lightgbm_gpu_categorical_fallback"](
        namespace["AIDE_AG_CONFIG"],
        train_frame,
        test_frame,
    )

    assert stats["triggered"] is False
    assert stats["reason"] == "lightgbm_not_gpu"
    assert train_out is train_frame
    assert test_out is test_frame
    assert ag_config["hyperparameters"]["GBM"][0]["ag_args_fit"] == {"num_gpus": 0}


def test_autogluon_wrapper_preserves_config_after_noop_lightgbm_fallback_update(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "full_boost"
    cfg.agent.autogluon.included_model_types = None
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)
    namespace = {}
    exec(code.replace("\nmain()\n", "\n"), namespace)

    train_frame = pd.DataFrame({"cat": [f"level_{idx}" for idx in range(513)]})
    test_frame = pd.DataFrame({"cat": ["new"]})
    ag_config_update, _train_out, _test_out, stats = namespace[
        "_apply_lightgbm_gpu_categorical_fallback"
    ](
        namespace["AIDE_AG_CONFIG"],
        train_frame,
        test_frame,
    )

    namespace["AIDE_AG_CONFIG"].clear()
    namespace["AIDE_AG_CONFIG"].update(ag_config_update)

    assert stats["reason"] == "lightgbm_not_gpu"
    assert namespace["_configured_metric"]() == "balanced_accuracy"


def test_autogluon_unknown_profile_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.autogluon.profile = "slow_magic"
    cfg.agent.autogluon.included_model_types = None

    with pytest.raises(ValueError, match="Unknown AutoGluon profile"):
        resolve_autogluon_included_model_types(cfg)


def test_agent_autogluon_draft_wraps_preprocess_response(tmp_path):
    cfg = _cfg(tmp_path)
    agent = Agent(
        task_desc="Predict `class`. The identifier column is `id`.",
        cfg=cfg,
        journal=Journal(),
    )
    agent.data_preview = (
        "class (object) has 3 unique values.\n"
        "id (int64) has range 1 - 2.\n"
        "redshift (float64) has useful signal.\n"
    )
    captured = {}

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return (
            "add a simple numeric ratio",
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['redshift_x2'] = df.get('redshift', 0) * 2\n"
            "    return df\n",
        )

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent._draft()

    assert "AutoGluon preprocess mode contract" in captured["prompt"]["Instructions"]
    prompt_text = str(captured["prompt"])
    assert "redshift" in prompt_text
    assert "class" in prompt_text
    assert "`id`" in prompt_text
    assert "__is_train__" not in prompt_text
    assert "__aide_row_id__" not in prompt_text
    assert "must replace the previous preprocess function" in prompt_text
    assert "Do not call `globals().get(\"preprocess\")`" in prompt_text
    assert "Mechanical simplifications are allowed only" in prompt_text
    assert "Do not optimize by changing algorithms" in prompt_text
    assert "dedicated timeout of 600 seconds" in prompt_text
    assert "Avoid expensive Python callbacks" in prompt_text
    assert "rolling.apply" in prompt_text
    assert "TabularPredictor" in node.code
    assert "redshift_x2" in node.code


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


def test_agent_code_web_search_option_writes_markdown_summary(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.agent.code.web_search = True
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    captured = {}

    def fake_query(**kwargs):
        captured["prompt"] = kwargs["system_message"]
        captured["web_search"] = kwargs.get("web_search")
        events = [
            {
                "type": "item.completed",
                "item": {
                    "type": "web_search",
                    "query": "playground s6e6 kaggle discussion",
                    "action": {
                        "type": "search",
                        "query": "playground s6e6 kaggle discussion",
                        "queries": [
                            "playground s6e6 kaggle discussion",
                            "playground s6e6 winning notebook",
                        ],
                    },
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "web_search",
                    "query": "https://www.kaggle.com/competitions/example/discussion/1",
                    "action": {"type": "other"},
                },
            },
        ]
        (artifact_dir / "codex_events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events),
            encoding="utf-8",
        )
        return (
            "Add a simple feature.\n"
            "```python\n"
            "def preprocess(df):\n"
            "    return df.copy()\n"
            "```"
        )

    monkeypatch.setitem(agent.plan_and_code_query.__globals__, "query", fake_query)
    agent._pending_llm_log_dir = artifact_dir

    plan, code = agent.plan_and_code_query({"Instructions": {}})

    assert plan == "Add a simple feature."
    assert "def preprocess" in code
    assert captured["web_search"] is True
    assert "Web search" in captured["prompt"]["Instructions"]
    assert any(
        "domain or method background relevant to the task" in item
        for item in captured["prompt"]["Instructions"]["Web search"]
    )
    assert any(
        "tentative preprocessing hypothesis" in item
        for item in captured["prompt"]["Instructions"]["Web search"]
    )
    assert any(
        "read their page content" in item
        for item in captured["prompt"]["Instructions"]["Web search"]
    )
    assert any(
        "support, change, or rule out" in item
        for item in captured["prompt"]["Instructions"]["Web search"]
    )
    summary = (artifact_dir / "web_search.md").read_text(encoding="utf-8")
    assert "# Web Search" in summary
    assert "## Search Queries" in summary
    assert "- playground s6e6 kaggle discussion" in summary
    assert "- playground s6e6 winning notebook" in summary
    assert "## Opened Pages" in summary
    assert "- https://www.kaggle.com/competitions/example/discussion/1" in summary
    assert '"type":' not in summary


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


def test_agent_autogluon_improve_prompt_keeps_memory_and_siblings_adjacent(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    parent = Node(
        plan="base preprocess",
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
    prior_child = Node(
        plan="prior child attempt",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['base_feature'] = 0\n"
            "    return df\n",
            cfg,
        ),
        parent=parent,
    )
    prior_child.metric = MetricValue(0.89, maximize=True)
    prior_child.is_buggy = False
    parent.children.add(prior_child)

    journal = Journal()
    journal.append(parent)
    journal.append(prior_child)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
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

    agent._improve(parent)

    keys = list(captured["prompt"])
    assert keys.index("Previous attempts from this parent") == keys.index("Memory") + 1
    assert "base preprocess" in captured["prompt"]["Memory"]
    assert (
        "prior child attempt"
        in captured["prompt"]["Previous attempts from this parent"]
    )
    history_rule = captured["prompt"]["Instructions"][
        "Experiment-history interpretation rule"
    ]
    history_rule_text = " ".join(history_rule)
    assert "scored experimental evidence" in history_rule_text
    assert "feature-mechanism families" in history_rule_text
    assert "Do not output this analysis" in history_rule_text
    assert "redshift" not in history_rule_text
    assert "aux" not in history_rule_text.lower()
    assert "class" not in history_rule_text.lower()


def test_agent_autogluon_improve_prompt_adds_other_improving_hypotheses(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    cfg.agent.search.hypothesis_min_improvement_epsilon = 0.1
    parent = Node(
        plan="base preprocess",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    return df.copy()\n",
            cfg,
        ),
    )
    parent.metric = MetricValue(0.9, maximize=True)
    parent.is_buggy = False
    prior_child = Node(
        plan="non improving current-tree child",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    return df.copy()\n",
            cfg,
        ),
        parent=parent,
    )
    prior_child.metric = MetricValue(0.89, maximize=True)
    prior_child.is_buggy = False

    other_parent = Node(
        plan="outside parent",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    return df.copy()\n",
            cfg,
        ),
    )
    other_parent.metric = MetricValue(0.7, maximize=True)
    other_parent.is_buggy = False
    other_child = Node(
        plan="small winning outside design below epsilon",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    return df.copy()\n",
            cfg,
        ),
        parent=other_parent,
    )
    other_child.metric = MetricValue(0.70001, maximize=True)
    other_child.is_buggy = False
    strong_other_child = Node(
        plan="winning outside design",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    return df.copy()\n",
            cfg,
        ),
        parent=other_parent,
    )
    strong_other_child.metric = MetricValue(0.81, maximize=True)
    strong_other_child.is_buggy = False
    duplicate_other_child = Node(
        plan="winning outside design",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    return df.copy()\n",
            cfg,
        ),
        parent=other_parent,
    )
    duplicate_other_child.metric = MetricValue(0.72, maximize=True)
    duplicate_other_child.is_buggy = False
    losing_other_child = Node(
        plan="losing outside design",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    return df.copy()\n",
            cfg,
        ),
        parent=other_parent,
    )
    losing_other_child.metric = MetricValue(0.69, maximize=True)
    losing_other_child.is_buggy = False

    journal = Journal()
    for node in [
        parent,
        prior_child,
        other_parent,
        other_child,
        strong_other_child,
        duplicate_other_child,
        losing_other_child,
    ]:
        journal.append(node)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    captured = {}

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return (
            "improve feature",
            "def preprocess(df):\n"
            "    return df.copy()\n",
        )

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(parent)

    keys = list(captured["prompt"])
    assert keys.index("Other improving hypotheses outside this node tree") == (
        keys.index("Memory") + 1
    )
    assert keys.index("Previous attempts from this parent") == keys.index(
        "Other improving hypotheses outside this node tree"
    ) + 1
    other = captured["prompt"]["Other improving hypotheses outside this node tree"]
    assert other.count("Design: winning outside design") == 1
    assert "small winning outside design below epsilon" not in other
    assert "losing outside design" not in other
    assert "non improving current-tree child" not in other
    assert "Step:" not in other
    assert "Validation Metric:" not in other


@pytest.mark.parametrize(
    ("child_count", "expected_key", "unexpected_key"),
    [
        (4, None, "Repeated non-improving sibling evidence"),
        (
            5,
            "Repeated non-improving sibling evidence",
            "Strong repeated-failure evidence",
        ),
        (
            11,
            "Strong repeated-failure evidence",
            "Repeated non-improving sibling evidence",
        ),
    ],
)
def test_agent_autogluon_improve_prompt_adds_repeated_failure_rule_by_count(
    tmp_path,
    child_count,
    expected_key,
    unexpected_key,
):
    cfg = _cfg(tmp_path)
    parent = Node(
        plan="base preprocess",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    return df.copy()\n",
            cfg,
        ),
    )
    parent.metric = MetricValue(0.9, maximize=True)
    parent.is_buggy = False

    journal = Journal()
    journal.append(parent)
    for idx in range(child_count):
        child = Node(
            plan=f"non improving child {idx}",
            code=build_autogluon_wrapper(
                "def preprocess(df):\n"
                "    return df.copy()\n",
                cfg,
            ),
            parent=parent,
        )
        child.metric = MetricValue(0.89 - idx / 1000.0, maximize=True)
        child.is_buggy = False
        parent.children.add(child)
        journal.append(child)

    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    captured = {}

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return (
            "improve feature",
            "def preprocess(df):\n"
            "    return df.copy()\n",
        )

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(parent)

    instructions = captured["prompt"]["Instructions"]
    if expected_key is None:
        assert "Repeated non-improving sibling evidence" not in instructions
        assert "Strong repeated-failure evidence" not in instructions
    else:
        assert expected_key in instructions
        assert unexpected_key not in instructions


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
