import pytest

from aide.utils.config import _load_cfg, prep_cfg, resolve_model_config


def _base_cfg(tmp_path, *, load_env: bool = False):
    cfg = _load_cfg(use_cli_args=False, load_env=load_env)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    return cfg


def test_resolve_model_config_splits_reasoning_effort_suffix():
    resolved = resolve_model_config("gpt-5.5:low", None)

    assert resolved.model == "gpt-5.5"
    assert resolved.reasoning_effort == "low"


@pytest.mark.parametrize(
    ("model", "reasoning_effort"),
    [
        ("gpt-5.6-luna:max", None),
        ("gpt-5.6-luna", "max"),
        ("gpt-future:ultra", None),
        ("gpt-future", "ultra"),
    ],
)
def test_resolve_model_config_accepts_backend_reasoning_effort(
    model, reasoning_effort
):
    resolved = resolve_model_config(model, reasoning_effort)

    assert resolved.model == model.split(":", 1)[0]
    assert resolved.reasoning_effort == (reasoning_effort or model.rsplit(":", 1)[1])


@pytest.mark.parametrize(
    ("model", "reasoning_effort"),
    [
        ("gpt-5.6-luna:", None),
        ("gpt-5.6-luna", ""),
    ],
)
def test_resolve_model_config_rejects_empty_reasoning_effort(model, reasoning_effort):
    with pytest.raises(ValueError, match="empty reasoning_effort|empty reasoning effort"):
        resolve_model_config(model, reasoning_effort)


def test_load_cfg_rejects_cli_model_suffix_and_explicit_reasoning_effort():
    with pytest.raises(ValueError, match="agent.code"):
        _load_cfg(
            cli_args=[
                "agent.code.model=gpt-5.5:low",
                "agent.code.reasoning_effort=high",
            ]
        )


def test_prep_cfg_resolves_default_models_to_gpt_5_4_mini_low(tmp_path):
    cfg = prep_cfg(_base_cfg(tmp_path))

    assert cfg.agent.code.model == "gpt-5.4-mini"
    assert cfg.agent.code.reasoning_effort == "low"
    assert cfg.agent.code.web_search is False
    assert cfg.agent.code.codex_branch_sessions is False
    assert cfg.agent.feedback.model == "gpt-5.4-mini"
    assert cfg.agent.feedback.reasoning_effort == "low"
    assert cfg.report.model == "gpt-5.4-mini"
    assert cfg.report.reasoning_effort == "low"
    assert cfg.research.root_hypothesis_model == "gpt-5.4-mini"
    assert cfg.research.reasoning_effort == "low"
    assert cfg.research.mode == "llm"
    assert cfg.research.manual_sample_size == 3
    assert cfg.research.manual_seed == 42
    assert cfg.research.hypothesis_root_limit == 100
    assert cfg.research.hypothesis_root_generate_workers == 1
    assert cfg.research.seed_scored_roots is True
    assert cfg.synthesis.model == "gpt-5.4-mini"
    assert cfg.synthesis.reasoning_effort == "low"
    assert cfg.agent.search.exploration_weight == 0.0
    assert cfg.agent.search.forced_root is None
    assert cfg.agent.search.hypothesis_max_non_improving_children_per_parent == 10
    assert cfg.agent.search.hypothesis_min_improvement_epsilon == 0.0002
    assert cfg.agent.search.plateau_block_epsilon == 0.00001
    assert cfg.agent.search.public_score_bonus_weight == 0.0
    assert cfg.agent.search.public_score_bonus_cap == 0.0005
    assert cfg.agent.memory_recent_steps == 50
    assert cfg.agent.memory_full_recent_steps == 10
    assert cfg.agent.include_parent_process_stdout is True
    assert cfg.agent.parent_process_stdout_max_bytes == 5000
    assert cfg.agent.gpu is False


def test_load_cfg_ignores_dotenv_by_default_under_pytest(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "AIDE_AGENT_CODE_MODEL=gpt-from-dotenv:high\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = prep_cfg(_base_cfg(tmp_path))

    assert cfg.agent.code.model == "gpt-5.4-mini"
    assert cfg.agent.code.reasoning_effort == "low"


def test_agent_code_model_env_accepts_backend_reasoning_effort(tmp_path, monkeypatch):
    monkeypatch.delenv("AIDE_AGENT_CODE_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "AIDE_AGENT_CODE_MODEL=gpt-5.6-luna:max\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = prep_cfg(_base_cfg(tmp_path, load_env=True), load_env=True)

    assert cfg.agent.code.model == "gpt-5.6-luna"
    assert cfg.agent.code.reasoning_effort == "max"


def test_research_root_hypothesis_model_env_overrides_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AIDE_RESEARCH_ROOT_HYPOTHESIS_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "AIDE_RESEARCH_ROOT_HYPOTHESIS_MODEL=gpt-root-hypothesis:medium\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = prep_cfg(_base_cfg(tmp_path, load_env=True), load_env=True)

    assert cfg.research.root_hypothesis_model == "gpt-root-hypothesis"
    assert cfg.research.reasoning_effort == "medium"


def test_agent_code_web_search_env_overrides_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AIDE_AGENT_CODE_WEB_SEARCH", raising=False)
    (tmp_path / ".env").write_text(
        "AIDE_AGENT_CODE_WEB_SEARCH=true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = prep_cfg(_base_cfg(tmp_path, load_env=True), load_env=True)

    assert cfg.agent.code.web_search is True


def test_agent_code_branch_sessions_env_overrides_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AIDE_AGENT_CODE_CODEX_BRANCH_SESSIONS", raising=False)
    (tmp_path / ".env").write_text(
        "AIDE_AGENT_CODE_CODEX_BRANCH_SESSIONS=true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = prep_cfg(_base_cfg(tmp_path, load_env=True), load_env=True)

    assert cfg.agent.code.codex_branch_sessions is True


def test_search_best_score_min_children_env_overrides_default(tmp_path, monkeypatch):
    monkeypatch.delenv(
        "AIDE_AGENT_SEARCH_BEST_SCORE_MIN_CHILDREN_BEFORE_EXPLORATION",
        raising=False,
    )
    (tmp_path / ".env").write_text(
        "AIDE_AGENT_SEARCH_BEST_SCORE_MIN_CHILDREN_BEFORE_EXPLORATION=6\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = prep_cfg(_base_cfg(tmp_path, load_env=True), load_env=True)

    assert cfg.agent.search.best_score_min_children_before_exploration == 6


def test_legacy_research_model_env_key_is_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("AIDE_RESEARCH_MODEL", raising=False)
    monkeypatch.delenv("AIDE_RESEARCH_ROOT_HYPOTHESIS_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "AIDE_RESEARCH_MODEL=gpt-legacy:high\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = prep_cfg(_base_cfg(tmp_path))

    assert cfg.research.root_hypothesis_model == "gpt-5.4-mini"
    assert cfg.research.reasoning_effort == "low"


def test_legacy_research_model_cli_key_is_rejected(tmp_path):
    cfg = _load_cfg(cli_args=["research.model=gpt-legacy"])
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")

    with pytest.raises(Exception, match="research.model"):
        prep_cfg(cfg)


def test_agent_memory_and_parent_stdout_options_can_be_overridden_from_cli():
    cfg = _load_cfg(
        cli_args=[
            "agent.memory_recent_steps=60",
            "agent.memory_full_recent_steps=12",
            "agent.include_parent_process_stdout=true",
            "agent.parent_process_stdout_max_bytes=2048",
        ]
    )

    assert cfg.agent.memory_recent_steps == 60
    assert cfg.agent.memory_full_recent_steps == 12
    assert cfg.agent.include_parent_process_stdout is True
    assert cfg.agent.parent_process_stdout_max_bytes == 2048


def test_full_boost_gpu_ensemble_profile_is_available():
    cfg = _load_cfg(use_cli_args=False)

    gpu_profile = cfg.agent.autogluon.profiles.full_boost_gpu
    ensemble_profile = cfg.agent.autogluon.profiles.full_boost_gpu_ens

    assert ensemble_profile.included_model_types == gpu_profile.included_model_types
    assert ensemble_profile.presets == gpu_profile.presets
    assert ensemble_profile.time_limit == gpu_profile.time_limit
    assert ensemble_profile.use_gpu is True
    assert ensemble_profile.fit_args.fit_weighted_ensemble is True
    assert gpu_profile.fit_args.fit_weighted_ensemble is False


def test_high_cv3_profile_combines_high_preset_with_three_fold_bagging():
    cfg = _load_cfg(use_cli_args=False)

    profile = cfg.agent.autogluon.profiles.boost_gpu_ens_high_cv3

    assert profile.presets == "high"
    assert profile.time_limit == 900
    assert profile.save_prediction_artifacts is False
    assert profile.use_gpu is True
    assert profile.validation_strategy == "autogluon"
    assert profile.class_balance == "balanced"
    assert profile.fit_args.fit_weighted_ensemble is True
    assert profile.fit_args.auto_stack is False
    assert profile.fit_args.num_bag_folds == 3
    assert profile.fit_args.num_bag_sets == 1
    assert profile.fit_args.num_stack_levels == 0


def test_prep_cfg_resolves_cli_model_suffix(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg.agent.code.model = "gpt-5.5:high"

    cfg = prep_cfg(cfg)

    assert cfg.agent.code.model == "gpt-5.5"
    assert cfg.agent.code.reasoning_effort == "high"


def test_cli_plain_model_override_clears_default_reasoning_effort(tmp_path):
    cfg = _load_cfg(
        cli_args=[
            f"data_dir={tmp_path}",
            "goal=test",
            f"log_dir={tmp_path / 'logs'}",
            f"workspace_dir={tmp_path / 'workspaces'}",
            "agent.code.model=gemma-4-31B",
        ]
    )

    cfg = prep_cfg(cfg)

    assert cfg.agent.code.model == "gemma-4-31B"
    assert cfg.agent.code.reasoning_effort is None


def test_cli_agent_gpu_override_is_loaded(tmp_path):
    cfg = _load_cfg(
        cli_args=[
            f"data_dir={tmp_path}",
            "goal=test",
            f"log_dir={tmp_path / 'logs'}",
            f"workspace_dir={tmp_path / 'workspaces'}",
            "agent.gpu=true",
        ]
    )

    cfg = prep_cfg(cfg)

    assert cfg.agent.gpu is True


def test_cli_can_disable_autogluon_prediction_artifacts(tmp_path):
    cfg = _load_cfg(
        cli_args=[
            f"data_dir={tmp_path}",
            "goal=test",
            f"log_dir={tmp_path / 'logs'}",
            f"workspace_dir={tmp_path / 'workspaces'}",
            "agent.autogluon.save_prediction_artifacts=false",
        ]
    )

    cfg = prep_cfg(cfg)

    assert cfg.agent.autogluon.save_prediction_artifacts is False


def test_load_cfg_preserves_unquoted_forced_root_cli_id(tmp_path):
    cfg = _load_cfg(
        cli_args=[
            f"data_dir={tmp_path}",
            "goal=test",
            f"log_dir={tmp_path / 'logs'}",
            f"workspace_dir={tmp_path / 'workspaces'}",
            "agent.search.forced_root=000405",
        ]
    )

    cfg = prep_cfg(cfg)

    assert cfg.agent.search.forced_root == "000405"


def test_load_cfg_accepts_short_forced_root_cli_alias(tmp_path):
    cfg = _load_cfg(
        cli_args=[
            f"data_dir={tmp_path}",
            "goal=test",
            f"log_dir={tmp_path / 'logs'}",
            f"workspace_dir={tmp_path / 'workspaces'}",
            "forced_root=000405",
        ]
    )

    cfg = prep_cfg(cfg)

    assert cfg.agent.search.forced_root == "000405"


def test_agent_mode_autogluon_alias_resolves_to_preprocess_mode(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg.agent.mode = "autogluon"

    cfg = prep_cfg(cfg)

    assert cfg.agent.mode == "autogluon_preprocess"
