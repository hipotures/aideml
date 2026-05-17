import pytest

from aide.utils.config import _load_cfg, prep_cfg, resolve_model_config


def _base_cfg(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    return cfg


def test_resolve_model_config_splits_reasoning_effort_suffix():
    resolved = resolve_model_config("gpt-5.5:low", None)

    assert resolved.model == "gpt-5.5"
    assert resolved.reasoning_effort == "low"


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
    assert cfg.agent.feedback.model == "gpt-5.4-mini"
    assert cfg.agent.feedback.reasoning_effort == "low"
    assert cfg.report.model == "gpt-5.4-mini"
    assert cfg.report.reasoning_effort == "low"
    assert cfg.research.model == "gpt-5.4-mini"
    assert cfg.research.reasoning_effort == "low"
    assert cfg.research.mode == "llm"
    assert cfg.research.manual_sample_size == 3
    assert cfg.research.manual_seed == 42
    assert cfg.research.hypothesis_root_limit == 100
    assert cfg.synthesis.model == "gpt-5.4-mini"
    assert cfg.synthesis.reasoning_effort == "low"
    assert cfg.agent.search.exploration_weight == 0.05
    assert cfg.agent.gpu is False


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


def test_agent_mode_autogluon_alias_resolves_to_preprocess_mode(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg.agent.mode = "autogluon"

    cfg = prep_cfg(cfg)

    assert cfg.agent.mode == "autogluon_preprocess"
