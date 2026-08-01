import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aide.agent import Agent
from aide.autogluon_generator import (
    AGENT_MODE,
    GENERATOR_EXPERIMENTS,
    build_generator_variant,
    completed_experiment_names,
    next_generator_experiment,
)
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.run import configure_autogluon_generator_mode
from aide.utils.artifact_manifest import parse_autogluon_config
from aide.utils.config import _load_cfg, prep_cfg
from aide.utils.metric import MetricValue


def _cfg(tmp_path: Path):
    cfg = _load_cfg(use_cli_args=False, load_env=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "ag-generator-test"
    cfg.agent.mode = AGENT_MODE
    return prep_cfg(cfg, load_env=False)


def _seed_code() -> str:
    return '''from __future__ import annotations

AIDE_AG_CONFIG = {
    "included_model_types": ["XGB", "GBM"],
    "hyperparameters": {
        "XGB": [{"ag_args_fit": {"num_gpus": 1}, "tree_method": "hist"}],
        "GBM": {"device": "cuda"},
    },
    "presets": "medium_quality",
    "time_limit": 600,
}

def main():
    return AIDE_AG_CONFIG
'''


@pytest.mark.parametrize("experiment", GENERATOR_EXPERIMENTS, ids=lambda item: item.name)
def test_generator_variant_adds_one_model_specific_generator(
    experiment,
    monkeypatch,
):
    class FakeGenerator:
        pass

    fake_generators = types.ModuleType("autogluon.features.generators")
    setattr(fake_generators, experiment.class_name, FakeGenerator)
    monkeypatch.setitem(sys.modules, "autogluon.features.generators", fake_generators)

    variant = build_generator_variant(_seed_code(), experiment)
    namespace = {}
    exec(compile(variant, "<variant>", "exec"), namespace)

    config = namespace["AIDE_AG_CONFIG"]
    assert config["ag_generator_experiment"] == experiment.metadata()
    assert parse_autogluon_config(variant)["ag_generator_experiment"] == (
        experiment.metadata()
    )
    xgb_config = config["hyperparameters"]["XGB"][0]
    gbm_config = config["hyperparameters"]["GBM"]
    assert xgb_config["ag_args_fit"]["num_gpus"] == 1
    for model_config in (xgb_config, gbm_config):
        generator_args = model_config["ag_args_fit"][
            "model_specific_feature_generator_kwargs"
        ]["feature_generators"]
        assert generator_args == [[FakeGenerator, experiment.kwargs]]


def test_generator_experiments_are_enumerated_once_in_fixed_order():
    journal = Journal()
    assert next_generator_experiment(journal).name == "groupby"

    for experiment in GENERATOR_EXPERIMENTS[:3]:
        journal.append(Node(code="", plan=experiment.plan()))

    assert completed_experiment_names(journal) == {
        "groupby",
        "rsfc",
        "arithmetic",
    }
    assert next_generator_experiment(journal).name == "categorical_interaction"


def test_agent_generates_from_seed_without_data_preview_or_llm(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    seed = Node(
        code=_seed_code(),
        plan="seed",
        status="ok",
        metric=MetricValue(0.9, maximize=True),
        run_stats={"seeded_from_manifest": True},
    )
    journal = Journal()
    journal.append(seed)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    monkeypatch.setattr(
        agent,
        "update_data_preview",
        lambda: pytest.fail("static generator mode does not need a data preview"),
    )
    monkeypatch.setattr(
        agent,
        "plan_and_code_query",
        lambda *_args, **_kwargs: pytest.fail("LLM must not be called"),
    )

    parent = agent.prepare_step()
    node = agent.generate_node(parent)

    assert parent is seed
    assert node.parent is seed
    assert "[groupby]" in node.plan
    assert parse_autogluon_config(node.code)["ag_generator_experiment"]["name"] == (
        "groupby"
    )


def test_generator_failure_is_reviewed_without_llm(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    experiment = GENERATOR_EXPERIMENTS[0]
    node = Node(code="raise RuntimeError('boom')", plan=experiment.plan())
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: pytest.fail("feedback LLM must not be called"),
    )

    agent.review_node(
        node,
        ExecutionResult(
            term_out=["RuntimeError: boom\n"],
            exec_time=1.0,
            exc_type="RuntimeError",
        ),
    )

    assert node.is_buggy is True
    assert node.metric.is_worst
    assert "no LLM review" in node.analysis
    assert node.run_stats["ag_generator_experiment"] == experiment.metadata()


def test_generator_mode_is_opt_in_and_disables_llm_paths(monkeypatch):
    cfg = OmegaConf.create(
        {
            "generate_report": True,
            "research": {"enabled": True},
            "synthesis": {"enabled": True},
            "refactor": {"enabled": True},
            "agent": {
                "mode": AGENT_MODE,
                "steps": 20,
                "hypotheses": 3,
                "search": {"code_ahead": 2},
            },
        }
    )
    source = types.SimpleNamespace(
        manifest={"status": "ok", "local_score": 0.9}
    )
    monkeypatch.setattr("aide.run.source_is_autogluon", lambda _source: True)

    configure_autogluon_generator_mode(
        cfg,
        seed_source=source,
        is_resume=False,
        cli_overrides=["agent.mode=ag_generator"],
    )

    assert cfg.agent.steps == len(GENERATOR_EXPERIMENTS)
    assert cfg.generate_report is False
    assert cfg.research.enabled is False
    assert cfg.synthesis.enabled is False
    assert cfg.refactor.enabled is False
    assert cfg.agent.hypotheses == 0
    assert cfg.agent.search.code_ahead == 0


def test_existing_mode_configuration_is_unchanged():
    cfg = OmegaConf.create(
        {
            "generate_report": True,
            "research": {"enabled": True},
            "synthesis": {"enabled": True},
            "refactor": {"enabled": True},
            "agent": {
                "mode": "autogluon_preprocess",
                "steps": 20,
                "hypotheses": 3,
                "search": {"code_ahead": 2},
            },
        }
    )
    before = OmegaConf.to_container(cfg, resolve=True)

    configure_autogluon_generator_mode(
        cfg,
        seed_source=None,
        is_resume=False,
        cli_overrides=[],
    )

    assert OmegaConf.to_container(cfg, resolve=True) == before


def test_generator_mode_requires_seed():
    cfg = OmegaConf.create(
        {
            "agent": {
                "mode": AGENT_MODE,
                "steps": 6,
                "hypotheses": 0,
                "search": {"code_ahead": 0},
            },
            "generate_report": False,
            "research": {"enabled": False},
            "synthesis": {"enabled": False},
            "refactor": {"enabled": False},
        }
    )

    with pytest.raises(ValueError, match="requires --seed-from-sha"):
        configure_autogluon_generator_mode(
            cfg,
            seed_source=None,
            is_resume=False,
            cli_overrides=["agent.mode=ag_generator"],
        )
