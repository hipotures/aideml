from __future__ import annotations

import ast
import pprint
from dataclasses import dataclass
from typing import Any

from .journal import Journal, Node


AGENT_MODE = "ag_generator"
PLAN_PREFIX = "Static AutoGluon generator experiment"


@dataclass(frozen=True)
class GeneratorExperiment:
    name: str
    class_name: str
    kwargs: dict[str, Any]
    description: str

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "kwargs": dict(self.kwargs),
        }

    def plan(self) -> str:
        return f"{PLAN_PREFIX} [{self.name}]: {self.description}"


GENERATOR_EXPERIMENTS = (
    GeneratorExperiment(
        name="groupby",
        class_name="GroupByFeatureGenerator",
        kwargs={"max_features": 64, "passthrough": True},
        description=(
            "add bounded categorical-by-numeric group statistics while preserving "
            "the seed features"
        ),
    ),
    GeneratorExperiment(
        name="rsfc",
        class_name="RandomSubsetFeatureCompressionGenerator",
        kwargs={
            "n_subsets": 8,
            "max_base_feats_to_consider": 32,
            "passthrough": True,
        },
        description=(
            "add eight deterministic target-aware random-subset compression "
            "features while preserving the seed features"
        ),
    ),
    GeneratorExperiment(
        name="arithmetic",
        class_name="ArithmeticFeatureGenerator",
        kwargs={
            "max_order": 2,
            "max_base_feats": 16,
            "max_new_feats": 64,
            "passthrough": True,
        },
        description=(
            "add a bounded set of pairwise numeric arithmetic interactions while "
            "preserving the seed features"
        ),
    ),
    GeneratorExperiment(
        name="categorical_interaction",
        class_name="CategoricalInteractionFeatureGenerator",
        kwargs={
            "max_order": 2,
            "max_new_feats": 32,
            "passthrough": True,
        },
        description=(
            "add bounded pairwise categorical interactions while preserving the "
            "seed features"
        ),
    ),
    GeneratorExperiment(
        name="oof_target_encoding",
        class_name="OOFTargetEncodingFeatureGenerator",
        kwargs={
            "n_splits": 5,
            "alpha": 10.0,
            "keep_original": False,
            "passthrough": True,
        },
        description=(
            "add smoothed five-fold out-of-fold target encodings while preserving "
            "the seed features"
        ),
    ),
    GeneratorExperiment(
        name="frequency",
        class_name="FrequencyFeatureGenerator",
        kwargs={
            "only_categorical": True,
            "keep_original": False,
            "passthrough": True,
        },
        description=(
            "add per-column categorical frequencies while preserving the seed "
            "features"
        ),
    ),
)


def is_autogluon_generator_mode(cfg: Any) -> bool:
    return getattr(cfg.agent, "mode", "legacy") == AGENT_MODE


def experiment_name_from_plan(plan: str | None) -> str | None:
    text = str(plan or "")
    prefix = f"{PLAN_PREFIX} ["
    if not text.startswith(prefix):
        return None
    name, separator, _rest = text[len(prefix) :].partition("]:")
    return name if separator and name else None


def completed_experiment_names(journal: Journal) -> set[str]:
    return {
        name
        for node in journal.nodes
        if (name := experiment_name_from_plan(node.plan)) is not None
    }


def next_generator_experiment(journal: Journal) -> GeneratorExperiment:
    completed = completed_experiment_names(journal)
    for experiment in GENERATOR_EXPERIMENTS:
        if experiment.name not in completed:
            return experiment
    raise ValueError("All configured AutoGluon generator experiments are complete.")


def generator_seed_node(journal: Journal) -> Node:
    candidates = [
        node
        for node in journal.nodes
        if node.parent is None
        and isinstance(node.run_stats, dict)
        and node.run_stats.get("seeded_from_manifest")
    ]
    if len(candidates) != 1:
        raise ValueError(
            "agent.mode=ag_generator requires exactly one artifact seed root."
        )
    return candidates[0]


def experiment_for_node(node: Node) -> GeneratorExperiment | None:
    name = experiment_name_from_plan(node.plan)
    if name is None:
        return None
    return next(
        (experiment for experiment in GENERATOR_EXPERIMENTS if experiment.name == name),
        None,
    )


def _autogluon_config_assignment(tree: ast.Module) -> ast.Assign:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "AIDE_AG_CONFIG":
            return node
    raise ValueError("Seed solution does not define a literal AIDE_AG_CONFIG.")


def _runtime_injection(experiment: GeneratorExperiment) -> str:
    return f'''
from autogluon.features.generators import {experiment.class_name} as _AIDE_GENERATOR_CLASS


def _aide_configure_generator_experiment():
    experiment = AIDE_AG_CONFIG["ag_generator_experiment"]
    hyperparameters = AIDE_AG_CONFIG.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        model_types = AIDE_AG_CONFIG.get("included_model_types") or []
        if not model_types:
            raise ValueError(
                "ag_generator requires explicit hyperparameters or included_model_types"
            )
        hyperparameters = {{model_type: [{{}}] for model_type in model_types}}
        AIDE_AG_CONFIG["hyperparameters"] = hyperparameters

    for model_type, raw_configs in hyperparameters.items():
        configs = raw_configs if isinstance(raw_configs, list) else [raw_configs]
        for model_config in configs:
            if not isinstance(model_config, dict):
                raise ValueError(
                    f"ag_generator cannot configure {{model_type}} model entry "
                    f"of type {{type(model_config).__name__}}"
                )
            ag_args_fit = dict(model_config.get("ag_args_fit") or {{}})
            if ag_args_fit.get("model_specific_feature_generator_kwargs") is not None:
                raise ValueError(
                    "ag_generator seed already defines model-specific feature generators"
                )
            ag_args_fit["model_specific_feature_generator_kwargs"] = {{
                "feature_generators": [
                    [_AIDE_GENERATOR_CLASS, dict(experiment["kwargs"])]
                ]
            }}
            model_config["ag_args_fit"] = ag_args_fit


_aide_configure_generator_experiment()
'''.strip()


def build_generator_variant(
    seed_code: str,
    experiment: GeneratorExperiment,
) -> str:
    try:
        tree = ast.parse(seed_code)
    except SyntaxError as exc:
        raise ValueError(f"Seed solution is not valid Python: {exc}") from exc

    assignment = _autogluon_config_assignment(tree)
    try:
        config = ast.literal_eval(assignment.value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Seed AIDE_AG_CONFIG must be a literal mapping.") from exc
    if not isinstance(config, dict):
        raise ValueError("Seed AIDE_AG_CONFIG must be a literal mapping.")
    if "ag_generator_experiment" in config:
        raise ValueError("Seed solution is already an ag_generator experiment.")

    config["ag_generator_experiment"] = experiment.metadata()
    replacement = "AIDE_AG_CONFIG = " + pprint.pformat(
        config,
        sort_dicts=True,
        width=88,
    )
    lines = seed_code.splitlines()
    start = assignment.lineno - 1
    end = assignment.end_lineno
    variant_lines = (
        lines[:start]
        + replacement.splitlines()
        + ["", _runtime_injection(experiment), ""]
        + lines[end:]
    )
    variant = "\n".join(variant_lines)
    if seed_code.endswith("\n"):
        variant += "\n"
    compile(variant, "<ag-generator-variant>", "exec")
    return variant
