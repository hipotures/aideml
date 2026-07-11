import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "heldout_prior_power.py"
SPEC = importlib.util.spec_from_file_location("heldout_prior_power", MODULE_PATH)
heldout_prior_power = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(heldout_prior_power)


def test_prior_power_formula_tau_zero_and_label_free_transform():
    probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])
    priors = np.array([0.75, 0.25])

    tau_zero = heldout_prior_power.transform_prior_power(probabilities, priors, 0.0)
    transformed = heldout_prior_power.transform_prior_power(probabilities, priors, 1.0)

    assert np.array_equal(tau_zero, probabilities)
    assert np.allclose(transformed, [[4 / 7, 3 / 7], [0.1, 0.9]])
    assert "target" not in heldout_prior_power.transform_prior_power.__code__.co_varnames
    assert "label" not in heldout_prior_power.transform_prior_power.__code__.co_varnames


def test_prior_power_rejects_invalid_schema_and_tau():
    with pytest.raises(ValueError, match="shape"):
        heldout_prior_power.transform_prior_power([[0.5, 0.5]], [1.0], 0.5)
    with pytest.raises(ValueError, match="tau"):
        heldout_prior_power.transform_prior_power([[0.5, 0.5]], [0.5, 0.5], 0.1)


def test_fixed_holdout_sweep_records_diagnostics_and_source_hashes(tmp_path):
    source = tmp_path / "probabilities.csv.gz"
    pd.DataFrame(
        {
            "row": [8, 3, 5],
            "target": ["a", "b", "a"],
            "a": [0.9, 0.4, 0.6],
            "b": [0.1, 0.6, 0.4],
        }
    ).to_csv(source, index=False, compression="gzip")

    result = heldout_prior_power.score_prior_power_file(
        source, class_counts={"a": 9, "b": 1}
    )

    assert result["kind"] == "fixed_heldout_fold_prior_power_sweep"
    assert result["reproduction"]["labels_used_only_for_scoring"] is True
    assert result["reproduction"]["fold_scope"] == "single_fixed_holdout_not_oof"
    assert result["taus"] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    assert set(result["results"]["0.0"]) == {
        "balanced_accuracy",
        "per_class_recall",
        "confusion_matrix",
        "prediction_counts",
        "prediction_proportions",
    }
    assert len(result["probability_source"]["sha256"]) == 64
    assert len(result["probability_source"]["row_sha256"]) == 64
    assert len(result["probability_source"]["target_sha256"]) == 64
    assert sum(result["results"]["0.0"]["prediction_proportions"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("counts", [{"a": 1.5, "b": 1}, {"a": True, "b": 1}])
def test_fixed_holdout_sweep_rejects_noninteger_class_counts(tmp_path, counts):
    source = tmp_path / "probabilities.csv"
    pd.DataFrame({"row": [0], "target": ["a"], "a": [0.8], "b": [0.2]}).to_csv(
        source, index=False
    )

    with pytest.raises(ValueError, match="positive integers"):
        heldout_prior_power.score_prior_power_file(source, class_counts=counts)


@pytest.mark.parametrize(
    "frame, message",
    [
        (
            pd.DataFrame({"row": [0], "target": ["a"], "a": [0.8], "b": [0.3]}),
            "sum to one",
        ),
        (
            pd.DataFrame({"row": [0, 0], "target": ["a", "b"], "a": [0.8, 0.2], "b": [0.2, 0.8]}),
            "unique",
        ),
        (
            pd.DataFrame({"row": [0], "target": ["other"], "a": [0.8], "b": [0.2]}),
            "class order",
        ),
    ],
)
def test_fixed_holdout_sweep_rejects_bad_probability_schema(tmp_path, frame, message):
    source = tmp_path / "probabilities.csv"
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match=message):
        heldout_prior_power.score_prior_power_file(source, class_counts={"a": 3, "b": 1})


def test_prior_power_cli_writes_machine_readable_output(tmp_path):
    source = tmp_path / "probabilities.csv"
    output = tmp_path / "result.json"
    pd.DataFrame(
        {"row": [0, 1], "target": ["a", "b"], "a": [0.8, 0.2], "b": [0.2, 0.8]}
    ).to_csv(source, index=False)

    assert heldout_prior_power.main(
        [
            "--probabilities",
            str(source),
            "--class-counts-json",
            json.dumps({"a": 3, "b": 1}),
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text())["class_counts"] == {"a": 3, "b": 1}
