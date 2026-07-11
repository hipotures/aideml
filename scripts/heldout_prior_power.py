"""Score fixed-holdout probability exports with a label-free prior-power transform."""
from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score


TAUS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def transform_prior_power(probabilities, priors, tau: float):
    """Return normalize(p / priors**tau); targets are deliberately not an input."""
    p = np.asarray(probabilities, dtype=float)
    pi = np.asarray(priors, dtype=float)
    if p.ndim != 2 or pi.ndim != 1 or p.shape[1] != len(pi):
        raise ValueError("probability/prior shape mismatch")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isfinite(pi).all() or (pi <= 0).any():
        raise ValueError("priors must be finite and positive")
    if tau not in TAUS:
        raise ValueError(f"tau must be one of {list(TAUS)}")
    if tau == 0:
        return p.copy()
    transformed = p / np.power(pi, tau)
    normalizer = transformed.sum(axis=1, keepdims=True)
    if not np.isfinite(normalizer).all() or (normalizer <= 0).any():
        raise ValueError("prior-power normalization is not finite and positive")
    return transformed / normalizer


def score_probabilities(probabilities, targets, labels):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets)
    if len(probabilities) != len(targets):
        raise ValueError("probability rows and targets must have equal length")
    if len(probabilities) == 0:
        raise ValueError("probability rows must not be empty")
    predictions = labels[probabilities.argmax(axis=1)]
    recalls = recall_score(targets, predictions, labels=labels, average=None, zero_division=0)
    prediction_counts = {
        str(label): int((predictions == label).sum()) for label in labels
    }
    return {
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "per_class_recall": {
            str(label): float(recall) for label, recall in zip(labels, recalls)
        },
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "prediction_counts": prediction_counts,
        "prediction_proportions": {
            label: count / len(predictions) for label, count in prediction_counts.items()
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_prior_power_file(
    probability_path: Path,
    *,
    class_counts: dict[str, int],
    target_column: str = "target",
    row_column: str = "row",
) -> dict:
    frame = pd.read_csv(probability_path)
    if target_column not in frame or row_column not in frame:
        raise ValueError("held-out export must include row and target columns")
    if frame[row_column].isna().any() or not frame[row_column].is_unique:
        raise ValueError("held-out export row IDs must be present and unique")
    class_order = list(class_counts)
    if not class_order or any(column not in frame for column in class_order):
        raise ValueError("class-count keys must exactly name probability columns")
    probability_columns = [column for column in frame if column not in {row_column, target_column}]
    if probability_columns != class_order:
        raise ValueError("probability columns must match explicit class-count order")
    if any(
        isinstance(class_counts[label], bool)
        or not isinstance(class_counts[label], Integral)
        or class_counts[label] <= 0
        for label in class_order
    ):
        raise ValueError("class counts must be positive integers")
    counts = np.asarray([class_counts[label] for label in class_order], dtype=float)
    probabilities = frame[class_order].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probability rows must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    if frame[target_column].isna().any() or not set(frame[target_column]).issubset(class_order):
        raise ValueError("target values must be present in class order")
    priors = counts / counts.sum()
    results = {}
    for tau in TAUS:
        transformed = transform_prior_power(probabilities, priors, tau)
        results[str(tau)] = score_probabilities(
            transformed,
            frame[target_column].to_numpy(),
            class_order,
        )
    return {
        "schema_version": 1,
        "kind": "fixed_heldout_fold_prior_power_sweep",
        "probability_source": {
            "path": str(probability_path),
            "sha256": _sha256(probability_path),
            "rows": int(len(frame)),
            "row_sha256": hashlib.sha256(
                json.dumps(frame[row_column].tolist(), separators=(",", ":")).encode()
            ).hexdigest(),
            "target_sha256": hashlib.sha256(
                json.dumps(frame[target_column].tolist(), separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "class_order": class_order,
        "class_counts": {label: int(class_counts[label]) for label in class_order},
        "priors": {label: float(prior) for label, prior in zip(class_order, priors)},
        "taus": list(TAUS),
        "results": results,
        "reproduction": {
            "transform": "normalize(probabilities / priors ** tau)",
            "labels_used_only_for_scoring": True,
            "fold_scope": "single_fixed_holdout_not_oof",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--class-counts-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        counts = json.loads(args.class_counts_json)
        if not isinstance(counts, dict):
            raise ValueError("--class-counts-json must be an object")
        result = score_prior_power_file(args.probabilities, class_counts=counts)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
