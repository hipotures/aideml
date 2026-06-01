from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


DEFAULT_DATA_DIR = Path("aide/example_tasks/playground-series-s6e6")
TARGET = "class"
ID_COL = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone AutoGluon baseline for playground-series-s6e6."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("submission_autogluon.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("autogluon_baseline_models"))
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    gz_path = data_dir / f"{name}.csv.gz"
    csv_path = data_dir / f"{name}.csv"
    if gz_path.exists():
        return pd.read_csv(gz_path)
    return pd.read_csv(csv_path)


def main() -> None:
    args = parse_args()
    train = read_csv(args.data_dir, "train")
    test = read_csv(args.data_dir, "test")
    sample_submission = read_csv(args.data_dir, "sample_submission")

    train = train.drop(columns=[ID_COL])
    test_features = test.drop(columns=[ID_COL])

    train_data, valid_data = train_test_split(
        train,
        test_size=0.2,
        random_state=args.seed,
        stratify=train[TARGET],
    )

    predictor = TabularPredictor(
        label=TARGET,
        problem_type="multiclass",
        eval_metric="balanced_accuracy",
        path=str(args.model_dir),
        verbosity=2,
    ).fit(
        train_data=train_data,
        tuning_data=valid_data,
        presets=args.presets,
        time_limit=args.time_limit,
    )

    valid_pred = predictor.predict(valid_data.drop(columns=[TARGET]))
    score = balanced_accuracy_score(valid_data[TARGET], valid_pred)
    print(f"Validation balanced accuracy: {score:.5f}")

    test_pred = predictor.predict(test_features)
    submission = sample_submission.copy()
    submission[TARGET] = test_pred.to_numpy()
    submission.to_csv(args.output, index=False)
    print(f"Submission written to: {args.output}")


if __name__ == "__main__":
    main()
