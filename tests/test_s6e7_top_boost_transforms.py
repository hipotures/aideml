from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "logs"
    / "experiments"
    / "s6e7_top_boost_transforms.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "s6e7_top_boost_transforms",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "diet_type": ["veg", "veg", "balanced"],
            "stress_level": ["low", "high", "medium"],
            "sleep_quality": ["good", "poor", "average"],
            "physical_activity_level": ["active", "sedentary", "moderate"],
            "smoking_alcohol": ["no", "yes", "occasional"],
            "gender": ["female", "female", "male"],
            "sleep_duration": [8.0, None, 6.5],
            "heart_rate": [70.0, 80.2, 90.0],
            "bmi": [22.0, 30.5, None],
            "calorie_expenditure": [2200.0, 1800.0, 2500.5],
            "step_count": [10000.0, 5000.0, 12000.0],
            "exercise_duration": [60.0, 0.0, 45.5],
            "water_intake": [2.5, 1.0, None],
        }
    )
    aux = df.copy()
    aux["health_condition"] = ["fit", "unhealthy", "at-risk"]
    return df, aux


def test_lightgbm_transform_adds_decimal_signature_features():
    module = _load_module()
    df, aux = _sample_frames()

    out = module.transform_for_lightgbm(df, aux)

    assert "categorical_profile_frequency" in out
    assert "sleep_duration_median_imputed" in out
    assert "sleep_duration_abs_decimal_residual" in out
    assert "sleep_duration_is_integer_like" in out
    assert "sleep_duration_is_tenth_like" in out
    assert out.loc[0, "sleep_duration_abs_decimal_residual"] == 0.0
    assert out.loc[0, "sleep_duration_is_integer_like"] == 1.0
    assert out.loc[2, "sleep_duration_is_tenth_like"] == 1.0


def test_catboost_transform_adds_integer_grid_and_boundary_features():
    module = _load_module()
    df, aux = _sample_frames()

    out = module.transform_for_catboost(df, aux)

    assert "numeric_exact_boundary_hit_count" in out
    assert "numeric_integer_grid_hit_fraction" in out
    assert out.loc[0, "numeric_exact_boundary_hit_count"] == 5.0
    assert round(float(out.loc[0, "numeric_integer_grid_hit_fraction"]), 6) == round(
        6 / 7,
        6,
    )


def test_xgboost_transform_adds_aux_profile_rates_and_numeric_distances():
    module = _load_module()
    df, aux = _sample_frames()

    out = module.transform_for_xgboost(df, aux)

    expected_columns = {
        "diet_type_frequency",
        "diet_type_ordinal",
        "sleep_duration_zscore",
        "healthy_behavior_score",
        "aux_profile_rate_fit",
        "aux_diet_type_rate_fit",
        "aux_numeric_distance_fit",
    }
    assert expected_columns.issubset(out.columns)
    assert round(float(out.loc[0, "diet_type_frequency"]), 6) == round(2 / 3, 6)
    assert out.loc[0, "diet_type_ordinal"] == 1.0
    assert out.loc[0, "aux_profile_rate_fit"] == 1.0
    assert out.loc[1, "aux_profile_rate_unhealthy"] == 1.0
    assert out["aux_numeric_distance_fit"].notna().all()


def test_parse_args_defaults_to_full_s6e7_experiment():
    module = _load_module()

    args = module.parse_args([])

    assert args.algorithm == "all"
    assert args.data_dir.name == "playground-series-s6e7"
    assert args.output_dir.name == "s6e7_top_boost_transforms"


def test_run_algorithm_trains_scores_and_writes_submission(tmp_path):
    module = _load_module()
    df, aux = _sample_frames()
    train = df.copy()
    train.insert(0, "id", [1, 2, 3])
    train["health_condition"] = ["fit", "unhealthy", "at-risk"]
    test = df.iloc[:2].copy()
    test.insert(0, "id", [10, 11])
    sample_submission = pd.DataFrame(
        {"id": [10, 11], "health_condition": ["fit", "fit"]}
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)
    aux.to_csv(data_dir / "student_health_dataset_50k.csv", index=False)
    output_dir = tmp_path / "out"
    created_predictors = []

    class FakePredictor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_kwargs = None
            created_predictors.append(self)

        def fit(self, **kwargs):
            self.fit_kwargs = kwargs
            return self

        def evaluate(self, valid_data, silent=True):
            assert silent is True
            assert "health_condition" in valid_data.columns
            return {"balanced_accuracy": 0.75}

        def predict(self, frame):
            assert "id" not in frame.columns
            assert "health_condition" not in frame.columns
            return pd.Series(["fit"] * len(frame))

        def leaderboard(self, silent=True):
            assert silent is True
            return pd.DataFrame(
                [{"model": "LightGBM", "score_val": 0.75, "stack_level": 1}]
            )

    def predictor_factory(**kwargs):
        return FakePredictor(**kwargs)

    result = module.run_algorithm(
        "lightgbm",
        data_dir=data_dir,
        output_dir=output_dir,
        predictor_factory=predictor_factory,
        time_limit=5,
        use_gpu=False,
    )

    assert result.algorithm == "lightgbm"
    assert result.cv_score == 0.75
    assert result.submission_path == output_dir / "lightgbm" / "submission.csv"
    assert result.metrics_path == output_dir / "lightgbm" / "metrics.json"
    submission = pd.read_csv(result.submission_path)
    assert submission.to_dict("list") == {
        "id": [10, 11],
        "health_condition": ["fit", "fit"],
    }
    predictor = created_predictors[0]
    assert predictor.kwargs["label"] == "health_condition"
    assert predictor.kwargs["eval_metric"] == "balanced_accuracy"
    assert predictor.fit_kwargs["included_model_types"] == ["GBM"]
    assert predictor.fit_kwargs["hyperparameters"] == {"GBM": [{}]}
    assert predictor.fit_kwargs["time_limit"] == 5
    assert "id" not in predictor.fit_kwargs["train_data"].columns
    assert "id" not in predictor.fit_kwargs["tuning_data"].columns
