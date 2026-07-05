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
