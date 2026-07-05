from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd


CATEGORICAL_COLUMNS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

NUMERIC_COLUMNS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

HEALTH_LABELS = ["at-risk", "unhealthy", "fit"]
MISSING_TOKEN = "__missing__"

TOP_BOOSTER_RESULTS = {
    "xgboost": {
        "run": "2-whimsical-albatross-from-camelot",
        "step": 29,
        "artifact": "20260705T211843-69dd79dc-29",
        "score_val": 0.950221642180,
        "transform": "auxiliary priors, category frequencies, ordinal maps, numeric robustness, and health composites",
    },
    "lightgbm": {
        "run": "2-whimsical-albatross-from-camelot",
        "step": 11,
        "artifact": "20260705T205955-28cad5f3-11",
        "score_val": 0.950617827755,
        "transform": "numeric decimal-part signatures",
    },
    "catboost": {
        "run": "2-whimsical-albatross-from-camelot",
        "step": 86,
        "artifact": "20260705T233447-83fbcb9f-86",
        "score_val": 0.950136772236,
        "transform": "integer-grid fraction with exact numeric boundary count",
    },
}


def transform_for_xgboost(df: pd.DataFrame, aux: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature set from the best XGBoost node in the s6e7 run."""
    out = df.copy()
    _require_columns(out, CATEGORICAL_COLUMNS + NUMERIC_COLUMNS)

    profile = _normalized_profile(out)
    _add_categorical_frequency_features(out, profile)
    _add_ordinal_features(out, profile)
    _coerce_numeric_columns(out)
    _add_numeric_robustness_features(out)
    _add_health_behavior_features(out)

    if aux is not None and "health_condition" in aux.columns:
        _add_auxiliary_rate_features(out, profile, aux)
        _add_auxiliary_numeric_distances(out, aux)

    return out


def transform_for_lightgbm(
    df: pd.DataFrame,
    aux: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature set from the best LightGBM node in the s6e7 run."""
    del aux
    out = df.copy()
    _require_columns(out, CATEGORICAL_COLUMNS + NUMERIC_COLUMNS)

    profile = _normalized_profile(out)
    out["categorical_profile_frequency"] = _profile_frequency(profile)
    _coerce_numeric_columns(out)
    medians = out[NUMERIC_COLUMNS].median()
    for col in NUMERIC_COLUMNS:
        filled = out[col].fillna(medians[col]).astype("float32")
        out[f"{col}_median_imputed"] = filled
        nearest_int = filled.round(0)
        nearest_tenth = (filled * 10.0).round(0) / 10.0
        out[f"{col}_abs_decimal_residual"] = (
            (filled - nearest_int).abs().astype("float32")
        )
        out[f"{col}_is_integer_like"] = ((filled - nearest_int).abs() < 1e-6).astype(
            "float32"
        )
        out[f"{col}_is_tenth_like"] = ((filled - nearest_tenth).abs() < 1e-6).astype(
            "float32"
        )

    return out


def transform_for_catboost(
    df: pd.DataFrame,
    aux: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature set from the best CatBoost node in the s6e7 run."""
    del aux
    out = df.copy()
    _require_columns(out, CATEGORICAL_COLUMNS + NUMERIC_COLUMNS)

    profile = _normalized_profile(out)
    out["categorical_profile_frequency"] = _profile_frequency(profile)
    _coerce_numeric_columns(out)
    medians = out[NUMERIC_COLUMNS].median()
    for col in NUMERIC_COLUMNS:
        out[f"{col}_median_imputed"] = out[col].fillna(medians[col]).astype("float32")

    observed = out[NUMERIC_COLUMNS]
    col_min = observed.min()
    col_max = observed.max()
    boundary_hits = observed.eq(col_min) | observed.eq(col_max)
    out["numeric_exact_boundary_hit_count"] = boundary_hits.sum(axis=1).astype(
        "float32"
    )

    nonmissing = observed.notna()
    integer_hits = observed.sub(observed.round()).abs().le(1e-6) & nonmissing
    denom = nonmissing.sum(axis=1).replace(0, 1)
    out["numeric_integer_grid_hit_fraction"] = (
        integer_hits.sum(axis=1) / denom
    ).astype("float32")

    return out


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")


def _normalized_profile(frame: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame(index=frame.index)
    for col in CATEGORICAL_COLUMNS:
        profile[col] = (
            frame[col].astype("string").str.strip().str.lower().fillna(MISSING_TOKEN)
        )
    return profile


def _profile_frequency(profile: pd.DataFrame) -> pd.Series:
    if len(profile) == 0:
        return pd.Series(dtype="float32", index=profile.index)
    profile_size = profile.groupby(CATEGORICAL_COLUMNS, sort=False)[
        CATEGORICAL_COLUMNS[0]
    ].transform("size")
    return (profile_size.astype("float32") / float(len(profile))).astype("float32")


def _add_categorical_frequency_features(
    out: pd.DataFrame,
    profile: pd.DataFrame,
) -> None:
    for col in CATEGORICAL_COLUMNS:
        out[f"{col}_missing"] = out[col].isna().astype("int8")
        out[f"{col}_frequency"] = (
            profile[col].map(profile[col].value_counts(normalize=True)).astype("float32")
        )
    out["categorical_profile_frequency"] = _profile_frequency(profile)


def _add_ordinal_features(out: pd.DataFrame, profile: pd.DataFrame) -> None:
    ordinal_maps = {
        "diet_type": {"non-veg": 0.0, "veg": 1.0, "balanced": 2.0},
        "stress_level": {"low": 0.0, "medium": 1.0, "high": 2.0},
        "sleep_quality": {"poor": 0.0, "average": 1.0, "good": 2.0},
        "physical_activity_level": {"sedentary": 0.0, "moderate": 1.0, "active": 2.0},
        "smoking_alcohol": {"no": 0.0, "occasional": 1.0, "yes": 2.0},
    }
    for col, mapping in ordinal_maps.items():
        out[f"{col}_ordinal"] = profile[col].map(mapping).fillna(-1.0).astype("float32")


def _coerce_numeric_columns(out: pd.DataFrame) -> None:
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")


def _add_numeric_robustness_features(out: pd.DataFrame) -> None:
    medians = out[NUMERIC_COLUMNS].median()
    for col in NUMERIC_COLUMNS:
        clean = out[col].fillna(medians[col])
        out[f"{col}_median_imputed"] = clean.astype("float32")
        out[f"{col}_missing"] = out[col].isna().astype("int8")
        spread = clean.std()
        if pd.notna(spread) and float(spread) > 0.0:
            out[f"{col}_zscore"] = ((clean - clean.mean()) / spread).astype("float32")
        else:
            out[f"{col}_zscore"] = 0.0


def _add_health_behavior_features(out: pd.DataFrame) -> None:
    sleep = out["sleep_duration_median_imputed"]
    heart_rate = out["heart_rate_median_imputed"]
    bmi = out["bmi_median_imputed"]
    calories = out["calorie_expenditure_median_imputed"]
    steps = out["step_count_median_imputed"]
    exercise = out["exercise_duration_median_imputed"]
    water = out["water_intake_median_imputed"]

    out["sleep_deficit"] = (7.0 - sleep).clip(lower=0.0).astype("float32")
    out["sleep_excess"] = (sleep - 9.0).clip(lower=0.0).astype("float32")
    out["bmi_under_healthy_distance"] = (18.5 - bmi).clip(lower=0.0).astype("float32")
    out["bmi_over_healthy_distance"] = (bmi - 24.9).clip(lower=0.0).astype("float32")
    out["bmi_healthy_distance"] = (
        out["bmi_under_healthy_distance"] + out["bmi_over_healthy_distance"]
    ).astype("float32")
    out["calories_per_1000_steps"] = (calories / (steps / 1000.0 + 1.0)).astype(
        "float32"
    )
    out["steps_per_exercise_minute"] = (steps / (exercise + 1.0)).astype("float32")
    out["exercise_minutes_per_sleep_hour"] = (exercise / (sleep + 1.0)).astype(
        "float32"
    )
    out["water_per_bmi"] = (water / bmi.clip(lower=1.0)).astype("float32")
    out["heart_rate_bmi_load"] = (heart_rate * bmi / 100.0).astype("float32")
    out["activity_volume"] = (steps / 10000.0 + exercise / 60.0).astype("float32")
    out["healthy_behavior_score"] = (
        out["diet_type_ordinal"]
        + out["sleep_quality_ordinal"]
        + out["physical_activity_level_ordinal"]
        + steps / 10000.0
        + exercise / 60.0
        + water / 2.5
        - out["stress_level_ordinal"]
        - out["smoking_alcohol_ordinal"]
        - out["bmi_healthy_distance"] / 5.0
    ).astype("float32")


def _add_auxiliary_rate_features(
    out: pd.DataFrame,
    profile: pd.DataFrame,
    aux: pd.DataFrame,
) -> None:
    _require_columns(aux, CATEGORICAL_COLUMNS + ["health_condition"])
    aux_profile = _normalized_profile(aux)
    aux_labels = aux["health_condition"].astype("string").str.strip().str.lower()
    aux_work = aux_profile.copy()
    aux_work["health_condition"] = aux_labels
    aux_global_rates = aux_work["health_condition"].value_counts(normalize=True)

    aux_profile_rates = (
        aux_work.groupby(CATEGORICAL_COLUMNS + ["health_condition"], sort=False)
        .size()
        .unstack("health_condition", fill_value=0)
    )
    aux_profile_rates = aux_profile_rates.div(
        aux_profile_rates.sum(axis=1), axis=0
    ).reset_index()
    for label in HEALTH_LABELS:
        if label not in aux_profile_rates.columns:
            aux_profile_rates[label] = 0.0
    aux_profile_rates = aux_profile_rates[CATEGORICAL_COLUMNS + HEALTH_LABELS].rename(
        columns={
            label: f"aux_profile_rate_{label.replace('-', '_')}"
            for label in HEALTH_LABELS
        }
    )
    merged_profile_rates = profile.reset_index(drop=True).merge(
        aux_profile_rates,
        on=CATEGORICAL_COLUMNS,
        how="left",
        sort=False,
    )
    for label in HEALTH_LABELS:
        feature = f"aux_profile_rate_{label.replace('-', '_')}"
        out[feature] = (
            merged_profile_rates[feature]
            .fillna(float(aux_global_rates.get(label, 0.0)))
            .astype("float32")
            .to_numpy()
        )

    for col in CATEGORICAL_COLUMNS:
        aux_cat_rates = (
            aux_work.groupby([col, "health_condition"], sort=False)
            .size()
            .unstack("health_condition", fill_value=0)
        )
        aux_cat_rates = aux_cat_rates.div(aux_cat_rates.sum(axis=1), axis=0).reset_index()
        for label in HEALTH_LABELS:
            if label not in aux_cat_rates.columns:
                aux_cat_rates[label] = 0.0
        aux_cat_rates = aux_cat_rates[[col] + HEALTH_LABELS].rename(
            columns={
                label: f"aux_{col}_rate_{label.replace('-', '_')}"
                for label in HEALTH_LABELS
            }
        )
        merged_cat_rates = (
            profile[[col]]
            .reset_index(drop=True)
            .merge(aux_cat_rates, on=col, how="left", sort=False)
        )
        for label in HEALTH_LABELS:
            feature = f"aux_{col}_rate_{label.replace('-', '_')}"
            out[feature] = (
                merged_cat_rates[feature]
                .fillna(float(aux_global_rates.get(label, 0.0)))
                .astype("float32")
                .to_numpy()
            )


def _add_auxiliary_numeric_distances(out: pd.DataFrame, aux: pd.DataFrame) -> None:
    _require_columns(aux, NUMERIC_COLUMNS + ["health_condition"])
    aux_labels = aux["health_condition"].astype("string").str.strip().str.lower()
    aux_num = aux[NUMERIC_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        aux_num[col] = pd.to_numeric(aux_num[col], errors="coerce")
    aux_scale = aux_num.std().replace(0.0, 1.0)

    for label in HEALTH_LABELS:
        label_mask = (aux_labels == label).fillna(False)
        centers = aux_num.loc[label_mask, NUMERIC_COLUMNS].median()
        distance = pd.Series(0.0, index=out.index)
        for col in NUMERIC_COLUMNS:
            scale = aux_scale[col]
            if pd.isna(scale) or float(scale) == 0.0:
                scale = 1.0
            center = centers[col]
            if pd.isna(center):
                center = aux_num[col].median()
            distance = distance + (
                (out[f"{col}_median_imputed"] - float(center)).abs() / float(scale)
            )
        out[f"aux_numeric_distance_{label.replace('-', '_')}"] = distance.astype(
            "float32"
        )


TRANSFORMS: dict[str, Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame]] = {
    "xgboost": transform_for_xgboost,
    "lightgbm": transform_for_lightgbm,
    "catboost": transform_for_catboost,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply top s6e7 feature transforms found per boosting algorithm.",
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--algorithm",
        choices=sorted(TRANSFORMS),
        required=True,
    )
    parser.add_argument("--aux-csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.input_csv)
    aux = pd.read_csv(args.aux_csv) if args.aux_csv is not None else None
    transformed = TRANSFORMS[args.algorithm](frame, aux)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(args.output_csv, index=False)
    print(
        f"wrote {args.algorithm} transform: rows={len(transformed)} cols={len(transformed.columns)} "
        f"path={args.output_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
