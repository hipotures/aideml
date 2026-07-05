from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, NamedTuple

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
CLASS_WEIGHT_COL = "__aide_class_weight__"
COMBINED_MODEL_NAME = "combined_top_features"
DEFAULT_PROFILE = "s6e6_boost_gpu"
COMPETITION = "playground-series-s6e7"
RESULT_MARKER = "AIDE_RESULT_JSON:"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_DATA_DIR = repo_root() / "aide" / "example_tasks" / "playground-series-s6e7"
DEFAULT_OUTPUT_DIR = repo_root() / "logs" / "experiments"

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
        out = _add_auxiliary_rate_features(out, profile, aux)
        out = _add_auxiliary_numeric_distances(out, aux)

    return out


def transform_combined_top_features(
    df: pd.DataFrame,
    aux: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Union of the top feature families found for XGBoost, LightGBM, and CatBoost."""
    out = df.copy()
    _require_columns(out, CATEGORICAL_COLUMNS + NUMERIC_COLUMNS)

    profile = _normalized_profile(out)
    _add_categorical_frequency_features(out, profile)
    _add_ordinal_features(out, profile)
    _coerce_numeric_columns(out)
    _add_numeric_robustness_features(out)
    _add_health_behavior_features(out)
    _add_decimal_signature_features(out)
    _add_numeric_grid_features(out)

    if aux is not None and "health_condition" in aux.columns:
        out = _add_auxiliary_rate_features(out, profile, aux)
        out = _add_auxiliary_numeric_distances(out, aux)

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
    _add_decimal_signature_features(out)

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
    _add_numeric_grid_features(out)

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


def _add_decimal_signature_features(out: pd.DataFrame) -> None:
    medians = out[NUMERIC_COLUMNS].median()
    for col in NUMERIC_COLUMNS:
        imputed_col = f"{col}_median_imputed"
        if imputed_col in out.columns:
            filled = out[imputed_col].astype("float32")
        else:
            filled = out[col].fillna(medians[col]).astype("float32")
            out[imputed_col] = filled
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


def _add_numeric_grid_features(out: pd.DataFrame) -> None:
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
) -> pd.DataFrame:
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
    new_features = {}
    for label in HEALTH_LABELS:
        feature = f"aux_profile_rate_{label.replace('-', '_')}"
        new_features[feature] = (
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
            new_features[feature] = (
                merged_cat_rates[feature]
                .fillna(float(aux_global_rates.get(label, 0.0)))
                .astype("float32")
                .to_numpy()
            )
    return pd.concat([out, pd.DataFrame(new_features, index=out.index)], axis=1).copy()


def _add_auxiliary_numeric_distances(out: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    _require_columns(aux, NUMERIC_COLUMNS + ["health_condition"])
    aux_labels = aux["health_condition"].astype("string").str.strip().str.lower()
    aux_num = aux[NUMERIC_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        aux_num[col] = pd.to_numeric(aux_num[col], errors="coerce")
    aux_scale = aux_num.std().replace(0.0, 1.0)

    new_features = {}
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
        new_features[f"aux_numeric_distance_{label.replace('-', '_')}"] = (
            distance.astype("float32").to_numpy()
        )
    return pd.concat([out, pd.DataFrame(new_features, index=out.index)], axis=1).copy()


TRANSFORMS: dict[str, Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame]] = {
    "xgboost": transform_for_xgboost,
    "lightgbm": transform_for_lightgbm,
    "catboost": transform_for_catboost,
}


class ExperimentResult(NamedTuple):
    name: str
    profile: str
    cv_score: float
    submission_path: Path
    artifact_dir: Path
    artifact_submission_path: Path
    metrics_path: Path
    model_dir: Path
    feature_count: int
    index_path: Path | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one s6e7 AutoGluon model using the combined top feature "
            "families found in 2-whimsical-albatross-from-camelot."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train/test/sample_submission and optional aux CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="AIDE log run directory where model, metrics, submission, and artifact are written.",
    )
    parser.add_argument(
        "--profile",
        default=_configured_default_profile(),
        help="AutoGluon profile name from aide/utils/config.yaml.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print available AutoGluon profiles and exit without training.",
    )
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--presets", default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Disable GPU hyperparameter settings and run AutoGluon on CPU.",
    )
    return parser.parse_args(argv)


def available_autogluon_profiles() -> list[str]:
    cfg = _load_aide_cfg()
    profiles = getattr(getattr(cfg.agent, "autogluon"), "profiles")
    return sorted(str(name) for name in profiles.keys())


def _configured_default_profile() -> str:
    try:
        cfg = _load_aide_cfg()
        profile = str(getattr(getattr(cfg.agent, "autogluon"), "profile"))
    except Exception:
        return DEFAULT_PROFILE
    return profile or DEFAULT_PROFILE


def _load_aide_cfg() -> Any:
    from aide.utils.config import _load_cfg

    return _load_cfg(use_cli_args=False)


def _resolve_autogluon_profile(
    profile: str,
    *,
    time_limit: int | None,
    presets: str | None,
    force_cpu: bool,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = _load_aide_cfg()
    profiles = getattr(getattr(cfg.agent, "autogluon"), "profiles")
    known_profiles = available_autogluon_profiles()
    if profile not in known_profiles:
        available = "\n".join(f"  - {name}" for name in known_profiles)
        raise ValueError(f"Unknown AutoGluon profile: {profile}\nAvailable profiles:\n{available}")

    settings = OmegaConf.to_container(profiles[profile], resolve=True)
    if not isinstance(settings, dict):
        settings = {}
    resolved = dict(settings)
    resolved["profile"] = profile
    if time_limit is not None:
        resolved["time_limit"] = time_limit
    if presets is not None:
        resolved["presets"] = presets
    if force_cpu:
        resolved["use_gpu"] = False
    resolved.setdefault("included_model_types", ["XGB", "GBM", "CAT"])
    resolved.setdefault("presets", "medium_quality")
    resolved.setdefault("time_limit", 600)
    resolved.setdefault("class_balance", "balanced")
    resolved.setdefault("fit_args", {})
    resolved["included_model_types"] = [
        str(model_type) for model_type in resolved["included_model_types"]
    ]
    validation_strategy = resolved.get("validation_strategy")
    if validation_strategy not in {None, "holdout", "autogluon"}:
        raise ValueError(
            "AutoGluon validation_strategy must be one of: holdout, autogluon"
        )
    return resolved


def run_combined_model(
    *,
    data_dir: Path,
    output_dir: Path,
    predictor_factory: Callable[..., Any] | None = None,
    profile: str = DEFAULT_PROFILE,
    time_limit: int | None = None,
    presets: str | None = None,
    validation_fraction: float = 0.2,
    seed: int = 42,
    use_gpu: bool | None = None,
    refresh_index: bool = True,
    index_refresher: Callable[..., Any] | None = None,
    timestamp: str | None = None,
) -> ExperimentResult:
    started_at = time.monotonic()
    predictor_factory = predictor_factory or _default_predictor_factory
    profile_settings = _resolve_autogluon_profile(
        profile,
        time_limit=time_limit,
        presets=presets,
        force_cpu=use_gpu is False,
    )
    if use_gpu is True:
        profile_settings["use_gpu"] = True
    profile = str(profile_settings["profile"])
    presets = str(profile_settings["presets"])
    time_limit = int(profile_settings["time_limit"])
    profile_use_gpu = bool(profile_settings.get("use_gpu", False))
    included_model_types = list(profile_settings["included_model_types"])
    fit_args = dict(profile_settings.get("fit_args") or {})
    class_balance = profile_settings.get("class_balance")
    sample_weight_col = CLASS_WEIGHT_COL if class_balance == "balanced" else None

    _log(
        "profile="
        f"{profile} presets={presets} time_limit={time_limit} "
        f"included_model_types={','.join(included_model_types)} "
        f"use_gpu={profile_use_gpu}"
    )
    _log(f"data_dir={data_dir}")
    _log(f"output_dir={output_dir}")

    train_df = _read_input_csv(data_dir, "train")
    test_df = _read_input_csv(data_dir, "test")
    sample_submission = _read_input_csv(data_dir, "sample_submission")
    aux_df = _read_optional_input_csv(data_dir, "student_health_dataset_50k")
    _log(
        f"loaded train rows={len(train_df)} cols={len(train_df.columns)} "
        f"test rows={len(test_df)} sample rows={len(sample_submission)}"
    )
    if aux_df is not None:
        _log(f"loaded aux student_health_dataset_50k rows={len(aux_df)} cols={len(aux_df.columns)}")
    else:
        _log("no auxiliary student_health_dataset_50k file found")

    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]
    if target_col not in train_df.columns:
        raise ValueError(f"Target column {target_col!r} not found in train data")

    y_train = train_df[target_col].reset_index(drop=True)
    train_features = train_df.drop(columns=[target_col, id_col], errors="ignore")
    test_features = test_df.drop(columns=[id_col], errors="ignore")
    combined = pd.concat([train_features, test_features], ignore_index=True, sort=False)
    _log("starting preprocess: combined top XGBoost + LightGBM + CatBoost feature families")
    transformed = transform_combined_top_features(combined, aux_df)
    _validate_transformed(transformed, combined)
    _log(f"finished preprocess rows={len(transformed)} cols={len(transformed.columns)}")

    train_fe = transformed.iloc[: len(train_df)].copy()
    test_fe = transformed.iloc[len(train_df) :].copy()
    train_model = train_fe.copy()
    train_model[target_col] = y_train.to_numpy()
    if sample_weight_col is not None:
        train_model[sample_weight_col] = _balanced_sample_weight(y_train)

    train_part, valid_part, validation_strategy_resolved = _training_frames(
        train_model,
        target_col=target_col,
        validation_fraction=validation_fraction,
        seed=seed,
        validation_strategy=profile_settings.get("validation_strategy"),
        presets=presets,
        fit_args=fit_args,
    )

    run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or _artifact_timestamp(run_dir / "artifacts")
    model_dir = run_dir / "models" / timestamp
    shutil.rmtree(model_dir, ignore_errors=True)
    predictor_kwargs = dict(
        label=target_col,
        eval_metric="balanced_accuracy",
        path=str(model_dir),
        verbosity=2,
        weight_evaluation=False,
    )
    if sample_weight_col is not None:
        predictor_kwargs["sample_weight"] = sample_weight_col
    predictor = predictor_factory(**predictor_kwargs)
    fit_kwargs = {
        "train_data": train_part,
        "presets": presets,
        "time_limit": time_limit,
        "included_model_types": included_model_types,
        "hyperparameters": _profile_hyperparameters(
            profile_settings,
            included_model_types=included_model_types,
            use_gpu=profile_use_gpu,
        ),
        "num_gpus": 1 if profile_use_gpu else 0,
    }
    if valid_part is not None:
        fit_kwargs["tuning_data"] = valid_part
    fit_kwargs.update(fit_args)
    _log("starting AutoGluon fit")
    predictor.fit(**fit_kwargs)
    _log("finished AutoGluon fit")

    _log("starting validation and test prediction")
    leaderboard = predictor.leaderboard(silent=True)
    if valid_part is not None:
        scores = predictor.evaluate(valid_part, silent=True)
        cv_score = float(scores["balanced_accuracy"])
    else:
        cv_score = _score_from_leaderboard(leaderboard)
    test_pred = predictor.predict(test_fe).reset_index(drop=True)
    submission = sample_submission.copy()
    submission[target_col] = test_pred.to_numpy()
    submission_path = run_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    _log(f"submission saved to {submission_path}")

    elapsed = time.monotonic() - started_at
    artifact_dir = run_dir / "artifacts" / timestamp
    artifact_submission_path = artifact_dir / "submission.csv"
    metrics = {
        "name": COMBINED_MODEL_NAME,
        "competition": COMPETITION,
        "profile": profile,
        "sources": TOP_BOOSTER_RESULTS,
        "cv_score": cv_score,
        "eval_metric": "balanced_accuracy",
        "feature_count": int(len(train_fe.columns)),
        "train_rows": int(len(train_part)),
        "validation_rows": int(len(valid_part)) if valid_part is not None else 0,
        "test_rows": int(len(test_fe)),
        "submission_path": str(submission_path),
        "artifact_submission_path": str(artifact_submission_path),
        "model_dir": str(model_dir),
        "autogluon": {
            "profile": profile,
            "presets": presets,
            "time_limit": time_limit,
            "included_model_types": included_model_types,
            "use_gpu": profile_use_gpu,
            "class_balance": class_balance,
            "fit_args": fit_args,
            "hyperparameters": fit_kwargs["hyperparameters"],
            "resolved_settings": profile_settings,
        },
        "run_stats": {
            "exec_time": elapsed,
            "feature_count": int(len(train_fe.columns)),
            "eval_metric": "balanced_accuracy",
            "validation_strategy": profile_settings.get("validation_strategy"),
            "validation_strategy_resolved": validation_strategy_resolved,
        },
        "leaderboard": _records_from_frame(leaderboard),
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    artifact_dir = _write_artifact_bundle(
        artifact_dir=artifact_dir,
        submission=submission,
        metrics=metrics,
        ag_config=metrics["autogluon"],
    )
    _log(f"artifact saved to {artifact_dir}")
    index_path = _refresh_submission_index(
        run_dir=run_dir,
        enabled=refresh_index,
        index_refresher=index_refresher,
    )
    if index_path is not None:
        _log(f"submission index refreshed at {index_path}")
    else:
        _log("submission index not refreshed")

    return ExperimentResult(
        name=COMBINED_MODEL_NAME,
        profile=profile,
        cv_score=cv_score,
        submission_path=submission_path,
        artifact_dir=artifact_dir,
        artifact_submission_path=artifact_submission_path,
        metrics_path=metrics_path,
        model_dir=model_dir,
        feature_count=int(len(train_fe.columns)),
        index_path=index_path,
    )


def _default_predictor_factory(**kwargs: Any) -> Any:
    from autogluon.tabular import TabularPredictor

    return TabularPredictor(**kwargs)


def _read_input_csv(data_dir: Path, stem: str) -> pd.DataFrame:
    path = _input_csv_path(data_dir, stem)
    if path is None:
        raise FileNotFoundError(
            f"Could not find {stem}.csv or {stem}.csv.gz under {data_dir}"
        )
    return pd.read_csv(path)


def _read_optional_input_csv(data_dir: Path, stem: str) -> pd.DataFrame | None:
    path = _input_csv_path(data_dir, stem)
    if path is None:
        return None
    return pd.read_csv(path)


def _input_csv_path(data_dir: Path, stem: str) -> Path | None:
    for suffix in (".csv", ".csv.gz"):
        path = data_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _validate_transformed(transformed: pd.DataFrame, original: pd.DataFrame) -> None:
    if len(transformed) != len(original):
        raise ValueError(
            f"Transform changed row count: original={len(original)} transformed={len(transformed)}"
        )
    forbidden = {"id", "health_condition", CLASS_WEIGHT_COL}
    overlap = forbidden.intersection(transformed.columns)
    if overlap:
        raise ValueError(f"Transform created forbidden columns: {sorted(overlap)}")


def _holdout_split(
    frame: pd.DataFrame,
    *,
    target_col: str,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    labels = frame[target_col]
    counts = labels.value_counts(dropna=False)
    stratify = labels if len(counts) > 1 and int(counts.min()) >= 2 else None
    return train_test_split(
        frame,
        test_size=validation_fraction,
        random_state=seed,
        stratify=stratify,
    )


def _training_frames(
    frame: pd.DataFrame,
    *,
    target_col: str,
    validation_fraction: float,
    seed: int,
    validation_strategy: str | None,
    presets: str,
    fit_args: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    if _autogluon_bagged_mode(presets=presets, fit_args=fit_args):
        _log(
            "bagged mode detected; using AutoGluon internal OOF validation "
            "without tuning_data"
        )
        return frame, None, "autogluon_oof"
    if validation_strategy == "holdout":
        train_part, valid_part = _holdout_split(
            frame,
            target_col=target_col,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        return train_part, valid_part, "holdout"

    _log("using AutoGluon internal validation without tuning_data")
    return frame, None, "autogluon"


def _autogluon_bagged_mode(*, presets: str, fit_args: dict[str, Any]) -> bool:
    if int(fit_args.get("num_bag_folds") or 0) > 0:
        return True
    if bool(fit_args.get("auto_stack")):
        return True
    preset_names = [str(value).lower() for value in _preset_values(presets)]
    return any(name in {"best_quality", "high_quality"} for name in preset_names)


def _preset_values(presets: str) -> list[str]:
    if isinstance(presets, list | tuple):
        return [str(value) for value in presets]
    return [str(presets)]


def _balanced_sample_weight(labels: pd.Series) -> pd.Series:
    counts = labels.value_counts(dropna=False)
    if counts.empty:
        raise ValueError("Cannot compute class weights for empty labels")
    weights_by_class = len(labels) / (len(counts) * counts.astype(float))
    return labels.map(weights_by_class).astype("float32")


def _log(message: str) -> None:
    print(f"AIDE AutoGluon: {message}", flush=True)


def _artifact_timestamp(artifacts_dir: Path) -> str:
    base = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    candidate = base
    counter = 1
    while (artifacts_dir / candidate).exists():
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def _profile_hyperparameters(
    profile_settings: dict[str, Any],
    *,
    included_model_types: list[str],
    use_gpu: bool,
) -> dict[str, list[dict[str, Any]]]:
    if not use_gpu:
        return {model_type: [{}] for model_type in included_model_types}

    profile_hyperparameters = profile_settings.get("hyperparameters")
    if isinstance(profile_hyperparameters, dict):
        return {
            model_type: list(profile_hyperparameters.get(model_type) or [{}])
            for model_type in included_model_types
        }

    fallback = _combined_hyperparameters(use_gpu=True)
    return {
        model_type: list(fallback.get(model_type) or [{}])
        for model_type in included_model_types
    }


def _write_artifact_bundle(
    *,
    artifact_dir: Path,
    submission: pd.DataFrame,
    metrics: dict[str, Any],
    ag_config: dict[str, Any],
) -> Path:
    from aide.utils.artifact_manifest import RESULT_MANIFEST_NAME, file_entry, sha256_file
    from aide.utils.artifact_manifest import write_json
    from aide.utils.path_portability import to_portable_path

    artifact_dir.mkdir(parents=True, exist_ok=True)
    submission_path = artifact_dir / "submission.csv"
    solution_path = artifact_dir / "solution.py"
    metrics_path = artifact_dir / "metrics.json"
    submission.to_csv(submission_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    solution_path.write_text(_solution_code_with_config(ag_config), encoding="utf-8")

    node_payload = {
        "id": f"{COMBINED_MODEL_NAME}-{artifact_dir.name}",
        "step": None,
        "ctime": artifact_dir.stat().st_ctime,
        "parent_id": None,
        "status": "ok",
        "origin": "manual_profile_eval",
        "plan": (
            "Combined the top s6e7 booster feature families into one AutoGluon "
            "training run."
        ),
        "analysis": "Single AutoGluon fit completed from combined top booster features.",
        "validity_warning": None,
        "is_buggy": False,
        "metric": {
            "value": metrics["cv_score"],
            "maximize": True,
            "name": metrics["eval_metric"],
        },
        "submission_validation": {"status": "not_run"},
    }
    manifest = {
        "schema_version": 2,
        "kind": "profile_eval",
        "competition": COMPETITION,
        "run": artifact_dir.parents[1].name,
        "timestamp": artifact_dir.name,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact_dir": to_portable_path(artifact_dir),
        "status": "ok",
        "local_score": metrics["cv_score"],
        "metric_maximize": True,
        "eval_metric": metrics["eval_metric"],
        "is_buggy": False,
        "sha256": sha256_file(submission_path),
        "profile": ag_config.get("profile"),
        "autogluon_presets": ag_config.get("presets"),
        "included_model_types": ag_config.get("included_model_types"),
        "time_limit": ag_config.get("time_limit"),
        "files": {
            "solution": file_entry(solution_path, base_dir=artifact_dir),
            "submission": file_entry(submission_path, base_dir=artifact_dir),
            "metrics": file_entry(metrics_path, base_dir=artifact_dir),
            "error": None,
            "oof_predictions": None,
            "test_predictions": None,
            "validation_predictions": None,
            "model_predictions": [],
        },
        "node": node_payload,
        "execution": {
            "exec_time": metrics["run_stats"]["exec_time"],
            "exc_type": None,
            "exc_info": None,
            "exc_stack": None,
        },
        "run_stats": metrics["run_stats"],
        "submission_validation": node_payload["submission_validation"],
        "autogluon": {
            "profile": ag_config.get("profile"),
            "presets": ag_config.get("presets"),
            "included_model_types": ag_config.get("included_model_types"),
            "time_limit": ag_config.get("time_limit"),
            "process_timeout": None,
            "use_gpu": ag_config.get("use_gpu"),
            "resolved_settings": ag_config,
        },
        "source": {
            "source_run": "2-whimsical-albatross-from-camelot",
            "source_node_id": None,
            "source_step": None,
            "source_timestamp": None,
            "source_sha256": None,
        },
    }
    write_json(artifact_dir / RESULT_MANIFEST_NAME, manifest)
    return artifact_dir


def _solution_code_with_config(ag_config: dict[str, Any]) -> str:
    source_path = Path(__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    config_header = (
        f"AIDE_AG_CONFIG = {ag_config!r}\n"
        f'RESULT_MARKER = "{RESULT_MARKER}"\n\n'
    )
    future_line = "from __future__ import annotations\n"
    if source.startswith(future_line):
        return future_line + "\n" + config_header + source[len(future_line) :]
    return config_header + source


def _refresh_submission_index(
    *,
    run_dir: Path,
    enabled: bool,
    index_refresher: Callable[..., Any] | None,
) -> Path | None:
    if not enabled:
        return None
    logs_dir = repo_root() / "logs"
    if run_dir.resolve().parent != logs_dir.resolve():
        _log(
            "cannot refresh submission index because output-dir is not a direct "
            f"child of {logs_dir}"
        )
        return None
    index_path = logs_dir / "submission_index.json"
    refresher = index_refresher or _default_index_refresher
    refresher(
        logs_dir=logs_dir,
        index_path=index_path,
        competition=COMPETITION,
        runs=[run_dir.name],
        reindex=True,
    )
    return index_path


def _default_index_refresher(**kwargs: Any) -> Any:
    from scripts.kaggle_submission_lab import refresh_index

    return refresh_index(**kwargs)


def _autogluon_model_type(algorithm: str) -> str:
    return {
        "xgboost": "XGB",
        "lightgbm": "GBM",
        "catboost": "CAT",
    }[algorithm]


def _autogluon_hyperparameters(
    algorithm: str,
    *,
    use_gpu: bool,
) -> dict[str, list[dict[str, Any]]]:
    model_type = _autogluon_model_type(algorithm)
    if not use_gpu:
        return {model_type: [{}]}
    return {
        "xgboost": {
            "XGB": [
                {
                    "ag_args": {"priority": 999},
                    "ag_args_fit": {"num_gpus": 1},
                    "device": "cuda",
                    "tree_method": "hist",
                }
            ]
        },
        "lightgbm": {"GBM": [{"ag_args_fit": {"num_gpus": 1}, "device": "cuda"}]},
        "catboost": {
            "CAT": [
                {
                    "ag_args_fit": {"num_gpus": 1},
                    "devices": "0",
                    "gpu_ram_part": 0.8,
                    "task_type": "GPU",
                }
            ]
        },
    }[algorithm]


def _combined_hyperparameters(*, use_gpu: bool) -> dict[str, list[dict[str, Any]]]:
    if not use_gpu:
        return {"XGB": [{}], "GBM": [{}], "CAT": [{}]}
    return {
        "XGB": [
            {
                "ag_args": {"priority": 999},
                "ag_args_fit": {"num_gpus": 1},
                "device": "cuda",
                "tree_method": "hist",
            }
        ],
        "GBM": [{"ag_args_fit": {"num_gpus": 1}, "device": "cuda"}],
        "CAT": [
            {
                "ag_args_fit": {"num_gpus": 1},
                "devices": "0",
                "gpu_ram_part": 0.8,
                "task_type": "GPU",
            }
        ],
    }


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for record in frame.to_dict("records"):
        records.append({key: _json_safe(value) for key, value in record.items()})
    return records


def _score_from_leaderboard(frame: pd.DataFrame) -> float:
    for score_col in ("score_val", "score_test"):
        if score_col in frame.columns and frame[score_col].notna().any():
            return float(frame[score_col].max())
    raise ValueError("AutoGluon leaderboard does not contain score_val or score_test")


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_profiles:
        print("Available AutoGluon profiles:", flush=True)
        for profile in available_autogluon_profiles():
            print(f"  - {profile}", flush=True)
        return 0

    result = run_combined_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        profile=args.profile,
        time_limit=args.time_limit,
        presets=args.presets,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        use_gpu=False if args.cpu else None,
    )
    report = {
        "name": result.name,
        "profile": result.profile,
        "cv_score": result.cv_score,
        "eval_metric": "balanced_accuracy",
        "feature_count": result.feature_count,
        "submission_path": str(result.submission_path),
        "artifact_submission_path": str(result.artifact_submission_path),
        "artifact_dir": str(result.artifact_dir),
        "metrics_path": str(result.metrics_path),
        "model_dir": str(result.model_dir),
        "index_path": str(result.index_path) if result.index_path else None,
    }
    print(f"Validation balanced_accuracy: {result.cv_score:.6f}", flush=True)
    print(f"Submission saved to: {result.submission_path}", flush=True)
    print(f"Artifact submission saved to: {result.artifact_submission_path}", flush=True)
    if result.index_path is not None:
        print(f"Submission index updated: {result.index_path}", flush=True)
    print(
        RESULT_MARKER + " " + json.dumps(report, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
