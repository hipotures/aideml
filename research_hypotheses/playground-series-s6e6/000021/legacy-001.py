import math
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from aide_solution_helpers import (
    aide_stage,
    load_competition_data,
    log_stage,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
N_SPLITS = 5
RAD_TO_ARCMIN = 180.0 * 60.0 / math.pi
BAND_NAMES = np.array(["u", "g", "r", "i", "z"])
ROT = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.44482963, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ],
    dtype=np.float64,
)


def choose_lightgbm_device() -> str:
    smoke_code = r"""
import numpy as np
from lightgbm import LGBMClassifier
X = np.random.RandomState(0).rand(48, 4)
y = np.array([0, 1, 2] * 16)
model = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=4,
    learning_rate=0.1,
    num_leaves=15,
    device="cuda",
    verbosity=-1,
)
model.fit(X, y)
print("LIGHTGBM_CUDA_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", smoke_code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and "LIGHTGBM_CUDA_OK" in result.stdout:
        log_stage("event=info|stage=build_features_stage|lightgbm_device=cuda")
        return "cuda"
    log_stage("event=info|stage=build_features_stage|lightgbm_device=cpu_fallback")
    return "cpu"


def add_cross(df: pd.DataFrame, left: str, right: str, out: str) -> None:
    df[out] = df[left].astype(str) + "__" + df[right].astype(str)


def add_triple_cross(df: pd.DataFrame, a: str, b: str, c: str, out: str) -> None:
    df[out] = df[a].astype(str) + "__" + df[b].astype(str) + "__" + df[c].astype(str)


def build_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base_cols = [
        "alpha",
        "delta",
        "u",
        "g",
        "r",
        "i",
        "z",
        "redshift",
        "spectral_type",
        "galaxy_population",
    ]
    n_train = len(train)

    # Transductive covariate-only features are built on train+test without any target column.
    df = pd.concat(
        [train[base_cols].copy(), test[base_cols].copy()],
        axis=0,
        ignore_index=True,
    )

    for col in ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]:
        df[col] = df[col].astype(np.float32)
    for col in ["spectral_type", "galaxy_population"]:
        df[col] = df[col].astype(str)

    df["u_g"] = (df["u"] - df["g"]).astype(np.float32)
    df["u_r"] = (df["u"] - df["r"]).astype(np.float32)
    df["u_i"] = (df["u"] - df["i"]).astype(np.float32)
    df["u_z"] = (df["u"] - df["z"]).astype(np.float32)
    df["g_r"] = (df["g"] - df["r"]).astype(np.float32)
    df["g_i"] = (df["g"] - df["i"]).astype(np.float32)
    df["g_z"] = (df["g"] - df["z"]).astype(np.float32)
    df["r_i"] = (df["r"] - df["i"]).astype(np.float32)
    df["r_z"] = (df["r"] - df["z"]).astype(np.float32)

    mags = df[["u", "g", "r", "i", "z"]].to_numpy(dtype=np.float32)
    k = np.arange(5, dtype=np.float32)
    sum_mags = mags.sum(axis=1)
    weighted_sum = mags @ k
    slope = (5.0 * weighted_sum - 10.0 * sum_mags) / 50.0
    intercept = (sum_mags - 10.0 * slope) / 5.0
    fitted = intercept[:, None] + slope[:, None] * k[None, :]
    residuals = mags - fitted
    d2 = np.diff(mags, n=2, axis=1)

    df["mag_sed_slope"] = slope.astype(np.float32)
    df["intercept"] = intercept.astype(np.float32)
    df["mag_sed_resid_mean_abs"] = np.abs(residuals).mean(axis=1).astype(np.float32)
    df["mag_sed_resid_std"] = residuals.std(axis=1).astype(np.float32)
    df["mag_sed_d2_mean"] = d2.mean(axis=1).astype(np.float32)
    df["mag_sed_d2_abs_mean"] = np.abs(d2).mean(axis=1).astype(np.float32)
    df["mag_sed_d2_std"] = d2.std(axis=1).astype(np.float32)
    df["mag_mean"] = mags.mean(axis=1).astype(np.float32)
    df["mag_std"] = mags.std(axis=1).astype(np.float32)
    df["mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype(np.float32)
    df["band_min_mag"] = mags.min(axis=1).astype(np.float32)
    df["band_max_mag"] = mags.max(axis=1).astype(np.float32)
    df["mag_min_band"] = pd.Categorical(BAND_NAMES[mags.argmin(axis=1)])
    df["mag_max_band"] = pd.Categorical(BAND_NAMES[mags.argmax(axis=1)])

    alpha_rad = np.deg2rad(df["alpha"].to_numpy(dtype=np.float64))
    delta_rad = np.deg2rad(df["delta"].to_numpy(dtype=np.float64))
    cos_delta = np.cos(delta_rad)
    sin_delta = np.sin(delta_rad)
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)

    df["alpha_sin"] = sin_alpha.astype(np.float32)
    df["alpha_cos"] = cos_alpha.astype(np.float32)
    df["delta_sin"] = sin_delta.astype(np.float32)
    df["delta_cos"] = cos_delta.astype(np.float32)

    sky_x = cos_delta * cos_alpha
    sky_y = cos_delta * sin_alpha
    sky_z = sin_delta
    df["sky_x"] = sky_x.astype(np.float32)
    df["sky_y"] = sky_y.astype(np.float32)
    df["sky_z"] = sky_z.astype(np.float32)
    df["sky_xy"] = (sky_x * sky_y).astype(np.float32)
    df["sky_xz"] = (sky_x * sky_z).astype(np.float32)
    df["sky_yz"] = (sky_y * sky_z).astype(np.float32)
    df["sky_x2_minus_y2"] = (sky_x**2 - sky_y**2).astype(np.float32)

    for order in [2, 3, 4, 5, 6, 7, 8]:
        df[f"alpha_sin{order}"] = np.sin(order * alpha_rad).astype(np.float32)
        df[f"alpha_cos{order}"] = np.cos(order * alpha_rad).astype(np.float32)
        df[f"delta_sin{order}"] = np.sin(order * delta_rad).astype(np.float32)
        df[f"delta_cos{order}"] = np.cos(order * delta_rad).astype(np.float32)

    alpha_wrapped = np.mod(df["alpha"].to_numpy(dtype=np.float64), 360.0)
    delta_vals = df["delta"].to_numpy(dtype=np.float64)

    alpha_bin_24 = np.clip(np.floor(alpha_wrapped / 15.0).astype(np.int16), 0, 23)
    delta_bin_18 = np.clip(np.floor((delta_vals + 90.0) / 10.0).astype(np.int16), 0, 17)
    sky_cell_bin = (alpha_bin_24 * 19 + delta_bin_18).astype(np.int32)

    alpha_bin_48 = np.clip(np.floor(alpha_wrapped / 7.5).astype(np.int16), 0, 47)
    delta_bin_36 = np.clip(np.floor((delta_vals + 90.0) / 5.0).astype(np.int16), 0, 35)
    sky_cell_bin_48 = (alpha_bin_48 * 36 + delta_bin_36).astype(np.int32)

    df["sky_alpha_bin_24"] = alpha_bin_24
    df["sky_delta_bin_18"] = delta_bin_18
    df["sky_cell_bin"] = sky_cell_bin
    df["sky_alpha_bin_48"] = alpha_bin_48
    df["sky_delta_bin_36"] = delta_bin_36
    df["sky_cell_bin_48"] = sky_cell_bin_48

    alpha_center = alpha_bin_24 * 15.0 + 7.5
    delta_center = delta_bin_18 * 10.0 - 85.0
    df["sky_alpha_bin_center_deg"] = alpha_center.astype(np.float32)
    df["sky_alpha_bin_offset_deg"] = (alpha_wrapped - alpha_center).astype(np.float32)
    df["sky_delta_bin_center_deg"] = delta_center.astype(np.float32)
    df["sky_delta_bin_offset_deg"] = (delta_vals - delta_center).astype(np.float32)

    coords = np.column_stack([delta_rad, alpha_rad])
    nn = NearestNeighbors(metric="haversine", algorithm="ball_tree", n_neighbors=17)
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)
    r5_arcmin = distances[:, 5] * RAD_TO_ARCMIN
    r10_arcmin = distances[:, 10] * RAD_TO_ARCMIN
    r16_arcmin = distances[:, 16] * RAD_TO_ARCMIN
    density5 = np.log1p(5.0 / np.maximum(r5_arcmin**2, 1e-6))
    density10 = np.log1p(10.0 / np.maximum(r10_arcmin**2, 1e-6))
    density16 = np.log1p(16.0 / np.maximum(r16_arcmin**2, 1e-6))
    ratio_5_16 = (5.0 / np.maximum(r5_arcmin**2, 1e-6)) / (
        16.0 / np.maximum(r16_arcmin**2, 1e-6)
    )

    df["sky_nn5_arcmin"] = r5_arcmin.astype(np.float32)
    df["sky_nn10_arcmin"] = r10_arcmin.astype(np.float32)
    df["sky_nn16_arcmin"] = r16_arcmin.astype(np.float32)
    df["sky_nn5_density"] = density5.astype(np.float32)
    df["sky_nn10_density"] = density10.astype(np.float32)
    df["sky_nn16_density"] = density16.astype(np.float32)
    df["sky_nn_density_ratio_5_16"] = ratio_5_16.astype(np.float32)

    redshift_vals = df["redshift"].to_numpy(dtype=np.float64)
    df["redshift_abs"] = np.abs(redshift_vals).astype(np.float32)
    df["redshift_sq"] = (redshift_vals**2).astype(np.float32)
    df["redshift_log_abs"] = np.log1p(np.abs(redshift_vals)).astype(np.float32)
    df["redshift_signed_log_abs"] = (
        np.sign(redshift_vals) * np.log1p(np.abs(redshift_vals))
    ).astype(np.float32)
    df["redshift_bin_20"] = pd.qcut(
        df["redshift"], q=20, labels=False, duplicates="drop"
    ).astype(np.int16)

    gal = np.column_stack([sky_x, sky_y, sky_z]) @ ROT.T
    gal_x = gal[:, 0]
    gal_y = gal[:, 1]
    gal_z = np.clip(gal[:, 2], -1.0, 1.0)
    gal_l = np.mod(np.degrees(np.arctan2(gal_y, gal_x)), 360.0)
    gal_b = np.degrees(np.arcsin(gal_z))

    df["galactic_l"] = gal_l.astype(np.float32)
    df["galactic_b"] = gal_b.astype(np.float32)
    df["galactic_l_sin"] = np.sin(np.deg2rad(gal_l)).astype(np.float32)
    df["galactic_l_cos"] = np.cos(np.deg2rad(gal_l)).astype(np.float32)
    df["galactic_b_sin"] = np.sin(np.deg2rad(gal_b)).astype(np.float32)
    df["galactic_b_cos"] = np.cos(np.deg2rad(gal_b)).astype(np.float32)
    df["galactic_b_abs"] = np.abs(gal_b).astype(np.float32)

    gal_l_bin_24 = np.clip(np.floor(gal_l / 15.0).astype(np.int16), 0, 23)
    gal_b_bin_12 = np.clip(np.floor((gal_b + 90.0) / 15.0).astype(np.int16), 0, 11)
    galactic_cell_bin = (gal_l_bin_24 * 12 + gal_b_bin_12).astype(np.int32)

    df["galactic_l_bin_24"] = gal_l_bin_24
    df["galactic_b_bin_12"] = gal_b_bin_12
    df["galactic_cell_bin"] = galactic_cell_bin

    add_cross(df, "spectral_type", "redshift_bin_20", "spectral_type__redshift_bin_20")
    add_cross(
        df,
        "galaxy_population",
        "redshift_bin_20",
        "galaxy_population__redshift_bin_20",
    )
    add_cross(
        df,
        "spectral_type",
        "galactic_cell_bin",
        "spectral_type__galactic_cell_bin",
    )
    add_cross(
        df,
        "galaxy_population",
        "galactic_cell_bin",
        "galaxy_population__galactic_cell_bin",
    )

    add_cross(
        df, "spectral_type", "galaxy_population", "spectral_type__galaxy_population"
    )
    add_cross(df, "spectral_type", "mag_min_band", "spectral_type__band_min_band")
    add_cross(
        df, "galaxy_population", "mag_max_band", "galaxy_population__band_max_band"
    )
    add_cross(
        df, "spectral_type", "sky_alpha_bin_24", "spectral_type__sky_alpha_bin_24"
    )
    add_cross(
        df,
        "galaxy_population",
        "sky_alpha_bin_24",
        "galaxy_population__sky_alpha_bin_24",
    )
    add_cross(df, "spectral_type", "sky_cell_bin", "spectral_type__sky_cell_bin")
    add_cross(
        df, "spectral_type", "sky_alpha_bin_48", "spectral_type__sky_alpha_bin_48"
    )
    add_cross(
        df,
        "galaxy_population",
        "sky_alpha_bin_48",
        "galaxy_population__sky_alpha_bin_48",
    )
    add_cross(df, "spectral_type", "sky_cell_bin_48", "spectral_type__sky_cell_bin_48")
    add_triple_cross(
        df,
        "spectral_type",
        "galaxy_population",
        "sky_delta_bin_18",
        "spectral_type__galaxy_population__sky_delta_bin_18",
    )

    freq_cols = [
        "sky_alpha_bin_24",
        "sky_delta_bin_18",
        "sky_cell_bin",
        "sky_alpha_bin_48",
        "sky_delta_bin_36",
        "sky_cell_bin_48",
        "redshift_bin_20",
        "galactic_l_bin_24",
        "galactic_b_bin_12",
        "galactic_cell_bin",
    ]
    for col in freq_cols:
        freq = df[col].value_counts(normalize=True, dropna=False)
        df[f"{col}_freq"] = df[col].map(freq).astype(np.float32)

    for col in ["u", "g", "r", "i", "z", "redshift"]:
        df[f"{col}_qrank"] = df[col].rank(method="average", pct=True).astype(np.float32)

    categorical_cols = [
        "spectral_type",
        "galaxy_population",
        "mag_min_band",
        "mag_max_band",
        "sky_alpha_bin_24",
        "sky_delta_bin_18",
        "sky_cell_bin",
        "sky_alpha_bin_48",
        "sky_delta_bin_36",
        "sky_cell_bin_48",
        "redshift_bin_20",
        "galactic_l_bin_24",
        "galactic_b_bin_12",
        "galactic_cell_bin",
        "spectral_type__redshift_bin_20",
        "galaxy_population__redshift_bin_20",
        "spectral_type__galactic_cell_bin",
        "galaxy_population__galactic_cell_bin",
        "spectral_type__galaxy_population",
        "spectral_type__band_min_band",
        "galaxy_population__band_max_band",
        "spectral_type__sky_alpha_bin_24",
        "galaxy_population__sky_alpha_bin_24",
        "spectral_type__sky_cell_bin",
        "spectral_type__sky_alpha_bin_48",
        "galaxy_population__sky_alpha_bin_48",
        "spectral_type__sky_cell_bin_48",
        "spectral_type__galaxy_population__sky_delta_bin_18",
    ]

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    numeric_cols = [c for c in df.columns if c not in categorical_cols]
    for col in numeric_cols:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(np.int32)

    train_features = df.iloc[:n_train].reset_index(drop=True)
    test_features = df.iloc[n_train:].reset_index(drop=True)
    return train_features, test_features, categorical_cols


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    categorical_cols: List[str],
    use_gpu: bool,
) -> CatBoostClassifier:
    params = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    if use_gpu:
        params.update(task_type="GPU", devices="0", gpu_ram_part=0.8)
    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=categorical_cols,
        use_best_model=True,
        verbose=False,
    )
    return model


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    sample_weight: np.ndarray,
    device: str,
    num_class: int,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=SEED,
        tree_method="hist",
        device=device,
        eval_metric="mlogloss",
        enable_categorical=True,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    categorical_cols: List[str],
    device: str,
    num_class: int,
) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="multiclass",
        num_class=num_class,
        class_weight="balanced",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=SEED,
        device=device,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        categorical_feature=categorical_cols,
        callbacks=[early_stopping(100, verbose=False)],
    )
    return model


def run_model_cv(
    model_name: str,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    categorical_cols: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    classes: np.ndarray,
    lgb_device: str,
) -> Dict[str, object]:
    num_class = len(classes)
    oof_proba = np.zeros((len(X), num_class), dtype=np.float32)
    test_proba = np.zeros((len(X_test), num_class), dtype=np.float32)
    fold_scores: List[float] = []

    cat_gpu_ok = True
    xgb_device = "cuda"
    lgb_runtime_device = lgb_device

    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        with aide_stage("fit_predict_fold_stage"):
            log_stage(
                f"event=info|stage=fit_predict_fold_stage|fold={fold_id}|model={model_name}"
            )
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            if model_name == "catboost":
                try:
                    model = fit_catboost(
                        X_train,
                        y_train,
                        X_valid,
                        y_valid,
                        categorical_cols,
                        use_gpu=cat_gpu_ok,
                    )
                except Exception as exc:
                    if cat_gpu_ok:
                        log_stage(
                            f"event=info|stage=fit_predict_fold_stage|fold={fold_id}|model={model_name}|gpu_fallback=cpu|error_type={exc.__class__.__name__}"
                        )
                        cat_gpu_ok = False
                        model = fit_catboost(
                            X_train,
                            y_train,
                            X_valid,
                            y_valid,
                            categorical_cols,
                            use_gpu=False,
                        )
                    else:
                        raise
            elif model_name == "xgboost":
                sample_weight = compute_sample_weight(
                    class_weight="balanced", y=y_train
                )
                try:
                    model = fit_xgboost(
                        X_train,
                        y_train,
                        X_valid,
                        y_valid,
                        sample_weight,
                        xgb_device,
                        num_class,
                    )
                except Exception as exc:
                    if xgb_device == "cuda":
                        log_stage(
                            f"event=info|stage=fit_predict_fold_stage|fold={fold_id}|model={model_name}|gpu_fallback=cpu|error_type={exc.__class__.__name__}"
                        )
                        xgb_device = "cpu"
                        model = fit_xgboost(
                            X_train,
                            y_train,
                            X_valid,
                            y_valid,
                            sample_weight,
                            xgb_device,
                            num_class,
                        )
                    else:
                        raise
            elif model_name == "lightgbm":
                try:
                    model = fit_lightgbm(
                        X_train,
                        y_train,
                        X_valid,
                        y_valid,
                        categorical_cols,
                        lgb_runtime_device,
                        num_class,
                    )
                except Exception as exc:
                    if lgb_runtime_device == "cuda":
                        log_stage(
                            f"event=info|stage=fit_predict_fold_stage|fold={fold_id}|model={model_name}|gpu_fallback=cpu|error_type={exc.__class__.__name__}"
                        )
                        lgb_runtime_device = "cpu"
                        model = fit_lightgbm(
                            X_train,
                            y_train,
                            X_valid,
                            y_valid,
                            categorical_cols,
                            lgb_runtime_device,
                            num_class,
                        )
                    else:
                        raise
            else:
                raise ValueError(model_name)

            valid_proba = model.predict_proba(X_valid)
            fold_pred = valid_proba.argmax(axis=1)
            fold_score = balanced_accuracy_score(y_valid, fold_pred)
            fold_scores.append(fold_score)
            oof_proba[valid_idx] = valid_proba.astype(np.float32)
            test_proba += model.predict_proba(X_test).astype(np.float32) / len(folds)

            print(
                f"{model_name} fold {fold_id} balanced_accuracy={fold_score:.6f}",
                flush=True,
            )

    return {
        "name": model_name,
        "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)),
        "oof_proba": oof_proba,
        "test_proba": test_proba,
    }


def main() -> None:
    train, test, sample_sub = load_competition_data()

    with aide_stage("build_features_stage"):
        log_stage(
            "event=info|stage=build_features_stage|message=building_explicit_covariate_only_feature_block"
        )
        X, X_test, categorical_cols = build_features(train, test)
        y_raw = train["class"].astype(str).to_numpy()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        classes = label_encoder.classes_
        lgb_device = choose_lightgbm_device()

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(skf.split(X, y))

    model_names = ["catboost", "xgboost", "lightgbm"]
    results: Dict[str, Dict[str, object]] = {}
    for model_name in model_names:
        results[model_name] = run_model_cv(
            model_name=model_name,
            X=X,
            X_test=X_test,
            y=y,
            categorical_cols=categorical_cols,
            folds=folds,
            classes=classes,
            lgb_device=lgb_device,
        )

    with aide_stage("score_stage"):
        for model_name in model_names:
            mean_score = results[model_name]["mean_score"]
            print(f"{model_name} mean_balanced_accuracy={mean_score:.6f}", flush=True)

        best_name = max(model_names, key=lambda name: results[name]["mean_score"])
        best_result = results[best_name]
        best_oof_pred = classes[np.argmax(best_result["oof_proba"], axis=1)]
        overall_oof_bal_acc = balanced_accuracy_score(
            train["class"].astype(str), best_oof_pred
        )

        print(f"best_model={best_name}", flush=True)
        print(
            f"best_model_mean_balanced_accuracy={best_result['mean_score']:.6f}",
            flush=True,
        )
        print(f"best_model_oof_balanced_accuracy={overall_oof_bal_acc:.6f}", flush=True)

    with aide_stage("write_outputs_stage"):
        submission = sample_sub[["id"]].copy()
        submission["class"] = classes[np.argmax(best_result["test_proba"], axis=1)]
        write_submission(submission)

        oof_frame = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=np.int32),
                "target": train["class"].astype(str).to_numpy(),
                "prediction": best_oof_pred,
            }
        )
        write_oof_predictions(oof_frame)

        test_pred_frame = sample_sub[["id"]].copy()
        for idx, cls in enumerate(classes):
            test_pred_frame[cls] = best_result["test_proba"][:, idx]
        write_test_predictions(test_pred_frame)


if __name__ == "__main__":
    main()
