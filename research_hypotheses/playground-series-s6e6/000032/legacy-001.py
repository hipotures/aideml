from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from aide_solution_helpers import (
    aidestage,
    load_competition_data,
    working_dir,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    log_stage,
)

# Note: aidestage is intentionally aliased to preserve helper naming in some environments.
# If not available with this exact name, fallback to aide_stage below.
try:
    from aide_solution_helpers import aide_stage
except Exception:  # pragma: no cover
    aide_stage = aidestage

SDSS_WAVELENGTHS_A = np.array(
    [3543.0, 4770.0, 6231.0, 7625.0, 9134.0], dtype=np.float32
)
MAG_COLS = ["u", "g", "r", "i", "z"]
NON_CAT_FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]
BB_CONST_A = 1.4387769e7
BB_TEMPERATURES = np.array(
    [
        2500.0,
        3000.0,
        3500.0,
        4500.0,
        6000.0,
        8500.0,
        12000.0,
        16000.0,
        22000.0,
        30000.0,
    ],
    dtype=np.float32,
)
PL_ALPHAS = np.array(
    [-3.0, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25],
    dtype=np.float32,
)
BANDS = np.array(list("ugriz"))


def mag_to_relative_flux(
    mag: np.ndarray, clip_min: float = -35.0, clip_max: float = 40.0
) -> np.ndarray:
    mag = np.clip(mag.astype(np.float32), clip_min, clip_max)
    return np.power(10.0, -0.4 * mag, dtype=np.float32)


def fit_template_family(
    flux: np.ndarray,
    z_factor: np.ndarray,
    wavelengths: np.ndarray,
    params: np.ndarray,
    family: str,
    prefix: str,
):
    n_rows, n_bands = flux.shape
    n_params = len(params)
    rows = np.arange(n_rows, dtype=np.int64)

    inv_wave = (1.0 / wavelengths).astype(np.float32)  # shape (5,)
    mse = np.empty((n_rows, n_params), dtype=np.float32)
    amp = np.empty((n_rows, n_params), dtype=np.float32)
    templates = np.empty((n_rows, n_params, n_bands), dtype=np.float32)

    if family == "blackbody":
        inv_wave_cubed = inv_wave**3
        z_cube = np.power(z_factor, 3.0).astype(np.float32)
        for j, t in enumerate(params):
            x = (BB_CONST_A * inv_wave[None, :] * z_factor[:, None]) / max(
                float(t), 1e-3
            )
            # x >= 0; clip to avoid inf/underflow issues when used inside expm1
            template = (inv_wave_cubed[None, :] * z_cube[:, None]) / np.expm1(
                np.clip(x, 1e-8, 1e3)
            )
            template = np.nan_to_num(template, nan=0.0, posinf=0.0, neginf=0.0)
            templates[:, j, :] = template.astype(np.float32)
            den = np.einsum("ij,ij->i", template, template)
            a = np.divide(
                np.einsum("ij,ij->i", flux, template),
                den,
                out=np.zeros(n_rows, dtype=np.float32),
                where=den > 1e-18,
            )
            a = np.where(a > 0, a, 0.0).astype(np.float32)
            amp[:, j] = a
            resid = flux - template * a[:, None]
            mse[:, j] = np.mean(resid * resid, axis=1).astype(np.float32)
    elif family == "powerlaw":
        for j, alpha in enumerate(params):
            # rest-frame handled by z_factor: nu_rest = nu_obs * (1+z)
            base = np.power(inv_wave, float(alpha))
            scale = np.power(z_factor, float(alpha))[:, None]
            template = base[None, :] * scale
            template = np.nan_to_num(template, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )
            templates[:, j, :] = template
            den = np.einsum("ij,ij->i", template, template)
            a = np.divide(
                np.einsum("ij,ij->i", flux, template),
                den,
                out=np.zeros(n_rows, dtype=np.float32),
                where=den > 1e-18,
            )
            a = np.where(a > 0, a, 0.0).astype(np.float32)
            amp[:, j] = a
            resid = flux - template * a[:, None]
            mse[:, j] = np.mean(resid * resid, axis=1).astype(np.float32)
    else:
        raise ValueError(f"unknown family: {family}")

    best_idx = np.argmin(mse, axis=1).astype(np.int16)
    best_param = params[best_idx]
    best_amp = amp[rows, best_idx]
    best_mse = mse[rows, best_idx]
    best_template = templates[rows, best_idx, :]
    best_resid = flux - best_template * best_amp[:, None]
    best_mae = np.mean(np.abs(best_resid), axis=1).astype(np.float32)
    best_max_abs = np.max(np.abs(best_resid), axis=1).astype(np.float32)

    feat = pd.DataFrame(index=np.arange(n_rows))
    feat[f"{prefix}_best_param"] = best_param.astype(np.float32)
    feat[f"{prefix}_best_idx"] = best_idx.astype(np.float32)
    feat[f"{prefix}_amp"] = best_amp.astype(np.float32)
    feat[f"{prefix}_mse"] = best_mse.astype(np.float32)
    feat[f"{prefix}_mae"] = best_mae
    feat[f"{prefix}_max_abs"] = best_max_abs

    for i, band in enumerate(BANDS):
        feat[f"{prefix}_res_{band}"] = best_resid[:, i].astype(np.float32)

    feat[f"{prefix}_res_curv_ugr"] = (
        best_resid[:, 0] - 2.0 * best_resid[:, 1] + best_resid[:, 2]
    ).astype(np.float32)
    feat[f"{prefix}_res_curv_gri"] = (
        best_resid[:, 1] - 2.0 * best_resid[:, 2] + best_resid[:, 3]
    ).astype(np.float32)
    feat[f"{prefix}_res_curv_riz"] = (
        best_resid[:, 2] - 2.0 * best_resid[:, 3] + best_resid[:, 4]
    ).astype(np.float32)
    feat[f"{prefix}_blue_excess"] = (best_resid[:, 0] - best_resid[:, 2]).astype(
        np.float32
    )
    feat[f"{prefix}_red_excess"] = (best_resid[:, 4] - best_resid[:, 2]).astype(
        np.float32
    )

    return feat, best_mse, best_resid.astype(np.float32)


def build_template_features(
    df: pd.DataFrame, flux_norm: np.ndarray, include_rest: bool = True
) -> pd.DataFrame:
    redshift = df["redshift"].astype(np.float32).to_numpy()
    z_obs = np.ones_like(redshift, dtype=np.float32)
    z_rest = np.clip(1.0 + redshift, 0.2, 6.0).astype(np.float32)

    # Observed-frame templates
    bb_obs, bb_obs_mse, bb_obs_res = fit_template_family(
        flux_norm, z_obs, SDSS_WAVELENGTHS_A, BB_TEMPERATURES, "blackbody", "bb_obs"
    )
    pl_obs, pl_obs_mse, pl_obs_res = fit_template_family(
        flux_norm, z_obs, SDSS_WAVELENGTHS_A, PL_ALPHAS, "powerlaw", "pl_obs"
    )
    feat = pd.concat([bb_obs, pl_obs], axis=1)

    feat["obs_bb_over_pl_mse_ratio"] = bb_obs_mse / (pl_obs_mse + 1e-12)
    feat["obs_pl_over_bb_mse_ratio"] = pl_obs_mse / (bb_obs_mse + 1e-12)
    for i, band in enumerate(BANDS):
        feat[f"obs_res_diff_{band}"] = bb_obs_res[:, i] - pl_obs_res[:, i]

    # Optional rest-frame template set (ablated/logged in this run as enabled)
    if include_rest:
        bb_rest, bb_rest_mse, bb_rest_res = fit_template_family(
            flux_norm,
            z_rest,
            SDSS_WAVELENGTHS_A,
            BB_TEMPERATURES,
            "blackbody",
            "bb_rest",
        )
        pl_rest, pl_rest_mse, pl_rest_res = fit_template_family(
            flux_norm, z_rest, SDSS_WAVELENGTHS_A, PL_ALPHAS, "powerlaw", "pl_rest"
        )
        feat = pd.concat([feat, bb_rest, pl_rest], axis=1)
        feat["rest_bb_over_pl_mse_ratio"] = bb_rest_mse / (pl_rest_mse + 1e-12)
        feat["rest_pl_over_bb_mse_ratio"] = pl_rest_mse / (bb_rest_mse + 1e-12)
        for i, band in enumerate(BANDS):
            feat[f"rest_res_diff_{band}"] = bb_rest_res[:, i] - pl_rest_res[:, i]
        feat["rest_obs_mse_gap_obs"] = np.abs(bb_obs_mse - bb_rest_mse).astype(
            np.float32
        )
        feat["rest_obs_mse_gap_rest"] = np.abs(pl_obs_mse - pl_rest_mse).astype(
            np.float32
        )

    feat.index = df.index
    return feat.astype(np.float32)


def build_features(
    df: pd.DataFrame, include_rest_templates: bool = True
) -> pd.DataFrame:
    base = pd.DataFrame(index=df.index)

    for col in NON_CAT_FEATURES:
        base[col] = df[col].astype(np.float32)

    base["color_u_g"] = (df["u"] - df["g"]).astype(np.float32)
    base["color_g_r"] = (df["g"] - df["r"]).astype(np.float32)
    base["color_r_i"] = (df["r"] - df["i"]).astype(np.float32)
    base["color_i_z"] = (df["i"] - df["z"]).astype(np.float32)

    mag = df[MAG_COLS].to_numpy(dtype=np.float32)
    flux = mag_to_relative_flux(mag)
    flux_sum = np.sum(flux, axis=1, keepdims=True)
    flux_norm = flux / np.maximum(flux_sum, 1e-12)

    base["flux_r_norm_u"] = (flux[:, 0] / np.maximum(flux[:, 2], 1e-12)).astype(
        np.float32
    )
    base["flux_r_norm_g"] = (flux[:, 1] / np.maximum(flux[:, 2], 1e-12)).astype(
        np.float32
    )
    base["flux_r_norm_r"] = (flux[:, 2] / np.maximum(flux[:, 2], 1e-12)).astype(
        np.float32
    )
    base["flux_r_norm_i"] = (flux[:, 3] / np.maximum(flux[:, 2], 1e-12)).astype(
        np.float32
    )
    base["flux_r_norm_z"] = (flux[:, 4] / np.maximum(flux[:, 2], 1e-12)).astype(
        np.float32
    )

    base["flux_sum"] = flux_sum.squeeze(axis=1).astype(np.float32)
    for i, band in enumerate(BANDS):
        base[f"flux_norm_{band}"] = flux_norm[:, i].astype(np.float32)
        base[f"log_flux_norm_{band}"] = np.log1p(
            np.maximum(flux_norm[:, i], 0.0)
        ).astype(np.float32)

    template_features = build_template_features(
        df, flux_norm, include_rest=include_rest_templates
    )
    base = pd.concat([base, template_features], axis=1)

    base = base.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return base.astype(np.float32)


def build_catboost_model(task_type: str = "GPU") -> CatBoostClassifier:
    common = {
        "iterations": 350,
        "learning_rate": 0.08,
        "depth": 8,
        "loss_function": "MultiClass",
        "auto_class_weights": "Balanced",
        "eval_metric": "MultiClass",
        "random_seed": 42,
        "verbose": False,
        "od_type": "Iter",
        "od_wait": 40,
    }

    if task_type == "GPU":
        common.update(
            {
                "task_type": "GPU",
                "devices": "0",
                "gpu_ram_part": 0.8,
            }
        )
    return CatBoostClassifier(**common)


def main() -> None:
    with aide_stage("build_features_stage"):
        log_stage(
            "Hypothesis 000032: analytic SED template residual feature block with observed/rest-frame templates."
        )
        # Load competition split via provided IO helper (no manual path resolution).
        train_df, test_df, sample_submission = load_competition_data()

        aux_path = Path("./input/star_classification.csv")
        if aux_path.exists():
            try:
                aux_df = pd.read_csv(aux_path)
                has_sentinal = bool((aux_df[MAG_COLS] < -9000).any().any())
                if has_sentinal:
                    log_stage(
                        "Auxiliary star_classification.csv found; sentinel magnitudes detected, but not merged due schema mismatch. "
                        "Using target-free feature engineering only."
                    )
                else:
                    log_stage(
                        "Auxiliary star_classification.csv loaded and inspected; not merged due schema mismatch."
                    )
            except Exception as exc:
                log_stage(
                    f"Auxiliary star_classification.csv load failed ({exc}); continuing without auxiliary usage."
                )
        else:
            log_stage(
                "Auxiliary star_classification.csv not present; proceeding with competition data only."
            )

        train_features = build_features(train_df, include_rest_templates=True)
        test_features = build_features(test_df, include_rest_templates=True)

        train_cat = pd.get_dummies(train_df[CATEGORICAL_COLS], dtype=np.float32)
        test_cat = pd.get_dummies(test_df[CATEGORICAL_COLS], dtype=np.float32)
        train_cat, test_cat = train_cat.align(
            test_cat, join="outer", axis=1, fill_value=0.0
        )

        train_features = pd.concat([train_features, train_cat], axis=1)
        test_features = pd.concat([test_features, test_cat], axis=1)
        test_features = test_features.reindex(
            columns=train_features.columns, fill_value=0.0
        )

        # Ensure stable numeric matrices
        X = train_features.to_numpy(dtype=np.float32, copy=False)
        X_test = test_features.to_numpy(dtype=np.float32, copy=False)

        y = LabelEncoder().fit_transform(train_df["class"].astype(str))
        class_names = LabelEncoder().fit(train_df["class"].astype(str)).classes_

        log_stage(f"Feature matrix: train={X.shape}, test={X_test.shape}")
        log_stage(
            f"Aux columns in train features: {train_features.shape[1] - train_df.shape[1] - 0}"
        )

    with aide_stage("make_folds_stage"):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(cv.split(X, y))

    oof_proba = np.zeros((len(train_df), len(class_names)), dtype=np.float32)
    test_proba = np.zeros((len(test_df), len(class_names)), dtype=np.float32)
    fold_scores = []

    with aide_stage("fit_predict_fold_stage"):
        use_gpu = True
        for fold_idx, (train_idx, valid_idx) in enumerate(folds, start=1):
            log_stage(
                f"Fold {fold_idx}/5: CatBoost fit (task={'GPU' if use_gpu else 'CPU'})"
            )
            X_tr, X_va = X[train_idx], X[valid_idx]
            y_tr, y_va = y[train_idx], y[valid_idx]

            if use_gpu:
                try:
                    model = build_catboost_model(task_type="GPU")
                    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
                except Exception as exc:
                    print(
                        f"[gpu_fallback] CatBoost GPU failed on fold {fold_idx}: {exc}. "
                        "Falling back to CPU for remaining folds.",
                        flush=True,
                    )
                    use_gpu = False
                    model = build_catboost_model(task_type="CPU")
                    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            else:
                model = build_catboost_model(task_type="CPU")
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

            val_proba = model.predict_proba(X_va)
            oof_proba[valid_idx] = val_proba
            fold_pred = np.argmax(val_proba, axis=1)
            fold_score = balanced_accuracy_score(
                y_va, fold_pred, labels=np.arange(len(class_names))
            )
            fold_scores.append(fold_score)
            print(f"Fold {fold_idx} balanced_accuracy: {fold_score:.6f}", flush=True)

            test_proba += model.predict_proba(X_test) / cv.get_n_splits()

    with aide_stage("score_stage"):
        oof_pred = np.argmax(oof_proba, axis=1)
        overall_score = balanced_accuracy_score(
            y, oof_pred, labels=np.arange(len(class_names))
        )
        print(
            f"OOF balanced_accuracy: {overall_score:.6f} "
            f"(fold mean {np.mean(fold_scores):.6f}, std {np.std(fold_scores):.6f})",
            flush=True,
        )

    with aide_stage("write_outputs_stage"):
        encoder_for_output = LabelEncoder().fit(train_df["class"].astype(str))
        oof_target = encoder_for_output.inverse_transform(y)
        oof_pred_label = encoder_for_output.inverse_transform(oof_pred)
        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train_df), dtype=np.int64),
                "target": oof_target,
                "prediction": oof_pred_label,
            }
        )
        write_oof_predictions(oof_df)

        submission_labels = encoder_for_output.inverse_transform(
            np.argmax(test_proba, axis=1)
        )
        submission_df = pd.DataFrame(
            {
                "id": test_df["id"].astype(np.int64),
                "class": submission_labels,
            }
        )
        write_submission(submission_df)

        test_pred_df = pd.DataFrame({"id": test_df["id"].astype(np.int64)})
        for i, cls in enumerate(encoder_for_output.classes_):
            test_pred_df[cls] = test_proba[:, i].astype(np.float32)
        write_test_predictions(test_pred_df)

        log_stage(
            f"Saved submission with {len(submission_df)} rows and test prediction artifact with columns: {list(test_pred_df.columns)[:3]}..."
        )


if __name__ == "__main__":
    main()
