import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 20260611
TARGET = "class"
ID_COL = "id"
CLASS_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
INT_TO_LABEL = {i: c for c, i in LABEL_TO_INT.items()}

BANDS = ["u", "g", "r", "i", "z"]
SDSS_WAVELENGTHS = np.array([3543.0, 4770.0, 6231.0, 7625.0, 9134.0], dtype=np.float64)
REST_GRID = np.array(
    [1300.0, 1700.0, 2200.0, 3000.0, 4000.0, 5200.0, 6800.0, 8500.0], dtype=np.float64
)


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def add_rest_frame_sed_warp_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    mags = out[BANDS].astype(np.float64).replace(-9999.0, np.nan)
    finite_band_medians = mags.replace([np.inf, -np.inf], np.nan).median(axis=0)
    mags = mags.fillna(finite_band_medians)

    z = (
        out["redshift"]
        .astype(np.float64)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )
    z_clip = np.clip(z, -0.95, 7.5)
    rest_waves = SDSS_WAVELENGTHS[None, :] / (1.0 + z_clip[:, None])

    mag_values = mags.to_numpy(dtype=np.float64)
    centered_mag = mag_values - mag_values.mean(axis=1, keepdims=True)
    rel_log_flux = -0.4 * centered_mag

    n = len(out)
    interp_mag = np.empty((n, len(REST_GRID)), dtype=np.float32)
    interp_flux = np.empty((n, len(REST_GRID)), dtype=np.float32)
    covered = np.empty((n, len(REST_GRID)), dtype=np.int8)

    for idx in range(n):
        order = np.argsort(rest_waves[idx])
        wave_i = rest_waves[idx, order]
        mag_i = centered_mag[idx, order]
        flux_i = rel_log_flux[idx, order]
        interp_mag[idx] = np.interp(REST_GRID, wave_i, mag_i).astype(np.float32)
        interp_flux[idx] = np.interp(REST_GRID, wave_i, flux_i).astype(np.float32)
        covered[idx] = ((REST_GRID >= wave_i[0]) & (REST_GRID <= wave_i[-1])).astype(
            np.int8
        )

    for j, wave in enumerate(REST_GRID.astype(int)):
        out[f"rest_mag_{wave}"] = interp_mag[:, j]
        out[f"rest_logflux_{wave}"] = interp_flux[:, j]
        out[f"rest_grid_covered_{wave}"] = covered[:, j]

    for j in range(len(REST_GRID) - 1):
        lo = int(REST_GRID[j])
        hi = int(REST_GRID[j + 1])
        delta_log_wave = np.log(REST_GRID[j + 1]) - np.log(REST_GRID[j])
        out[f"rest_color_mag_{lo}_{hi}"] = interp_mag[:, j] - interp_mag[:, j + 1]
        out[f"rest_flux_slope_{lo}_{hi}"] = (
            interp_flux[:, j + 1] - interp_flux[:, j]
        ) / delta_log_wave

    for j in range(1, len(REST_GRID) - 1):
        mid = int(REST_GRID[j])
        out[f"rest_mag_curvature_{mid}"] = (
            interp_mag[:, j - 1] - 2.0 * interp_mag[:, j] + interp_mag[:, j + 1]
        )
        out[f"rest_flux_curvature_{mid}"] = (
            interp_flux[:, j - 1] - 2.0 * interp_flux[:, j] + interp_flux[:, j + 1]
        )

    blue_mask = REST_GRID <= 3000.0
    red_mask = REST_GRID >= 5200.0
    break_blue = REST_GRID <= 4000.0
    break_red = REST_GRID > 4000.0
    out["rest_uv_to_red_flux_contrast"] = interp_flux[:, blue_mask].mean(
        axis=1
    ) - interp_flux[:, red_mask].mean(axis=1)
    out["rest_4000a_flux_break_proxy"] = interp_flux[:, break_red].mean(
        axis=1
    ) - interp_flux[:, break_blue].mean(axis=1)
    out["rest_covered_grid_count"] = covered.sum(axis=1)
    out["rest_extrapolated_grid_count"] = (
        len(REST_GRID) - out["rest_covered_grid_count"]
    )

    for band, wave in zip(BANDS, SDSS_WAVELENGTHS):
        rest_band_wave = wave / (1.0 + z_clip)
        out[f"{band}_rest_wave"] = rest_band_wave
        out[f"{band}_crosses_4000a"] = (rest_band_wave < 4000.0).astype(np.int8)
        out[f"{band}_crosses_3000a"] = (rest_band_wave < 3000.0).astype(np.int8)
        out[f"{band}_crosses_5200a"] = (rest_band_wave < 5200.0).astype(np.int8)

    out["rest_wave_span"] = rest_waves.max(axis=1) - rest_waves.min(axis=1)
    out["rest_wave_min"] = rest_waves.min(axis=1)
    out["rest_wave_max"] = rest_waves.max(axis=1)
    return out


def align_proba(proba: np.ndarray, model_classes) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(CLASS_ORDER)), dtype=np.float64)
    for j, cls in enumerate(model_classes):
        if isinstance(cls, str):
            aligned[:, LABEL_TO_INT[cls]] = proba[:, j]
        else:
            aligned[:, int(cls)] = proba[:, j]
    return aligned


def main():
    with aide_stage("load_data_stage"):
        train, test, sample_sub = load_competition_data()

    with aide_stage("build_features_stage"):
        y_labels = train[TARGET].astype(str).to_numpy()
        y = np.array([LABEL_TO_INT[v] for v in y_labels], dtype=np.int64)

        train_x = train.drop(columns=[TARGET])
        test_x = test.copy()

        all_x = pd.concat([train_x, test_x], axis=0, ignore_index=True)
        all_feat = add_rest_frame_sed_warp_features(all_x)
        X = all_feat.iloc[: len(train_x)].reset_index(drop=True)
        X_test = all_feat.iloc[len(train_x) :].reset_index(drop=True)

        cat_cols = [c for c in ["spectral_type", "galaxy_population"] if c in X.columns]
        numeric_cols = [c for c in X.columns if c not in cat_cols and c != ID_COL]

        for col in cat_cols:
            X[col] = X[col].astype("object").fillna("missing").astype(str)
            X_test[col] = X_test[col].astype("object").fillna("missing").astype(str)

        X[numeric_cols] = (
            X[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(X[numeric_cols].median())
        )
        X_test[numeric_cols] = (
            X_test[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(X[numeric_cols].median())
        )

        model_features = numeric_cols + cat_cols
        X_model = X[model_features].copy()
        X_test_model = X_test[model_features].copy()

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        folds = list(skf.split(X_model, y))

    oof_proba = np.zeros((len(X_model), len(CLASS_ORDER)), dtype=np.float64)
    test_proba = np.zeros((len(X_test_model), len(CLASS_ORDER)), dtype=np.float64)
    model_scores = {"logistic": [], "random_forest": [], "catboost": []}

    with aide_stage("fit_predict_fold_stage"):
        for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
            X_tr = X_model.iloc[tr_idx].copy()
            X_va = X_model.iloc[va_idx].copy()
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    ("cat", _one_hot_encoder(), cat_cols),
                ],
                remainder="drop",
            )

            log_stage(f"Fold {fold}: fitting balanced multinomial logistic regression")
            logistic = Pipeline(
                steps=[
                    ("prep", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            solver="saga",
                            multi_class="multinomial",
                            class_weight="balanced",
                            max_iter=600,
                            n_jobs=max(1, min(8, os.cpu_count() or 1)),
                            random_state=RANDOM_STATE + fold,
                        ),
                    ),
                ]
            )
            logistic.fit(X_tr, y_tr)
            va_log = align_proba(
                logistic.predict_proba(X_va), logistic.named_steps["model"].classes_
            )
            te_log = align_proba(
                logistic.predict_proba(X_test_model),
                logistic.named_steps["model"].classes_,
            )
            model_scores["logistic"].append(
                balanced_accuracy_score(y_va, va_log.argmax(axis=1))
            )

            rf_preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", _one_hot_encoder(), cat_cols),
                ],
                remainder="passthrough",
            )
            log_stage(f"Fold {fold}: fitting balanced shallow random forest")
            rf = Pipeline(
                steps=[
                    ("prep", rf_preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=220,
                            max_depth=14,
                            min_samples_leaf=20,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=RANDOM_STATE + 100 + fold,
                        ),
                    ),
                ]
            )
            rf.fit(X_tr, y_tr)
            va_rf = align_proba(
                rf.predict_proba(X_va), rf.named_steps["model"].classes_
            )
            te_rf = align_proba(
                rf.predict_proba(X_test_model), rf.named_steps["model"].classes_
            )
            model_scores["random_forest"].append(
                balanced_accuracy_score(y_va, va_rf.argmax(axis=1))
            )

            log_stage(f"Fold {fold}: fitting CatBoost GPU")
            cat_feature_indices = [model_features.index(c) for c in cat_cols]
            cat_model = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="TotalF1",
                iterations=900,
                learning_rate=0.055,
                depth=7,
                l2_leaf_reg=6.0,
                random_seed=RANDOM_STATE + 200 + fold,
                auto_class_weights="Balanced",
                task_type="GPU",
                devices="0",
                gpu_ram_part=0.8,
                verbose=150,
                allow_writing_files=False,
            )
            train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
            valid_pool = Pool(X_va, y_va, cat_features=cat_feature_indices)
            test_pool = Pool(X_test_model, cat_features=cat_feature_indices)
            cat_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            va_cat = align_proba(
                cat_model.predict_proba(valid_pool), cat_model.classes_
            )
            te_cat = align_proba(cat_model.predict_proba(test_pool), cat_model.classes_)
            model_scores["catboost"].append(
                balanced_accuracy_score(y_va, va_cat.argmax(axis=1))
            )

            fold_va = (va_log + va_rf + va_cat) / 3.0
            fold_te = (te_log + te_rf + te_cat) / 3.0
            oof_proba[va_idx] = fold_va
            test_proba += fold_te / len(folds)

            fold_score = balanced_accuracy_score(y_va, fold_va.argmax(axis=1))
            print(f"Fold {fold} panel balanced_accuracy: {fold_score:.6f}", flush=True)

    with aide_stage("score_stage"):
        oof_pred_int = oof_proba.argmax(axis=1)
        cv_score = balanced_accuracy_score(y, oof_pred_int)
        print(f"OOF balanced_accuracy: {cv_score:.6f}", flush=True)
        for name, scores in model_scores.items():
            print(
                f"{name} mean fold balanced_accuracy: {np.mean(scores):.6f}", flush=True
            )

    with aide_stage("write_outputs_stage"):
        pred_labels = np.array([INT_TO_LABEL[i] for i in test_proba.argmax(axis=1)])

        submission = pd.DataFrame(
            {
                ID_COL: sample_sub[ID_COL].to_numpy(),
                TARGET: pred_labels,
            }
        )

        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=np.int64),
                "target": y_labels,
                "prediction": np.array([INT_TO_LABEL[i] for i in oof_pred_int]),
            }
        )

        test_pred_df = pd.DataFrame({ID_COL: sample_sub[ID_COL].to_numpy()})
        for cls in CLASS_ORDER:
            test_pred_df[f"prob_{cls}"] = test_proba[:, LABEL_TO_INT[cls]]
        test_pred_df[TARGET] = pred_labels

        write_submission(submission)
        write_oof_predictions(oof_df)
        write_test_predictions(test_pred_df)


if __name__ == "__main__":
    main()
