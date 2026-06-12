import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aide_solution_helpers import (
    aide_stage,
    load_competition_data,
    log_stage,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "class"
ID_COL = "id"
CLASSES = np.array(["GALAXY", "QSO", "STAR"])
BANDS = ["u", "g", "r", "i", "z"]
WAVELENGTHS = np.array([3543.0, 4770.0, 6231.0, 7625.0, 9134.0], dtype=float)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in BANDS + ["redshift", "alpha", "delta"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    mag = out[BANDS].copy()
    mag = mag.where(np.isfinite(mag), np.nan)
    mag = mag.mask(mag <= -100.0, np.nan)
    mag = mag.clip(lower=-5.0, upper=40.0)

    flux = np.power(10.0, -0.4 * mag)
    flux = flux.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total_flux = flux.sum(axis=1).astype(float)
    safe_total = total_flux.replace(0.0, np.nan)

    for band in BANDS:
        out[f"flux_{band}"] = flux[band].astype(float)
        out[f"log_flux_{band}"] = np.log1p(flux[band].astype(float) * 1e12)
        out[f"flux_share_{band}"] = (flux[band] / safe_total).fillna(0.0).astype(float)

    out["total_flux"] = total_flux
    out["log_total_flux"] = np.log1p(total_flux * 1e12)

    eps = 1e-300
    ratio_pairs = [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("u", "r"),
        ("g", "i"),
        ("r", "z"),
        ("u", "z"),
    ]
    for a, b in ratio_pairs:
        out[f"log_flux_ratio_{a}_{b}"] = np.log((flux[a] + eps) / (flux[b] + eps))

    share_cols = [f"flux_share_{b}" for b in BANDS]
    shares = out[share_cols].to_numpy(dtype=float)
    wave_mean = (shares * WAVELENGTHS.reshape(1, -1)).sum(axis=1)
    centered = WAVELENGTHS.reshape(1, -1) - wave_mean.reshape(-1, 1)
    wave_var = (shares * centered * centered).sum(axis=1)
    wave_std = np.sqrt(np.maximum(wave_var, 0.0))

    out["flux_weighted_wave_mean"] = wave_mean
    out["flux_weighted_wave_std"] = wave_std
    out["flux_wave_moment_3"] = (shares * centered**3).sum(axis=1) / np.maximum(
        wave_std, 1.0
    ) ** 3
    out["flux_wave_moment_4"] = (shares * centered**4).sum(axis=1) / np.maximum(
        wave_std, 1.0
    ) ** 4
    out["flux_entropy"] = -(shares * np.log(np.clip(shares, 1e-12, 1.0))).sum(axis=1)
    out["flux_concentration"] = (shares * shares).sum(axis=1)
    out["blue_red_flux_balance"] = (
        (flux["u"] + flux["g"]) - (flux["i"] + flux["z"])
    ) / (total_flux + eps)

    redshift = (
        out["redshift"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-0.99)
    )
    log1pz = np.log1p(redshift)
    zscale = np.square(1.0 + redshift)
    out["log1p_redshift"] = log1pz
    out["total_flux_z2"] = total_flux * zscale
    out["log_total_flux_plus_2log1pz"] = out["log_total_flux"] + 2.0 * log1pz
    for band in BANDS:
        out[f"log_flux_{band}_plus_2log1pz"] = out[f"log_flux_{band}"] + 2.0 * log1pz
        out[f"flux_{band}_z2"] = flux[band] * zscale

    for a, b in zip(BANDS[:-1], BANDS[1:]):
        out[f"color_{a}_{b}"] = out[a] - out[b]
    out["color_u_z"] = out["u"] - out["z"]
    out["color_g_i"] = out["g"] - out["i"]

    for cat in ["spectral_type", "galaxy_population"]:
        if cat in out.columns:
            out[cat] = out[cat].astype("category")

    return out.drop(columns=[TARGET], errors="ignore")


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in ["spectral_type", "galaxy_population"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols and c != ID_COL]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def class_recall_text(y_true, y_pred):
    recalls = recall_score(
        y_true, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    return ", ".join(f"{cls}:{rec:.5f}" for cls, rec in zip(CLASSES, recalls))


def main():
    np.random.seed(RANDOM_STATE)

    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        y = train[TARGET].astype(str).to_numpy()
        test_ids = sample_sub[ID_COL].to_numpy()

        X = make_features(train)
        X_test = make_features(test)

        feature_cols = [c for c in X.columns if c != ID_COL]
        X = X[feature_cols]
        X_test = X_test[feature_cols]

        cat_cols = [c for c in ["spectral_type", "galaxy_population"] if c in X.columns]
        cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]
        for c in cat_cols:
            X[c] = X[c].astype(str).fillna("missing")
            X_test[c] = X_test[c].astype(str).fillna("missing")

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )

    model_names = ["linear", "extratrees", "catboost"]
    oof_proba = {
        name: np.zeros((len(X), len(CLASSES)), dtype=float) for name in model_names
    }
    test_proba = {
        name: np.zeros((len(X_test), len(CLASSES)), dtype=float) for name in model_names
    }

    with aide_stage("fit_predict_fold_stage"):
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            y_tr, y_va = y[tr_idx], y[va_idx]

            log_stage(f"Fold {fold}/{N_SPLITS}: fitting balanced logistic regression")
            linear_model = Pipeline(
                [
                    ("prep", make_preprocessor(X)),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=400,
                            multi_class="auto",
                            solver="lbfgs",
                            n_jobs=max(1, min(8, os.cpu_count() or 1)),
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            linear_model.fit(X_tr, y_tr)
            oof_proba["linear"][va_idx] = linear_model.predict_proba(X_va)
            test_proba["linear"] += linear_model.predict_proba(X_test) / N_SPLITS

            log_stage(f"Fold {fold}/{N_SPLITS}: fitting balanced ExtraTrees")
            tree_prep = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        SimpleImputer(strategy="median"),
                        [c for c in X.columns if c not in cat_cols],
                    ),
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        cat_cols,
                    ),
                ],
                remainder="drop",
                sparse_threshold=0.3,
            )
            et_model = Pipeline(
                [
                    ("prep", tree_prep),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=350,
                            max_features="sqrt",
                            min_samples_leaf=3,
                            class_weight="balanced",
                            random_state=RANDOM_STATE + fold,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            et_model.fit(X_tr, y_tr)
            oof_proba["extratrees"][va_idx] = et_model.predict_proba(X_va)
            test_proba["extratrees"] += et_model.predict_proba(X_test) / N_SPLITS

            log_stage(f"Fold {fold}/{N_SPLITS}: fitting CatBoost GPU")
            from catboost import CatBoostClassifier, Pool

            cat_model = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="BalancedAccuracy",
                iterations=1800,
                learning_rate=0.045,
                depth=6,
                l2_leaf_reg=5.0,
                random_seed=RANDOM_STATE + fold,
                auto_class_weights="Balanced",
                task_type="GPU",
                devices="0",
                gpu_ram_part=0.8,
                verbose=200,
                allow_writing_files=False,
            )
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_feature_indices)
            te_pool = Pool(X_test, cat_features=cat_feature_indices)
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            oof_proba["catboost"][va_idx] = cat_model.predict_proba(va_pool)
            test_proba["catboost"] += cat_model.predict_proba(te_pool) / N_SPLITS

    with aide_stage("score_stage"):
        scores = {}
        for name in model_names:
            pred = CLASSES[np.argmax(oof_proba[name], axis=1)]
            score = balanced_accuracy_score(y, pred)
            scores[name] = score
            print(f"{name} OOF balanced_accuracy: {score:.6f}", flush=True)
            print(f"{name} per-class recall: {class_recall_text(y, pred)}", flush=True)

        primary_model = max(scores, key=scores.get)
        print(
            f"Primary model selected by OOF balanced accuracy: {primary_model}",
            flush=True,
        )

    with aide_stage("write_outputs_stage"):
        final_oof = oof_proba[primary_model]
        final_test = test_proba[primary_model]
        final_oof_labels = CLASSES[np.argmax(final_oof, axis=1)]
        final_test_labels = CLASSES[np.argmax(final_test, axis=1)]

        submission = pd.DataFrame({ID_COL: test_ids, TARGET: final_test_labels})
        write_submission(submission)

        oof_df = pd.DataFrame(
            {"row": np.arange(len(train)), "target": y, "prediction": final_oof_labels}
        )
        write_oof_predictions(oof_df)

        test_pred_df = pd.DataFrame({ID_COL: test_ids, TARGET: final_test_labels})
        for j, cls in enumerate(CLASSES):
            test_pred_df[f"prob_{cls}"] = final_test[:, j]
        write_test_predictions(test_pred_df)


if __name__ == "__main__":
    main()
