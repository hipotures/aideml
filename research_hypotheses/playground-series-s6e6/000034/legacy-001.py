import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)

C_LIGHT = 299_792.458
H0 = 70.0
OMEGA_M = 0.30


def add_cosmology_features(train: pd.DataFrame, test: pd.DataFrame):
    y = train["class"].copy()
    train_base = train.drop(columns=["class"]).copy()
    combined = pd.concat([train_base, test], axis=0, ignore_index=True)

    cat_cols = ["spectral_type", "galaxy_population"]
    cat_encoded = pd.get_dummies(combined[cat_cols], prefix=cat_cols, dtype=np.float32)

    feats = combined.drop(columns=cat_cols + ["id"]).copy()
    feats = pd.concat(
        [feats.reset_index(drop=True), cat_encoded.reset_index(drop=True)], axis=1
    )

    z = feats["redshift"].to_numpy(dtype=np.float64)
    redshift_invalid = (z <= 0).astype(np.float32)
    redshift_near_zero = ((z > 0) & (z <= 1e-3)).astype(np.float32)
    redshift_low = (z < 0.1).astype(np.float32)
    redshift_mid = ((z >= 0.1) & (z < 2.0)).astype(np.float32)
    redshift_high = (z >= 2.0).astype(np.float32)

    feats["redshift_invalid"] = redshift_invalid
    feats["redshift_near_zero"] = redshift_near_zero
    feats["redshift_low"] = redshift_low
    feats["redshift_mid"] = redshift_mid
    feats["redshift_high"] = redshift_high
    feats["redshift_high_plus"] = (z >= 4.0).astype(np.float32)

    z_for_distance = np.clip(z, 1e-6, 10.0)
    z_cap = float(np.max(z_for_distance))
    grid_n = max(2048, int((z_cap + 1.0) * 600))
    z_grid = np.linspace(0.0, z_cap, grid_n, dtype=np.float64)

    e_inv = 1.0 / np.sqrt(OMEGA_M * (1.0 + z_grid) ** 3 + (1.0 - OMEGA_M))
    dz = np.diff(z_grid)
    integral = np.concatenate([[0.0], np.cumsum((e_inv[:-1] + e_inv[1:]) * 0.5 * dz)])
    lum_proxy_grid = (C_LIGHT / H0) * (1.0 + z_grid) * integral

    lum_proxy = np.interp(z_for_distance, z_grid, lum_proxy_grid)
    dist_modulus = 5.0 * np.log10(np.maximum(lum_proxy, 1e-6)) + 25.0

    feats["luminosity_distance_proxy"] = lum_proxy.astype(np.float32)
    feats["log_luminosity_distance"] = np.log1p(lum_proxy).astype(np.float32)
    feats["distance_modulus"] = dist_modulus.astype(np.float32)

    for band in ["u", "g", "r", "i", "z"]:
        feats[f"M_{band}"] = (feats[band] - dist_modulus).astype(np.float32)

    mabs_cols = [f"M_{band}" for band in ["u", "g", "r", "i", "z"]]
    feats["mabs_mean"] = feats[mabs_cols].mean(axis=1).astype(np.float32)
    feats["mabs_std"] = feats[mabs_cols].std(axis=1).astype(np.float32)
    feats["mabs_range"] = (
        feats[mabs_cols].max(axis=1) - feats[mabs_cols].min(axis=1)
    ).astype(np.float32)

    feats["g_minus_r"] = (feats["g"] - feats["r"]).astype(np.float32)
    feats["u_minus_g"] = (feats["u"] - feats["g"]).astype(np.float32)
    feats["r_minus_i"] = (feats["r"] - feats["i"]).astype(np.float32)
    feats["i_minus_z"] = (feats["i"] - feats["z"]).astype(np.float32)

    feats["Mr_times_gr_color"] = (feats["M_r"] * feats["g_minus_r"]).astype(np.float32)
    feats["Mr_times_ug_color"] = (feats["M_r"] * feats["u_minus_g"]).astype(np.float32)
    feats["Mi_times_ri_color"] = (feats["M_i"] * feats["r_minus_i"]).astype(np.float32)
    feats["Mg_minus_Mr"] = (feats["M_g"] - feats["M_r"]).astype(np.float32)

    for col in cat_encoded.columns:
        feats[f"{col}_z_low"] = (feats[col] * feats["redshift_low"]).astype(np.float32)
        feats[f"{col}_z_mid"] = (feats[col] * feats["redshift_mid"]).astype(np.float32)
        feats[f"{col}_z_high"] = (feats[col] * feats["redshift_high"]).astype(
            np.float32
        )

    feats = feats.astype(np.float32)

    X_train = feats.iloc[: len(train)].copy().reset_index(drop=True)
    X_test = feats.iloc[len(train) :].reset_index(drop=True).copy()
    return X_train, X_test, y.reset_index(drop=True)


def build_catboost_model(use_gpu: bool = True):
    params = {
        "iterations": 900,
        "learning_rate": 0.07,
        "depth": 8,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": 42,
        "auto_class_weights": "Balanced",
        "verbose": 100,
    }
    if use_gpu:
        params.update(
            {
                "task_type": "GPU",
                "devices": "0",
                "gpu_ram_part": 0.8,
            }
        )
    return CatBoostClassifier(**params)


def main():
    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        X_train, X_test, y = add_cosmology_features(train, test)

    with aide_stage("make_folds_stage"):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        classes = le.classes_.tolist()
        n_classes = len(classes)
        n_train = len(y_enc)
        n_test = len(X_test)

        oof_pred_proba = np.zeros((n_train, n_classes), dtype=np.float32)
        test_pred_proba = np.zeros((n_test, n_classes), dtype=np.float32)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with aide_stage("fit_predict_fold_stage"):
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_enc), start=1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

            log_stage(f"Fold {fold_idx}/5: fitting CatBoost (GPU first)")

            try:
                model = build_catboost_model(use_gpu=True)
                model.fit(X_tr, y_tr)
            except Exception as e:
                log_stage(
                    f"Fold {fold_idx}: GPU CatBoost failed ({type(e).__name__}: {e}); falling back to CPU."
                )
                model = build_catboost_model(use_gpu=False)
                model.fit(X_tr, y_tr)

            va_prob = model.predict_proba(X_va)
            oof_pred_proba[va_idx] = va_prob
            test_pred_proba += model.predict_proba(X_test) / skf.n_splits

            fold_pred = np.argmax(va_prob, axis=1)
            fold_bal_acc = balanced_accuracy_score(y_va, fold_pred)
            log_stage(
                f"Fold {fold_idx} validation balanced accuracy: {fold_bal_acc:.6f}"
            )

    with aide_stage("score_stage"):
        oof_pred = np.argmax(oof_pred_proba, axis=1)
        oof_pred_labels = le.inverse_transform(oof_pred)
        oof_true_labels = y
        valid_bal_acc = balanced_accuracy_score(oof_true_labels, oof_pred_labels)
        print(f"OOF balanced accuracy: {valid_bal_acc:.6f}", flush=True)

        test_pred_labels = le.inverse_transform(np.argmax(test_pred_proba, axis=1))
        test_prob_cols = [f"{cls}_prob" for cls in classes]
        test_prob_df = pd.concat(
            [
                sample_sub[["id"]].reset_index(drop=True).copy(),
                pd.DataFrame(test_pred_proba, columns=test_prob_cols),
            ],
            axis=1,
        )

        with aide_stage("write_outputs_stage"):
            oof_df = pd.DataFrame(
                {
                    "row": np.arange(len(train), dtype=np.int64),
                    "target": oof_true_labels.to_numpy(),
                    "prediction": oof_pred_labels,
                }
            )
            write_oof_predictions(oof_df)

            write_test_predictions(test_prob_df)

            submission = sample_sub[["id"]].copy().reset_index(drop=True)
            submission["class"] = test_pred_labels
            write_submission(submission)


if __name__ == "__main__":
    main()
