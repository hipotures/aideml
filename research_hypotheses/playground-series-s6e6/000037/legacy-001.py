import numpy as np
import pandas as pd
from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)

TARGET_COL = "class"
ID_COL = "id"
SEED = 42


def _build_galactic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # J2000 Galactic pole / zero point constants (degrees)
    ra_gp = np.deg2rad(192.859508)
    dec_gp = np.deg2rad(27.128336)
    l_node = np.deg2rad(32.931918)

    ra = np.deg2rad(out["alpha"].to_numpy(dtype=np.float64))
    dec = np.deg2rad(out["delta"].to_numpy(dtype=np.float64))

    sin_b = np.sin(dec) * np.sin(dec_gp) + np.cos(dec) * np.cos(dec_gp) * np.cos(
        ra - ra_gp
    )
    b_rad = np.arcsin(np.clip(sin_b, -1.0, 1.0))
    l_rad = (
        np.arctan2(
            np.cos(dec) * np.sin(ra - ra_gp),
            np.sin(dec) * np.cos(dec_gp)
            - np.cos(dec) * np.sin(dec_gp) * np.cos(ra - ra_gp),
        )
        + l_node
    )

    b_deg = np.degrees(b_rad)
    l_deg = np.degrees(l_rad) % 360.0

    out["galactic_l_deg"] = l_deg
    out["galactic_b_deg"] = b_deg
    abs_b = np.abs(b_deg)

    # Analytic dust-column proxy from latitude (crude synthetic/sky-signal-safe approximation)
    # floor at 0.5 deg avoids blow-up at equator while preserving relative low-latitude structure
    sin_floor = np.sin(np.deg2rad(0.5))
    dust_proxy = 1.0 / np.maximum(np.sin(np.deg2rad(abs_b)), sin_floor)
    dust_proxy = np.clip(dust_proxy, 1.0, 120.0)
    out["dust_proxy"] = dust_proxy
    out["dust_proxy_log"] = np.log1p(out["dust_proxy"])
    out["dust_proxy_tanh"] = np.tanh(out["dust_proxy"] / 15.0)
    out["dust_proxy_rel"] = out["dust_proxy"] / out["dust_proxy"].quantile(0.95)

    out["abs_galactic_b_deg"] = abs_b
    out["low_lat_3deg"] = (abs_b < 3.0).astype(np.int8)
    out["low_lat_6deg"] = (abs_b < 6.0).astype(np.int8)
    out["low_lat_10deg"] = (abs_b < 10.0).astype(np.int8)
    out["low_lat_20deg"] = (abs_b < 20.0).astype(np.int8)
    out["dust_proxy_x_lowlat3"] = out["dust_proxy"] * out["low_lat_3deg"].astype(
        np.float32
    )

    # Fixed SDSS ugriz extinction coefficients (SFD/Schlafly-Finkbeiner-style values)
    # Au, Ag, Ar, Ai, Az
    coeff = {"u": 5.155, "g": 3.793, "r": 2.751, "i": 2.086, "z": 1.479}

    for band in ["u", "g", "r", "i", "z"]:
        out[f"m0_{band}"] = (
            out[band].astype(np.float32) - coeff[band] * out["dust_proxy"]
        )

    # Adjacent and broader colors (observed + dereddened)
    color_pairs = [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("u", "r"),
        ("g", "i"),
        ("i", "z"),
        ("u", "i"),
        ("u", "z"),
        ("g", "z"),
    ]
    for c1, c2 in color_pairs:
        out[f"color_obs_{c1}_{c2}"] = out[c1].astype(np.float32) - out[c2].astype(
            np.float32
        )
        out[f"color_dered_{c1}_{c2}"] = out[f"m0_{c1}"] - out[f"m0_{c2}"]

    # Projection of adjacent-color vector onto fixed reddening direction
    adj_obs = np.column_stack(
        [
            out["color_obs_u_g"],
            out["color_obs_g_r"],
            out["color_obs_r_i"],
            out["color_obs_i_z"],
        ]
    ).astype(np.float32)
    adj_dered = np.column_stack(
        [
            out["color_dered_u_g"],
            out["color_dered_g_r"],
            out["color_dered_r_i"],
            out["color_dered_i_z"],
        ]
    ).astype(np.float32)

    redd_dir = np.array(
        [
            coeff["u"] - coeff["g"],
            coeff["g"] - coeff["r"],
            coeff["r"] - coeff["i"],
            coeff["i"] - coeff["z"],
        ],
        dtype=np.float64,
    )
    redd_unit = redd_dir / np.linalg.norm(redd_dir)

    proj_obs = adj_obs @ redd_unit
    proj_dered = adj_dered @ redd_unit
    orth_vec = adj_obs - np.outer(proj_obs, redd_dir / np.linalg.norm(redd_dir))
    out["reddening_projection_strength"] = proj_obs.astype(np.float32)
    out["reddening_projection_shift"] = (proj_obs - proj_dered).astype(np.float32)
    out["reddening_orthogonal_norm"] = np.linalg.norm(orth_vec, axis=1).astype(
        np.float32
    )

    # Compact color summaries likely useful for class separation
    out["blue_color_dered_mean"] = (
        out["color_dered_u_g"] + out["color_dered_g_r"]
    ) / 2.0
    out["red_color_dered_mean"] = (
        out["color_dered_r_i"] + out["color_dered_i_z"]
    ) / 2.0
    out["blue_color_obs_mean"] = (out["color_obs_u_g"] + out["color_obs_g_r"]) / 2.0
    out["red_color_obs_mean"] = (out["color_obs_r_i"] + out["color_obs_i_z"]) / 2.0

    # Explicit interactions with redshift and categories (target-free)
    out["dust_x_redshift"] = out["dust_proxy"] * out["redshift"].astype(np.float32)

    sp_d = pd.get_dummies(out["spectral_type"].astype("string"), prefix="spectral_type")
    gp_d = pd.get_dummies(
        out["galaxy_population"].astype("string"), prefix="galaxy_population"
    )
    out = out.drop(columns=["spectral_type", "galaxy_population"])
    out = pd.concat([out, sp_d, gp_d], axis=1)

    for c in sp_d.columns:
        out[f"dust_x_{c}"] = out["dust_proxy"] * out[c].astype(np.float32)
        out[f"redshift_x_{c}"] = out["redshift"].astype(np.float32) * out[c].astype(
            np.float32
        )
    for c in gp_d.columns:
        out[f"dust_x_{c}"] = out["dust_proxy"] * out[c].astype(np.float32)
        out[f"redshift_x_{c}"] = out["redshift"].astype(np.float32) * out[c].astype(
            np.float32
        )

    return out


def _make_model(name: str, n_classes: int, random_state: int = SEED):
    if name == "catboost":
        from catboost import CatBoostClassifier

        def fit_predict(X_tr, y_tr, X_va, X_te, sample_weight=None):
            try:
                log_stage(
                    "CatBoost GPU training requested (task_type=GPU, devices=0, gpu_ram_part=0.8)"
                )
                model = CatBoostClassifier(
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    iterations=320,
                    learning_rate=0.06,
                    depth=8,
                    random_seed=random_state,
                    verbose=False,
                    auto_class_weights="Balanced",
                    task_type="GPU",
                    devices="0",
                    gpu_ram_part=0.8,
                )
                model.fit(X_tr, y_tr)
            except Exception as exc:
                log_stage(
                    f"CatBoost GPU unavailable or failed: {exc}. Falling back to CPU."
                )
                model = CatBoostClassifier(
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    iterations=320,
                    learning_rate=0.06,
                    depth=8,
                    random_seed=random_state,
                    verbose=False,
                    auto_class_weights="Balanced",
                )
                model.fit(X_tr, y_tr)
            return model.predict_proba(X_va), model.predict_proba(X_te)

        return fit_predict

    if name == "xgboost":
        from xgboost import XGBClassifier

        def fit_predict(X_tr, y_tr, X_va, X_te, sample_weight=None):
            X_tr_np = X_tr.astype(np.float32).to_numpy()
            X_va_np = X_va.astype(np.float32).to_numpy()
            X_te_np = X_te.astype(np.float32).to_numpy()

            try:
                log_stage(
                    "XGBoost GPU training requested (tree_method=hist, device=cuda)"
                )
                model = XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    n_estimators=280,
                    learning_rate=0.06,
                    max_depth=8,
                    subsample=0.90,
                    colsample_bytree=0.90,
                    reg_lambda=1.0,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                    tree_method="hist",
                    device="cuda",
                )
                model.fit(X_tr_np, y_tr, sample_weight=sample_weight)
            except Exception as exc:
                log_stage(
                    f"XGBoost GPU unavailable or failed: {exc}. Falling back to CPU (tree_method=hist)."
                )
                model = XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    n_estimators=280,
                    learning_rate=0.06,
                    max_depth=8,
                    subsample=0.90,
                    colsample_bytree=0.90,
                    reg_lambda=1.0,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                    tree_method="hist",
                )
                model.fit(X_tr_np, y_tr, sample_weight=sample_weight)
            return model.predict_proba(X_va_np), model.predict_proba(X_te_np)

        return fit_predict

    raise ValueError(f"Unknown model: {name}")


def main():
    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        y_text = train[TARGET_COL].astype(str).values
        classes = np.array(sorted(np.unique(y_text)))
        class_to_id = {c: i for i, c in enumerate(classes)}
        id_to_class = {i: c for c, i in class_to_id.items()}
        y = np.array([class_to_id[v] for v in y_text], dtype=np.int32)

        # Auxiliary dataset is intentionally not used to avoid any target/domain mixing risk and to stay faithful to covariate-only hypothesis.
        train_features_raw = train.drop(columns=[TARGET_COL, ID_COL]).copy()
        test_features_raw = test.drop(columns=[ID_COL]).copy()

        all_features = pd.concat(
            [train_features_raw, test_features_raw], axis=0, ignore_index=True
        )
        all_features = _build_galactic_features(all_features)

        # remove raw identifiers from feature set
        if "id" in all_features.columns:
            all_features = all_features.drop(columns=["id"])

        # use consistent train/test order from concatenation
        X_all = all_features.astype(np.float32)
        X_train = X_all.iloc[: len(train)].reset_index(drop=True)
        X_test = X_all.iloc[len(train) :].reset_index(drop=True)
        y = y

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        n_train = len(X_train)
        n_test = len(X_test)
        n_classes = len(classes)

        model_names = ["catboost", "xgboost"]
        oof_per_model = {}
        test_per_model = {}
        score_per_model = {}

    for model_name in model_names:
        with aide_stage("fit_predict_fold_stage"):
            oof_proba = np.zeros((n_train, n_classes), dtype=np.float64)
            test_proba = np.zeros((n_test, n_classes), dtype=np.float64)
            fold_scores = []

            fit_predict = _make_model(
                model_name, n_classes=n_classes, random_state=SEED
            )

            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), start=1):
                X_tr = X_train.iloc[tr_idx]
                X_va = X_train.iloc[va_idx]
                y_tr = y[tr_idx]
                y_va = y[va_idx]

                sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr)

                log_stage(f"[{model_name}] fold {fold_idx}/5 start")
                va_proba, te_proba = fit_predict(
                    X_tr,
                    y_tr,
                    X_va,
                    X_test,
                    sample_weight=sample_weight,
                )
                oof_proba[va_idx] = va_proba
                test_proba += te_proba / skf.n_splits

                va_pred = np.argmax(va_proba, axis=1)
                fold_score = balanced_accuracy_score(y_va, va_pred)
                fold_scores.append(float(fold_score))
                log_stage(
                    f"[{model_name}] fold {fold_idx}/5 balanced_accuracy={fold_score:.6f}"
                )

            mean_score = float(np.mean(fold_scores))
            oof_per_model[model_name] = oof_proba
            test_per_model[model_name] = test_proba
            score_per_model[model_name] = mean_score

    with aide_stage("score_stage"):
        for name in model_names:
            print(
                f"{name} CV balanced_accuracy (5-fold): {score_per_model[name]:.6f}",
                flush=True,
            )

        best_model = max(score_per_model, key=score_per_model.get)
        print(
            f"Best model: {best_model} (CV balanced_accuracy={score_per_model[best_model]:.6f})",
            flush=True,
        )

        best_oof = oof_per_model[best_model]
        best_test = test_per_model[best_model]

        oof_pred = np.argmax(best_oof, axis=1)
        oof_acc = balanced_accuracy_score(y, oof_pred)
        print(
            f"Primary validation metric (balanced_accuracy from OOF of best model {best_model}): {oof_acc:.6f}",
            flush=True,
        )

    with aide_stage("write_outputs_stage"):
        oof_pred_text = np.array(
            [id_to_class[i] for i in np.argmax(best_oof, axis=1)], dtype=object
        )
        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=np.int64),
                "target": y_text,
                "prediction": oof_pred_text,
            }
        )
        write_oof_predictions(oof_df)

        test_pred_text = np.array(
            [id_to_class[i] for i in np.argmax(best_test, axis=1)], dtype=object
        )
        submission = pd.DataFrame(
            {
                "id": sample_sub[ID_COL].values,
                "class": test_pred_text,
            }
        )
        write_submission(submission)

        test_pred_probs = pd.DataFrame(
            {
                "id": sample_sub[ID_COL].values,
                classes[0]: best_test[:, 0],
                classes[1]: best_test[:, 1],
                classes[2]: best_test[:, 2],
            }
        )
        write_test_predictions(test_pred_probs)


if __name__ == "__main__":
    main()
