import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight

from aide_solution_helpers import (
    load_competition_data,
    working_dir,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    write_validation_predictions,
    aide_stage,
    log_stage,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
CLASS_NAMES = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def join_parts(*parts: pd.Series) -> pd.Series:
    out = parts[0].astype("string").fillna("__MISSING__")
    for part in parts[1:]:
        out = out + "__" + part.astype("string").fillna("__MISSING__")
    return out


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    numeric_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    base_cat_cols = ["spectral_type", "galaxy_population"]
    keep_cols = numeric_cols + base_cat_cols

    log_stage("Covariate-only combined train+test features; no labels used")
    combined = pd.concat(
        [train_df[keep_cols].copy(), test_df[keep_cols].copy()],
        axis=0,
        ignore_index=True,
    )

    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype(np.float64)
    for col in base_cat_cols:
        combined[col] = combined[col].astype("string").fillna("__MISSING__")

    mags = combined[["u", "g", "r", "i", "z"]].to_numpy(dtype=np.float64)
    band_names = np.array(["u", "g", "r", "i", "z"], dtype=object)

    combined["u_g"] = combined["u"] - combined["g"]
    combined["u_r"] = combined["u"] - combined["r"]
    combined["u_i"] = combined["u"] - combined["i"]
    combined["u_z"] = combined["u"] - combined["z"]
    combined["g_r"] = combined["g"] - combined["r"]
    combined["g_i"] = combined["g"] - combined["i"]
    combined["g_z"] = combined["g"] - combined["z"]
    combined["r_i"] = combined["r"] - combined["i"]
    combined["r_z"] = combined["r"] - combined["z"]

    k = np.arange(5, dtype=np.float64)
    mag_sum = mags.sum(axis=1)
    mag_sed_slope = (5.0 * (mags * k).sum(axis=1) - 10.0 * mag_sum) / 50.0
    mag_sed_intercept = (mag_sum - 10.0 * mag_sed_slope) / 5.0
    mag_fit = mag_sed_intercept[:, None] + mag_sed_slope[:, None] * k[None, :]
    mag_resid = mags - mag_fit
    mag_d2 = np.diff(mags, n=2, axis=1)

    min_idx = np.argmin(mags, axis=1)
    max_idx = np.argmax(mags, axis=1)

    combined["mag_sed_slope"] = mag_sed_slope
    combined["mag_sed_intercept"] = mag_sed_intercept
    combined["mag_sed_resid_mean_abs"] = np.abs(mag_resid).mean(axis=1)
    combined["mag_sed_resid_std"] = mag_resid.std(axis=1)
    combined["mag_sed_d2_mean"] = mag_d2.mean(axis=1)
    combined["mag_sed_d2_abs_mean"] = np.abs(mag_d2).mean(axis=1)
    combined["mag_sed_d2_std"] = mag_d2.std(axis=1)
    combined["mag_mean"] = mags.mean(axis=1)
    combined["mag_std"] = mags.std(axis=1)
    combined["mag_range"] = mags.max(axis=1) - mags.min(axis=1)
    combined["band_min_mag"] = mags.min(axis=1)
    combined["band_max_mag"] = mags.max(axis=1)
    combined["mag_min_band"] = band_names[min_idx]
    combined["mag_max_band"] = band_names[max_idx]
    combined["band_min_band"] = combined["mag_min_band"]
    combined["band_max_band"] = combined["mag_max_band"]

    alpha_deg = np.mod(combined["alpha"].to_numpy(dtype=np.float64), 360.0)
    delta_deg = combined["delta"].to_numpy(dtype=np.float64)
    alpha_rad = np.deg2rad(alpha_deg)
    delta_rad = np.deg2rad(delta_deg)

    alpha_sin = np.sin(alpha_rad)
    alpha_cos = np.cos(alpha_rad)
    delta_sin = np.sin(delta_rad)
    delta_cos = np.cos(delta_rad)

    combined["alpha_sin"] = alpha_sin
    combined["alpha_cos"] = alpha_cos
    combined["delta_sin"] = delta_sin
    combined["delta_cos"] = delta_cos

    sky_x = delta_cos * alpha_cos
    sky_y = delta_cos * alpha_sin
    sky_z = delta_sin

    combined["sky_x"] = sky_x
    combined["sky_y"] = sky_y
    combined["sky_z"] = sky_z
    combined["sky_xy"] = sky_x * sky_y
    combined["sky_xz"] = sky_x * sky_z
    combined["sky_yz"] = sky_y * sky_z
    combined["sky_x2_minus_y2"] = sky_x**2 - sky_y**2

    for order in [2, 3, 4, 5, 6, 7, 8]:
        combined[f"alpha_sin{order}"] = np.sin(order * alpha_rad)
        combined[f"alpha_cos{order}"] = np.cos(order * alpha_rad)
        combined[f"delta_sin{order}"] = np.sin(order * delta_rad)
        combined[f"delta_cos{order}"] = np.cos(order * delta_rad)

    sky_alpha_bin_24 = np.floor(alpha_deg / 15.0).astype(np.int16)
    sky_delta_bin_18 = np.clip(np.floor((delta_deg + 90.0) / 10.0), 0, 17).astype(
        np.int16
    )
    sky_cell_bin = (sky_alpha_bin_24 * 19 + sky_delta_bin_18).astype(np.int32)

    sky_alpha_bin_48 = np.floor(alpha_deg / 7.5).astype(np.int16)
    sky_delta_bin_36 = np.clip(np.floor((delta_deg + 90.0) / 5.0), 0, 35).astype(
        np.int16
    )
    sky_cell_bin_48 = (sky_alpha_bin_48 * 36 + sky_delta_bin_36).astype(np.int32)

    combined["sky_alpha_bin_24"] = sky_alpha_bin_24
    combined["sky_delta_bin_18"] = sky_delta_bin_18
    combined["sky_cell_bin"] = sky_cell_bin
    combined["sky_alpha_bin_48"] = sky_alpha_bin_48
    combined["sky_delta_bin_36"] = sky_delta_bin_36
    combined["sky_cell_bin_48"] = sky_cell_bin_48

    sky_alpha_bin_center_deg = (sky_alpha_bin_24 + 0.5) * 15.0
    sky_delta_bin_center_deg = (sky_delta_bin_18 + 0.5) * 10.0 - 90.0
    combined["sky_alpha_bin_center_deg"] = sky_alpha_bin_center_deg
    combined["sky_alpha_bin_offset_deg"] = alpha_deg - sky_alpha_bin_center_deg
    combined["sky_delta_bin_center_deg"] = sky_delta_bin_center_deg
    combined["sky_delta_bin_offset_deg"] = delta_deg - sky_delta_bin_center_deg

    sky_points = np.column_stack([delta_rad, alpha_rad])
    nn_model = NearestNeighbors(
        n_neighbors=17,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    nn_model.fit(sky_points)
    nn_dist, _ = nn_model.kneighbors(sky_points)
    nn_arcmin = nn_dist * (180.0 / np.pi) * 60.0

    r5 = nn_arcmin[:, 5]
    r10 = nn_arcmin[:, 10]
    r16 = nn_arcmin[:, 16]
    d5 = np.log1p(5.0 / np.maximum(r5**2, 1e-6))
    d10 = np.log1p(10.0 / np.maximum(r10**2, 1e-6))
    d16 = np.log1p(16.0 / np.maximum(r16**2, 1e-6))

    combined["sky_nn5_arcmin"] = r5
    combined["sky_nn10_arcmin"] = r10
    combined["sky_nn16_arcmin"] = r16
    combined["sky_nn5_density"] = d5
    combined["sky_nn10_density"] = d10
    combined["sky_nn16_density"] = d16
    combined["sky_nn_density_ratio_5_16"] = d5 / np.maximum(d16, 1e-6)

    redshift = combined["redshift"].to_numpy(dtype=np.float64)
    combined["redshift_abs"] = np.abs(redshift)
    combined["redshift_sq"] = redshift**2
    combined["redshift_log_abs"] = np.log1p(np.abs(redshift))
    combined["redshift_signed_log_abs"] = np.sign(redshift) * np.log1p(np.abs(redshift))
    redshift_bin_20 = pd.qcut(
        combined["redshift"], q=20, labels=False, duplicates="drop"
    )
    combined["redshift_bin_20"] = pd.Series(
        redshift_bin_20, index=combined.index
    ).astype(np.int16)

    rotation = np.array(
        [
            [-0.0548755604, -0.8734370902, -0.4838350155],
            [0.4941094279, -0.44482963, 0.7469822445],
            [-0.8676661490, -0.1980763734, 0.4559837762],
        ],
        dtype=np.float64,
    )
    sky_xyz = np.column_stack([sky_x, sky_y, sky_z])
    gal_xyz = sky_xyz @ rotation.T
    gal_x = gal_xyz[:, 0]
    gal_y = gal_xyz[:, 1]
    gal_z = np.clip(gal_xyz[:, 2], -1.0, 1.0)

    galactic_l_rad = np.mod(np.arctan2(gal_y, gal_x), 2.0 * np.pi)
    galactic_b_rad = np.arcsin(gal_z)
    galactic_l = np.rad2deg(galactic_l_rad)
    galactic_b = np.rad2deg(galactic_b_rad)

    combined["galactic_l"] = galactic_l
    combined["galactic_b"] = galactic_b
    combined["galactic_l_sin"] = np.sin(galactic_l_rad)
    combined["galactic_l_cos"] = np.cos(galactic_l_rad)
    combined["galactic_b_sin"] = np.sin(galactic_b_rad)
    combined["galactic_b_cos"] = np.cos(galactic_b_rad)
    combined["galactic_b_abs"] = np.abs(galactic_b)

    combined["galactic_l_bin_24"] = np.floor(galactic_l / 15.0).astype(np.int16)
    combined["galactic_b_bin_12"] = np.clip(
        np.floor((galactic_b + 90.0) / 15.0), 0, 11
    ).astype(np.int16)
    combined["galactic_cell_bin"] = (
        combined["galactic_l_bin_24"] * 12 + combined["galactic_b_bin_12"]
    ).astype(np.int32)

    combined["spectral_type__redshift_bin_20"] = join_parts(
        combined["spectral_type"], combined["redshift_bin_20"]
    )
    combined["galaxy_population__redshift_bin_20"] = join_parts(
        combined["galaxy_population"], combined["redshift_bin_20"]
    )
    combined["spectral_type__galactic_cell_bin"] = join_parts(
        combined["spectral_type"], combined["galactic_cell_bin"]
    )
    combined["galaxy_population__galactic_cell_bin"] = join_parts(
        combined["galaxy_population"], combined["galactic_cell_bin"]
    )

    combined["spectral_type__galaxy_population"] = join_parts(
        combined["spectral_type"], combined["galaxy_population"]
    )
    combined["spectral_type__band_min_band"] = join_parts(
        combined["spectral_type"], combined["band_min_band"]
    )
    combined["galaxy_population__band_max_band"] = join_parts(
        combined["galaxy_population"], combined["band_max_band"]
    )
    combined["spectral_type__sky_alpha_bin_24"] = join_parts(
        combined["spectral_type"], combined["sky_alpha_bin_24"]
    )
    combined["galaxy_population__sky_alpha_bin_24"] = join_parts(
        combined["galaxy_population"], combined["sky_alpha_bin_24"]
    )
    combined["spectral_type__sky_cell_bin"] = join_parts(
        combined["spectral_type"], combined["sky_cell_bin"]
    )
    combined["spectral_type__sky_alpha_bin_48"] = join_parts(
        combined["spectral_type"], combined["sky_alpha_bin_48"]
    )
    combined["galaxy_population__sky_alpha_bin_48"] = join_parts(
        combined["galaxy_population"], combined["sky_alpha_bin_48"]
    )
    combined["spectral_type__sky_cell_bin_48"] = join_parts(
        combined["spectral_type"], combined["sky_cell_bin_48"]
    )
    combined["spectral_type__galaxy_population__sky_delta_bin_18"] = join_parts(
        combined["spectral_type"],
        combined["galaxy_population"],
        combined["sky_delta_bin_18"],
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
        freq_map = combined[col].value_counts(normalize=True, dropna=False)
        combined[f"{col}_freq"] = combined[col].map(freq_map).astype(np.float64)

    for col in ["u", "g", "r", "i", "z", "redshift"]:
        combined[f"{col}_qrank"] = (
            combined[col].rank(method="average", pct=True).astype(np.float64)
        )

    categorical_cols = [
        "spectral_type",
        "galaxy_population",
        "mag_min_band",
        "mag_max_band",
        "band_min_band",
        "band_max_band",
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
        if pd.api.types.is_numeric_dtype(combined[col]):
            combined[col] = (
                pd.to_numeric(combined[col], errors="coerce")
                .fillna(-1)
                .astype(np.int32)
            )
        else:
            values = combined[col].astype("string").fillna("__MISSING__")
            codes, _ = pd.factorize(values, sort=False)
            combined[col] = codes.astype(np.int32)

    for col in combined.columns:
        if col in categorical_cols:
            continue
        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype(np.float32)

    n_train = len(train_df)
    train_feat = combined.iloc[:n_train].reset_index(drop=True)
    test_feat = combined.iloc[n_train:].reset_index(drop=True)
    return train_feat, test_feat, categorical_cols


def fit_catboost(X_train, y_train, X_valid, y_valid, X_test, categorical_cols):
    cat_idx = [X_train.columns.get_loc(col) for col in categorical_cols]
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=RANDOM_STATE,
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.8,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=False,
    )
    return model.predict_proba(X_valid), model.predict_proba(X_test)


def fit_xgboost(X_train, y_train, X_valid, y_valid, X_test):
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1.0,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        early_stopping_rounds=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model.predict_proba(X_valid), model.predict_proba(X_test)


def fit_lightgbm(X_train, y_train, X_valid, y_valid, X_test, categorical_cols):
    cat_idx = [X_train.columns.get_loc(col) for col in categorical_cols]
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASS_NAMES),
        class_weight="balanced",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        categorical_feature=cat_idx,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model.predict_proba(X_valid), model.predict_proba(X_test)


def fit_histgb(X_train, y_train, X_valid, y_valid, X_test):
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=12,
        max_leaf_nodes=127,
        max_iter=500,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_valid), model.predict_proba(X_test)


def main():
    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        y_labels = train["class"].astype(str).to_numpy()
        y = np.array([CLASS_TO_INT[label] for label in y_labels], dtype=np.int32)

        train_feat, test_feat, categorical_cols = build_features(train, test)
        train_feat = train_feat.astype({col: np.int32 for col in categorical_cols})
        test_feat = test_feat.astype({col: np.int32 for col in categorical_cols})

        _ = working_dir()

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        folds = list(skf.split(train_feat, y))

    model_names = ["catboost", "xgboost", "lightgbm", "histgb"]
    oof_proba = {
        name: np.zeros((len(train), len(CLASS_NAMES)), dtype=np.float32)
        for name in model_names
    }
    test_proba = {
        name: np.zeros((len(test), len(CLASS_NAMES)), dtype=np.float32)
        for name in model_names
    }
    fold_scores = {name: [] for name in model_names}

    with aide_stage("fit_predict_fold_stage"):
        for fold_idx, (train_idx, valid_idx) in enumerate(folds, start=1):
            X_tr = train_feat.iloc[train_idx]
            X_va = train_feat.iloc[valid_idx]
            y_tr = y[train_idx]
            y_va = y[valid_idx]

            log_stage(f"Fold {fold_idx}/{N_SPLITS} - CatBoost")
            cb_valid_proba, cb_test_proba = fit_catboost(
                X_tr, y_tr, X_va, y_va, test_feat, categorical_cols
            )
            oof_proba["catboost"][valid_idx] = cb_valid_proba
            test_proba["catboost"] += cb_test_proba / N_SPLITS
            fold_scores["catboost"].append(
                balanced_accuracy_score(y_va, cb_valid_proba.argmax(axis=1))
            )

            log_stage(f"Fold {fold_idx}/{N_SPLITS} - XGBoost")
            xgb_valid_proba, xgb_test_proba = fit_xgboost(
                X_tr, y_tr, X_va, y_va, test_feat
            )
            oof_proba["xgboost"][valid_idx] = xgb_valid_proba
            test_proba["xgboost"] += xgb_test_proba / N_SPLITS
            fold_scores["xgboost"].append(
                balanced_accuracy_score(y_va, xgb_valid_proba.argmax(axis=1))
            )

            log_stage(f"Fold {fold_idx}/{N_SPLITS} - LightGBM")
            lgb_valid_proba, lgb_test_proba = fit_lightgbm(
                X_tr, y_tr, X_va, y_va, test_feat, categorical_cols
            )
            oof_proba["lightgbm"][valid_idx] = lgb_valid_proba
            test_proba["lightgbm"] += lgb_test_proba / N_SPLITS
            fold_scores["lightgbm"].append(
                balanced_accuracy_score(y_va, lgb_valid_proba.argmax(axis=1))
            )

            log_stage(f"Fold {fold_idx}/{N_SPLITS} - HistGradientBoosting")
            hist_valid_proba, hist_test_proba = fit_histgb(
                X_tr, y_tr, X_va, y_va, test_feat
            )
            oof_proba["histgb"][valid_idx] = hist_valid_proba
            test_proba["histgb"] += hist_test_proba / N_SPLITS
            fold_scores["histgb"].append(
                balanced_accuracy_score(y_va, hist_valid_proba.argmax(axis=1))
            )

    with aide_stage("score_stage"):
        mean_scores = {}
        for model_name in model_names:
            mean_score = float(np.mean(fold_scores[model_name]))
            mean_scores[model_name] = mean_score
            print(
                f"{model_name} mean_balanced_accuracy={mean_score:.6f} "
                f"folds={[round(x, 6) for x in fold_scores[model_name]]}",
                flush=True,
            )

        best_model = max(mean_scores, key=mean_scores.get)
        best_score = mean_scores[best_model]
        best_oof_labels = CLASS_NAMES[oof_proba[best_model].argmax(axis=1)]
        best_test_labels = CLASS_NAMES[test_proba[best_model].argmax(axis=1)]

        print(
            f"PRIMARY_VALIDATION_METRIC balanced_accuracy={best_score:.6f} best_model={best_model}",
            flush=True,
        )

    with aide_stage("write_outputs_stage"):
        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=np.int64),
                "target": y_labels,
                "prediction": best_oof_labels,
            }
        )

        test_pred_df = pd.DataFrame({"id": sample_sub["id"].to_numpy()})
        for class_idx, class_name in enumerate(CLASS_NAMES):
            test_pred_df[class_name] = test_proba[best_model][:, class_idx]

        submission_df = pd.DataFrame(
            {
                "id": sample_sub["id"].to_numpy(),
                "class": best_test_labels,
            }
        )

        write_oof_predictions(oof_df)
        write_test_predictions(test_pred_df)
        write_submission(submission_df)


if __name__ == "__main__":
    main()
