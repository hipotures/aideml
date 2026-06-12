import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

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
CLASSES = ["GALAXY", "QSO", "STAR"]
BANDS = ["u", "g", "r", "i", "z"]
COLOR_PAIRS = [
    ("u", "g"),
    ("g", "r"),
    ("r", "i"),
    ("i", "z"),
    ("u", "r"),
    ("g", "i"),
    ("r", "z"),
]
SKY_FEATURES = BANDS + [f"{a}_{b}" for a, b in COLOR_PAIRS]


def add_base_features(df):
    out = df.copy()
    for a, b in COLOR_PAIRS:
        out[f"{a}_{b}"] = out[a] - out[b]
    out["u_g_r"] = out["u_g"] - out["g_r"]
    out["g_r_i"] = out["g_r"] - out["r_i"]
    out["r_i_z"] = out["r_i"] - out["i_z"]
    out["mag_mean"] = out[BANDS].mean(axis=1)
    out["mag_std"] = out[BANDS].std(axis=1)
    out["mag_range"] = out[BANDS].max(axis=1) - out[BANDS].min(axis=1)
    out["redshift_abs"] = out["redshift"].abs()
    out["redshift_log1p_abs"] = np.log1p(out["redshift_abs"])
    out["alpha_sin"] = np.sin(np.deg2rad(out["alpha"]))
    out["alpha_cos"] = np.cos(np.deg2rad(out["alpha"]))
    out["delta_sin"] = np.sin(np.deg2rad(out["delta"]))
    out["delta_cos"] = np.cos(np.deg2rad(out["delta"]))
    return out


def add_sky_cells(df, alpha_bins=72, delta_bins=36):
    out = df.copy()
    alpha_bin = np.floor(
        np.clip(out["alpha"].to_numpy(), 0, np.nextafter(360.0, 0.0))
        / (360.0 / alpha_bins)
    ).astype(int)
    delta_clipped = np.clip(out["delta"].to_numpy(), -90.0, np.nextafter(90.0, -90.0))
    delta_bin = np.floor((delta_clipped + 90.0) / (180.0 / delta_bins)).astype(int)
    delta_bin = np.clip(delta_bin, 0, delta_bins - 1)
    out["sky_cell"] = alpha_bin * delta_bins + delta_bin
    out["sky_alpha_bin"] = alpha_bin
    out["sky_delta_bin"] = delta_bin
    return out


def build_sky_table(reference):
    global_medians = reference[SKY_FEATURES].median()
    global_scales = (
        reference[SKY_FEATURES].quantile(0.75) - reference[SKY_FEATURES].quantile(0.25)
    ).replace(0, 1.0)

    grouped = reference.groupby("sky_cell", sort=False)
    med = grouped[SKY_FEATURES].median()
    q75 = grouped[SKY_FEATURES].quantile(0.75)
    q25 = grouped[SKY_FEATURES].quantile(0.25)
    scale = (q75 - q25).replace(0, np.nan)
    counts = grouped.size().rename("sky_cell_count")

    table = pd.DataFrame(index=med.index)
    table["sky_cell_count"] = counts
    table["sky_cell_log_count"] = np.log1p(counts)

    for col in SKY_FEATURES:
        table[f"{col}_cell_median"] = med[col]
        table[f"{col}_cell_offset"] = med[col] - global_medians[col]
        table[f"{col}_cell_scale"] = scale[col].fillna(global_scales[col])
        table[f"{col}_cell_scale_ratio"] = table[f"{col}_cell_scale"] / (
            global_scales[col] + 1e-6
        )

    defaults = {"sky_cell_count": 0.0, "sky_cell_log_count": 0.0}
    for col in SKY_FEATURES:
        defaults[f"{col}_cell_median"] = global_medians[col]
        defaults[f"{col}_cell_offset"] = 0.0
        defaults[f"{col}_cell_scale"] = global_scales[col]
        defaults[f"{col}_cell_scale_ratio"] = 1.0
    return table, defaults


def apply_sky_features(df, table, defaults):
    joined = df.join(table, on="sky_cell")
    for col, value in defaults.items():
        joined[col] = joined[col].fillna(value)

    for col in SKY_FEATURES:
        joined[f"{col}_minus_cell_median"] = joined[col] - joined[f"{col}_cell_median"]
        joined[f"{col}_local_z"] = joined[f"{col}_minus_cell_median"] / (
            joined[f"{col}_cell_scale"] + 1e-6
        )

    return joined.drop(columns=["sky_cell"])


def encode_categoricals(train_df, valid_df, test_df, categorical_cols):
    for col in categorical_cols:
        combined = (
            pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0)
            .astype(str)
            .fillna("__MISSING__")
        )
        categories = {value: idx for idx, value in enumerate(pd.unique(combined))}
        train_df[col] = (
            train_df[col]
            .astype(str)
            .fillna("__MISSING__")
            .map(categories)
            .astype("int32")
        )
        valid_df[col] = (
            valid_df[col]
            .astype(str)
            .fillna("__MISSING__")
            .map(categories)
            .astype("int32")
        )
        test_df[col] = (
            test_df[col]
            .astype(str)
            .fillna("__MISSING__")
            .map(categories)
            .astype("int32")
        )
    return train_df, valid_df, test_df


def make_model_panel():
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=2.0,
                class_weight="balanced",
                multi_class="multinomial",
                max_iter=400,
                solver="lbfgs",
                n_jobs=min(8, os.cpu_count() or 1),
                random_state=RANDOM_STATE,
            ),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350,
            max_depth=22,
            min_samples_leaf=3,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def make_catboost():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=1600,
        learning_rate=0.055,
        depth=8,
        l2_leaf_reg=7.0,
        random_seed=RANDOM_STATE,
        auto_class_weights="Balanced",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.8,
        verbose=250,
        allow_writing_files=False,
    )


def align_proba(model, proba, label_encoder):
    aligned = np.zeros((proba.shape[0], len(label_encoder.classes_)), dtype=np.float64)
    model_classes = getattr(model, "classes_", np.arange(proba.shape[1]))
    for src_idx, cls in enumerate(model_classes):
        if isinstance(cls, str):
            dst_idx = int(np.where(label_encoder.classes_ == cls)[0][0])
        else:
            dst_idx = int(cls)
        aligned[:, dst_idx] = proba[:, src_idx]
    return aligned


with aide_stage("build_features_stage"):
    train, test, sample_sub = load_competition_data()
    y_raw = train[TARGET].copy()
    test_ids = sample_sub[ID_COL].copy()

    train_base = add_sky_cells(add_base_features(train.drop(columns=[TARGET])))
    test_base = add_sky_cells(add_base_features(test))

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_order = list(le.classes_)

with aide_stage("make_folds_stage"):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(train_base, y))

    oof_cat = np.zeros((len(train_base), len(class_order)), dtype=np.float64)
    oof_panel = {
        name: np.zeros_like(oof_cat) for name in ["logistic", "extra_trees", "catboost"]
    }
    test_cat = np.zeros((len(test_base), len(class_order)), dtype=np.float64)
    fold_scores = []
    panel_scores = {name: [] for name in oof_panel}

for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
    with aide_stage("fit_predict_fold_stage"):
        log_stage(f"Fold {fold}: building fold-aware sky-local residual features")
        fold_train_base = train_base.iloc[tr_idx].copy()
        fold_valid_base = train_base.iloc[va_idx].copy()
        fold_test_base = test_base.copy()

        # The sky calibration reference concatenates fold-train and test covariates only;
        # it contains no target column, labels, OOF predictions, or model predictions.
        sky_reference = pd.concat(
            [fold_train_base, fold_test_base], axis=0, ignore_index=True
        )
        sky_table, sky_defaults = build_sky_table(sky_reference)

        x_train = apply_sky_features(fold_train_base, sky_table, sky_defaults)
        x_valid = apply_sky_features(fold_valid_base, sky_table, sky_defaults)
        x_test = apply_sky_features(fold_test_base, sky_table, sky_defaults)

        categorical_cols = [
            c for c in ["spectral_type", "galaxy_population"] if c in x_train.columns
        ]
        x_train, x_valid, x_test = encode_categoricals(
            x_train, x_valid, x_test, categorical_cols
        )

        x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(-999.0)
        x_valid = x_valid.replace([np.inf, -np.inf], np.nan).fillna(-999.0)
        x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(-999.0)

        feature_cols = [c for c in x_train.columns if c != ID_COL]
        X_tr = x_train[feature_cols]
        X_va = x_valid[feature_cols]
        X_te = x_test[feature_cols]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        for name, model in make_model_panel().items():
            log_stage(f"Fold {fold}: fitting {name}")
            if name == "extra_trees":
                model.fit(
                    X_tr,
                    y_tr,
                    sample_weight=compute_sample_weight(
                        class_weight="balanced", y=y_tr
                    ),
                )
            else:
                model.fit(X_tr, y_tr)
            va_proba = align_proba(model, model.predict_proba(X_va), le)
            oof_panel[name][va_idx] = va_proba
            score = balanced_accuracy_score(y_va, va_proba.argmax(axis=1))
            panel_scores[name].append(score)
            print(f"Fold {fold} {name} balanced_accuracy={score:.6f}", flush=True)

        log_stage(f"Fold {fold}: fitting catboost on GPU")
        cat_model = make_catboost()
        try:
            cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        except Exception as exc:
            print(
                f"CatBoost GPU training failed on fold {fold}; falling back to CPU. Reason: {exc}",
                flush=True,
            )
            from catboost import CatBoostClassifier

            cat_model = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="TotalF1",
                iterations=1200,
                learning_rate=0.055,
                depth=8,
                l2_leaf_reg=7.0,
                random_seed=RANDOM_STATE,
                auto_class_weights="Balanced",
                task_type="CPU",
                verbose=250,
                allow_writing_files=False,
            )
            cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

        va_cat = cat_model.predict_proba(X_va)
        te_cat = cat_model.predict_proba(X_te)
        oof_cat[va_idx] = va_cat
        oof_panel["catboost"][va_idx] = va_cat
        test_cat += te_cat / N_SPLITS

        cat_score = balanced_accuracy_score(y_va, va_cat.argmax(axis=1))
        panel_scores["catboost"].append(cat_score)
        fold_scores.append(cat_score)
        print(f"Fold {fold} catboost balanced_accuracy={cat_score:.6f}", flush=True)

with aide_stage("score_stage"):
    print("Panel CV balanced accuracy:", flush=True)
    for name, preds in oof_panel.items():
        score = balanced_accuracy_score(y, preds.argmax(axis=1))
        fold_text = ", ".join(f"{s:.6f}" for s in panel_scores[name])
        print(f"{name}: mean_oof={score:.6f}; folds=[{fold_text}]", flush=True)

    final_oof_score = balanced_accuracy_score(y, oof_cat.argmax(axis=1))
    print(
        f"Primary validation metric: 5-fold OOF CatBoost balanced_accuracy={final_oof_score:.6f}",
        flush=True,
    )

with aide_stage("write_outputs_stage"):
    oof_labels = le.inverse_transform(oof_cat.argmax(axis=1))
    test_labels = le.inverse_transform(test_cat.argmax(axis=1))

    submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_labels})
    write_submission(submission)

    oof_df = pd.DataFrame(
        {
            "row": np.arange(len(train_base)),
            "target": y_raw.to_numpy(),
            "prediction": oof_labels,
        }
    )
    write_oof_predictions(oof_df)

    test_pred_df = pd.DataFrame({ID_COL: test_ids})
    for idx, cls in enumerate(class_order):
        test_pred_df[cls] = test_cat[:, idx]
    test_pred_df[TARGET] = test_labels
    write_test_predictions(test_pred_df)
