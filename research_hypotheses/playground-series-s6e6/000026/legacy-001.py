import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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


SEED = 20260611
N_SPLITS = 5
TARGET = "class"
ID_COL = "id"
CLASS_LABELS = np.array(["GALAXY", "QSO", "STAR"])
AUX_TARGETS = ["spectral_type", "galaxy_population"]
BASE_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CAT_COLS = ["spectral_type", "galaxy_population"]


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_basic_features(df):
    out = df.copy()
    bands = ["u", "g", "r", "i", "z"]
    for col in bands:
        out[col] = out[col].replace(-9999, np.nan)

    color_pairs = [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("u", "r"),
        ("g", "i"),
        ("r", "z"),
    ]
    for a, b in color_pairs:
        out[f"{a}_minus_{b}"] = out[a] - out[b]

    out["ug_over_gr"] = out["u_minus_g"] / (np.abs(out["g_minus_r"]) + 1e-3)
    out["ri_over_iz"] = out["r_minus_i"] / (np.abs(out["i_minus_z"]) + 1e-3)
    out["redshift_abs"] = np.abs(out["redshift"])
    out["redshift_log1p_abs"] = np.log1p(np.abs(out["redshift"]))
    out["sky_sin_alpha"] = np.sin(np.deg2rad(out["alpha"]))
    out["sky_cos_alpha"] = np.cos(np.deg2rad(out["alpha"]))
    out["sky_sin_delta"] = np.sin(np.deg2rad(out["delta"]))
    out["sky_cos_delta"] = np.cos(np.deg2rad(out["delta"]))
    return out


def entropy_from_proba(proba):
    clipped = np.clip(proba, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def margin_from_proba(proba):
    if proba.shape[1] == 1:
        return np.ones(proba.shape[0])
    part = np.partition(proba, -2, axis=1)
    return part[:, -1] - part[:, -2]


def fit_aux_model(x_train, y_train, seed_offset=0):
    classes, counts = np.unique(y_train.astype(str), return_counts=True)
    min_count = counts.min() if len(counts) else 0
    if len(classes) <= 1 or min_count < 2:
        return None, classes

    model = HistGradientBoostingClassifier(
        max_iter=90,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=SEED + seed_offset,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=10,
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_train.astype(str))
    model.fit(x_train, y_train.astype(str), sample_weight=weights)
    return model, model.classes_


def predict_aux_proba(model, classes, x_pred):
    if model is None:
        proba = np.ones((len(x_pred), len(classes)), dtype=float) / max(len(classes), 1)
        return classes, proba
    return model.classes_, model.predict_proba(x_pred)


def append_aux_features(base_df, aux_feature_frames):
    out = base_df.copy()
    for frame in aux_feature_frames:
        for col in frame.columns:
            out[col] = frame[col].values
    return out


def build_aux_feature_frame(observed_df, proba_map):
    features = pd.DataFrame(index=observed_df.index)

    actual_prob_cols = []
    mismatch_cols = []
    entropy_cols = []
    margin_cols = []
    maxprob_cols = []

    for target_name, (classes, proba) in proba_map.items():
        safe_name = target_name
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = classes[pred_idx].astype(str)
        observed = observed_df[target_name].astype(str).fillna("__MISSING__").values

        for j, cls in enumerate(classes):
            features[f"aux_{safe_name}_prob_{cls}"] = proba[:, j]

        class_to_idx = {str(cls): j for j, cls in enumerate(classes)}
        actual_prob = np.array(
            [
                (
                    proba[i, class_to_idx.get(str(val), -1)]
                    if str(val) in class_to_idx
                    else 0.0
                )
                for i, val in enumerate(observed)
            ]
        )

        features[f"aux_{safe_name}_max_prob"] = proba.max(axis=1)
        features[f"aux_{safe_name}_entropy"] = entropy_from_proba(proba)
        features[f"aux_{safe_name}_margin"] = margin_from_proba(proba)
        features[f"aux_{safe_name}_actual_prob"] = actual_prob
        features[f"aux_{safe_name}_mismatch"] = (pred_labels != observed).astype(float)

        actual_prob_cols.append(f"aux_{safe_name}_actual_prob")
        mismatch_cols.append(f"aux_{safe_name}_mismatch")
        entropy_cols.append(f"aux_{safe_name}_entropy")
        margin_cols.append(f"aux_{safe_name}_margin")
        maxprob_cols.append(f"aux_{safe_name}_max_prob")

    if len(actual_prob_cols) == 2:
        features["aux_metadata_actual_prob_product"] = (
            features[actual_prob_cols[0]] * features[actual_prob_cols[1]]
        )
        features["aux_metadata_actual_prob_min"] = features[actual_prob_cols].min(
            axis=1
        )
        features["aux_metadata_any_mismatch"] = features[mismatch_cols].max(axis=1)
        features["aux_metadata_both_mismatch"] = features[mismatch_cols].prod(axis=1)
        features["aux_metadata_entropy_sum"] = features[entropy_cols].sum(axis=1)
        features["aux_metadata_margin_min"] = features[margin_cols].min(axis=1)
        features["aux_metadata_maxprob_product"] = (
            features[maxprob_cols[0]] * features[maxprob_cols[1]]
        )

    return features


def make_preprocessor(feature_df):
    categorical = [c for c in CAT_COLS if c in feature_df.columns]
    numeric = [c for c in feature_df.columns if c not in categorical and c != ID_COL]
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
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_encoder()),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_linear_model(feature_df):
    return Pipeline(
        steps=[
            ("prep", make_preprocessor(feature_df)),
            (
                "model",
                LogisticRegression(
                    C=1.2,
                    class_weight="balanced",
                    max_iter=800,
                    multi_class="auto",
                    solver="lbfgs",
                    n_jobs=max(1, min(8, os.cpu_count() or 1)),
                    random_state=SEED,
                ),
            ),
        ]
    )


def make_tree_model(feature_df):
    categorical = [c for c in CAT_COLS if c in feature_df.columns]
    numeric = [c for c in feature_df.columns if c not in categorical and c != ID_COL]
    prep = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_encoder()),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("prep", prep),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=350,
                    max_depth=18,
                    min_samples_leaf=8,
                    max_features=0.7,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def catboost_available():
    try:
        import catboost  # noqa: F401

        return True
    except Exception:
        return False


def fit_predict_catboost(x_train, y_train, x_valid, x_test, cat_cols, fold):
    from catboost import CatBoostClassifier, Pool

    cat_features = [c for c in cat_cols if c in x_train.columns]
    x_train_cb = x_train.copy()
    x_valid_cb = x_valid.copy()
    x_test_cb = x_test.copy()

    for c in cat_features:
        x_train_cb[c] = x_train_cb[c].astype(str).fillna("__MISSING__")
        x_valid_cb[c] = x_valid_cb[c].astype(str).fillna("__MISSING__")
        x_test_cb[c] = x_test_cb[c].astype(str).fillna("__MISSING__")

    train_pool = Pool(x_train_cb, y_train, cat_features=cat_features)
    valid_pool = Pool(x_valid_cb, y_valid=None, cat_features=cat_features)
    test_pool = Pool(x_test_cb, cat_features=cat_features)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=900,
        learning_rate=0.055,
        depth=6,
        l2_leaf_reg=8.0,
        random_seed=SEED + fold,
        auto_class_weights="Balanced",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.8,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool, verbose=False)
    valid_proba = model.predict_proba(valid_pool)
    test_proba = model.predict_proba(test_pool)
    return model.classes_.astype(str), valid_proba, test_proba


def align_proba(classes, proba, target_classes):
    out = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    for j, cls in enumerate(target_classes):
        if str(cls) in class_to_idx:
            out[:, j] = proba[:, class_to_idx[str(cls)]]
    row_sum = out.sum(axis=1, keepdims=True)
    out = np.divide(
        out, row_sum, out=np.ones_like(out) / len(target_classes), where=row_sum > 0
    )
    return out


with aide_stage("load_data_stage"):
    train, test, sample_sub = load_competition_data()
    train = train.copy()
    test = test.copy()
    y = train[TARGET].astype(str).values
    test_ids = sample_sub[ID_COL].values

with aide_stage("build_features_stage"):
    train_base = add_basic_features(train.drop(columns=[TARGET]))
    test_base = add_basic_features(test)
    aux_input_cols = [c for c in train_base.columns if c not in [ID_COL] + CAT_COLS]
    raw_feature_cols = [c for c in train_base.columns if c != ID_COL]

with aide_stage("make_folds_stage"):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(train_base, y))

oof_panel = np.zeros((len(train_base), len(CLASS_LABELS)), dtype=float)
oof_panel_raw = np.zeros((len(train_base), len(CLASS_LABELS)), dtype=float)
test_panel_sum = np.zeros((len(test_base), len(CLASS_LABELS)), dtype=float)

aux_acc = {name: [] for name in AUX_TARGETS}
use_catboost = catboost_available()

with aide_stage("fit_predict_fold_stage"):
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        log_stage(
            f"Fold {fold}/{N_SPLITS}: building fold-safe auxiliary metadata consistency features"
        )
        x_aux_tr = train_base.iloc[tr_idx][aux_input_cols]
        x_aux_va = train_base.iloc[va_idx][aux_input_cols]
        x_aux_test = test_base[aux_input_cols]

        fold_aux_frames = {}
        test_aux_frames = {}

        for aux_i, aux_target in enumerate(AUX_TARGETS):
            aux_model, aux_classes = fit_aux_model(
                x_aux_tr,
                train_base.iloc[tr_idx][aux_target],
                seed_offset=fold * 100 + aux_i,
            )
            va_classes, va_proba = predict_aux_proba(aux_model, aux_classes, x_aux_va)

            pred_aux = va_classes[np.argmax(va_proba, axis=1)].astype(str)
            acc = accuracy_score(
                train_base.iloc[va_idx][aux_target].astype(str).values, pred_aux
            )
            aux_acc[aux_target].append(acc)

            full_aux_model, full_aux_classes = fit_aux_model(
                train_base.iloc[tr_idx][aux_input_cols],
                train_base.iloc[tr_idx][aux_target],
                seed_offset=fold * 100 + aux_i + 50,
            )
            test_classes, test_proba = predict_aux_proba(
                full_aux_model, full_aux_classes, x_aux_test
            )

            fold_aux_frames[aux_target] = (va_classes.astype(str), va_proba)
            test_aux_frames[aux_target] = (test_classes.astype(str), test_proba)

        va_aux_features = build_aux_feature_frame(
            train_base.iloc[va_idx], fold_aux_frames
        )
        test_aux_features = build_aux_feature_frame(test_base, test_aux_frames)

        x_train_full = append_aux_features(
            train_base.iloc[tr_idx][raw_feature_cols].reset_index(drop=True),
            [
                build_aux_feature_frame(
                    train_base.iloc[tr_idx],
                    {
                        name: predict_aux_proba(
                            fit_aux_model(
                                train_base.iloc[tr_idx][aux_input_cols],
                                train_base.iloc[tr_idx][name],
                                seed_offset=fold * 200 + i,
                            )[0],
                            fit_aux_model(
                                train_base.iloc[tr_idx][aux_input_cols],
                                train_base.iloc[tr_idx][name],
                                seed_offset=fold * 200 + i,
                            )[1],
                            train_base.iloc[tr_idx][aux_input_cols],
                        )
                        for i, name in enumerate(AUX_TARGETS)
                    },
                ).reset_index(drop=True)
            ],
        )

        x_valid_full = append_aux_features(
            train_base.iloc[va_idx][raw_feature_cols].reset_index(drop=True),
            [va_aux_features.reset_index(drop=True)],
        )
        x_test_full = append_aux_features(
            test_base[raw_feature_cols].reset_index(drop=True),
            [test_aux_features.reset_index(drop=True)],
        )

        x_train_raw = train_base.iloc[tr_idx][raw_feature_cols].reset_index(drop=True)
        x_valid_raw = train_base.iloc[va_idx][raw_feature_cols].reset_index(drop=True)
        y_train = y[tr_idx]
        y_valid = y[va_idx]

        fold_valid_probas = []
        fold_test_probas = []
        raw_valid_probas = []

        log_stage(f"Fold {fold}/{N_SPLITS}: fitting balanced logistic regression")
        linear = make_linear_model(x_train_full)
        linear.fit(x_train_full, y_train)
        fold_valid_probas.append(
            align_proba(
                linear.named_steps["model"].classes_,
                linear.predict_proba(x_valid_full),
                CLASS_LABELS,
            )
        )
        fold_test_probas.append(
            align_proba(
                linear.named_steps["model"].classes_,
                linear.predict_proba(x_test_full),
                CLASS_LABELS,
            )
        )

        linear_raw = make_linear_model(x_train_raw)
        linear_raw.fit(x_train_raw, y_train)
        raw_valid_probas.append(
            align_proba(
                linear_raw.named_steps["model"].classes_,
                linear_raw.predict_proba(x_valid_raw),
                CLASS_LABELS,
            )
        )

        log_stage(f"Fold {fold}/{N_SPLITS}: fitting balanced shallow ExtraTrees")
        tree = make_tree_model(x_train_full)
        tree.fit(x_train_full, y_train)
        fold_valid_probas.append(
            align_proba(
                tree.named_steps["model"].classes_,
                tree.predict_proba(x_valid_full),
                CLASS_LABELS,
            )
        )
        fold_test_probas.append(
            align_proba(
                tree.named_steps["model"].classes_,
                tree.predict_proba(x_test_full),
                CLASS_LABELS,
            )
        )

        tree_raw = make_tree_model(x_train_raw)
        tree_raw.fit(x_train_raw, y_train)
        raw_valid_probas.append(
            align_proba(
                tree_raw.named_steps["model"].classes_,
                tree_raw.predict_proba(x_valid_raw),
                CLASS_LABELS,
            )
        )

        if use_catboost:
            log_stage(
                f"Fold {fold}/{N_SPLITS}: fitting GPU CatBoost balanced multiclass model"
            )
            try:
                cb_classes, cb_valid, cb_test = fit_predict_catboost(
                    x_train_full,
                    y_train,
                    x_valid_full,
                    x_test_full,
                    CAT_COLS,
                    fold,
                )
                fold_valid_probas.append(
                    align_proba(cb_classes, cb_valid, CLASS_LABELS)
                )
                fold_test_probas.append(align_proba(cb_classes, cb_test, CLASS_LABELS))
            except Exception as exc:
                print(
                    f"Fold {fold}: CatBoost GPU failed, continuing with linear/tree panel only. Reason: {exc}",
                    flush=True,
                )

        valid_avg = np.mean(fold_valid_probas, axis=0)
        test_avg = np.mean(fold_test_probas, axis=0)
        raw_valid_avg = np.mean(raw_valid_probas, axis=0)

        oof_panel[va_idx] = valid_avg
        oof_panel_raw[va_idx] = raw_valid_avg
        test_panel_sum += test_avg / N_SPLITS

        fold_score = balanced_accuracy_score(
            y_valid, CLASS_LABELS[np.argmax(valid_avg, axis=1)]
        )
        raw_fold_score = balanced_accuracy_score(
            y_valid, CLASS_LABELS[np.argmax(raw_valid_avg, axis=1)]
        )
        print(
            f"Fold {fold} balanced_accuracy={fold_score:.6f} raw_baseline={raw_fold_score:.6f} delta={fold_score - raw_fold_score:.6f}",
            flush=True,
        )

with aide_stage("score_stage"):
    oof_pred = CLASS_LABELS[np.argmax(oof_panel, axis=1)]
    raw_oof_pred = CLASS_LABELS[np.argmax(oof_panel_raw, axis=1)]
    cv_score = balanced_accuracy_score(y, oof_pred)
    raw_cv_score = balanced_accuracy_score(y, raw_oof_pred)
    print(f"Primary 5-fold OOF balanced_accuracy={cv_score:.6f}", flush=True)
    print(
        f"Raw categorical one-hot baseline balanced_accuracy={raw_cv_score:.6f}",
        flush=True,
    )
    print(f"Metadata consistency delta={cv_score - raw_cv_score:.6f}", flush=True)
    for name in AUX_TARGETS:
        print(f"Auxiliary {name} OOF accuracy={np.mean(aux_acc[name]):.6f}", flush=True)

with aide_stage("write_outputs_stage"):
    test_pred = CLASS_LABELS[np.argmax(test_panel_sum, axis=1)]

    submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred})
    write_submission(submission)

    oof_df = pd.DataFrame(
        {
            "row": np.arange(len(train_base)),
            "target": y,
            "prediction": oof_pred,
        }
    )
    write_oof_predictions(oof_df)

    test_pred_df = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred})
    for i, cls in enumerate(CLASS_LABELS):
        test_pred_df[f"prob_{cls}"] = test_panel_sum[:, i]
    write_test_predictions(test_pred_df)
