import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from aide_solution_helpers import (
    aide_stage,
    load_competition_data,
    log_stage,
    working_dir,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
    write_validation_predictions,
)

warnings.filterwarnings("ignore")

MAG_COLS = ["u", "g", "r", "i", "z"]
BASE_NUMERIC_COLS = ["alpha", "delta", "redshift"] + MAG_COLS
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]
RANDOM_STATE = 42
N_SPLITS = 5


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    for col in BASE_NUMERIC_COLS:
        feat[col] = pd.to_numeric(df[col], errors="coerce")

    # Sentinel cleanup is applied row-wise only; auxiliary data is intentionally unused
    # so the measured change reflects the photometric color-stack hypothesis itself.
    for col in MAG_COLS:
        feat.loc[feat[col] <= -1000, col] = np.nan

    color_defs = [
        ("u_g", "u", "g"),
        ("g_r", "g", "r"),
        ("r_i", "r", "i"),
        ("i_z", "i", "z"),
        ("u_r", "u", "r"),
        ("g_i", "g", "i"),
        ("r_z", "r", "z"),
        ("u_z", "u", "z"),
    ]
    for name, left, right in color_defs:
        feat[name] = feat[left] - feat[right]

    feat["ug_minus_gr"] = feat["u_g"] - feat["g_r"]
    feat["gr_minus_ri"] = feat["g_r"] - feat["r_i"]
    feat["ri_minus_iz"] = feat["r_i"] - feat["i_z"]
    feat["slope_balance"] = feat["u_g"] - feat["i_z"]
    feat["blue_curvature"] = feat["u"] - 2.0 * feat["g"] + feat["r"]
    feat["mid_curvature"] = feat["g"] - 2.0 * feat["r"] + feat["i"]
    feat["red_curvature"] = feat["r"] - 2.0 * feat["i"] + feat["z"]

    mag_frame = feat[MAG_COLS]
    color_cols = [name for name, _, _ in color_defs]
    color_frame = feat[color_cols]
    adjacent_color_frame = feat[["u_g", "g_r", "r_i", "i_z"]]

    feat["mag_mean"] = mag_frame.mean(axis=1)
    feat["mag_min"] = mag_frame.min(axis=1)
    feat["mag_max"] = mag_frame.max(axis=1)
    feat["mag_std"] = mag_frame.std(axis=1)
    feat["mag_range"] = feat["mag_max"] - feat["mag_min"]

    feat["color_mean"] = color_frame.mean(axis=1)
    feat["color_std"] = color_frame.std(axis=1)
    feat["color_range"] = color_frame.max(axis=1) - color_frame.min(axis=1)
    feat["adjacent_color_mean"] = adjacent_color_frame.mean(axis=1)
    feat["adjacent_color_std"] = adjacent_color_frame.std(axis=1)
    feat["color_energy"] = (
        feat["u_g"].pow(2)
        + feat["g_r"].pow(2)
        + feat["r_i"].pow(2)
        + feat["i_z"].pow(2)
    )

    feat["ug_gr_ratio"] = feat["u_g"] / (1.0 + feat["g_r"].abs())
    feat["ri_iz_ratio"] = feat["r_i"] / (1.0 + feat["i_z"].abs())
    feat["uz_over_range"] = feat["u_z"] / (1.0 + feat["mag_range"].abs())

    red = feat["redshift"].fillna(0.0)
    feat["redshift_abs"] = red.abs()
    feat["redshift_log1p_abs"] = np.log1p(red.abs())
    feat["redshift_signed_log1p"] = np.sign(red) * np.log1p(red.abs())
    feat["redshift_sq"] = red.pow(2)
    feat["redshift_sqrt_abs"] = np.sqrt(red.abs())
    feat["redshift_color_mix"] = red * feat["g_r"].fillna(0.0)
    feat["redshift_color_gap"] = red * feat["u_g"].fillna(0.0)

    for col in CATEGORICAL_COLS:
        feat[col] = df[col].astype("string")

    return feat


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def make_logreg(numeric_cols, categorical_cols):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="saga",
                    max_iter=200,
                    class_weight="balanced",
                    multi_class="multinomial",
                    n_jobs=min(16, os.cpu_count() or 1),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_extra_trees(numeric_cols, categorical_cols):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=min(16, os.cpu_count() or 1),
                ),
            ),
        ]
    )


def make_xgb(numeric_cols, categorical_cols, num_class):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=num_class,
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                    device="cuda",
                    eval_metric="mlogloss",
                    n_jobs=min(16, os.cpu_count() or 1),
                    verbosity=0,
                ),
            ),
        ]
    )


def main():
    _ = working_dir()

    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        X_train = build_features(train)
        X_test = build_features(test)
        y = train["class"].astype(str).to_numpy()
        test_ids = sample_sub["id"].copy()
        numeric_cols = [c for c in X_train.columns if c not in CATEGORICAL_COLS]

    with aide_stage("make_folds_stage"):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
        skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        folds = list(skf.split(X_train, y_encoded))

    model_builders = OrderedDict(
        [
            ("balanced_logreg", lambda: make_logreg(numeric_cols, CATEGORICAL_COLS)),
            (
                "shallow_extratrees",
                lambda: make_extra_trees(numeric_cols, CATEGORICAL_COLS),
            ),
            (
                "balanced_xgb_gpu",
                lambda: make_xgb(numeric_cols, CATEGORICAL_COLS, len(class_names)),
            ),
        ]
    )

    panel_state = {
        name: {
            "oof_proba": np.zeros((len(train), len(class_names)), dtype=np.float32),
            "test_proba": np.zeros((len(test), len(class_names)), dtype=np.float32),
            "fold_scores": [],
        }
        for name in model_builders
    }

    with aide_stage("fit_predict_fold_stage"):
        for fold_idx, (train_idx, valid_idx) in enumerate(folds, start=1):
            X_tr = X_train.iloc[train_idx]
            X_va = X_train.iloc[valid_idx]
            y_tr = y_encoded[train_idx]
            y_va = y_encoded[valid_idx]
            y_va_labels = y[valid_idx]

            for model_name, builder in model_builders.items():
                log_stage(f"Fold {fold_idx}/{N_SPLITS} - fitting {model_name}")
                model = builder()
                fit_kwargs = {}

                if model_name == "balanced_xgb_gpu":
                    fit_kwargs["model__sample_weight"] = compute_sample_weight(
                        class_weight="balanced", y=y_tr
                    )

                model.fit(X_tr, y_tr, **fit_kwargs)

                valid_proba = model.predict_proba(X_va).astype(np.float32)
                test_proba = model.predict_proba(X_test).astype(np.float32)

                panel_state[model_name]["oof_proba"][valid_idx] = valid_proba
                panel_state[model_name]["test_proba"] += test_proba / N_SPLITS

                valid_pred = class_names[np.argmax(valid_proba, axis=1)]
                fold_score = balanced_accuracy_score(y_va_labels, valid_pred)
                panel_state[model_name]["fold_scores"].append(fold_score)

                print(
                    f"fold={fold_idx} model={model_name} balanced_accuracy={fold_score:.6f}",
                    flush=True,
                )

    with aide_stage("score_stage"):
        best_model_name = None
        best_score = -np.inf
        best_oof_pred = None

        for model_name, state in panel_state.items():
            oof_pred = class_names[np.argmax(state["oof_proba"], axis=1)]
            oof_score = balanced_accuracy_score(y, oof_pred)
            mean_fold_score = float(np.mean(state["fold_scores"]))
            print(
                f"model={model_name} mean_fold_balanced_accuracy={mean_fold_score:.6f} "
                f"oof_balanced_accuracy={oof_score:.6f}",
                flush=True,
            )
            if oof_score > best_score:
                best_score = oof_score
                best_model_name = model_name
                best_oof_pred = oof_pred

        print(
            f"primary_validation_metric_balanced_accuracy={best_score:.6f}", flush=True
        )

    with aide_stage("write_outputs_stage"):
        best_test_proba = panel_state[best_model_name]["test_proba"]
        test_pred = class_names[np.argmax(best_test_proba, axis=1)]

        submission_df = pd.DataFrame({"id": test_ids, "class": test_pred})
        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=np.int64),
                "target": y,
                "prediction": best_oof_pred,
            }
        )
        test_pred_df = pd.DataFrame(best_test_proba, columns=class_names)
        test_pred_df.insert(0, "id", test_ids.to_numpy())

        write_submission(submission_df)
        write_oof_predictions(oof_df)
        write_test_predictions(test_pred_df)


if __name__ == "__main__":
    main()
