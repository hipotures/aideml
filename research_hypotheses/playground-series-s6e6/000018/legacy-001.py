import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from aide_solution_helpers import (
    aide_stage,
    load_competition_data,
    log_stage,
    working_dir,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "class"
ID_COL = "id"
MAG_COLS = ["u", "g", "r", "i", "z"]
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]
COLOR_FEATURES = [
    ("u_minus_g", "u", "g"),
    ("g_minus_r", "g", "r"),
    ("r_minus_i", "r", "i"),
    ("i_minus_z", "i", "z"),
    ("u_minus_r", "u", "r"),
    ("g_minus_i", "g", "i"),
    ("r_minus_z", "r", "z"),
]
RESIDUAL_FEATURES = [
    "predicted_redshift",
    "redshift_minus_predicted",
    "redshift_abs_residual",
    "redshift_squared_residual",
    "redshift_scaled_residual",
    "photoz_residual_sign",
    "photoz_abs_residual_gt_0p05",
    "photoz_abs_residual_gt_0p10",
    "photoz_abs_residual_gt_0p20",
    "photoz_consistent_within_5pct_scale",
]


def add_color_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for feature_name, left, right in COLOR_FEATURES:
        frame[feature_name] = frame[left] - frame[right]
    frame["redshift_log1p"] = np.log1p(frame["redshift"].clip(lower=-0.999999))
    frame["magnitude_mean"] = frame[MAG_COLS].mean(axis=1)
    frame["magnitude_std"] = frame[MAG_COLS].std(axis=1)
    return frame


def prepare_base_features(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    for frame in (train, test):
        for col in MAG_COLS:
            frame[col] = frame[col].replace(-9999, np.nan)
        for col in CATEGORICAL_COLS:
            frame[col] = frame[col].fillna("missing").astype(str)

    train = add_color_features(train)
    test = add_color_features(test)

    # This concatenation uses only covariates and no target column, so it is safe for one-hot alignment.
    combined = pd.concat(
        [train.drop(columns=[TARGET]), test],
        axis=0,
        ignore_index=True,
    )
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLS, dummy_na=False)

    train_base = combined.iloc[: len(train)].reset_index(drop=True)
    test_base = combined.iloc[len(train) :].reset_index(drop=True)

    train_base = train_base.drop(columns=[ID_COL]).astype(np.float32)
    test_base = test_base.drop(columns=[ID_COL]).astype(np.float32)

    photoz_feature_cols = [
        col
        for col in train_base.columns
        if col != "redshift" and not col.startswith("redshift_")
    ]
    classifier_feature_cols = list(train_base.columns)
    return train_base, test_base, photoz_feature_cols, classifier_feature_cols


def fit_photoz_oof(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    photoz_feature_cols,
    folds,
):
    x_photoz_train = train_base[photoz_feature_cols].copy()
    x_photoz_test = test_base[photoz_feature_cols].copy()
    fill_values = x_photoz_train.median(axis=0)
    x_photoz_train = x_photoz_train.fillna(fill_values)
    x_photoz_test = x_photoz_test.fillna(fill_values)

    y_redshift = train_base["redshift"].values
    oof_pred = np.zeros(len(train_base), dtype=np.float32)

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=6,
        max_iter=150,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        log_stage(
            f"event=info|stage=fit_predict_fold_stage|fold={fold}|model=photoz_regressor"
        )
        regressor.fit(x_photoz_train.iloc[train_idx], y_redshift[train_idx])
        oof_pred[valid_idx] = regressor.predict(x_photoz_train.iloc[valid_idx]).astype(
            np.float32
        )

    regressor.fit(x_photoz_train, y_redshift)
    test_pred = regressor.predict(x_photoz_test).astype(np.float32)
    return oof_pred, test_pred


def build_photoz_features(observed_redshift, predicted_redshift) -> pd.DataFrame:
    observed_redshift = np.asarray(observed_redshift, dtype=np.float32)
    predicted_redshift = np.asarray(predicted_redshift, dtype=np.float32)

    residual = observed_redshift - predicted_redshift
    abs_residual = np.abs(residual)
    scale = 1.0 + np.abs(observed_redshift)

    return pd.DataFrame(
        {
            "predicted_redshift": predicted_redshift,
            "redshift_minus_predicted": residual,
            "redshift_abs_residual": abs_residual,
            "redshift_squared_residual": residual**2,
            "redshift_scaled_residual": residual / scale,
            "photoz_residual_sign": np.sign(residual).astype(np.float32),
            "photoz_abs_residual_gt_0p05": (abs_residual > 0.05).astype(np.float32),
            "photoz_abs_residual_gt_0p10": (abs_residual > 0.10).astype(np.float32),
            "photoz_abs_residual_gt_0p20": (abs_residual > 0.20).astype(np.float32),
            "photoz_consistent_within_5pct_scale": (
                abs_residual <= 0.05 * scale
            ).astype(np.float32),
        }
    )


def build_classifier_matrices(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    train_photoz: pd.DataFrame,
    test_photoz: pd.DataFrame,
):
    x_train = pd.concat(
        [train_base.reset_index(drop=True), train_photoz.reset_index(drop=True)], axis=1
    )
    x_test = pd.concat(
        [test_base.reset_index(drop=True), test_photoz.reset_index(drop=True)], axis=1
    )

    fill_values = x_train.median(axis=0)
    x_train = x_train.fillna(fill_values).astype(np.float32)
    x_test = x_test.fillna(fill_values).astype(np.float32)
    return x_train, x_test


def make_model_builders():
    return {
        "ridge": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeClassifier(alpha=2.0, class_weight="balanced")),
            ]
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "catboost": lambda: CatBoostClassifier(
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            iterations=1200,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=5.0,
            random_seed=RANDOM_STATE,
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.8,
            od_type="Iter",
            od_wait=100,
            verbose=False,
        ),
    }


def main():
    working_dir()
    train, test, sample_sub = load_competition_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train[TARGET].values)
    class_names = list(label_encoder.classes_)

    with aide_stage("build_features_stage"):
        train_base, test_base, photoz_feature_cols, classifier_feature_cols = (
            prepare_base_features(train, test)
        )

    with aide_stage("make_folds_stage"):
        splitter = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        folds = list(splitter.split(train_base, y))

    with aide_stage("fit_predict_fold_stage"):
        photoz_oof, photoz_test = fit_photoz_oof(
            train_base=train_base,
            test_base=test_base,
            photoz_feature_cols=photoz_feature_cols,
            folds=folds,
        )

        train_photoz = build_photoz_features(train_base["redshift"].values, photoz_oof)
        test_photoz = build_photoz_features(test_base["redshift"].values, photoz_test)

        x_train, x_test = build_classifier_matrices(
            train_base=train_base[classifier_feature_cols],
            test_base=test_base[classifier_feature_cols],
            train_photoz=train_photoz,
            test_photoz=test_photoz,
        )
        feature_names = x_train.columns.tolist()

        model_builders = make_model_builders()
        model_outputs = {}
        catboost_importance_sum = np.zeros(len(feature_names), dtype=np.float64)
        catboost_importance_folds = 0

        for model_name, model_builder in model_builders.items():
            has_proba = model_name != "ridge"
            oof_pred = np.zeros(len(train), dtype=np.int64)
            test_proba_sum = (
                np.zeros((len(test), len(class_names)), dtype=np.float32)
                if has_proba
                else None
            )
            test_vote_sum = np.zeros((len(test), len(class_names)), dtype=np.float32)

            for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
                log_stage(
                    f"event=info|stage=fit_predict_fold_stage|fold={fold}|model={model_name}"
                )
                model = model_builder()

                x_fold_train = x_train.iloc[train_idx]
                y_fold_train = y[train_idx]
                x_fold_valid = x_train.iloc[valid_idx]
                y_fold_valid = y[valid_idx]

                if model_name == "catboost":
                    model.fit(
                        x_fold_train,
                        y_fold_train,
                        eval_set=(x_fold_valid, y_fold_valid),
                        use_best_model=True,
                    )
                    catboost_importance_sum += model.get_feature_importance()
                    catboost_importance_folds += 1
                else:
                    model.fit(x_fold_train, y_fold_train)

                valid_pred = (
                    np.asarray(model.predict(x_fold_valid)).reshape(-1).astype(np.int64)
                )
                oof_pred[valid_idx] = valid_pred

                if has_proba:
                    test_proba_sum += (
                        model.predict_proba(x_test).astype(np.float32) / N_SPLITS
                    )
                else:
                    test_pred = (
                        np.asarray(model.predict(x_test)).reshape(-1).astype(np.int64)
                    )
                    test_vote_sum[np.arange(len(test_pred)), test_pred] += 1.0

            model_outputs[model_name] = {
                "oof_pred": oof_pred,
                "test_proba": test_proba_sum,
                "test_votes": test_vote_sum,
            }

    with aide_stage("score_stage"):
        scores = {}
        recalls_by_model = {}
        ridge_recalls = None

        galaxy_idx = class_names.index("GALAXY")
        qso_idx = class_names.index("QSO")

        for model_name, output in model_outputs.items():
            score = balanced_accuracy_score(y, output["oof_pred"])
            recalls = recall_score(
                y, output["oof_pred"], average=None, labels=np.arange(len(class_names))
            )
            cm = confusion_matrix(
                y,
                output["oof_pred"],
                labels=np.arange(len(class_names)),
                normalize="true",
            )

            scores[model_name] = score
            recalls_by_model[model_name] = recalls

            print(f"CV balanced_accuracy [{model_name}] = {score:.6f}", flush=True)
            print(
                "Per-class recall [{}]: ".format(model_name)
                + ", ".join(
                    f"{class_names[i]}={recalls[i]:.6f}"
                    for i in range(len(class_names))
                ),
                flush=True,
            )
            print(
                f"QSO_vs_GALAXY [{model_name}] "
                f"qso_recall={recalls[qso_idx]:.6f} "
                f"galaxy_recall={recalls[galaxy_idx]:.6f} "
                f"qso_to_galaxy={cm[qso_idx, galaxy_idx]:.6f} "
                f"galaxy_to_qso={cm[galaxy_idx, qso_idx]:.6f}",
                flush=True,
            )

            if model_name == "ridge":
                ridge_recalls = recalls

        if ridge_recalls is not None:
            for model_name, recalls in recalls_by_model.items():
                if model_name == "ridge":
                    continue
                print(
                    "Recall delta vs ridge [{}]: ".format(model_name)
                    + ", ".join(
                        f"{class_names[i]}={recalls[i] - ridge_recalls[i]:+.6f}"
                        for i in range(len(class_names))
                    ),
                    flush=True,
                )

        if catboost_importance_folds > 0:
            mean_importance = catboost_importance_sum / catboost_importance_folds
            ranked_features = sorted(
                zip(feature_names, mean_importance),
                key=lambda item: item[1],
                reverse=True,
            )
            residual_rank_rows = []
            for rank, (feature_name, importance) in enumerate(ranked_features, start=1):
                if feature_name in RESIDUAL_FEATURES:
                    residual_rank_rows.append(f"{rank}:{feature_name}:{importance:.6f}")
            print(
                "Residual feature importance ranks [catboost]: "
                + ", ".join(residual_rank_rows[:10]),
                flush=True,
            )

        best_model_name = max(scores, key=scores.get)
        best_output = model_outputs[best_model_name]
        print(f"Selected model = {best_model_name}", flush=True)
        print(
            f"Primary CV balanced_accuracy = {scores[best_model_name]:.6f}", flush=True
        )

        if best_output["test_proba"] is not None:
            test_proba = best_output["test_proba"]
            test_pred = test_proba.argmax(axis=1)
            test_pred_labels = label_encoder.inverse_transform(test_pred)
            test_predictions_frame = pd.DataFrame({ID_COL: sample_sub[ID_COL].values})
            for class_index, class_name in enumerate(class_names):
                test_predictions_frame[class_name] = test_proba[:, class_index]
        else:
            test_pred = best_output["test_votes"].argmax(axis=1)
            test_pred_labels = label_encoder.inverse_transform(test_pred)
            test_predictions_frame = pd.DataFrame(
                {
                    ID_COL: sample_sub[ID_COL].values,
                    TARGET: test_pred_labels,
                }
            )

        oof_pred_labels = label_encoder.inverse_transform(best_output["oof_pred"])

    with aide_stage("write_outputs_stage"):
        submission = pd.DataFrame(
            {
                ID_COL: sample_sub[ID_COL].values,
                TARGET: test_pred_labels,
            }
        )
        oof_frame = pd.DataFrame(
            {
                "row": np.arange(len(train)),
                "target": train[TARGET].values,
                "prediction": oof_pred_labels,
            }
        )

        write_submission(submission)
        write_oof_predictions(oof_frame)
        write_test_predictions(test_predictions_frame)


if __name__ == "__main__":
    main()
