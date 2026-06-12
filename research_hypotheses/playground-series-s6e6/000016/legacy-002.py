import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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

RANDOM_STATE = 42
N_SPLITS = 5
ID_COL = "id"
TARGET_COL = "class"
MAG_COLS = ["u", "g", "r", "i", "z"]
SHARED_NUMERIC_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR_COLS = ["u_g", "g_r", "r_i", "i_z", "u_z", "g_z"]
BASE_NUMERIC_COLS = SHARED_NUMERIC_COLS + COLOR_COLS + ["mag_mean", "log_redshift"]
TRANSFER_COLS = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
    "u_g",
    "g_r",
    "r_i",
    "i_z",
    "u_z",
    "log_redshift",
]
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]


def clean_and_engineer(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in SHARED_NUMERIC_COLS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in MAG_COLS:
        frame.loc[frame[col] <= -999.0, col] = np.nan
    frame["u_g"] = frame["u"] - frame["g"]
    frame["g_r"] = frame["g"] - frame["r"]
    frame["r_i"] = frame["r"] - frame["i"]
    frame["i_z"] = frame["i"] - frame["z"]
    frame["u_z"] = frame["u"] - frame["z"]
    frame["g_z"] = frame["g"] - frame["z"]
    frame["mag_mean"] = frame[MAG_COLS].mean(axis=1)
    frame["log_redshift"] = np.sign(frame["redshift"]) * np.log1p(
        np.abs(frame["redshift"])
    )
    return frame


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def min_distance_to_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1)[None, :]
    d2 = x_norm + c_norm - 2.0 * (x @ centers.T)
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0)).astype(np.float32)


def fit_auxiliary_transfer(aux_df: pd.DataFrame, class_names: list[str]):
    aux_values = aux_df[TRANSFER_COLS].replace([np.inf, -np.inf], np.nan)
    aux_medians = aux_values.median()
    aux_filled = aux_values.fillna(aux_medians)

    scaler = StandardScaler()
    aux_scaled = scaler.fit_transform(aux_filled).astype(np.float32)

    centroids = {}
    spreads = {}
    prototypes = {}
    aux_labels = aux_df[TARGET_COL].astype(str).to_numpy()

    for class_name in class_names:
        class_matrix = aux_scaled[aux_labels == class_name]
        centroids[class_name] = class_matrix.mean(axis=0).astype(np.float32)
        spreads[class_name] = np.clip(class_matrix.std(axis=0), 0.05, None).astype(
            np.float32
        )
        n_clusters = int(min(4, max(1, class_matrix.shape[0])))
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            batch_size=4096,
            n_init=10,
        )
        kmeans.fit(class_matrix)
        prototypes[class_name] = kmeans.cluster_centers_.astype(np.float32)

    return aux_medians, scaler, centroids, spreads, prototypes


def make_auxiliary_features(
    frame: pd.DataFrame,
    class_names: list[str],
    aux_bundle,
) -> pd.DataFrame:
    aux_medians, scaler, centroids, spreads, prototypes = aux_bundle
    values = frame[TRANSFER_COLS].replace([np.inf, -np.inf], np.nan).fillna(aux_medians)
    scaled = scaler.transform(values).astype(np.float32)

    feature_data = {}
    centroid_scores = []
    prototype_scores = []

    for class_name in class_names:
        center = centroids[class_name]
        spread = spreads[class_name]
        diff = scaled - center
        centroid_dist = np.sqrt(np.mean(diff * diff, axis=1)).astype(np.float32)
        pattern_maha = np.sqrt(np.mean((diff / spread) ** 2, axis=1)).astype(np.float32)
        proto_dist = min_distance_to_centers(scaled, prototypes[class_name])

        feature_data[f"aux_centroid_dist_{class_name}"] = centroid_dist
        feature_data[f"aux_pattern_maha_{class_name}"] = pattern_maha
        feature_data[f"aux_proto_dist_{class_name}"] = proto_dist

        centroid_scores.append((-centroid_dist).reshape(-1, 1))
        prototype_scores.append((-proto_dist).reshape(-1, 1))

    centroid_probs = softmax_rows(np.hstack(centroid_scores))
    prototype_probs = softmax_rows(np.hstack(prototype_scores))

    for idx, class_name in enumerate(class_names):
        feature_data[f"aux_centroid_prob_{class_name}"] = centroid_probs[:, idx].astype(
            np.float32
        )
        feature_data[f"aux_proto_prob_{class_name}"] = prototype_probs[:, idx].astype(
            np.float32
        )

    return pd.DataFrame(feature_data, index=frame.index)


def run_model_cv(
    model_name: str,
    x: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
):
    oof_proba = np.zeros((x.shape[0], n_classes), dtype=np.float32)
    test_proba = np.zeros((x_test.shape[0], n_classes), dtype=np.float32)
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        log_stage(f"model={model_name}|fold={fold}|event=fit_start")
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]

        if model_name == "logreg":
            scaler = StandardScaler()
            x_train_fit = scaler.fit_transform(x_train)
            x_valid_fit = scaler.transform(x_valid)
            x_test_fit = scaler.transform(x_test)
            model = LogisticRegression(
                C=1.0,
                max_iter=400,
                solver="lbfgs",
                multi_class="multinomial",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
            model.fit(x_train_fit, y_train)
            valid_proba = model.predict_proba(x_valid_fit).astype(np.float32)
            test_fold_proba = model.predict_proba(x_test_fit).astype(np.float32)

        elif model_name == "extratrees":
            model = ExtraTreesClassifier(
                n_estimators=220,
                max_depth=24,
                min_samples_leaf=4,
                max_features=0.8,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE + fold,
            )
            model.fit(x_train, y_train)
            valid_proba = model.predict_proba(x_valid).astype(np.float32)
            test_fold_proba = model.predict_proba(x_test).astype(np.float32)

        elif model_name == "xgboost":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            model = XGBClassifier(
                n_estimators=3000,
                learning_rate=0.04,
                max_depth=8,
                min_child_weight=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                device="cuda",
                max_bin=256,
                random_state=RANDOM_STATE + fold,
                verbosity=0,
            )
            model.fit(
                x_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(x_valid, y_valid)],
                verbose=False,
            )
            valid_proba = model.predict_proba(x_valid).astype(np.float32)
            test_fold_proba = model.predict_proba(x_test).astype(np.float32)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        oof_proba[valid_idx] = valid_proba
        test_proba += test_fold_proba / len(folds)

        valid_pred = valid_proba.argmax(axis=1)
        fold_score = balanced_accuracy_score(y_valid, valid_pred)
        fold_scores.append(fold_score)
        print(
            f"{model_name} fold {fold} balanced_accuracy={fold_score:.6f}", flush=True
        )
        log_stage(
            f"model={model_name}|fold={fold}|event=fit_end|balanced_accuracy={fold_score:.6f}"
        )

    oof_pred = oof_proba.argmax(axis=1)
    oof_score = balanced_accuracy_score(y, oof_pred)
    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "fold_scores": fold_scores,
        "oof_score": oof_score,
    }


def main():
    working_dir()

    train, test, sample_sub = load_competition_data()

    with aide_stage("build_features_stage"):
        aux = pd.read_csv(Path("./input/star_classification.csv"))
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(train[TARGET_COL].astype(str))
        class_names = list(label_encoder.classes_)

        train_features = clean_and_engineer(train.drop(columns=[TARGET_COL]))
        test_features = clean_and_engineer(test)
        aux_features = clean_and_engineer(aux)

        aux_bundle = fit_auxiliary_transfer(aux_features, class_names)
        train_aux = make_auxiliary_features(train_features, class_names, aux_bundle)
        test_aux = make_auxiliary_features(test_features, class_names, aux_bundle)

        train_features = pd.concat([train_features, train_aux], axis=1)
        test_features = pd.concat([test_features, test_aux], axis=1)

        aux_feature_cols = list(train_aux.columns)
        feature_cols = BASE_NUMERIC_COLS + aux_feature_cols + CATEGORICAL_COLS

        # Train and test are concatenated only for covariate-only imputation and dummy alignment.
        combined = pd.concat(
            [train_features[feature_cols], test_features[feature_cols]],
            axis=0,
            ignore_index=True,
        )
        numeric_cols = [col for col in feature_cols if col not in CATEGORICAL_COLS]
        combined[numeric_cols] = combined[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        combined[numeric_cols] = combined[numeric_cols].fillna(
            combined[numeric_cols].median()
        )
        combined = pd.get_dummies(
            combined,
            columns=CATEGORICAL_COLS,
            dummy_na=True,
            dtype=np.float32,
        )
        combined = combined.astype(np.float32)

        x = combined.iloc[: len(train_features)].to_numpy(dtype=np.float32, copy=False)
        x_test = combined.iloc[len(train_features) :].to_numpy(
            dtype=np.float32, copy=False
        )

    with aide_stage("make_folds_stage"):
        splitter = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        folds = list(splitter.split(x, y))

    model_results = {}
    with aide_stage("fit_predict_fold_stage"):
        for model_name in ["logreg", "extratrees", "xgboost"]:
            print(f"Training model={model_name}", flush=True)
            model_results[model_name] = run_model_cv(
                model_name=model_name,
                x=x,
                x_test=x_test,
                y=y,
                folds=folds,
                n_classes=len(class_names),
            )

    with aide_stage("score_stage"):
        for model_name, result in model_results.items():
            mean_score = float(np.mean(result["fold_scores"]))
            std_score = float(np.std(result["fold_scores"]))
            print(
                f"{model_name} cv_mean_balanced_accuracy={mean_score:.6f} "
                f"cv_std={std_score:.6f} oof_balanced_accuracy={result['oof_score']:.6f}",
                flush=True,
            )

        best_model_name = max(
            model_results, key=lambda name: model_results[name]["oof_score"]
        )
        best_result = model_results[best_model_name]
        best_oof_pred = best_result["oof_proba"].argmax(axis=1)
        best_test_pred = best_result["test_proba"].argmax(axis=1)
        primary_score = balanced_accuracy_score(y, best_oof_pred)

        print(f"selected_model={best_model_name}", flush=True)
        print(f"primary_cv_balanced_accuracy={primary_score:.6f}", flush=True)

    with aide_stage("write_outputs_stage"):
        submission = sample_sub[[ID_COL]].copy()
        submission[TARGET_COL] = label_encoder.inverse_transform(best_test_pred)
        write_submission(submission)

        oof_frame = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=np.int64),
                "target": train[TARGET_COL].astype(str).to_numpy(),
                "prediction": label_encoder.inverse_transform(best_oof_pred),
            }
        )
        write_oof_predictions(oof_frame)

        test_prediction_frame = pd.DataFrame({ID_COL: sample_sub[ID_COL].to_numpy()})
        for class_idx, class_name in enumerate(class_names):
            test_prediction_frame[class_name] = best_result["test_proba"][:, class_idx]
        write_test_predictions(test_prediction_frame)


if __name__ == "__main__":
    main()
