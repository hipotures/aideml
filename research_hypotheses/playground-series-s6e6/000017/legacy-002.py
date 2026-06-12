import gc
import os

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
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


def build_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    mag_cols = ["u", "g", "r", "i", "z"]
    for col in mag_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        df.loc[df[col] <= -999.0, col] = np.nan
    df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce").astype(np.float32)

    features = pd.DataFrame(
        {
            "u": df["u"],
            "g": df["g"],
            "r": df["r"],
            "i": df["i"],
            "z": df["z"],
            "redshift": df["redshift"],
            "u_g": df["u"] - df["g"],
            "g_r": df["g"] - df["r"],
            "r_i": df["r"] - df["i"],
            "i_z": df["i"] - df["z"],
            "u_r": df["u"] - df["r"],
            "g_i": df["g"] - df["i"],
            "r_z": df["r"] - df["z"],
            "u_z": df["u"] - df["z"],
            "g_z": df["g"] - df["z"],
            "u_minus_redshift": df["u"] - df["redshift"],
            "r_minus_redshift": df["r"] - df["redshift"],
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    return features.astype(np.float32)


def build_categorical_dummies(frame: pd.DataFrame) -> pd.DataFrame:
    cat_df = frame[["spectral_type", "galaxy_population"]].copy()
    cat_df = cat_df.fillna("missing").astype(str)
    return pd.get_dummies(cat_df, dtype=np.float32).astype(np.float32)


def soft_affinity(distances: np.ndarray) -> np.ndarray:
    scale = max(float(np.std(distances)), 1e-6)
    logits = -distances / scale
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def nearest_margin_features(distances: np.ndarray, prefix: str) -> pd.DataFrame:
    smallest = np.partition(distances, kth=1, axis=1)[:, :2]
    smallest.sort(axis=1)
    return pd.DataFrame(
        {
            f"{prefix}_nearest": smallest[:, 0].astype(np.float32),
            f"{prefix}_second": smallest[:, 1].astype(np.float32),
            f"{prefix}_margin": (smallest[:, 1] - smallest[:, 0]).astype(np.float32),
            f"{prefix}_ratio": (
                smallest[:, 0] / np.maximum(smallest[:, 1], 1e-6)
            ).astype(np.float32),
        }
    )


def approximate_density_features(
    embedding: np.ndarray,
    max_neighbors: int = 50,
    ref_size: int = 120_000,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n_rows = embedding.shape[0]
    ref_idx = rng.choice(n_rows, size=min(ref_size, n_rows), replace=False)
    ref = embedding[ref_idx]

    nn = NearestNeighbors(
        n_neighbors=max_neighbors,
        metric="euclidean",
        algorithm="auto",
        n_jobs=max(1, min(16, os.cpu_count() or 1)),
    )
    nn.fit(ref)

    out = np.empty((n_rows, 4), dtype=np.float32)
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        distances, _ = nn.kneighbors(embedding[start:stop], return_distance=True)
        out[start:stop, 0] = distances[:, 0]
        out[start:stop, 1] = distances[:, :10].mean(axis=1)
        out[start:stop, 2] = distances[:, :25].mean(axis=1)
        out[start:stop, 3] = distances[:, :50].mean(axis=1)
        print(f"density_features rows {start}:{stop}", flush=True)

    return pd.DataFrame(
        out,
        columns=["nn_dist_1", "nn_mean_dist_10", "nn_mean_dist_25", "nn_mean_dist_50"],
    )


def build_full_feature_matrix(train: pd.DataFrame, test: pd.DataFrame):
    # Combined train+test covariates are used only for label-free unsupervised features.
    combined = pd.concat(
        [train.drop(columns=["class"]), test], axis=0, ignore_index=True
    )

    num_df = build_numeric_features(combined)
    cat_df = build_categorical_dummies(combined)

    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(num_df).astype(np.float32)
    unsup_input = np.concatenate(
        [scaled_num, cat_df.to_numpy(dtype=np.float32, copy=False)],
        axis=1,
    ).astype(np.float32, copy=False)

    pca = PCA(n_components=8, svd_solver="randomized", random_state=RANDOM_STATE)
    pca_features = pca.fit_transform(unsup_input).astype(np.float32)

    km12 = MiniBatchKMeans(
        n_clusters=12,
        random_state=RANDOM_STATE,
        batch_size=8192,
        n_init=5,
    )
    dist12 = km12.fit_transform(pca_features).astype(np.float32)
    aff12 = soft_affinity(dist12)
    margin12 = nearest_margin_features(dist12, "km12")

    km24 = MiniBatchKMeans(
        n_clusters=24,
        random_state=RANDOM_STATE,
        batch_size=8192,
        n_init=5,
    )
    dist24 = km24.fit_transform(pca_features).astype(np.float32)
    margin24 = nearest_margin_features(dist24, "km24")

    density_df = approximate_density_features(pca_features)

    feature_df = pd.concat(
        [
            num_df.reset_index(drop=True),
            cat_df.reset_index(drop=True),
            pd.DataFrame(pca_features, columns=[f"pca_{i}" for i in range(8)]),
            pd.DataFrame(aff12, columns=[f"km12_aff_{i}" for i in range(12)]),
            pd.DataFrame(dist24, columns=[f"km24_dist_{i}" for i in range(24)]),
            margin12.reset_index(drop=True),
            margin24.reset_index(drop=True),
            density_df.reset_index(drop=True),
        ],
        axis=1,
    ).astype(np.float32)

    n_train = len(train)
    x_train = feature_df.iloc[:n_train].to_numpy(dtype=np.float32, copy=False)
    x_test = feature_df.iloc[n_train:].to_numpy(dtype=np.float32, copy=False)

    del combined, num_df, cat_df, scaled_num, unsup_input, pca_features
    del dist12, aff12, dist24, margin12, margin24, density_df, feature_df
    gc.collect()
    return x_train, x_test


def build_model(model_name: str):
    if model_name == "logreg":
        return LogisticRegression(
            C=2.0,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=300,
            criterion="gini",
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            auto_class_weights="Balanced",
            iterations=800,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5.0,
            random_seed=RANDOM_STATE,
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.8,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            od_type="Iter",
            od_wait=100,
            allow_writing_files=False,
            verbose=False,
        )
    raise ValueError(f"Unknown model: {model_name}")


def fit_predict_model(model_name, x_train, y_train, x_test, folds, n_classes):
    oof_probs = np.zeros((x_train.shape[0], n_classes), dtype=np.float32)
    test_probs = np.zeros((x_test.shape[0], n_classes), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        log_stage(f"model={model_name}|fold={fold}|event=fit_start")
        x_tr = x_train[tr_idx]
        y_tr = y_train[tr_idx]
        x_va = x_train[va_idx]
        y_va = y_train[va_idx]

        if model_name == "logreg":
            scaler = StandardScaler()
            x_tr_fit = scaler.fit_transform(x_tr)
            x_va_fit = scaler.transform(x_va)
            x_te_fit = scaler.transform(x_test)
            model = build_model(model_name)
            model.fit(x_tr_fit, y_tr)
            va_probs = model.predict_proba(x_va_fit).astype(np.float32)
            te_probs = model.predict_proba(x_te_fit).astype(np.float32)
        elif model_name == "catboost":
            model = build_model(model_name)
            model.fit(
                x_tr, y_tr, eval_set=(x_va, y_va), use_best_model=True, verbose=False
            )
            va_probs = model.predict_proba(x_va).astype(np.float32)
            te_probs = model.predict_proba(x_test).astype(np.float32)
        else:
            model = build_model(model_name)
            model.fit(x_tr, y_tr)
            va_probs = model.predict_proba(x_va).astype(np.float32)
            te_probs = model.predict_proba(x_test).astype(np.float32)

        oof_probs[va_idx] = va_probs
        test_probs += te_probs / len(folds)
        log_stage(f"model={model_name}|fold={fold}|event=fit_end")

        del x_tr, y_tr, x_va, y_va, model, va_probs, te_probs
        gc.collect()

    return oof_probs, test_probs


def summarize_predictions(y_true_labels, pred_labels, class_names, model_name):
    bal_acc = balanced_accuracy_score(y_true_labels, pred_labels)
    recalls = recall_score(
        y_true_labels,
        pred_labels,
        labels=class_names,
        average=None,
        zero_division=0,
    )
    recall_text = ", ".join(
        f"recall_{cls}={rec:.6f}" for cls, rec in zip(class_names, recalls)
    )
    print(f"{model_name} balanced_accuracy={bal_acc:.6f}, {recall_text}", flush=True)
    return bal_acc


def main():
    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        _ = working_dir()
        x_train, x_test = build_full_feature_matrix(train, test)

    with aide_stage("make_folds_stage"):
        y_labels = train["class"].astype(str).to_numpy()
        le = LabelEncoder()
        y = le.fit_transform(y_labels)
        class_names = le.classes_
        folds = list(
            StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            ).split(x_train, y)
        )

    model_names = ["logreg", "extratrees", "catboost"]
    results = {}

    with aide_stage("fit_predict_fold_stage"):
        for model_name in model_names:
            oof_probs, test_probs = fit_predict_model(
                model_name=model_name,
                x_train=x_train,
                y_train=y,
                x_test=x_test,
                folds=folds,
                n_classes=len(class_names),
            )
            oof_pred = le.inverse_transform(np.argmax(oof_probs, axis=1))
            score = summarize_predictions(y_labels, oof_pred, class_names, model_name)
            results[model_name] = {
                "score": score,
                "oof_probs": oof_probs,
                "test_probs": test_probs,
            }

    with aide_stage("score_stage"):
        best_model_name = max(results, key=lambda name: results[name]["score"])
        best_oof_probs = results[best_model_name]["oof_probs"]
        best_test_probs = results[best_model_name]["test_probs"]
        best_oof_pred = le.inverse_transform(np.argmax(best_oof_probs, axis=1))
        best_score = summarize_predictions(
            y_labels,
            best_oof_pred,
            class_names,
            f"selected_{best_model_name}",
        )
        print(f"primary_metric_balanced_accuracy={best_score:.6f}", flush=True)

    with aide_stage("write_outputs_stage"):
        submission = sample_sub[["id"]].copy()
        submission["class"] = le.inverse_transform(np.argmax(best_test_probs, axis=1))
        write_submission(submission)

        oof_frame = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=np.int64),
                "target": y_labels,
                "prediction": best_oof_pred,
            }
        )
        write_oof_predictions(oof_frame)

        test_pred_frame = pd.DataFrame({"id": sample_sub["id"].to_numpy()})
        test_pred_frame["class"] = submission["class"].to_numpy()
        for idx, cls in enumerate(class_names):
            test_pred_frame[f"class_{cls}"] = best_test_probs[:, idx]
        write_test_predictions(test_pred_frame)


if __name__ == "__main__":
    main()
