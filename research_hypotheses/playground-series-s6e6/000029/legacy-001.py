import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA

from catboost import CatBoostClassifier

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)

warnings.filterwarnings("ignore")


SEED = 20260611
N_FOLDS = 5
CLASS_ORDER = ["GALAXY", "QSO", "STAR"]


def add_photometric_features(df):
    out = df.copy()
    bands = ["u", "g", "r", "i", "z"]

    for b in bands:
        out[f"{b}_is_sentinel"] = (out[b] <= -9000).astype("int8")
        out.loc[out[b] <= -9000, b] = np.nan

    adjacent = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
    broad = [("u", "r"), ("u", "i"), ("u", "z"), ("g", "i"), ("g", "z"), ("r", "z")]
    for a, b in adjacent + broad:
        out[f"{a}_{b}_color"] = out[a] - out[b]

    out["mag_mean"] = out[bands].mean(axis=1)
    out["mag_std"] = out[bands].std(axis=1)
    out["mag_min"] = out[bands].min(axis=1)
    out["mag_max"] = out[bands].max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]
    out["redshift_abs"] = out["redshift"].abs()
    out["redshift_log1p_abs"] = np.log1p(out["redshift_abs"])
    out["alpha_sin"] = np.sin(np.deg2rad(out["alpha"]))
    out["alpha_cos"] = np.cos(np.deg2rad(out["alpha"]))
    out["delta_sin"] = np.sin(np.deg2rad(out["delta"]))
    out["delta_cos"] = np.cos(np.deg2rad(out["delta"]))

    return out


def get_ohe_kwargs():
    try:
        OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:
        return {"sparse": False, "handle_unknown": "ignore"}


def build_base_matrix(train, test):
    train_f = add_photometric_features(train.drop(columns=["class"]))
    test_f = add_photometric_features(test)

    feature_cols = [c for c in train_f.columns if c != "id"]
    cat_cols = [c for c in ["spectral_type", "galaxy_population"] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(**get_ohe_kwargs())),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    combined = pd.concat(
        [train_f[feature_cols], test_f[feature_cols]], axis=0, ignore_index=True
    )
    combined_matrix = preprocessor.fit_transform(combined)
    combined_matrix = np.asarray(combined_matrix, dtype=np.float32)

    return combined_matrix[: len(train)], combined_matrix[len(train) :], train_f, test_f


def make_embedding(combined_matrix):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(combined_matrix)

    try:
        import umap

        log_stage(
            "Fitting UMAP manifold embedding on combined train+test covariates only"
        )
        reducer = umap.UMAP(
            n_components=6,
            n_neighbors=40,
            min_dist=0.08,
            metric="euclidean",
            random_state=SEED,
            low_memory=True,
            transform_seed=SEED,
            verbose=False,
        )
        emb = reducer.fit_transform(x_scaled).astype(np.float32)
        print("Embedding method: umap", flush=True)
        return emb
    except Exception as exc:
        print(
            f"Embedding method fallback: umap unavailable or failed ({exc}); using Nystroem+PCA",
            flush=True,
        )
        n_components = min(256, max(32, x_scaled.shape[1] * 4))
        kernel = Nystroem(
            kernel="rbf",
            gamma=1.0 / max(1, x_scaled.shape[1]),
            n_components=n_components,
            random_state=SEED,
        )
        z = kernel.fit_transform(x_scaled)
        emb = PCA(n_components=6, random_state=SEED).fit_transform(z).astype(np.float32)
        return emb


def add_embedding_features(base_train, base_test):
    # This transductive block concatenates train and test covariates only; no target, labels,
    # OOF predictions, model predictions, or target-derived information are present.
    combined_matrix = np.vstack([base_train, base_test]).astype(np.float32)
    emb = make_embedding(combined_matrix)

    centroid = emb.mean(axis=0, keepdims=True)
    q25 = np.percentile(emb, 25, axis=0, keepdims=True)
    q75 = np.percentile(emb, 75, axis=0, keepdims=True)

    dist_mean = np.linalg.norm(emb - centroid, axis=1, keepdims=True)
    dist_q25 = np.linalg.norm(emb - q25, axis=1, keepdims=True)
    dist_q75 = np.linalg.norm(emb - q75, axis=1, keepdims=True)
    emb_full = np.hstack([emb, dist_mean, dist_q25, dist_q75]).astype(np.float32)

    train_emb = emb_full[: base_train.shape[0]]
    test_emb = emb_full[base_train.shape[0] :]

    return (
        np.hstack([base_train, train_emb]).astype(np.float32),
        np.hstack([base_test, test_emb]).astype(np.float32),
    )


def make_catboost():
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=900,
        learning_rate=0.055,
        depth=7,
        l2_leaf_reg=8.0,
        random_seed=SEED,
        auto_class_weights="Balanced",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.8,
        bootstrap_type="Bayesian",
        bagging_temperature=0.7,
        verbose=150,
        allow_writing_files=False,
    )


def main():
    np.random.seed(SEED)

    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        y_text = train["class"].astype(str).values

        label_encoder = LabelEncoder()
        label_encoder.fit(CLASS_ORDER)
        y = label_encoder.transform(y_text)

        base_train, base_test, _, _ = build_base_matrix(train, test)
        x_train, x_test = add_embedding_features(base_train, base_test)
        print(f"Feature matrix: train={x_train.shape}, test={x_test.shape}", flush=True)

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = list(skf.split(x_train, y))

    oof_proba = np.zeros((len(train), len(CLASS_ORDER)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(CLASS_ORDER)), dtype=np.float32)
    fold_scores = []

    with aide_stage("fit_predict_fold_stage"):
        for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
            log_stage(
                f"Training fold {fold}/{N_FOLDS}: CatBoost GPU balanced multiclass"
            )
            model = make_catboost()
            model.fit(
                x_train[tr_idx],
                y[tr_idx],
                eval_set=(x_train[va_idx], y[va_idx]),
                use_best_model=True,
            )

            va_proba = model.predict_proba(x_train[va_idx]).astype(np.float32)
            te_proba = model.predict_proba(x_test).astype(np.float32)

            oof_proba[va_idx] = va_proba
            test_proba += te_proba / N_FOLDS

            va_pred = va_proba.argmax(axis=1)
            score = balanced_accuracy_score(y[va_idx], va_pred)
            fold_scores.append(score)
            print(f"Fold {fold} balanced_accuracy={score:.6f}", flush=True)

    with aide_stage("score_stage"):
        oof_pred_idx = oof_proba.argmax(axis=1)
        cv_score = balanced_accuracy_score(y, oof_pred_idx)
        print(f"Mean fold balanced_accuracy={np.mean(fold_scores):.6f}", flush=True)
        print(f"OOF balanced_accuracy={cv_score:.6f}", flush=True)

    with aide_stage("write_outputs_stage"):
        pred_idx = test_proba.argmax(axis=1)
        pred_labels = label_encoder.inverse_transform(pred_idx)

        submission = pd.DataFrame(
            {
                sample_sub.columns[0]: sample_sub.iloc[:, 0].values,
                sample_sub.columns[1]: pred_labels,
            }
        )
        write_submission(submission)

        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=np.int64),
                "target": y_text,
                "prediction": label_encoder.inverse_transform(oof_pred_idx),
            }
        )
        write_oof_predictions(oof_df)

        test_pred_df = pd.DataFrame(
            {sample_sub.columns[0]: sample_sub.iloc[:, 0].values}
        )
        for cls_idx, cls_name in enumerate(label_encoder.classes_):
            test_pred_df[cls_name] = test_proba[:, cls_idx]
        test_pred_df[sample_sub.columns[1]] = pred_labels
        write_test_predictions(test_pred_df)


if __name__ == "__main__":
    main()
