import os
import warnings

import numpy as np
import pandas as pd

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")


RANDOM_STATE = 2026
N_SPLITS = 5
K_VALUES = (16, 48, 96)
CLASS_NAMES = ["GALAXY", "QSO", "STAR"]


def add_base_features(df):
    out = df.copy()
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["r_z"] = out["r"] - out["z"]
    out["ugriz_sum"] = out[["u", "g", "r", "i", "z"]].sum(axis=1)
    out["ugriz_mean"] = out[["u", "g", "r", "i", "z"]].mean(axis=1)
    out["ugriz_std"] = out[["u", "g", "r", "i", "z"]].std(axis=1)
    out["redshift_abs"] = out["redshift"].abs()
    return out


def make_preprocessor(train_df):
    feature_cols = [c for c in train_df.columns if c not in ("id", "class")]
    cat_cols = [
        c
        for c in feature_cols
        if train_df[c].dtype == "object"
        or str(train_df[c].dtype).startswith("category")
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", encoder, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor, feature_cols


def posterior_entropy(p):
    clipped = np.clip(p, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def knn_density_features(x_query, x_ref, y_ref, class_count, k_values):
    max_k = max(k_values)
    nn = NearestNeighbors(
        n_neighbors=max_k, algorithm="auto", metric="euclidean", n_jobs=-1
    )
    nn.fit(x_ref)
    distances, indices = nn.kneighbors(x_query, return_distance=True)
    neighbor_y = y_ref[indices]
    weights_all = 1.0 / (distances + 1e-6)

    blocks = []
    for k in k_values:
        labels_k = neighbor_y[:, :k]
        weights_k = weights_all[:, :k]
        probs = np.zeros((x_query.shape[0], class_count), dtype=np.float32)
        votes = np.zeros((x_query.shape[0], class_count), dtype=np.float32)

        for cls in range(class_count):
            mask = labels_k == cls
            probs[:, cls] = (weights_k * mask).sum(axis=1)
            votes[:, cls] = mask.mean(axis=1)

        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        entropy = posterior_entropy(probs)
        nearest = distances[:, 0]
        mean_dist = distances[:, :k].mean(axis=1)

        block = [
            probs,
            votes,
            margin[:, None],
            entropy[:, None],
            nearest[:, None],
            mean_dist[:, None],
        ]
        blocks.append(np.hstack(block).astype(np.float32))

    class_distance = np.zeros((x_query.shape[0], class_count), dtype=np.float32)
    for cls in range(class_count):
        cls_mask = neighbor_y == cls
        masked = np.where(cls_mask, distances, np.nan)
        class_distance[:, cls] = np.nanmean(masked, axis=1)
        fallback = np.nanmax(distances, axis=1) * 1.5
        class_distance[:, cls] = np.where(
            np.isfinite(class_distance[:, cls]), class_distance[:, cls], fallback
        )

    blocks.append(class_distance)
    return np.hstack(blocks).astype(np.float32)


def knn_feature_names(class_names, k_values):
    names = []
    for k in k_values:
        names.extend([f"knn{k}_prob_{c}" for c in class_names])
        names.extend([f"knn{k}_vote_{c}" for c in class_names])
        names.extend(
            [
                f"knn{k}_margin",
                f"knn{k}_entropy",
                f"knn{k}_nearest_dist",
                f"knn{k}_mean_dist",
            ]
        )
    names.extend([f"knn_mean_dist_to_{c}" for c in class_names])
    return names


def make_panel_models():
    models = {
        "linear_balanced": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            multi_class="auto",
            solver="saga",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "rf_balanced": RandomForestClassifier(
            n_estimators=260,
            max_depth=18,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from catboost import CatBoostClassifier

        models["catboost_balanced"] = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=700,
            learning_rate=0.055,
            depth=6,
            l2_leaf_reg=6.0,
            random_seed=RANDOM_STATE,
            auto_class_weights="Balanced",
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.8,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception as exc:
        print(f"CatBoost import failed, skipping CatBoost: {exc}", flush=True)

    return models


with aide_stage("build_features_stage"):
    train, test, sample_sub = load_competition_data()
    train_base = add_base_features(train)
    test_base = add_base_features(test)

    y_text = train_base["class"].astype(str).values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    ordered_class_names = list(label_encoder.classes_)
    class_count = len(ordered_class_names)

    base_preprocessor, base_feature_cols = make_preprocessor(train_base)
    full_base_matrix = base_preprocessor.fit_transform(
        train_base[base_feature_cols]
    ).astype(np.float32)
    full_test_base_matrix = base_preprocessor.transform(
        test_base[base_feature_cols]
    ).astype(np.float32)

    knn_names = knn_feature_names(ordered_class_names, K_VALUES)
    oof_knn = np.zeros((len(train_base), len(knn_names)), dtype=np.float32)
    test_knn_sum = np.zeros((len(test_base), len(knn_names)), dtype=np.float32)

with aide_stage("make_folds_stage"):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(train_base, y))

with aide_stage("fit_predict_fold_stage"):
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        log_stage(f"fold {fold}: fitting fold-safe KNN posterior features")
        fold_preprocessor, _ = make_preprocessor(train_base.iloc[tr_idx])
        x_tr = fold_preprocessor.fit_transform(
            train_base.iloc[tr_idx][base_feature_cols]
        ).astype(np.float32)
        x_va = fold_preprocessor.transform(
            train_base.iloc[va_idx][base_feature_cols]
        ).astype(np.float32)
        x_te = fold_preprocessor.transform(test_base[base_feature_cols]).astype(
            np.float32
        )

        oof_knn[va_idx] = knn_density_features(
            x_va, x_tr, y[tr_idx], class_count, K_VALUES
        )
        test_knn_sum += (
            knn_density_features(x_te, x_tr, y[tr_idx], class_count, K_VALUES)
            / N_SPLITS
        )

    x_train_final = np.hstack([full_base_matrix, oof_knn]).astype(np.float32)
    x_test_final = np.hstack([full_test_base_matrix, test_knn_sum]).astype(np.float32)

    panel_models = make_panel_models()
    panel_oof_pred = {}
    panel_oof_proba = {}
    panel_test_proba = {}

    for model_name, model in panel_models.items():
        oof_pred = np.zeros(len(train_base), dtype=np.int32)
        oof_proba = np.zeros((len(train_base), class_count), dtype=np.float32)
        test_proba_sum = np.zeros((len(test_base), class_count), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            log_stage(f"fold {fold}: fitting {model_name}")
            estimator = clone(model)

            try:
                if model_name == "linear_balanced":
                    pipe = Pipeline(
                        steps=[
                            ("scale", StandardScaler()),
                            ("model", estimator),
                        ]
                    )
                    pipe.fit(x_train_final[tr_idx], y[tr_idx])
                    proba = pipe.predict_proba(x_train_final[va_idx])
                    test_proba = pipe.predict_proba(x_test_final)
                elif model_name == "catboost_balanced":
                    estimator.fit(
                        x_train_final[tr_idx],
                        y[tr_idx],
                        eval_set=(x_train_final[va_idx], y[va_idx]),
                        use_best_model=True,
                        early_stopping_rounds=80,
                    )
                    proba = estimator.predict_proba(x_train_final[va_idx])
                    test_proba = estimator.predict_proba(x_test_final)
                else:
                    weights = compute_sample_weight(
                        class_weight="balanced", y=y[tr_idx]
                    )
                    estimator.fit(
                        x_train_final[tr_idx], y[tr_idx], sample_weight=weights
                    )
                    proba = estimator.predict_proba(x_train_final[va_idx])
                    test_proba = estimator.predict_proba(x_test_final)
            except Exception as exc:
                if model_name == "catboost_balanced":
                    print(
                        f"CatBoost GPU training failed, falling back to CPU for this fold: {exc}",
                        flush=True,
                    )
                    from catboost import CatBoostClassifier

                    estimator = CatBoostClassifier(
                        loss_function="MultiClass",
                        eval_metric="TotalF1",
                        iterations=700,
                        learning_rate=0.055,
                        depth=6,
                        l2_leaf_reg=6.0,
                        random_seed=RANDOM_STATE,
                        auto_class_weights="Balanced",
                        task_type="CPU",
                        verbose=False,
                        allow_writing_files=False,
                    )
                    estimator.fit(
                        x_train_final[tr_idx],
                        y[tr_idx],
                        eval_set=(x_train_final[va_idx], y[va_idx]),
                        use_best_model=True,
                        early_stopping_rounds=80,
                    )
                    proba = estimator.predict_proba(x_train_final[va_idx])
                    test_proba = estimator.predict_proba(x_test_final)
                else:
                    raise

            oof_proba[va_idx] = proba
            oof_pred[va_idx] = np.argmax(proba, axis=1)
            test_proba_sum += test_proba / N_SPLITS

        panel_oof_pred[model_name] = oof_pred
        panel_oof_proba[model_name] = oof_proba
        panel_test_proba[model_name] = test_proba_sum

with aide_stage("score_stage"):
    scores = {}
    for model_name, pred in panel_oof_pred.items():
        score = balanced_accuracy_score(y, pred)
        scores[model_name] = score
        recalls = recall_score(y, pred, average=None, labels=np.arange(class_count))
        print(f"{model_name} 5-fold balanced_accuracy: {score:.6f}", flush=True)
        for cls_name, rec in zip(ordered_class_names, recalls):
            print(f"{model_name} recall_{cls_name}: {rec:.6f}", flush=True)

    final_model_name = (
        "catboost_balanced"
        if "catboost_balanced" in panel_test_proba
        else max(scores, key=scores.get)
    )
    final_oof_pred = panel_oof_pred[final_model_name]
    final_test_proba = panel_test_proba[final_model_name]
    final_test_pred = np.argmax(final_test_proba, axis=1)
    final_test_labels = label_encoder.inverse_transform(final_test_pred)
    final_oof_labels = label_encoder.inverse_transform(final_oof_pred)

    print(f"Selected final model: {final_model_name}", flush=True)

with aide_stage("write_outputs_stage"):
    submission = sample_sub.copy()
    submission["class"] = final_test_labels
    write_submission(submission)

    oof_df = pd.DataFrame(
        {
            "row": np.arange(len(train_base)),
            "target": y_text,
            "prediction": final_oof_labels,
        }
    )
    write_oof_predictions(oof_df)

    test_pred_df = pd.DataFrame({"id": sample_sub["id"].values})
    for idx, cls_name in enumerate(ordered_class_names):
        test_pred_df[f"prob_{cls_name}"] = final_test_proba[:, idx]
    test_pred_df["class"] = final_test_labels
    write_test_predictions(test_pred_df)
