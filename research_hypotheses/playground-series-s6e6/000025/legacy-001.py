import os
import warnings

import numpy as np
import pandas as pd

from aide_solution_helpers import (
    aide_stage,
    load_competition_data,
    log_stage,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier
except Exception as exc:
    raise RuntimeError("catboost is required for this solution") from exc


RANDOM_STATE = 2026
N_SPLITS = 5
TARGET = "class"
ID_COL = "id"
CLASSES = np.array(["GALAXY", "QSO", "STAR"])
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}


def add_base_features(df):
    out = df.copy()
    mag_cols = ["u", "g", "r", "i", "z"]
    for col in mag_cols:
        out[col] = out[col].replace(-9999, np.nan)

    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["r_z"] = out["r"] - out["z"]
    out["u_z"] = out["u"] - out["z"]
    out["gr_iz_ratio"] = out["g_r"] / (np.abs(out["i_z"]) + 1e-3)
    out["ug_ri_ratio"] = out["u_g"] / (np.abs(out["r_i"]) + 1e-3)
    out["mag_mean"] = out[mag_cols].mean(axis=1)
    out["mag_std"] = out[mag_cols].std(axis=1)
    out["mag_min"] = out[mag_cols].min(axis=1)
    out["mag_max"] = out[mag_cols].max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]
    out["redshift_abs"] = np.abs(out["redshift"])
    out["redshift_log1p"] = np.log1p(np.clip(out["redshift"], a_min=0, a_max=None))
    out["alpha_sin"] = np.sin(np.deg2rad(out["alpha"]))
    out["alpha_cos"] = np.cos(np.deg2rad(out["alpha"]))
    out["delta_sin"] = np.sin(np.deg2rad(out["delta"]))
    out["delta_cos"] = np.cos(np.deg2rad(out["delta"]))

    for col in out.columns:
        if out[col].dtype.kind in "fc":
            med = out[col].median()
            out[col] = out[col].fillna(med)
    return out


def make_graph_preprocessor(feature_df):
    cat_cols = [
        c for c in ["spectral_type", "galaxy_population"] if c in feature_df.columns
    ]
    graph_num_cols = [
        "alpha",
        "delta",
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
        "u_r",
        "g_i",
        "r_z",
        "u_z",
        "gr_iz_ratio",
        "ug_ri_ratio",
        "mag_mean",
        "mag_std",
        "mag_range",
        "redshift_abs",
        "redshift_log1p",
        "alpha_sin",
        "alpha_cos",
        "delta_sin",
        "delta_cos",
    ]
    graph_num_cols = [c for c in graph_num_cols if c in feature_df.columns]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), graph_num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def weighted_knn_label_features(
    x_labeled, y_labeled_int, x_query, prefix, n_neighbors=80
):
    n_neighbors = min(n_neighbors, len(y_labeled_int))
    nn = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean", algorithm="auto", n_jobs=-1
    )
    nn.fit(x_labeled)
    distances, indices = nn.kneighbors(x_query, return_distance=True)

    weights = 1.0 / (distances + 1e-3)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    probs = np.zeros((x_query.shape[0], len(CLASSES)), dtype=np.float32)
    neighbor_labels = y_labeled_int[indices]
    for class_idx in range(len(CLASSES)):
        probs[:, class_idx] = (weights * (neighbor_labels == class_idx)).sum(axis=1)

    sorted_probs = np.sort(probs, axis=1)
    max_prob = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1) / np.log(
        len(CLASSES)
    )
    top_class = probs.argmax(axis=1)
    agreement = (neighbor_labels == top_class[:, None]).mean(axis=1)
    coverage = weights[:, : min(20, weights.shape[1])].sum(axis=1)

    data = {
        f"{prefix}_prob_{cls.lower()}": probs[:, i] for i, cls in enumerate(CLASSES)
    }
    data[f"{prefix}_max_prob"] = max_prob
    data[f"{prefix}_margin"] = margin
    data[f"{prefix}_entropy"] = entropy
    data[f"{prefix}_neighbor_agreement"] = agreement
    data[f"{prefix}_labeled_coverage"] = coverage
    return pd.DataFrame(data)


def make_model_preprocessor(feature_df):
    cat_cols = [
        c for c in ["spectral_type", "galaxy_population"] if c in feature_df.columns
    ]
    num_cols = [c for c in feature_df.columns if c not in cat_cols and c != ID_COL]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def build_models(feature_df):
    preprocessor = make_model_preprocessor(feature_df)
    linear = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=600,
                    solver="saga",
                    multi_class="multinomial",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    forest = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=260,
                    max_depth=18,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    catboost = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="BalancedAccuracy",
        iterations=1400,
        learning_rate=0.055,
        depth=7,
        l2_leaf_reg=6.0,
        random_seed=RANDOM_STATE,
        auto_class_weights="Balanced",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.8,
        verbose=False,
        allow_writing_files=False,
    )
    return {"linear": linear, "forest": forest, "catboost": catboost}


def predict_proba_aligned(model, x):
    proba = model.predict_proba(x)
    if hasattr(model, "classes_"):
        model_classes = model.classes_
    elif hasattr(model, "named_steps") and hasattr(
        model.named_steps["model"], "classes_"
    ):
        model_classes = model.named_steps["model"].classes_
    else:
        model_classes = CLASSES

    aligned = np.zeros((len(x), len(CLASSES)), dtype=np.float32)
    for j, cls in enumerate(model_classes):
        cls_label = (
            INT_TO_CLASS[int(cls)] if isinstance(cls, (np.integer, int)) else str(cls)
        )
        aligned[:, CLASS_TO_INT[cls_label]] = proba[:, j]
    return aligned


def fit_catboost(model, x_train, y_train, x_valid=None, y_valid=None):
    cat_features = [
        i
        for i, c in enumerate(x_train.columns)
        if c in ["spectral_type", "galaxy_population"]
    ]
    fit_kwargs = {"cat_features": cat_features}
    if x_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = (x_valid, y_valid)
    try:
        model.fit(x_train, y_train, **fit_kwargs)
        return model
    except Exception as exc:
        print(f"CatBoost GPU failed; retrying on CPU. Reason: {exc}", flush=True)
        cpu_model = model.copy()
        cpu_model.set_params(task_type="CPU")
        cpu_model.fit(x_train, y_train, **fit_kwargs)
        return cpu_model


def evaluate_panel(base_train, base_test, y, graph_oof, graph_test, folds):
    model_results = {}
    best_key = None
    best_score = -np.inf

    feature_sets = {
        "base": (base_train, base_test),
        "graph": (
            pd.concat(
                [base_train.reset_index(drop=True), graph_oof.reset_index(drop=True)],
                axis=1,
            ),
            pd.concat(
                [base_test.reset_index(drop=True), graph_test.reset_index(drop=True)],
                axis=1,
            ),
        ),
    }

    for feature_name, (train_features, test_features) in feature_sets.items():
        models = build_models(train_features)
        for model_name, base_model in models.items():
            key = f"{model_name}_{feature_name}"
            oof_proba = np.zeros((len(train_features), len(CLASSES)), dtype=np.float32)
            fold_scores = []

            for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
                log_stage(f"fit fold {fold}/{N_SPLITS}: {key}")
                x_tr = train_features.iloc[tr_idx].reset_index(drop=True)
                x_va = train_features.iloc[va_idx].reset_index(drop=True)
                y_tr = y[tr_idx]
                y_va = y[va_idx]

                if model_name == "catboost":
                    model = fit_catboost(clone(base_model), x_tr, y_tr, x_va, y_va)
                else:
                    model = clone(base_model)
                    if model_name == "linear":
                        sample_weight = compute_sample_weight(
                            class_weight="balanced", y=y_tr
                        )
                        model.fit(x_tr, y_tr, model__sample_weight=sample_weight)
                    else:
                        model.fit(x_tr, y_tr)

                fold_proba = predict_proba_aligned(model, x_va)
                oof_proba[va_idx] = fold_proba
                fold_pred = CLASSES[fold_proba.argmax(axis=1)]
                fold_score = balanced_accuracy_score(y_va, fold_pred)
                fold_scores.append(fold_score)
                print(
                    f"{key} fold {fold} balanced_accuracy={fold_score:.6f}", flush=True
                )

            oof_pred = CLASSES[oof_proba.argmax(axis=1)]
            score = balanced_accuracy_score(y, oof_pred)
            recalls = recall_score(y, oof_pred, labels=CLASSES, average=None)
            print(
                f"{key} CV balanced_accuracy={score:.6f}; "
                + " ".join(
                    f"recall_{cls}={recalls[i]:.6f}" for i, cls in enumerate(CLASSES)
                ),
                flush=True,
            )
            model_results[key] = {
                "score": score,
                "oof_proba": oof_proba,
                "feature_name": feature_name,
                "model_name": model_name,
                "train_features": train_features,
                "test_features": test_features,
                "base_model": base_model,
            }
            if score > best_score:
                best_score = score
                best_key = key

    return best_key, model_results[best_key]


def main():
    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        y = train[TARGET].astype(str).values
        y_int = np.array([CLASS_TO_INT[v] for v in y], dtype=np.int32)

        train_base = add_base_features(train.drop(columns=[TARGET]))
        test_base = add_base_features(test)

        # Combined train/test covariates are used only for unsupervised scaling context and final
        # unlabeled graph queries; the target column is absent, so no label information is leaked.
        graph_oof = pd.DataFrame(index=np.arange(len(train_base)))
        graph_test_accum = np.zeros((len(test_base), 8), dtype=np.float32)
        graph_cols = None

    with aide_stage("make_folds_stage"):
        folds = list(
            StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            ).split(train_base, y)
        )

    with aide_stage("fit_predict_fold_stage"):
        for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
            log_stage(
                f"building fold-safe graph label-propagation features fold {fold}/{N_SPLITS}"
            )
            prep = make_graph_preprocessor(train_base)
            x_tr_graph = prep.fit_transform(train_base.iloc[tr_idx])
            x_va_graph = prep.transform(train_base.iloc[va_idx])
            x_te_graph = prep.transform(test_base)

            fold_graph_valid = weighted_knn_label_features(
                x_tr_graph, y_int[tr_idx], x_va_graph, "graph"
            )
            graph_oof.iloc[va_idx, :] = (
                fold_graph_valid.values if len(graph_oof.columns) else np.nan
            )
            if graph_cols is None:
                graph_cols = list(fold_graph_valid.columns)
                graph_oof = pd.DataFrame(
                    np.zeros((len(train_base), len(graph_cols)), dtype=np.float32),
                    columns=graph_cols,
                )
                graph_oof.iloc[va_idx, :] = fold_graph_valid.values

            fold_graph_test = weighted_knn_label_features(
                x_tr_graph, y_int[tr_idx], x_te_graph, "graph"
            )
            graph_test_accum += (
                fold_graph_test[graph_cols].values.astype(np.float32) / N_SPLITS
            )

        graph_test_cv = pd.DataFrame(graph_test_accum, columns=graph_cols)

        log_stage("building final all-train graph label-propagation features for test")
        final_prep = make_graph_preprocessor(train_base)
        x_all_graph = final_prep.fit_transform(train_base)
        x_test_graph = final_prep.transform(test_base)
        graph_test_final = weighted_knn_label_features(
            x_all_graph, y_int, x_test_graph, "graph"
        )[graph_cols]

        best_key, best_result = evaluate_panel(
            train_base, test_base, y, graph_oof, graph_test_cv, folds
        )

        log_stage(f"refitting best model on full training data: {best_key}")
        final_model = clone(best_result["base_model"])
        x_full = best_result["train_features"]
        x_test = (
            pd.concat(
                [
                    test_base.reset_index(drop=True),
                    graph_test_final.reset_index(drop=True),
                ],
                axis=1,
            )
            if best_result["feature_name"] == "graph"
            else test_base
        )

        if best_result["model_name"] == "catboost":
            final_model = fit_catboost(final_model, x_full, y)
        else:
            if best_result["model_name"] == "linear":
                sample_weight = compute_sample_weight(class_weight="balanced", y=y)
                final_model.fit(x_full, y, model__sample_weight=sample_weight)
            else:
                final_model.fit(x_full, y)

        test_proba = predict_proba_aligned(final_model, x_test)
        test_pred = CLASSES[test_proba.argmax(axis=1)]

    with aide_stage("score_stage"):
        oof_pred = CLASSES[best_result["oof_proba"].argmax(axis=1)]
        cv_score = balanced_accuracy_score(y, oof_pred)
        recalls = recall_score(y, oof_pred, labels=CLASSES, average=None)
        print(f"BEST_MODEL={best_key}", flush=True)
        print(f"CV balanced_accuracy={cv_score:.6f}", flush=True)
        for cls, val in zip(CLASSES, recalls):
            print(f"CV recall_{cls}={val:.6f}", flush=True)

    with aide_stage("write_outputs_stage"):
        submission = sample_sub[[ID_COL]].copy()
        submission[TARGET] = test_pred
        write_submission(submission)

        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train_base)),
                "target": y,
                "prediction": oof_pred,
            }
        )
        write_oof_predictions(oof_df)

        test_pred_df = sample_sub[[ID_COL]].copy()
        for i, cls in enumerate(CLASSES):
            test_pred_df[f"prob_{cls}"] = test_proba[:, i]
        test_pred_df[TARGET] = test_pred
        write_test_predictions(test_pred_df)


if __name__ == "__main__":
    main()
