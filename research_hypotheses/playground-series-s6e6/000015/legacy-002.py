import gc
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
)

warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
INNER_SPLITS = 4
CLASS_NAMES = np.array(["GALAXY", "QSO", "STAR"])
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def build_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    base = df[["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]].copy()
    base["ug"] = df["u"] - df["g"]
    base["gr"] = df["g"] - df["r"]
    base["ri"] = df["r"] - df["i"]
    base["iz"] = df["i"] - df["z"]
    base["ur"] = df["u"] - df["r"]
    base["gi"] = df["g"] - df["i"]
    base["spectral_type"] = df["spectral_type"].fillna("missing").astype(str)
    base["galaxy_population"] = df["galaxy_population"].fillna("missing").astype(str)
    return base


def make_bin_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(series.to_numpy(dtype=float), quantiles))
    if edges.size < 2:
        return np.array([-np.inf, np.inf], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(float)


def apply_bins(series: pd.Series, edges: np.ndarray, prefix: str) -> pd.Series:
    codes = pd.cut(series, bins=edges, labels=False, include_lowest=True)
    codes = pd.Series(codes, index=series.index).fillna(-1).astype(int).astype(str)
    return prefix + "_" + codes


def build_likelihood_maps(
    cat_series: pd.Series, y: np.ndarray, priors: np.ndarray, alpha: float
):
    stats = pd.crosstab(cat_series, y).reindex(
        columns=np.arange(len(CLASS_NAMES)), fill_value=0
    )
    totals = stats.sum(axis=1).astype(float)
    maps = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        probs = (stats[class_idx] + alpha * priors[class_idx]) / (totals + alpha)
        maps[class_name] = probs.to_dict()
    return maps


def add_fold_safe_encodings(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols,
    alpha: float = 20.0,
):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    priors = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float64)
    priors /= priors.sum()

    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)

    for col in cat_cols:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        freq_col = f"{col}_freq"
        train_df[freq_col] = train_df[col].map(freq_map).fillna(0.0).astype(np.float32)
        valid_df[freq_col] = valid_df[col].map(freq_map).fillna(0.0).astype(np.float32)
        test_df[freq_col] = test_df[col].map(freq_map).fillna(0.0).astype(np.float32)

        inner_encoded = {
            class_name: np.empty(len(train_df), dtype=np.float32)
            for class_name in CLASS_NAMES
        }
        for inner_train_idx, inner_valid_idx in inner_cv.split(train_df, y_train):
            inner_train = train_df.iloc[inner_train_idx]
            inner_valid = train_df.iloc[inner_valid_idx]
            inner_y = y_train[inner_train_idx]
            maps = build_likelihood_maps(inner_train[col], inner_y, priors, alpha)
            for class_idx, class_name in enumerate(CLASS_NAMES):
                inner_encoded[class_name][inner_valid_idx] = (
                    inner_valid[col]
                    .map(maps[class_name])
                    .fillna(priors[class_idx])
                    .to_numpy(np.float32)
                )

        full_maps = build_likelihood_maps(train_df[col], y_train, priors, alpha)
        for class_idx, class_name in enumerate(CLASS_NAMES):
            enc_col = f"{col}_p_{class_name.lower()}"
            train_df[enc_col] = inner_encoded[class_name]
            valid_df[enc_col] = (
                valid_df[col]
                .map(full_maps[class_name])
                .fillna(priors[class_idx])
                .astype(np.float32)
            )
            test_df[enc_col] = (
                test_df[col]
                .map(full_maps[class_name])
                .fillna(priors[class_idx])
                .astype(np.float32)
            )

    return train_df, valid_df, test_df


def engineer_fold_features(train_base, valid_base, test_base, y_train):
    train_feat = train_base.copy().reset_index(drop=True)
    valid_feat = valid_base.copy().reset_index(drop=True)
    test_feat = test_base.copy().reset_index(drop=True)

    bin_specs = {
        "redshift_bin": ("redshift", 8),
        "u_bin": ("u", 8),
        "g_bin": ("g", 8),
        "r_bin": ("r", 8),
        "color_gr_bin": ("gr", 8),
    }
    for new_col, (src_col, n_bins) in bin_specs.items():
        edges = make_bin_edges(train_feat[src_col], n_bins)
        train_feat[new_col] = apply_bins(train_feat[src_col], edges, new_col)
        valid_feat[new_col] = apply_bins(valid_feat[src_col], edges, new_col)
        test_feat[new_col] = apply_bins(test_feat[src_col], edges, new_col)

    for frame in (train_feat, valid_feat, test_feat):
        frame["spectral_x_redshift_bin"] = (
            frame["spectral_type"] + "__" + frame["redshift_bin"]
        )
        frame["population_x_color_bin"] = (
            frame["galaxy_population"] + "__" + frame["color_gr_bin"]
        )

    train_feat, valid_feat, test_feat = add_fold_safe_encodings(
        train_feat,
        valid_feat,
        test_feat,
        y_train,
        cat_cols=["spectral_type", "galaxy_population"],
        alpha=20.0,
    )

    cat_cols = [
        "spectral_type",
        "galaxy_population",
        "redshift_bin",
        "u_bin",
        "g_bin",
        "r_bin",
        "color_gr_bin",
        "spectral_x_redshift_bin",
        "population_x_color_bin",
    ]
    numeric_cols = [
        "alpha",
        "delta",
        "u",
        "g",
        "r",
        "i",
        "z",
        "redshift",
        "ug",
        "gr",
        "ri",
        "iz",
        "ur",
        "gi",
        "spectral_type_freq",
        "galaxy_population_freq",
        "spectral_type_p_galaxy",
        "spectral_type_p_qso",
        "spectral_type_p_star",
        "galaxy_population_p_galaxy",
        "galaxy_population_p_qso",
        "galaxy_population_p_star",
    ]

    for frame in (train_feat, valid_feat, test_feat):
        frame[numeric_cols] = frame[numeric_cols].astype(np.float32)
        for col in cat_cols:
            frame[col] = frame[col].astype(str)

    return train_feat, valid_feat, test_feat, numeric_cols, cat_cols


def make_sparse_preprocessor(numeric_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
                cat_cols,
            ),
        ],
        sparse_threshold=1.0,
    )


def fit_catboost(train_df, y_train, valid_df, y_valid, cat_cols, fold_idx):
    params = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        iterations=700,
        learning_rate=0.06,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=SEED + fold_idx,
        verbose=False,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
    )
    train_pool = Pool(train_df, label=y_train, cat_features=cat_cols)
    valid_pool = Pool(valid_df, label=y_valid, cat_features=cat_cols)
    try:
        model = CatBoostClassifier(
            **params,
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.8,
        )
        model.fit(train_pool, eval_set=valid_pool, verbose=False)
    except Exception as exc:
        print(
            f"Fold {fold_idx}: CatBoost GPU failed with {exc.__class__.__name__}, retrying on CPU.",
            flush=True,
        )
        model = CatBoostClassifier(**params, task_type="CPU")
        model.fit(train_pool, eval_set=valid_pool, verbose=False)
    return model, train_pool, valid_pool


def fit_xgboost(X_train, y_train, X_valid, y_valid, sample_weight, fold_idx):
    gpu_params = dict(
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        eval_metric="mlogloss",
        n_estimators=450,
        learning_rate=0.07,
        max_depth=8,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=SEED + fold_idx,
        tree_method="hist",
        device="cuda",
        verbosity=0,
    )
    cpu_params = dict(gpu_params)
    cpu_params["device"] = "cpu"

    try:
        model = XGBClassifier(**gpu_params)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    except Exception as exc:
        print(
            f"Fold {fold_idx}: XGBoost CUDA failed with {exc.__class__.__name__}, retrying on CPU.",
            flush=True,
        )
        model = XGBClassifier(**cpu_params)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    return model


def probs_to_labels(probs: np.ndarray) -> np.ndarray:
    return CLASS_NAMES[np.argmax(probs, axis=1)]


with aide_stage("build_features_stage"):
    workdir = working_dir()
    train, test, sample_sub = load_competition_data()
    # Keep this experiment isolated to hypothesis 000015 on the competition data only.
    base_train = build_base_frame(train)
    base_test = build_base_frame(test)
    y = train["class"].map(CLASS_TO_INT).to_numpy()

with aide_stage("make_folds_stage"):
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(base_train, y))

model_names = ["logistic_regression", "catboost", "xgboost"]
oof_probs = {
    name: np.zeros((len(train), len(CLASS_NAMES)), dtype=np.float32)
    for name in model_names
}
test_probs = {
    name: np.zeros((len(test), len(CLASS_NAMES)), dtype=np.float32)
    for name in model_names
}

with aide_stage("fit_predict_fold_stage"):
    for fold_idx, (train_idx, valid_idx) in enumerate(folds, start=1):
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        fold_train_base = base_train.iloc[train_idx]
        fold_valid_base = base_train.iloc[valid_idx]

        train_feat, valid_feat, test_feat, numeric_cols, cat_cols = (
            engineer_fold_features(fold_train_base, fold_valid_base, base_test, y_train)
        )

        preprocessor = make_sparse_preprocessor(numeric_cols, cat_cols)
        feature_cols = numeric_cols + cat_cols
        X_train_sparse = preprocessor.fit_transform(train_feat[feature_cols])
        X_valid_sparse = preprocessor.transform(valid_feat[feature_cols])
        X_test_sparse = preprocessor.transform(test_feat[feature_cols])

        log_stage(
            f"event=info|stage=fit_predict_fold_stage|fold={fold_idx}|model=logistic_regression"
        )
        logistic = LogisticRegression(
            C=1.5,
            solver="saga",
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=600,
            random_state=SEED + fold_idx,
        )
        logistic.fit(X_train_sparse, y_train)
        valid_pred = logistic.predict_proba(X_valid_sparse).astype(np.float32)
        test_pred = logistic.predict_proba(X_test_sparse).astype(np.float32)
        oof_probs["logistic_regression"][valid_idx] = valid_pred
        test_probs["logistic_regression"] += test_pred / N_SPLITS

        log_stage(
            f"event=info|stage=fit_predict_fold_stage|fold={fold_idx}|model=catboost"
        )
        cat_model, _, valid_pool = fit_catboost(
            train_feat[feature_cols],
            y_train,
            valid_feat[feature_cols],
            y_valid,
            cat_cols,
            fold_idx,
        )
        valid_pred = cat_model.predict_proba(valid_pool).astype(np.float32)
        test_pool = Pool(test_feat[feature_cols], cat_features=cat_cols)
        test_pred = cat_model.predict_proba(test_pool).astype(np.float32)
        oof_probs["catboost"][valid_idx] = valid_pred
        test_probs["catboost"] += test_pred / N_SPLITS

        log_stage(
            f"event=info|stage=fit_predict_fold_stage|fold={fold_idx}|model=xgboost"
        )
        xgb_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        xgb_model = fit_xgboost(
            X_train_sparse,
            y_train,
            X_valid_sparse,
            y_valid,
            xgb_weights,
            fold_idx,
        )
        valid_pred = xgb_model.predict_proba(X_valid_sparse).astype(np.float32)
        test_pred = xgb_model.predict_proba(X_test_sparse).astype(np.float32)
        oof_probs["xgboost"][valid_idx] = valid_pred
        test_probs["xgboost"] += test_pred / N_SPLITS

        for model_name in model_names:
            fold_labels = probs_to_labels(oof_probs[model_name][valid_idx])
            fold_score = balanced_accuracy_score(
                train.iloc[valid_idx]["class"], fold_labels
            )
            print(
                f"Fold {fold_idx} {model_name} balanced_accuracy: {fold_score:.6f}",
                flush=True,
            )

        del (
            train_feat,
            valid_feat,
            test_feat,
            preprocessor,
            X_train_sparse,
            X_valid_sparse,
            X_test_sparse,
            logistic,
            cat_model,
            xgb_model,
        )
        gc.collect()

with aide_stage("score_stage"):
    scores = {}
    for model_name in model_names:
        labels = probs_to_labels(oof_probs[model_name])
        score = balanced_accuracy_score(train["class"], labels)
        scores[model_name] = score
        print(f"OOF balanced_accuracy {model_name}: {score:.6f}", flush=True)

    best_model = max(scores, key=scores.get)
    final_oof_probs = oof_probs[best_model]
    final_test_probs = test_probs[best_model]
    final_oof_labels = probs_to_labels(final_oof_probs)
    final_test_labels = probs_to_labels(final_test_probs)
    primary_score = balanced_accuracy_score(train["class"], final_oof_labels)
    print(f"Selected model: {best_model}", flush=True)
    print(f"Primary CV balanced_accuracy: {primary_score:.6f}", flush=True)

with aide_stage("write_outputs_stage"):
    submission = sample_sub[["id"]].copy()
    submission["class"] = final_test_labels
    write_submission(submission)

    oof_frame = pd.DataFrame(
        {
            "row": np.arange(len(train), dtype=np.int64),
            "target": train["class"].to_numpy(),
            "prediction": final_oof_labels,
        }
    )
    write_oof_predictions(oof_frame)

    test_pred_frame = sample_sub[["id"]].copy()
    test_pred_frame["class"] = final_test_labels
    for class_idx, class_name in enumerate(CLASS_NAMES):
        test_pred_frame[class_name] = final_test_probs[:, class_idx]
    write_test_predictions(test_pred_frame)
