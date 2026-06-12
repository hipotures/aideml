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
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp

from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


RANDOM_STATE = 2026
N_SPLITS = 5
CLASS_ORDER = np.array(["GALAXY", "QSO", "STAR"])


def make_color_frame(df):
    out = pd.DataFrame(index=df.index)
    mag_cols = ["u", "g", "r", "i", "z"]
    for c in mag_cols + ["redshift", "alpha", "delta"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["r_z"] = out["r"] - out["z"]
    out["u_z"] = out["u"] - out["z"]
    out["g_z"] = out["g"] - out["z"]
    out["redshift_abs"] = out["redshift"].abs()
    out["redshift_log1p_abs"] = np.log1p(out["redshift_abs"].clip(lower=0))

    for c in ["spectral_type", "galaxy_population"]:
        s = df[c].astype("string").fillna("missing")
        dummies = pd.get_dummies(s, prefix=c, dtype=float)
        out = pd.concat([out, dummies], axis=1)

    return out.replace([np.inf, -np.inf], np.nan)


def make_base_features(train_df, test_df):
    train_x = train_df.drop(columns=["class"]).copy()
    test_x = test_df.copy()

    for df in (train_x, test_x):
        df["u_g"] = df["u"] - df["g"]
        df["g_r"] = df["g"] - df["r"]
        df["r_i"] = df["r"] - df["i"]
        df["i_z"] = df["i"] - df["z"]
        df["u_r"] = df["u"] - df["r"]
        df["g_i"] = df["g"] - df["i"]
        df["r_z"] = df["r"] - df["z"]
        df["u_z"] = df["u"] - df["z"]
        df["redshift_abs"] = df["redshift"].abs()
        df["redshift_log1p_abs"] = np.log1p(df["redshift_abs"].clip(lower=0))

    return train_x.replace([np.inf, -np.inf], np.nan), test_x.replace(
        [np.inf, -np.inf], np.nan
    )


def fit_density_block(train_small, y_train, valid_small, class_order):
    num_cols = train_small.columns.tolist()
    pre = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_tr = pre.fit_transform(train_small[num_cols])
    x_va = pre.transform(valid_small[num_cols])

    priors = np.array([(y_train == cls).mean() for cls in class_order], dtype=float)
    priors = np.clip(priors, 1e-8, 1.0)
    log_priors = np.log(priors)

    ll_valid = np.zeros((x_va.shape[0], len(class_order)), dtype=float)
    models = []

    for j, cls in enumerate(class_order):
        x_cls = x_tr[y_train == cls]
        n_components = int(min(4, max(1, x_cls.shape[0] // 20000)))
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=1e-4,
            max_iter=120,
            n_init=1,
            random_state=RANDOM_STATE + j,
        )
        model.fit(x_cls)
        ll_valid[:, j] = model.score_samples(x_va)
        models.append(model)

    return pre, models, density_features_from_ll(ll_valid, log_priors, class_order)


def transform_density_block(
    pre, models, train_small, y_train, target_small, class_order
):
    x_target = pre.transform(target_small[train_small.columns.tolist()])
    priors = np.array([(y_train == cls).mean() for cls in class_order], dtype=float)
    priors = np.clip(priors, 1e-8, 1.0)
    log_priors = np.log(priors)

    ll = np.column_stack([m.score_samples(x_target) for m in models])
    return density_features_from_ll(ll, log_priors, class_order)


def density_features_from_ll(ll, log_priors, class_order):
    ll = np.clip(ll, -1e6, 1e6)
    log_post = ll + log_priors.reshape(1, -1)
    log_norm = logsumexp(log_post, axis=1, keepdims=True)
    post = np.exp(log_post - log_norm)
    post = np.clip(post, 1e-12, 1.0)

    feats = pd.DataFrame(index=np.arange(ll.shape[0]))
    for j, cls in enumerate(class_order):
        feats[f"density_ll_{cls}"] = ll[:, j]
        feats[f"density_post_{cls}"] = post[:, j]

    idx = {cls: i for i, cls in enumerate(class_order)}
    pairs = [("QSO", "STAR"), ("GALAXY", "QSO"), ("GALAXY", "STAR")]
    for a, b in pairs:
        feats[f"density_llr_{a}_vs_{b}"] = ll[:, idx[a]] - ll[:, idx[b]]
        feats[f"density_post_diff_{a}_vs_{b}"] = post[:, idx[a]] - post[:, idx[b]]

    sorted_post = np.sort(post, axis=1)
    feats["density_post_max"] = sorted_post[:, -1]
    feats["density_post_margin_top2"] = sorted_post[:, -1] - sorted_post[:, -2]
    feats["density_post_entropy"] = -(post * np.log(post)).sum(axis=1)
    feats["density_ll_max"] = ll.max(axis=1)
    feats["density_ll_range"] = ll.max(axis=1) - ll.min(axis=1)

    return feats


def build_preprocessor(x):
    categorical_cols = [
        c
        for c in x.columns
        if x[c].dtype == "object" or str(x[c].dtype).startswith("string")
    ]
    numeric_cols = [c for c in x.columns if c not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        sparse_threshold=0.35,
    )


def make_models():
    return {
        "logreg": Pipeline(
            steps=[
                ("prep", None),
                (
                    "model",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        multi_class="multinomial",
                        solver="saga",
                        max_iter=700,
                        n_jobs=max(1, min(8, os.cpu_count() or 1)),
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            steps=[
                ("prep", None),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=280,
                        max_depth=18,
                        min_samples_leaf=8,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "catboost": CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=900,
            learning_rate=0.055,
            depth=7,
            l2_leaf_reg=7.0,
            random_seed=RANDOM_STATE,
            auto_class_weights="Balanced",
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.8,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def align_proba(model, proba):
    model_classes = np.array(model.classes_)
    aligned = np.zeros((proba.shape[0], len(CLASS_ORDER)), dtype=float)
    for j, cls in enumerate(CLASS_ORDER):
        src = np.where(model_classes == cls)[0][0]
        aligned[:, j] = proba[:, src]
    return aligned


def main():
    with aide_stage("load_data_stage"):
        train, test, sample_sub = load_competition_data()

    y = train["class"].astype(str).values

    with aide_stage("build_features_stage"):
        base_train, base_test = make_base_features(train, test)
        density_train_source = make_color_frame(train)
        density_test_source = make_color_frame(test)

        all_density_cols = sorted(
            set(density_train_source.columns) | set(density_test_source.columns)
        )
        density_train_source = density_train_source.reindex(
            columns=all_density_cols, fill_value=0.0
        )
        density_test_source = density_test_source.reindex(
            columns=all_density_cols, fill_value=0.0
        )

        density_oof = pd.DataFrame(index=train.index)
        density_test_folds = []

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        folds = list(skf.split(base_train, y))

    model_oof_sum = np.zeros((len(train), len(CLASS_ORDER)), dtype=float)
    model_test_sum = np.zeros((len(test), len(CLASS_ORDER)), dtype=float)
    model_scores = {}

    with aide_stage("fit_density_stage"):
        for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
            log_stage(
                f"Fitting fold {fold} class-conditional Gaussian density features"
            )
            _, _, fold_feats = fit_density_block(
                density_train_source.iloc[tr_idx].reset_index(drop=True),
                y[tr_idx],
                density_train_source.iloc[va_idx].reset_index(drop=True),
                CLASS_ORDER,
            )
            fold_feats.index = train.index[va_idx]
            density_oof = pd.concat([density_oof, fold_feats], axis=0)

        density_oof = density_oof.sort_index()

        log_stage(
            "Refitting class-conditional Gaussian density features on full training data for test"
        )
        full_pre, full_density_models, _ = fit_density_block(
            density_train_source.reset_index(drop=True),
            y,
            density_train_source.iloc[:1000].reset_index(drop=True),
            CLASS_ORDER,
        )
        density_test = transform_density_block(
            full_pre,
            full_density_models,
            density_train_source.reset_index(drop=True),
            y,
            density_test_source.reset_index(drop=True),
            CLASS_ORDER,
        )
        density_test.index = test.index

        train_aug = pd.concat(
            [base_train.reset_index(drop=True), density_oof.reset_index(drop=True)],
            axis=1,
        )
        test_aug = pd.concat(
            [base_test.reset_index(drop=True), density_test.reset_index(drop=True)],
            axis=1,
        )

    with aide_stage("fit_predict_fold_stage"):
        for model_name, base_model in make_models().items():
            log_stage(f"Starting 5-fold CV for {model_name}")
            oof = np.zeros((len(train_aug), len(CLASS_ORDER)), dtype=float)
            test_pred = np.zeros((len(test_aug), len(CLASS_ORDER)), dtype=float)

            for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
                log_stage(f"Fold {fold}/{N_SPLITS}: fitting {model_name}")
                if model_name in ("logreg", "rf"):
                    model = clone(base_model)
                    model.set_params(prep=build_preprocessor(train_aug.iloc[tr_idx]))
                    model.fit(train_aug.iloc[tr_idx], y[tr_idx])
                    va_proba = align_proba(
                        model.named_steps["model"],
                        model.predict_proba(train_aug.iloc[va_idx]),
                    )
                    te_proba = align_proba(
                        model.named_steps["model"], model.predict_proba(test_aug)
                    )
                else:
                    cat_cols = [
                        i
                        for i, c in enumerate(train_aug.columns)
                        if train_aug[c].dtype == "object"
                        or str(train_aug[c].dtype).startswith("string")
                    ]
                    x_tr = train_aug.iloc[tr_idx].copy()
                    x_va = train_aug.iloc[va_idx].copy()
                    x_te = test_aug.copy()
                    for c in train_aug.columns[cat_cols]:
                        x_tr[c] = x_tr[c].astype(str).fillna("missing")
                        x_va[c] = x_va[c].astype(str).fillna("missing")
                        x_te[c] = x_te[c].astype(str).fillna("missing")

                    model = clone(base_model)
                    model.fit(x_tr, y[tr_idx], cat_features=cat_cols)
                    va_proba = align_proba(model, model.predict_proba(x_va))
                    te_proba = align_proba(model, model.predict_proba(x_te))

                oof[va_idx] = va_proba
                test_pred += te_proba / N_SPLITS

            pred_labels = CLASS_ORDER[np.argmax(oof, axis=1)]
            score = balanced_accuracy_score(y, pred_labels)
            model_scores[model_name] = score
            model_oof_sum += oof / 3.0
            model_test_sum += test_pred / 3.0
            print(f"{model_name} CV balanced_accuracy: {score:.6f}", flush=True)

    with aide_stage("score_stage"):
        final_oof_labels = CLASS_ORDER[np.argmax(model_oof_sum, axis=1)]
        cv_score = balanced_accuracy_score(y, final_oof_labels)
        print(f"Ensemble CV balanced_accuracy: {cv_score:.6f}", flush=True)

    with aide_stage("write_outputs_stage"):
        final_test_labels = CLASS_ORDER[np.argmax(model_test_sum, axis=1)]

        submission = pd.DataFrame(
            {"id": sample_sub["id"].values, "class": final_test_labels}
        )
        write_submission(submission)

        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train)),
                "target": y,
                "prediction": final_oof_labels,
            }
        )
        write_oof_predictions(oof_df)

        test_pred_df = pd.DataFrame({"id": sample_sub["id"].values})
        for j, cls in enumerate(CLASS_ORDER):
            test_pred_df[f"prob_{cls}"] = model_test_sum[:, j]
        test_pred_df["class"] = final_test_labels
        write_test_predictions(test_pred_df)


if __name__ == "__main__":
    main()
