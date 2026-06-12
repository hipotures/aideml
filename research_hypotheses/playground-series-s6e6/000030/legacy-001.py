import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from catboost import CatBoostClassifier

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)


class StellarLocusResidualTransformer:
    """Build target-free locus-residual features from ugriz colors."""

    def __init__(self, n_bins: int = 80, smooth_window: int = 7):
        self.n_bins = max(20, int(n_bins))
        self.smooth_window = max(3, int(smooth_window))
        self.bin_edges = None
        self.bin_count = None
        self.color_names = ("u_g", "g_r", "r_i", "i_z")
        self.median_curves = {}
        self.mad_curves = {}

    def _compute_color_series(self, df):
        u = df["u"].to_numpy(dtype=float)
        g = df["g"].to_numpy(dtype=float)
        r = df["r"].to_numpy(dtype=float)
        i = df["i"].to_numpy(dtype=float)
        z = df["z"].to_numpy(dtype=float)
        return {
            "u_g": u - g,
            "g_r": g - r,
            "r_i": r - i,
            "i_z": i - z,
        }

    def fit(self, df):
        colors = self._compute_color_series(df)
        coord = (df["g"].to_numpy(dtype=float) - df["i"].to_numpy(dtype=float)).astype(
            float
        )

        finite = np.isfinite(coord)
        if finite.sum() < 1000:
            coord_clean = np.linspace(-2.5, 2.5, len(coord))
            finite = np.isfinite(coord_clean)
            coord = coord_clean
        q = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(coord[finite], q)
        edges = np.unique(edges.astype(float))
        if len(edges) < 3:
            mn = float(np.nanmin(coord[finite]))
            mx = float(np.nanmax(coord[finite]))
            if np.isclose(mx, mn):
                mx = mn + 1.0
            edges = np.linspace(mn, mx, self.n_bins + 1)
        edges = np.asarray(edges, dtype=float)
        self.bin_edges = edges
        self.bin_count = len(edges) - 1

        bin_idx = np.clip(
            np.digitize(coord, self.bin_edges[1:-1], right=True), 0, self.bin_count - 1
        )
        for name in self.color_names:
            arr = colors[name].astype(float)
            med = np.full(self.bin_count, np.nan, dtype=float)
            mad = np.full(self.bin_count, np.nan, dtype=float)
            for b in range(self.bin_count):
                vals = arr[(bin_idx == b) & np.isfinite(arr)]
                if vals.size == 0:
                    continue
                m = np.nanmedian(vals)
                med[b] = m
                mad[b] = np.nanmedian(np.abs(vals - m)) * 1.4826
            med_series = pd.Series(med).astype(float)
            mad_series = pd.Series(mad).astype(float)
            med_series = med_series.rolling(
                self.smooth_window, min_periods=1, center=True
            ).median()
            mad_series = mad_series.rolling(
                self.smooth_window, min_periods=1, center=True
            ).median()
            med_series = med_series.fillna(method="bfill").fillna(method="ffill")
            mad_series = mad_series.fillna(method="bfill").fillna(method="ffill")
            self.median_curves[name] = med_series.to_numpy(dtype=float)
            self.mad_curves[name] = np.maximum(mad_series.to_numpy(dtype=float), 1e-6)

        self.coord_global = float(np.nanmedian(coord[finite])) if finite.any() else 0.0
        self.coord_fallback_bin = min(self.bin_count - 1, self.bin_count // 2)
        return self

    def transform(self, df):
        colors = self._compute_color_series(df)
        coord = (df["g"].to_numpy(dtype=float) - df["i"].to_numpy(dtype=float)).astype(
            float
        )

        finite = np.isfinite(coord)
        bin_idx = np.clip(
            np.digitize(
                np.where(finite, coord, self.coord_global),
                self.bin_edges[1:-1],
                right=True,
            ),
            0,
            self.bin_count - 1,
        )
        bin_idx[~finite] = self.coord_fallback_bin

        out = pd.DataFrame(index=df.index)
        total_sq = np.zeros(len(df), dtype=float)
        abs_scores = []
        norm_scores = []

        for name in self.color_names:
            med = self.median_curves[name][bin_idx]
            mad = self.mad_curves[name][bin_idx]
            resid = colors[name] - med
            resid_norm = resid / np.maximum(mad, 1e-6)
            out[f"residual_{name}"] = resid.astype(np.float32)
            out[f"residual_norm_{name}"] = resid_norm.astype(np.float32)
            out[f"abs_residual_{name}"] = np.abs(resid).astype(np.float32)
            out[f"abs_residual_norm_{name}"] = np.abs(resid_norm).astype(np.float32)
            total_sq += resid_norm.astype(float) ** 2
            abs_scores.append(np.abs(resid_norm))
            norm_scores.append(resid_norm)

        out["locus_coord"] = coord.astype(np.float32)
        out["locus_total_norm_distance"] = np.sqrt(total_sq).astype(np.float32)
        out["locus_max_abs_norm_residual"] = np.maximum.reduce(abs_scores).astype(
            np.float32
        )
        out["locus_l1_norm_abs_residual"] = np.sum(np.abs(norm_scores), axis=0).astype(
            np.float32
        )
        out["uv_excess_score"] = np.maximum(-out["residual_norm_u_g"], 0.0).astype(
            np.float32
        )
        out["red_excess_score"] = np.maximum(
            out["residual_norm_r_i"], out["residual_norm_i_z"]
        ).astype(np.float32)
        return out


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_raw_features(df, residue_transformer):
    base = pd.DataFrame(index=df.index)
    numeric_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    for c in numeric_cols:
        base[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        cvals = base[c].replace([np.inf, -np.inf], np.nan)
        base[c] = cvals.fillna(cvals.median())

    base["spectral_type_cat"] = (
        df["spectral_type"].astype(str).str.strip().replace("nan", "missing")
    )
    base["galaxy_population_cat"] = (
        df["galaxy_population"].astype(str).str.strip().replace("nan", "missing")
    )
    base["spectral_type_num"] = (
        pd.to_numeric(df["spectral_type"], errors="coerce").fillna(-1.0).astype(float)
    )
    base["galaxy_population_num"] = (
        pd.to_numeric(df["galaxy_population"], errors="coerce")
        .fillna(-1.0)
        .astype(float)
    )

    base["u_g"] = base["u"] - base["g"]
    base["g_r"] = base["g"] - base["r"]
    base["r_i"] = base["r"] - base["i"]
    base["i_z"] = base["i"] - base["z"]

    locus = residue_transformer.transform(df)
    features = pd.concat([base, locus], axis=1)

    # Interactions with redshift and spectral-type as requested by hypothesis.
    features["spectral_redshift"] = features["spectral_type_num"] * features["redshift"]
    features["redshift_x_ug"] = features["redshift"] * features["residual_u_g"]
    features["redshift_x_gr"] = features["redshift"] * features["residual_g_r"]
    features["redshift_x_ri"] = features["redshift"] * features["residual_r_i"]
    features["redshift_x_iz"] = features["redshift"] * features["residual_i_z"]
    features["spectral_x_ug_norm"] = (
        features["spectral_type_num"] * features["residual_norm_u_g"]
    )
    features["spectral_x_gr_norm"] = (
        features["spectral_type_num"] * features["residual_norm_g_r"]
    )

    for c in ["u_g", "g_r", "r_i", "i_z"]:
        features[c] = features[c].astype(np.float32)
    for c in [
        "alpha",
        "delta",
        "u",
        "g",
        "r",
        "i",
        "z",
        "redshift",
        "spectral_type_num",
        "galaxy_population_num",
    ]:
        features[c] = features[c].astype(np.float32)

    return features


def make_preprocessor(numeric_cols, cat_cols, scale_numeric):
    num_transform = StandardScaler() if scale_numeric else "passthrough"
    ct = ColumnTransformer(
        transformers=[
            ("numeric", num_transform, numeric_cols),
            ("cat", make_encoder(), cat_cols),
        ],
        remainder="drop",
    )
    return ct


def align_proba(proba, model_classes, target_classes):
    model_classes = np.asarray(model_classes)
    target_classes = np.asarray(target_classes)
    if np.array_equal(model_classes, target_classes):
        return proba
    idx = [int(np.where(model_classes == cls)[0][0]) for cls in target_classes]
    return proba[:, idx]


def fit_catboost(X_train, y_train, random_state, fold_id):
    params = dict(
        loss_function="MultiClass",
        iterations=220,
        depth=8,
        learning_rate=0.09,
        l2_leaf_reg=4.0,
        random_seed=random_state,
        auto_class_weights="Balanced",
        verbose=False,
        thread_count=-1,
    )
    try:
        log_stage(f"Fold {fold_id}: CatBoost training on GPU (task_type=GPU)")
        gpu_model = CatBoostClassifier(
            task_type="GPU", devices="0", gpu_ram_part=0.8, **params
        )
        gpu_model.fit(X_train, y_train)
        return gpu_model
    except Exception as exc:
        print(
            f"Fold {fold_id}: CatBoost GPU training failed, falling back to CPU: {exc}",
            flush=True,
        )
        cpu_model = CatBoostClassifier(task_type="CPU", **params)
        cpu_model.fit(X_train, y_train)
        return cpu_model


def main():
    train, test, _ = load_competition_data()
    seed = 42

    # Auxiliary SDSS-like `star_classification.csv` is intentionally not merged;
    # hypothesis is validated target-free using competition covariates only.
    target = train["class"].astype(str).values

    feat_cols_for_locus = ["u", "g", "r", "i", "z"]
    residue_transformer = StellarLocusResidualTransformer(n_bins=80, smooth_window=9)

    with aide_stage("build_features_stage"):
        locus_df = pd.concat(
            [train[feat_cols_for_locus], test[feat_cols_for_locus]],
            axis=0,
            ignore_index=True,
        )
        residue_transformer.fit(locus_df)

        X_train = build_raw_features(train, residue_transformer)
        X_test = build_raw_features(test, residue_transformer)

        # Replace any remaining infinities or NaNs.
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(
            X_train.median(numeric_only=True)
        )
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(
            X_train.median(numeric_only=True)
        )

    cat_cols = ["spectral_type_cat", "galaxy_population_cat"]
    numeric_cols = [c for c in X_train.columns if c not in cat_cols]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)
    class_values = label_encoder.transform(label_encoder.classes_)
    n_classes = len(label_encoder.classes_)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_proba = np.zeros((len(train), n_classes), dtype=float)
    test_proba = np.zeros((len(test), n_classes), dtype=float)

    model_weights = {
        "logreg": 0.33,
        "extrees": 0.28,
        "catboost": 0.39,
    }

    with aide_stage("make_folds_stage"):
        folds = list(skf.split(X_train, y))

    with aide_stage("fit_predict_fold_stage"):
        for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
            log_stage(f"Fold {fold_idx}: preparing preprocessing and split")
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]

            # Logistic path (scaled numeric + one-hot categorical)
            pre_lin = make_preprocessor(numeric_cols, cat_cols, scale_numeric=True)
            X_tr_lin = pre_lin.fit_transform(X_tr)
            X_va_lin = pre_lin.transform(X_va)
            X_test_lin = pre_lin.transform(X_test)

            # Tree path (one-hot categorical, passthrough numeric)
            pre_tree = make_preprocessor(numeric_cols, cat_cols, scale_numeric=False)
            X_tr_tree = pre_tree.fit_transform(X_tr)
            X_va_tree = pre_tree.transform(X_va)
            X_test_tree = pre_tree.transform(X_test)

            # 1) Regularized multinomial linear model
            log_stage(f"Fold {fold_idx}: fitting logistic regression")
            logreg = Pipeline(
                steps=[
                    (
                        "clf",
                        LogisticRegression(
                            multi_class="multinomial",
                            solver="saga",
                            max_iter=350,
                            n_jobs=-1,
                            C=1.0,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    )
                ]
            )
            logreg.fit(X_tr_lin, y[tr_idx])
            val_pred = logreg.predict_proba(X_va_lin)
            test_pred = logreg.predict_proba(X_test_lin)
            oof_proba[va_idx] += model_weights["logreg"] * align_proba(
                val_pred, logreg.named_steps["clf"].classes_, class_values
            )
            test_proba += model_weights["logreg"] * align_proba(
                test_pred, logreg.named_steps["clf"].classes_, class_values
            )

            # 2) Shallow class-balanced tree model
            log_stage(f"Fold {fold_idx}: fitting ExtraTrees")
            et = ExtraTreesClassifier(
                n_estimators=140,
                max_depth=16,
                min_samples_leaf=16,
                n_jobs=-1,
                random_state=seed,
                class_weight="balanced",
            )
            et.fit(X_tr_tree, y[tr_idx])
            val_pred = et.predict_proba(X_va_tree)
            test_pred = et.predict_proba(X_test_tree)
            oof_proba[va_idx] += model_weights["extrees"] * align_proba(
                val_pred, et.classes_, class_values
            )
            test_proba += model_weights["extrees"] * align_proba(
                test_pred, et.classes_, class_values
            )

            # 3) Class-balanced gradient-boosted tree model (CatBoost)
            log_stage(f"Fold {fold_idx}: fitting CatBoost")
            cb = fit_catboost(X_tr_tree, y[tr_idx], seed + fold_idx, fold_id=fold_idx)
            val_pred = cb.predict_proba(X_va_tree)
            test_pred = cb.predict_proba(X_test_tree)
            oof_proba[va_idx] += model_weights["catboost"] * align_proba(
                val_pred, cb.classes_, class_values
            )
            test_proba += model_weights["catboost"] * align_proba(
                test_pred, cb.classes_, class_values
            )

            va_pred_idx = np.argmax(
                model_weights["logreg"]
                * align_proba(
                    logreg.predict_proba(X_va_lin),
                    logreg.named_steps["clf"].classes_,
                    class_values,
                )
                + model_weights["extrees"]
                * align_proba(et.predict_proba(X_va_tree), et.classes_, class_values)
                + model_weights["catboost"]
                * align_proba(cb.predict_proba(X_va_tree), cb.classes_, class_values),
                axis=1,
            )
            fold_bacc = balanced_accuracy_score(y[va_idx], va_pred_idx)
            fold_recall = recall_score(
                y[va_idx],
                va_pred_idx,
                labels=class_values,
                average=None,
                zero_division=0,
            )
            fold_class_scores = {
                cls: float(r) for cls, r in zip(label_encoder.classes_, fold_recall)
            }
            print(
                f"Fold {fold_idx} BALANCED_ACCURACY={fold_bacc:.6f} | "
                + " | ".join([f"{k}={v:.6f}" for k, v in fold_class_scores.items()]),
                flush=True,
            )

    # Normalize blend weights if they do not sum to 1 exactly due to floating point.
    weight_sum = sum(model_weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        oof_proba /= weight_sum
        test_proba /= weight_sum

    with aide_stage("score_stage"):
        oof_pred_idx = np.argmax(oof_proba, axis=1)
        overall = balanced_accuracy_score(y, oof_pred_idx)
        recall = recall_score(
            y, oof_pred_idx, labels=class_values, average=None, zero_division=0
        )
        per_class = {cls: float(r) for cls, r in zip(label_encoder.classes_, recall)}
        print(f"OOF BALANCED_ACCURACY={overall:.6f}", flush=True)
        for cls, val in per_class.items():
            print(f"OOF Recall({cls})={val:.6f}", flush=True)

    with aide_stage("write_outputs_stage"):
        oof_df = pd.DataFrame(
            {
                "row": train.index.to_numpy(),
                "target": train["class"].astype(str).values,
                "prediction": label_encoder.inverse_transform(oof_pred_idx),
            }
        )
        write_oof_predictions(oof_df)

        test_pred_df = pd.DataFrame({"id": test["id"].astype(str).values})
        for i, cls in enumerate(label_encoder.classes_):
            test_pred_df[cls] = test_proba[:, i]
        write_test_predictions(test_pred_df)

        final_pred = label_encoder.inverse_transform(np.argmax(test_proba, axis=1))
        sub = pd.DataFrame({"id": test["id"], "class": final_pred})
        write_submission(sub)


if __name__ == "__main__":
    main()
