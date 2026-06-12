import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
    working_dir,
)


def build_line_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy(deep=False)

    feats = df[["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]].copy()
    feats["color_ug"] = feats["u"] - feats["g"]
    feats["color_gr"] = feats["g"] - feats["r"]
    feats["color_ri"] = feats["r"] - feats["i"]
    feats["color_iz"] = feats["i"] - feats["z"]

    n = len(df)
    band_names = np.array(["u", "g", "r", "i", "z"], dtype=object)
    band_centers = np.array([3543.0, 4770.0, 6231.0, 7625.0, 9134.0], dtype=np.float32)
    band_widths = np.array([560.0, 1380.0, 1380.0, 1530.0, 1510.0], dtype=np.float32)
    band_sigmas = band_widths / 2.35482

    band_low = band_centers - band_widths / 2.0
    band_high = band_centers + band_widths / 2.0
    edges = np.sort(np.concatenate([band_low, band_high]))
    optical_min = band_low.min()
    optical_max = band_high.max()

    rest_lines = np.array(
        [1216, 1549, 1909, 2798, 3727, 3934, 3969, 4861, 5007, 6563], dtype=np.float32
    )
    line_names = [
        "LyA_1216",
        "CIV_1549",
        "CIII_1909",
        "MgII_2798",
        "OII_3727",
        "CaH_3934",
        "CaK_3969",
        "Hbeta_4861",
        "OIII_5007",
        "Halpha_6563",
    ]
    line_mask_blue = rest_lines <= 4000

    z = np.clip(df["redshift"].to_numpy(dtype=np.float32), -0.5, 7.0)
    band_affinity_sum = np.zeros((n, len(band_names)), dtype=np.float32)
    per_line_affinity_total = np.zeros((n, len(rest_lines)), dtype=np.float32)
    per_line_min_edge = np.zeros((n, len(rest_lines)), dtype=np.float32)
    per_line_min_center = np.zeros((n, len(rest_lines)), dtype=np.float32)
    per_line_visible = np.zeros((n, len(rest_lines)), dtype=np.float32)

    for li, (wl, ln) in enumerate(zip(rest_lines, line_names)):
        observed = wl * (1.0 + z)
        delta = observed[:, None] - band_centers[None, :]
        abs_delta = np.abs(delta)
        dist_norm = abs_delta / band_widths[None, :]
        affinity = np.exp(-0.5 * (delta / band_sigmas[None, :]) ** 2)

        band_affinity_sum += affinity.astype(np.float32)
        line_band_aff = affinity.sum(axis=1)
        per_line_affinity_total[:, li] = line_band_aff.astype(np.float32)

        nearest_band = np.argmin(abs_delta, axis=1)
        per_line_min_center[:, li] = abs_delta[np.arange(n), nearest_band].astype(
            np.float32
        )
        min_edge = np.min(np.abs(observed[:, None] - edges[None, :]), axis=1).astype(
            np.float32
        )
        per_line_min_edge[:, li] = min_edge
        visible = ((observed >= optical_min) & (observed <= optical_max)).astype(
            np.float32
        )
        per_line_visible[:, li] = visible

        for bi, band in enumerate(band_names):
            feats[f"line_{ln}_{band}_dist_norm"] = dist_norm[:, bi].astype(np.float32)
            feats[f"line_{ln}_{band}_affinity"] = affinity[:, bi].astype(np.float32)
            feats[f"line_{ln}_nearest_{band}"] = (nearest_band == bi).astype(np.float32)

        feats[f"line_{ln}_min_center_normdist"] = (
            per_line_min_center[:, li] / band_widths[nearest_band]
        ).astype(np.float32)
        feats[f"line_{ln}_min_edge_distance"] = min_edge
        feats[f"line_{ln}_visible"] = visible
        feats[f"line_{ln}_affinity_sum_all_bands"] = line_band_aff.astype(np.float32)

    feats["line_visible_count"] = per_line_visible.sum(axis=1).astype(np.float32)
    feats["line_visible_ratio"] = feats["line_visible_count"] / float(len(rest_lines))
    feats["line_gap_distance_mean"] = per_line_min_edge.mean(axis=1).astype(np.float32)
    feats["line_gap_distance_min"] = per_line_min_edge.min(axis=1).astype(np.float32)
    feats["line_center_dist_mean"] = per_line_min_center.mean(axis=1).astype(np.float32)

    blue_sum = per_line_affinity_total[:, line_mask_blue].sum(axis=1)
    red_sum = per_line_affinity_total[:, ~line_mask_blue].sum(axis=1)
    feats["line_affinity_sum_blue"] = blue_sum.astype(np.float32)
    feats["line_affinity_sum_red"] = red_sum.astype(np.float32)
    feats["line_blue_red_balance"] = (blue_sum / (red_sum + 1e-6)).astype(np.float32)
    feats["line_affinity_sum_all"] = per_line_affinity_total.sum(axis=1).astype(
        np.float32
    )

    for bi, band in enumerate(band_names):
        band_aff = band_affinity_sum[:, bi].astype(np.float32)
        feats[f"affinity_{band}_sum"] = band_aff
        feats[f"affinity_{band}_sum_x_{band}_mag"] = band_aff * feats[band].to_numpy(
            dtype=np.float32
        )

    color_pairs = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
    for bi, (left, right) in enumerate(color_pairs):
        aff = band_affinity_sum[:, bi].astype(np.float32)
        col = feats[f"color_{left}_{right}"].to_numpy(dtype=np.float32)
        feats[f"affinity_{left}_sum_x_color_{left}_{right}"] = aff * col
        feats[f"affinity_{right}_sum_x_color_{left}_{right}"] = (
            band_affinity_sum[:, bi + 1].astype(np.float32) * col
        )

    # Keep all generated features numeric/floating for the tree model.
    return feats.astype(np.float32)


def make_model(use_gpu: bool) -> CatBoostClassifier:
    base = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=600,
        learning_rate=0.08,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        auto_class_weights="Balanced",
        verbose=False,
        thread_count=-1,
    )
    if use_gpu:
        return CatBoostClassifier(
            task_type="GPU", devices="0", gpu_ram_part=0.8, **base
        )
    return CatBoostClassifier(task_type="CPU", **base)


def main() -> None:
    with aide_stage("load_data_stage"):
        train, test, sample_sub = load_competition_data()
        _ = working_dir()
        # Hypothesis 000035 uses only competition-provided train/test covariates in this
        # verification run (no auxiliary star_classification merge).

        le = LabelEncoder()
        y = le.fit_transform(train["class"])
        class_names = le.classes_.tolist()

    raw_all = pd.concat(
        [train.drop(columns=["class"]), test], axis=0, ignore_index=True
    )

    with aide_stage("build_features_stage"):
        line_features_all = build_line_features(
            raw_all[
                [
                    "alpha",
                    "delta",
                    "u",
                    "g",
                    "r",
                    "i",
                    "z",
                    "redshift",
                    "spectral_type",
                    "galaxy_population",
                ]
            ]
        )
        cat_features = pd.get_dummies(
            raw_all[["spectral_type", "galaxy_population"]].astype("category"),
            prefix=["spectral_type", "galaxy_population"],
            dtype=np.float32,
        )
        features_all = pd.concat(
            [
                line_features_all.reset_index(drop=True),
                cat_features.reset_index(drop=True),
            ],
            axis=1,
        ).astype(np.float32)
        n_train = len(train)
        X_train = features_all.iloc[:n_train].reset_index(drop=True)
        X_test = features_all.iloc[n_train:].reset_index(drop=True)
        print(
            f"Feature matrix shape: train={X_train.shape}, test={X_test.shape}",
            flush=True,
        )
        print(f"Built {X_train.shape[1]} features (Hypothesis 000035).", flush=True)

    oof_pred = np.zeros(len(train), dtype=np.int32)
    oof_proba = np.zeros((len(train), len(class_names)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(class_names)), dtype=np.float32)
    fold_scores = []
    importances = np.zeros(X_train.shape[1], dtype=np.float64)

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X_train, y))

    with aide_stage("fit_predict_fold_stage"):
        for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
            xtr, xva = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr, yva = y[tr_idx], y[val_idx]

            model = make_model(use_gpu=True)
            log_stage(
                f"fit_predict_fold_stage: fold={fold_idx}/5 model=CatBoostClassifier GPU"
            )
            try:
                model.fit(xtr, ytr, eval_set=(xva, yva), verbose=False)
            except Exception as exc:
                print(
                    f"Fold {fold_idx}: GPU training failed ({exc}). Falling back to CPU.",
                    flush=True,
                )
                model = make_model(use_gpu=False)
                log_stage(
                    f"fit_predict_fold_stage: fold={fold_idx}/5 model=CatBoostClassifier CPU fallback"
                )
                model.fit(xtr, ytr, eval_set=(xva, yva), verbose=False)

            val_prob = model.predict_proba(xva)
            val_pred = np.argmax(val_prob, axis=1).astype(np.int32)

            oof_pred[val_idx] = val_pred
            oof_proba[val_idx] = val_prob.astype(np.float32)
            fold_score = balanced_accuracy_score(yva, val_pred)
            fold_scores.append(float(fold_score))
            print(f"Fold {fold_idx} balanced_accuracy={fold_score:.6f}", flush=True)

            test_fold = model.predict_proba(X_test).astype(np.float32)
            test_proba += test_fold / float(len(folds))
            importances += np.asarray(model.get_feature_importance(), dtype=np.float64)

    with aide_stage("score_stage"):
        oof_label = le.inverse_transform(oof_pred)
        true_label = le.inverse_transform(y)
        oof_score = balanced_accuracy_score(true_label, oof_label)
        print("OOF balanced_accuracy:", float(oof_score), flush=True)
        print(
            "Per-fold balanced_accuracy:",
            [round(v, 6) for v in fold_scores],
            flush=True,
        )

        mean_importance = importances / float(len(folds))
        imp_rank = pd.Series(mean_importance, index=X_train.columns).sort_values(
            ascending=False
        )
        print("Top feature importances (mean over folds):", flush=True)
        print(imp_rank.head(30).to_string(), flush=True)

    with aide_stage("write_outputs_stage"):
        write_oof_predictions(
            pd.DataFrame(
                {
                    "row": np.arange(len(train), dtype=np.int32),
                    "target": true_label,
                    "prediction": oof_label,
                }
            )
        )

        test_pred_probs = pd.DataFrame(test_proba, columns=class_names)
        write_test_predictions(
            pd.concat(
                [sample_sub[["id"]].reset_index(drop=True), test_pred_probs], axis=1
            )
        )

        final_test_pred = le.inverse_transform(np.argmax(test_proba, axis=1))
        submission = pd.DataFrame(
            {"id": sample_sub["id"].to_numpy(), "class": final_test_pred}
        )
        write_submission(submission)


if __name__ == "__main__":
    main()
