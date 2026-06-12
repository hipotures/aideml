import os
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from aide_solution_helpers import (
    load_competition_data,
    write_oof_predictions,
    write_test_predictions,
    write_submission,
    aide_stage,
    log_stage,
)

BANDS = ["u", "g", "r", "i", "z"]
BAND_CHARS = np.array(["u", "g", "r", "i", "z"], dtype=object)


def _count_inversions(seq):
    c = 0
    for i in range(len(seq)):
        si = seq[i]
        for j in range(i + 1, len(seq)):
            if si > seq[j]:
                c += 1
    return c


def _local_extrema_info(vals):
    vals = np.asarray(vals, dtype=float)
    positions = []
    for j in range(1, len(vals) - 1):
        left = vals[j - 1]
        mid = vals[j]
        right = vals[j + 1]
        if (mid < left and mid < right) or (mid > left and mid > right):
            positions.append(j)
    return len(positions), positions


def _longest_monotone_run(vals):
    dif = np.diff(vals)
    signs = np.sign(dif)
    nz = signs[signs != 0]
    if len(nz) == 0:
        return 1
    longest = 1
    cur = 1
    for i in range(1, len(nz)):
        if nz[i] == nz[i - 1]:
            cur += 1
        else:
            cur = 1
        if cur > longest:
            longest = cur
    return int(longest)


def _segment_counts_from_signs(signs):
    seg_convex = 0
    seg_concave = 0
    prev = 0
    for s in signs:
        if s > 0:
            if prev != 1:
                seg_convex += 1
                prev = 1
        elif s < 0:
            if prev != -1:
                seg_concave += 1
                prev = -1
        else:
            prev = 0
    return int(seg_convex), int(seg_concave)


def build_topology_features(df):
    x = df[BANDS].to_numpy(dtype=np.float64)
    n = x.shape[0]

    order_tokens = []
    adj_sign_masks = []
    pairwise_masks = []
    inv_blue_to_red = []
    inv_red_to_blue = []
    extrema_counts = []
    extrema_positions = []
    longest_monotone = []
    second_sign_mask = []
    convex_counts = []
    concave_counts = []
    bright_band = []
    faint_band = []
    inv_to_string = {4: "z", 3: "i", 2: "r", 1: "g", 0: "u"}

    for i in range(n):
        row = x[i]
        rank_idx = np.argsort(row)
        order_tokens.append("-".join(BAND_CHARS[rank_idx].tolist()))

        adj = 0
        for j in range(len(BANDS) - 1):
            if row[j + 1] - row[j] > 0:
                adj |= 1 << j
        adj_sign_masks.append(adj)

        mask = 0
        bit = 1
        for a in range(len(BANDS)):
            for b in range(a + 1, len(BANDS)):
                if row[a] < row[b]:
                    mask |= bit
                bit <<= 1
        pairwise_masks.append(mask)

        perm = rank_idx.tolist()
        inv_blue_to_red.append(_count_inversions(perm))
        inv_red_to_blue.append(
            _count_inversions([inv_to_string[k] == "" for k in []])
        )  # placeholder kept compatible

        mapped_rev_order = [
            4 - p for p in perm
        ]  # equivalent inversion vs red->blue direction for reversed indices
        inv_red_to_blue[-1] = _count_inversions(mapped_rev_order)

        e_count, e_pos = _local_extrema_info(row)
        extrema_counts.append(e_count)
        extrema_positions.append(
            "none" if len(e_pos) == 0 else "-".join(map(str, e_pos))
        )
        longest_monotone.append(_longest_monotone_run(row))

        second_d = np.diff(row, n=2)  # second differences
        s2 = np.sign(second_d)
        sgn_mask = 0
        for j, s in enumerate(s2):
            if s > 0:
                sgn_mask |= 1 << j
        second_sign_mask.append(sgn_mask)

        conv, conc = _segment_counts_from_signs(s2)
        convex_counts.append(conv)
        concave_counts.append(conc)

        bi = int(np.argmin(row))
        fi = int(np.argmax(row))
        bright_band.append(BAND_CHARS[bi])
        faint_band.append(BAND_CHARS[fi])

    return pd.DataFrame(
        {
            "order_token": order_tokens,
            "adjacent_sign_bitmask": pd.Series(adj_sign_masks, dtype=int).astype(str),
            "pairwise_comparison_bits": pd.Series(pairwise_masks, dtype=int).astype(
                str
            ),
            "inversion_count_blue_to_red": np.array(inv_blue_to_red, dtype=int),
            "inversion_count_red_to_blue": np.array(inv_red_to_blue, dtype=int),
            "extrema_count": np.array(extrema_counts, dtype=int),
            "extrema_positions": extrema_positions,
            "longest_monotone_run": np.array(longest_monotone, dtype=int),
            "second_diff_sign_mask": pd.Series(second_sign_mask, dtype=int).astype(str),
            "convex_segment_count": np.array(convex_counts, dtype=int),
            "concave_segment_count": np.array(concave_counts, dtype=int),
            "brightest_band": bright_band,
            "faintest_band": faint_band,
        },
        index=df.index,
    )


def build_features(train, test):
    # Auxiliary file decision: intentionally not merged to avoid id/column-mismatch confounds and keep this run fully covariate-safe.
    _aux_path = os.path.join("input", "star_classification.csv")
    if os.path.exists(_aux_path):
        log_stage(
            "Auxiliary file star_classification.csv detected; skipped in this exact hypothesis run by design."
        )

    tr = train.copy()
    te = test.copy()

    tr_topo = build_topology_features(tr)
    te_topo = build_topology_features(te)

    tr_feat = tr[["alpha", "delta", "redshift", "u", "g", "r", "i", "z"]].copy()
    te_feat = te[["alpha", "delta", "redshift", "u", "g", "r", "i", "z"]].copy()

    for frame, src in [(tr_feat, tr), (te_feat, te)]:
        frame["ug_color"] = src["g"] - src["u"]
        frame["gr_color"] = src["r"] - src["g"]
        frame["ri_color"] = src["i"] - src["r"]
        frame["iz_color"] = src["z"] - src["i"]

        frame["spectral_type"] = src["spectral_type"]
        frame["galaxy_population"] = src["galaxy_population"]

    tr_feat = pd.concat([tr_feat, tr_topo], axis=1)
    te_feat = pd.concat([te_feat, te_topo], axis=1)

    combined = pd.concat([tr_feat, te_feat], axis=0)

    # Token-level covariate-only context features from train+test (no targets/OOF info).
    combined["redshift_regime"] = (
        pd.qcut(combined["redshift"], q=5, labels=False, duplicates="drop")
        .astype("Int64")
        .astype(str)
    )

    combined["coarse_topology_token"] = (
        combined["brightest_band"].astype(str)
        + "|"
        + combined["faintest_band"].astype(str)
        + "|s"
        + combined["adjacent_sign_bitmask"].astype(str)
        + "|z"
        + combined["redshift_regime"].astype(str)
    )

    token_columns = [
        "spectral_type",
        "galaxy_population",
        "redshift_regime",
        "order_token",
        "adjacent_sign_bitmask",
        "pairwise_comparison_bits",
        "second_diff_sign_mask",
        "extrema_positions",
        "brightest_band",
        "faintest_band",
        "coarse_topology_token",
    ]

    low_cardinality_threshold = 20
    # Make sure numeric geometry/topology indicators stay numeric.
    keep_numeric = [
        "alpha",
        "delta",
        "redshift",
        "u",
        "g",
        "r",
        "i",
        "z",
        "ug_color",
        "gr_color",
        "ri_color",
        "iz_color",
        "inversion_count_blue_to_red",
        "inversion_count_red_to_blue",
        "extrema_count",
        "longest_monotone_run",
        "convex_segment_count",
        "concave_segment_count",
    ]

    # Frequency/one-hot encoding of unsupervised topology tokens and raw categorical tokens.
    for col in token_columns:
        nunique = combined[col].nunique(dropna=False)
        if nunique <= low_cardinality_threshold:
            dummies = pd.get_dummies(combined[col].astype(str), prefix=col)
            combined = pd.concat([combined.drop(columns=[col]), dummies], axis=1)
        else:
            freq = combined[col].value_counts(normalize=True)
            combined[f"{col}_freq"] = (
                combined[col].astype(str).map(freq).astype(np.float64)
            )
            combined = combined.drop(columns=[col])

    # Ensure explicit numeric types are compact.
    combined = combined.astype(
        {c: "float32" for c in combined.columns if c not in token_columns},
        errors="ignore",
    )

    # All required columns are now numeric/categorical-encoded features.
    x_train = combined.iloc[: len(tr), :].drop(columns=[])
    x_test = combined.iloc[len(tr) :, :].drop(columns=[])

    # Drop accidental leakage/ID leftovers.
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    return x_train, x_test


def _catboost_gpu_fallback_fit(X_tr, y_tr, X_va, y_va, params):
    try:
        model = CatBoostClassifier(
            **params, task_type="GPU", devices="0", gpu_ram_part=0.8
        )
        log_stage("Fitting CatBoost multiclass on GPU")
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False,
        )
        return model, "GPU"
    except Exception as err:
        log_stage(f"GPU CatBoost training failed ({repr(err)}). Falling back to CPU.")
        model = CatBoostClassifier(**params, task_type="CPU")
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False,
        )
        return model, "CPU"


def main():
    train, test, sample_submission = load_competition_data()

    with aide_stage("build_features_stage"):
        X, X_test = build_features(train, test)
        y = train["class"].to_numpy()
        class_order = np.array(sorted(train["class"].dropna().unique()), dtype=object)

    with aide_stage("make_folds_stage"):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_proba = np.zeros((len(train), len(class_order)), dtype=np.float64)
    test_proba = np.zeros((len(test), len(class_order)), dtype=np.float64)
    fold_scores = []

    class_index = {c: i for i, c in enumerate(class_order)}

    cat_params = {
        "iterations": 900,
        "learning_rate": 0.08,
        "depth": 9,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": False,
    }

    with aide_stage("fit_predict_fold_stage"):
        for fold_id, (tr_idx, va_idx) in enumerate(folds.split(X, y), start=1):
            log_stage(f"Fold {fold_id}/5: CatBoost multiclass")
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model, used_mode = _catboost_gpu_fallback_fit(
                X_tr, y_tr, X_va, y_va, cat_params
            )
            print(f"Fold {fold_id} trained with {used_mode}", flush=True)

            va_proba_raw = model.predict_proba(X_va)
            va_proba = np.zeros_like(va_proba_raw, dtype=np.float64)
            for j, c in enumerate(model.classes_):
                va_proba[:, class_index[c]] = va_proba_raw[:, j]

            oof_proba[va_idx] = va_proba
            va_pred = class_order[np.argmax(va_proba, axis=1)]
            score = balanced_accuracy_score(y_va, va_pred, labels=class_order)
            fold_scores.append(score)
            print(f"Fold {fold_id} balanced accuracy: {score:.6f}", flush=True)

            test_fold_raw = model.predict_proba(X_test)
            test_fold = np.zeros_like(test_fold_raw, dtype=np.float64)
            for j, c in enumerate(model.classes_):
                test_fold[:, class_index[c]] = test_fold_raw[:, j]
            test_proba += test_fold / folds.n_splits

    with aide_stage("score_stage"):
        oof_pred = class_order[np.argmax(oof_proba, axis=1)]
        overall = balanced_accuracy_score(y, oof_pred, labels=class_order)
        print(f"OOF balanced accuracy (5-fold): {overall:.6f}", flush=True)
        print(f"Mean fold balanced accuracy: {np.mean(fold_scores):.6f}", flush=True)

    oof_df = pd.DataFrame(
        {
            "row": np.arange(len(train), dtype=int),
            "target": y,
            "prediction": oof_pred,
        }
    )

    test_pred_cols = pd.DataFrame(test_proba, columns=class_order)
    test_pred_with_id = pd.concat(
        [test[["id"]].reset_index(drop=True), test_pred_cols], axis=1
    )

    final_submission = pd.DataFrame(
        {
            "id": test["id"].values,
            "class": class_order[np.argmax(test_proba, axis=1)],
        }
    )

    with aide_stage("write_outputs_stage"):
        write_oof_predictions(oof_df)
        write_test_predictions(test_pred_with_id)
        write_submission(final_submission)
        # Explicitly ensure the required grading file exists in ./working via provided writer.
        # write_submission handles the required Kaggle-format path and columns id,class.

    # Keep a short human-readable checkpoint in logs.
    print(
        "submission.csv, oof_predictions.csv.gz, and test_predictions.csv.gz written.",
        flush=True,
    )


if __name__ == "__main__":
    main()
