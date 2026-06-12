from __future__ import annotations

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
)

# Feature tuning for Hypothesis 000033
BASE_NUMERIC_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CAT_COLS = ["spectral_type", "galaxy_population"]
MAG_LIMITS = {"u": 22.0, "g": 22.2, "r": 22.2, "i": 21.3, "z": 20.5}
SAT_LIMITS = {"u": 16.0, "g": 14.5, "r": 14.5, "i": 14.5}
MARGIN_SCALE = 2.0
MID_WINDOW = 0.35
MID_CENTER = 19.5
N_SPLITS = 5
RANDOM_STATE = 42
EVAL_BASELINE = True


def build_feature_frame(
    df: pd.DataFrame, cat_levels: dict[str, list[str]], add_reliability: bool
) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    # Raw numeric baseline features
    feats[BASE_NUMERIC_COLS] = df[BASE_NUMERIC_COLS].astype(np.float32)

    # One-hot encoded categorical features (target-free)
    for c in CAT_COLS:
        col = df[c].astype(str).fillna("__missing__")
        levels = cat_levels[c]
        for level in levels:
            feats[f"{c}_{level}"] = (col == level).astype(np.float32)
        if "__missing__" in levels:
            feats[f"{c}__missing__"] = (col == "__missing__").astype(np.float32)

    if not add_reliability:
        return feats

    margin = {}
    reliability = {}

    # Per-band limits and flags
    for band, limit in MAG_LIMITS.items():
        m = limit - df[band].astype(np.float32)
        margin[band] = m
        feats[f"{band}_depth_margin"] = m
        feats[f"{band}_is_fainter_than_limit"] = (df[band] > limit).astype(np.uint8)
        feats[f"{band}_is_not_fainter_than_limit"] = (df[band] <= limit).astype(
            np.uint8
        )
        reliability[band] = np.clip(m / MARGIN_SCALE, 0.0, 1.0).astype(np.float32)

    over_u = feats["u_is_fainter_than_limit"].astype(np.uint8)
    over_g = feats["g_is_fainter_than_limit"].astype(np.uint8)
    over_r = feats["r_is_fainter_than_limit"].astype(np.uint8)
    over_i = feats["i_is_fainter_than_limit"].astype(np.uint8)
    over_z = feats["z_is_fainter_than_limit"].astype(np.uint8)

    # Count/bitmask-style summaries
    feats["overlimit_count_blue_bands"] = (over_u + over_g).astype(np.uint8)
    feats["overlimit_count_middle_bands"] = (over_r + over_i).astype(np.uint8)
    feats["overlimit_count_red_bands"] = over_z.astype(np.uint8)
    feats["overlimit_count_all_bands"] = (
        over_u + over_g + over_r + over_i + over_z
    ).astype(np.uint8)

    feats["overlimit_mask_blue"] = (over_u * 1 + over_g * 2).astype(np.uint8)
    feats["overlimit_mask_middle"] = (over_r * 4 + over_i * 8).astype(np.uint8)
    feats["overlimit_mask_red"] = (over_z * 16).astype(np.uint8)
    feats["overlimit_mask_all"] = (
        over_u * 1 + over_g * 2 + over_r * 4 + over_i * 8 + over_z * 16
    ).astype(np.uint8)

    # Saturation/quality regime flags (target-free)
    feats["sat_u"] = (df["u"] < SAT_LIMITS["u"]).astype(np.uint8)
    feats["sat_g"] = (df["g"] < SAT_LIMITS["g"]).astype(np.uint8)
    feats["sat_r"] = (df["r"] < SAT_LIMITS["r"]).astype(np.uint8)
    feats["sat_i"] = (df["i"] < SAT_LIMITS["i"]).astype(np.uint8)
    feats["sat_count"] = (
        feats["sat_u"] + feats["sat_g"] + feats["sat_r"] + feats["sat_i"]
    ).astype(np.uint8)

    feats["mid_u"] = (np.abs(df["u"] - MID_CENTER) <= MID_WINDOW).astype(np.uint8)
    feats["mid_g"] = (np.abs(df["g"] - MID_CENTER) <= MID_WINDOW).astype(np.uint8)
    feats["mid_r"] = (np.abs(df["r"] - MID_CENTER) <= MID_WINDOW).astype(np.uint8)
    feats["mid_i"] = (np.abs(df["i"] - MID_CENTER) <= MID_WINDOW).astype(np.uint8)
    feats["mid_band_count_19_5"] = (
        feats["mid_u"] + feats["mid_g"] + feats["mid_r"] + feats["mid_i"]
    ).astype(np.uint8)

    # Margin summaries across bands
    margin_matrix = np.column_stack(
        [margin[b].to_numpy() for b in ["u", "g", "r", "i", "z"]]
    ).astype(np.float32)
    feats["depth_margin_min"] = np.min(margin_matrix, axis=1)
    feats["depth_margin_max"] = np.max(margin_matrix, axis=1)
    feats["depth_margin_mean"] = np.mean(margin_matrix, axis=1)
    feats["depth_margin_std"] = np.std(margin_matrix, axis=1)

    # Reliability-weighted adjacent and broad colors
    adjacent_pairs = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
    broad_pairs = [("u", "r"), ("g", "i"), ("r", "z")]
    for b1, b2 in adjacent_pairs + broad_pairs:
        color = (df[b1] - df[b2]).astype(np.float32)
        w = np.minimum(reliability[b1], reliability[b2]).astype(np.float32)
        feats[f"{b1}_{b2}_color"] = color
        feats[f"{b1}_{b2}_rel_weighted_color"] = color * w
        feats[f"{b1}_{b2}_min_reliability"] = w

    feats = feats.astype(np.float32)
    return feats


def make_cat_model(seed: int, use_gpu: bool) -> CatBoostClassifier:
    kwargs = dict(
        iterations=350,
        depth=10,
        learning_rate=0.1,
        random_seed=seed,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        verbose=False,
    )
    if use_gpu:
        kwargs.update(task_type="GPU", devices="0", gpu_ram_part=0.8)
    else:
        kwargs.update(task_type="CPU")
    return CatBoostClassifier(**kwargs)


def run_cv_eval(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y_train_enc: np.ndarray,
    y_train_raw: pd.Series,
    class_names: np.ndarray,
    tag: str = "baseline",
) -> tuple[float, np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_prob = np.zeros((len(train_x), len(class_names)), dtype=np.float32)
    test_prob = np.zeros((len(test_x), len(class_names)), dtype=np.float32)
    fold_scores: list[float] = []

    with aide_stage("fit_predict_fold_stage"):
        for fold_id, (tr_idx, val_idx) in enumerate(skf.split(train_x, y_train_enc), 1):
            x_tr = train_x.iloc[tr_idx]
            x_val = train_x.iloc[val_idx]
            y_tr = y_train_enc[tr_idx]
            y_val = y_train_raw.iloc[val_idx].to_numpy()

            msg = f"[{tag}] Fold {fold_id}/{N_SPLITS}: training CatBoost"
            log_stage(msg)
            print(msg, flush=True)

            seed = RANDOM_STATE + fold_id
            try:
                model = make_cat_model(seed=seed, use_gpu=True)
                model.fit(x_tr, y_tr)
            except Exception as err:
                reason = f"[{tag}] Fold {fold_id}: GPU unavailable/failing -> fallback to CPU. reason={type(err).__name__}: {err}"
                log_stage(reason)
                print(reason, flush=True)
                model = make_cat_model(seed=seed, use_gpu=False)
                model.fit(x_tr, y_tr)

            pred_val_prob = model.predict_proba(x_val)
            oof_prob[val_idx] = pred_val_prob
            pred_val = class_names[np.argmax(pred_val_prob, axis=1)]
            fold_scores.append(balanced_accuracy_score(y_val, pred_val))

            pred_test_prob = model.predict_proba(test_x)
            test_prob += pred_test_prob.astype(np.float32) / N_SPLITS

            msg = f"[{tag}] Fold {fold_id} balanced_accuracy={fold_scores[-1]:.6f}"
            log_stage(msg)
            print(msg, flush=True)

    oof_pred = class_names[np.argmax(oof_prob, axis=1)]
    with aide_stage("score_stage"):
        overall = balanced_accuracy_score(y_train_raw.to_numpy(), oof_pred)
    log_stage(f"[{tag}] OOF balanced_accuracy={overall:.6f}")
    print(f"[{tag}] OOF balanced_accuracy={overall:.6f}", flush=True)
    return overall, oof_prob, test_prob


def main() -> None:
    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        y_raw = train["class"].astype(str)
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw)
        class_names = le.classes_

        cat_levels = {
            c: sorted(train[c].astype(str).fillna("__missing__").unique().tolist())
            for c in CAT_COLS
        }

        x_train_base = build_feature_frame(
            train.drop(columns=["class"]), cat_levels, add_reliability=False
        )
        x_test_base = build_feature_frame(test, cat_levels, add_reliability=False)
        x_train_rel = build_feature_frame(
            train.drop(columns=["class"]), cat_levels, add_reliability=True
        )
        x_test_rel = build_feature_frame(test, cat_levels, add_reliability=True)

    base_score = rel_score = None
    if EVAL_BASELINE:
        with aide_stage("make_folds_stage"):
            base_score, base_oof_prob, base_test_prob = run_cv_eval(
                x_train_base, x_test_base, y_enc, y_raw, class_names, tag="baseline"
            )
        print(f"Baseline raw+onehot OOF balanced_accuracy={base_score:.6f}", flush=True)

    with aide_stage("make_folds_stage"):
        rel_score, rel_oof_prob, rel_test_prob = run_cv_eval(
            x_train_rel,
            x_test_rel,
            y_enc,
            y_raw,
            class_names,
            tag="reliability_augmented",
        )
    if base_score is not None:
        delta = rel_score - base_score
        print(
            f"Incremental effect (reliability features): {rel_score:.6f} - {base_score:.6f} = {delta:.6f}",
            flush=True,
        )

    # Save required artifacts from final hypothesis feature set
    rel_oof_pred = class_names[np.argmax(rel_oof_prob, axis=1)]
    rel_test_pred = class_names[np.argmax(rel_test_prob, axis=1)]

    oof_df = pd.DataFrame(
        {
            "row": train.index.to_numpy(),
            "target": y_raw.to_numpy(),
            "prediction": rel_oof_pred,
        }
    )

    test_probs_df = pd.DataFrame(
        rel_test_prob, columns=class_names, index=sample_sub.index
    )
    test_probs_df = pd.concat(
        [
            sample_sub[["id"]].reset_index(drop=True),
            test_probs_df.reset_index(drop=True),
        ],
        axis=1,
    )

    submission = sample_sub[["id"]].copy().reset_index(drop=True)
    submission["class"] = rel_test_pred

    with aide_stage("write_outputs_stage"):
        write_submission(submission)
        write_oof_predictions(oof_df)
        write_test_predictions(test_probs_df)

    print(f"Primary CV balanced_accuracy (selected): {rel_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
