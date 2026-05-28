import os
import random
import warnings
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

try:
    from scipy.special import ndtri
except Exception:
    ndtri = None

try:
    from colorama import Fore, Style
except ImportError:

    class _Empty:
        BLACK = ""
        GREEN = ""
        BRIGHT = ""
        RESET_ALL = ""

    Fore = Style = _Empty()

warnings.filterwarnings("ignore")

ID = "id"
TARGET = "PitNextLap"


class CFG:
    FOLDS = 5
    SEED = 42
    SEEDS = [42, 829]
    ALT_CV_SEED = 2027
    ALT_CV_MODEL_SEED = 42
    ALT_CV_BLEND_WEIGHT = 0.20
    CAT_SEED = 1379
    CAT_BLEND_WEIGHT = 0.15
    TE = True


PARAMS = {
    "random_state": 42,
    "verbosity": 2,
    "val_metric_name": "1-auc_ovr",
    "n_ens": 40,
    "n_epochs": 5,
    "batch_size": 256,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    "lr": 0.019,
    "wd": 0.01,
    "sq_mom": 0.99,
    "lr_sched": "lin_cos_log_15",
    "first_layer_lr_factor": 0.25,
    "embedding_size": 6,
    "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128],
    "act": "silu",
    "p_drop": 0.05,
    "p_drop_sched": "invsqrtp1e-3",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_act_name": "gelu",
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "ls_eps": 0.01,
    "ls_eps_sched": "sqrt_cos",
    "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": [
        "one_hot",
        "median_center",
        "robust_scale",
        "smooth_clip",
        "embedding",
        "l2_normalize",
    ],
}


CAT_PARAMS = {
    "iterations": 1800,
    "learning_rate": 0.035,
    "depth": 6,
    "l2_leaf_reg": 8.0,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": CFG.CAT_SEED,
    "task_type": "GPU",
    "devices": "0",
    "gpu_ram_part": 0.8,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "od_type": "Iter",
    "od_wait": 120,
    "allow_writing_files": False,
    "verbose": 200,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv(input_dir: Path, name: str) -> pd.DataFrame:
    for path in [input_dir / name, input_dir / f"{name}.gz"]:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find {name} or {name}.gz in {input_dir.resolve()}"
    )


def load_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = read_csv(input_dir, "train.csv")
    test = read_csv(input_dir, "test.csv")
    original_path = input_dir / "f1_strategy_dataset_v4.csv"
    if not original_path.exists():
        raise FileNotFoundError(
            f"Missing original F1 strategy dataset: {original_path.resolve()}"
        )
    orig = pd.read_csv(original_path)
    sample = read_csv(input_dir, "sample_submission.csv")
    if list(sample.columns) != [ID, TARGET]:
        raise ValueError(
            f"sample_submission columns must be {[ID, TARGET]}, got {list(sample.columns)}"
        )
    if len(sample) != len(test):
        raise ValueError(
            f"sample_submission rows ({len(sample)}) do not match test rows ({len(test)})"
        )
    print("Train shape:", train.shape)
    print("Test shape :", test.shape)
    print("Orig shape :", orig.shape)
    return train, test, orig


def prepare_features(train, test, orig):
    orig = orig.drop(["Normalized_TyreLife"], axis=1)
    y_orig = orig[TARGET]
    orig = orig.drop([TARGET], axis=1)

    X = train.drop([ID, TARGET], axis=1)
    train_id = train[ID]
    y = train[TARGET]
    X_test = test.drop([ID], axis=1)
    test_id = test[ID]

    print("X      init shape:", X.shape)
    print("X_test init shape:", X_test.shape)
    print("orig   init shape:", orig.shape, "\n")

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    print("init len(cat_cols):", len(cat_cols))
    print("init len(num_cols):", len(num_cols), "\n")

    category_map = {}
    important_combos = [("Race", "Compound"), ("Race", "Year")]

    def feature_engineering(df: pd.DataFrame, fit: bool = False):
        df = df.copy()

        df["_LapNumber_/_RaceProgress"] = (
            df["LapNumber"] / (df["RaceProgress"] + 1e-6)
        ).astype("float32")
        df["_TyreLife_/_LapNumber"] = (
            df["TyreLife"] / df["LapNumber"].clip(lower=1)
        ).astype("float32")
        df["_LapTime (s)_*_Cumulative_Degradation"] = (
            df["LapTime (s)"] * df["Cumulative_Degradation"]
        ).astype("float32")
        df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (
            df["LapTime (s)"] * df["Cumulative_Degradation"].abs()
        ).astype("float32")
        df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (
            df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)
        ).astype("float32")

        for col in num_cols + ["_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"]:
            cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
            if fit:
                codes, uniques = np.floor(df[col]).factorize()
                category_map[col] = uniques
            else:
                uniques = category_map[col]
                code_map = {cat: i for i, cat in enumerate(uniques)}
                codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
            df[cat_name] = codes.astype(str)

        for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
            count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
            if fit:
                count_map = df[col].value_counts()
                category_map[count_name] = count_map
            else:
                count_map = category_map[count_name]
            df[count_name] = df[col].map(count_map).fillna(0).astype("int32")

        bin_config = {"RaceProgress": [200], "LapTime (s)": [7]}
        for col, bins_list in bin_config.items():
            for n_bins in bins_list:
                bin_name = f"{col}_{n_bins}_quantile_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode="ordinal",
                        strategy="quantile",
                        subsample=None,
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype("int32")
                df[bin_name] = binned.astype(str)

        combo_names = []
        for cols in important_combos:
            combo_name = "_".join(cols) + "_"
            combo_names.append(combo_name)
            combo_series = df[cols[0]].astype(str)
            for col in cols[1:]:
                combo_series = combo_series + "_" + df[col].astype(str)
            if fit:
                codes, uniques = pd.factorize(combo_series, sort=False)
                category_map[combo_name] = uniques
            else:
                uniques = category_map[combo_name]
                code_map = {cat: i for i, cat in enumerate(uniques)}
                codes = combo_series.map(code_map).fillna(-1).astype("int32")
            df[combo_name] = codes.astype(str)

        new_cat_cols = [col for col in df.columns if col.endswith("_")]
        new_num_cols = [col for col in df.columns if col.startswith("_")]
        return df, new_cat_cols, new_num_cols, combo_names

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
    X_test, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X_test, fit=False
    )
    orig, new_cat_cols, new_num_cols, combo_names = feature_engineering(orig, fit=False)

    cat_cols += new_cat_cols
    num_cols += new_num_cols
    print("len(new_cat_cols):", len(new_cat_cols))
    print("len(new_num_cols):", len(new_num_cols), "\n")
    print("prep len(cat_cols):", len(cat_cols))
    print("prep len(num_cols):", len(num_cols), "\n")
    print("X      prep shape:", X.shape)
    print("X_test prep shape:", X_test.shape)
    print("orig   prep shape:", orig.shape, "\n")

    return X, y, train_id, X_test, test_id, orig, y_orig, combo_names


def add_fold_target_encoding(X_tr, y_tr, X_val, X_tst, combo_names):
    te_cols = combo_names
    target_encoder = TargetEncoder(
        cv=CFG.FOLDS, smooth="auto", shuffle=True, random_state=CFG.SEED
    )
    tr_enc = target_encoder.fit_transform(X_tr[te_cols], y_tr)
    val_enc = target_encoder.transform(X_val[te_cols])
    tst_enc = target_encoder.transform(X_tst[te_cols])
    te_names = [f"_{col}TE" for col in te_cols]
    X_tr[te_names] = tr_enc
    X_val[te_names] = val_enc
    X_tst[te_names] = tst_enc
    return X_tr, X_val, X_tst


def train_one_seed(
    X,
    y,
    X_test,
    orig,
    y_orig,
    combo_names,
    model_seed,
    cv_seed=CFG.SEED,
    stream_name=None,
):
    seed_everything(model_seed)
    stream_name = stream_name or f"Seed {model_seed}"
    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=cv_seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, ((tr_idx, val_idx), (or_tr_idx, _)) in enumerate(
        zip(skf.split(X, y), skf.split(orig, y_orig)), 1
    ):
        X_tr = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(
            drop=True
        )
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]
        X_tst = X_test.copy()

        if CFG.TE:
            X_tr, X_val, X_tst = add_fold_target_encoding(
                X_tr, y_tr, X_val, X_tst, combo_names
            )

        if fold == 1:
            print("len(FEATURES):", len(X_tr.columns.tolist()), "\n")
        print("#" * 16)
        print(f"### {stream_name} | CV seed {cv_seed} | Fold {fold}/{CFG.FOLDS} ...")
        print("#" * 16)

        params = PARAMS.copy()
        params["random_state"] = model_seed
        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds = model.predict_proba(X_val)[:, 1]
        fold_test_preds = model.predict_proba(X_tst)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.FOLDS

        fold_score = roc_auc_score(y_val, val_preds)
        print(
            f"{Fore.GREEN}{Style.BRIGHT}{stream_name} | CV seed {cv_seed} | Fold {fold} | AUC Score: {fold_score:.5f}{Style.RESET_ALL}\n"
        )
        torch.cuda.empty_cache()

    seed_score = roc_auc_score(y, oof_preds)
    print(
        f"{Fore.GREEN}{Style.BRIGHT}{stream_name} | CV seed {cv_seed} | Overall OOF AUC: {seed_score:.5f}{Style.RESET_ALL}\n"
    )
    return oof_preds, test_preds


def make_catboost_pool(X_df, y=None):
    X_cb = X_df.copy()
    cat_features = X_cb.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_features:
        X_cb[col] = X_cb[col].astype("string").fillna("__MISSING__").astype(str)
    return Pool(X_cb, label=y, cat_features=cat_features)


def fit_catboost_with_fallback(params, train_pool, val_pool):
    try:
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        return model
    except Exception as exc:
        if params.get("task_type") != "GPU":
            raise
        print(f"CatBoost GPU training failed, retrying on CPU: {exc}")
        cpu_params = params.copy()
        cpu_params["task_type"] = "CPU"
        cpu_params.pop("devices", None)
        cpu_params.pop("gpu_ram_part", None)
        cpu_params["thread_count"] = os.cpu_count() or -1
        model = CatBoostClassifier(**cpu_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        return model


def train_catboost_expert(X, y, X_test, orig, y_orig, combo_names):
    seed_everything(CFG.CAT_SEED)
    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, ((tr_idx, val_idx), (or_tr_idx, _)) in enumerate(
        zip(skf.split(X, y), skf.split(orig, y_orig)), 1
    ):
        X_tr = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(
            drop=True
        )
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]
        X_tst = X_test.copy()

        if CFG.TE:
            X_tr, X_val, X_tst = add_fold_target_encoding(
                X_tr, y_tr, X_val, X_tst, combo_names
            )

        print("#" * 16)
        print(f"### CatBoost seed {CFG.CAT_SEED} | Fold {fold}/{CFG.FOLDS} ...")
        print("#" * 16)

        train_pool = make_catboost_pool(X_tr, y_tr)
        val_pool = make_catboost_pool(X_val, y_val)
        test_pool = make_catboost_pool(X_tst)
        model = fit_catboost_with_fallback(CAT_PARAMS.copy(), train_pool, val_pool)

        val_preds = model.predict_proba(val_pool)[:, 1]
        fold_test_preds = model.predict_proba(test_pool)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.FOLDS

        fold_score = roc_auc_score(y_val, val_preds)
        print(
            f"{Fore.GREEN}{Style.BRIGHT}CatBoost Fold {fold} | AUC Score: {fold_score:.5f}{Style.RESET_ALL}\n"
        )

    cat_score = roc_auc_score(y, oof_preds)
    print(
        f"{Fore.GREEN}{Style.BRIGHT}CatBoost Overall OOF AUC: {cat_score:.5f}{Style.RESET_ALL}\n"
    )
    return oof_preds, test_preds


def clipped_logit(preds):
    clipped = np.clip(preds, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def rank_percentile(preds):
    values = np.asarray(preds, dtype=np.float64)
    if len(values) == 0:
        return values.copy()
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    percentiles = (ranks - 0.5) / float(len(values))
    return np.clip(percentiles, 1e-6, 1.0 - 1e-6)


def rankit_scores(preds):
    percentiles = rank_percentile(preds)
    if ndtri is None:
        return clipped_logit(percentiles)
    return ndtri(percentiles)


def transformed_blend_streams(mode, realmlp_core_preds, alt_preds, cat_preds):
    if mode == "raw":
        return realmlp_core_preds, alt_preds, cat_preds
    if mode == "logit":
        return (
            clipped_logit(realmlp_core_preds),
            clipped_logit(alt_preds),
            clipped_logit(cat_preds),
        )
    if mode == "rankit":
        return (
            rankit_scores(realmlp_core_preds),
            rankit_scores(alt_preds),
            rankit_scores(cat_preds),
        )
    raise ValueError(f"Unknown blend mode: {mode}")


def blend_from_transformed(mode, weights, transformed_streams):
    core_w, alt_w, cat_w = weights
    core_values, alt_values, cat_values = transformed_streams
    blended = core_w * core_values + alt_w * alt_values + cat_w * cat_values
    if mode in ("logit", "rankit"):
        return sigmoid(blended)
    return blended


def blend_three_streams(mode, weights, realmlp_core_preds, alt_preds, cat_preds):
    return blend_from_transformed(
        mode,
        weights,
        transformed_blend_streams(mode, realmlp_core_preds, alt_preds, cat_preds),
    )


def build_three_stream_blend_candidates():
    parent_alt_w = (1.0 - CFG.CAT_BLEND_WEIGHT) * CFG.ALT_CV_BLEND_WEIGHT
    parent_core_w = 1.0 - CFG.CAT_BLEND_WEIGHT - parent_alt_w
    parent = (
        round(parent_core_w, 6),
        round(parent_alt_w, 6),
        round(CFG.CAT_BLEND_WEIGHT, 6),
    )

    candidates = []
    for alt_w in np.round(np.arange(0.10, 0.261, 0.01), 3):
        for cat_w in np.round(np.arange(0.08, 0.221, 0.01), 3):
            core_w = round(1.0 - float(alt_w) - float(cat_w), 6)
            if core_w >= 0.55:
                candidates.append(
                    (core_w, round(float(alt_w), 6), round(float(cat_w), 6))
                )

    if parent not in candidates:
        candidates.append(parent)
    return candidates


def score_blend_candidates(y, realmlp_core_oof, alt_oof, cat_oof, candidates, parent):
    precomputed = {
        mode: transformed_blend_streams(mode, realmlp_core_oof, alt_oof, cat_oof)
        for mode in sorted({candidate[0] for candidate in candidates})
    }

    def score_candidate(candidate):
        mode, weights = candidate
        core_w, alt_w, cat_w = weights
        preds = blend_from_transformed(mode, weights, precomputed[mode])
        score = float(roc_auc_score(y, preds))
        distance = (
            abs(core_w - parent[0]) + abs(alt_w - parent[1]) + abs(cat_w - parent[2])
        )
        return mode, core_w, alt_w, cat_w, score, distance

    workers = min(16, os.cpu_count() or 1)
    try:
        from joblib import Parallel, delayed

        print(f"Evaluating {len(candidates)} blend candidates with {workers} workers")
        return Parallel(n_jobs=workers, prefer="threads")(
            delayed(score_candidate)(candidate) for candidate in candidates
        )
    except Exception as exc:
        print(f"Blend candidate parallel evaluation unavailable, using serial: {exc}")
        print(f"Evaluating {len(candidates)} blend candidates with 1 workers")
        return [score_candidate(candidate) for candidate in candidates]


def select_three_stream_blend(y, realmlp_core_oof, alt_oof, cat_oof):
    weight_candidates = build_three_stream_blend_candidates()
    candidates = [
        (mode, weights)
        for mode in ("raw", "logit", "rankit")
        for weights in weight_candidates
    ]

    parent_alt_w = (1.0 - CFG.CAT_BLEND_WEIGHT) * CFG.ALT_CV_BLEND_WEIGHT
    parent_core_w = 1.0 - CFG.CAT_BLEND_WEIGHT - parent_alt_w
    parent = (parent_core_w, parent_alt_w, CFG.CAT_BLEND_WEIGHT)
    mode_priority = {"raw": 2, "logit": 1, "rankit": 0}

    results = score_blend_candidates(
        y, realmlp_core_oof, alt_oof, cat_oof, candidates, parent
    )
    best = max(
        results,
        key=lambda item: (
            round(item[4], 12),
            -item[5],
            mode_priority.get(str(item[0]), 0),
        ),
    )
    best_mode = str(best[0])
    best_weights = (float(best[1]), float(best[2]), float(best[3]))
    best_score = float(best[4])
    print(
        "Selected three-stream blend "
        f"mode={best_mode}, weights core={best_weights[0]:.3f}, "
        f"alt_cv={best_weights[1]:.3f}, catboost={best_weights[2]:.3f} "
        f"| OOF AUC: {best_score:.5f}"
    )
    return best_mode, best_weights, best_score, len(candidates)


def nested_select_three_stream_blend(
    y, realmlp_core_oof, alt_oof, cat_oof, realmlp_core_test, alt_test, cat_test
):
    y_arr = np.asarray(y)

    meta_skf = StratifiedKFold(
        n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED + 913
    )
    meta_oof = np.zeros(len(y_arr))
    fold_bagged_test = np.zeros(len(realmlp_core_test))
    fold_modes = []
    fold_weights = []

    for fold, (meta_tr_idx, meta_val_idx) in enumerate(
        meta_skf.split(realmlp_core_oof.reshape(-1, 1), y_arr), 1
    ):
        fold_mode, fold_weights_tuple, fold_score, _ = select_three_stream_blend(
            y_arr[meta_tr_idx],
            realmlp_core_oof[meta_tr_idx],
            alt_oof[meta_tr_idx],
            cat_oof[meta_tr_idx],
        )
        meta_oof[meta_val_idx] = blend_three_streams(
            fold_mode,
            fold_weights_tuple,
            realmlp_core_oof[meta_val_idx],
            alt_oof[meta_val_idx],
            cat_oof[meta_val_idx],
        )
        fold_bagged_test += (
            blend_three_streams(
                fold_mode,
                fold_weights_tuple,
                realmlp_core_test,
                alt_test,
                cat_test,
            )
            / CFG.FOLDS
        )
        fold_modes.append(fold_mode)
        fold_weights.append(fold_weights_tuple)
        fold_auc = roc_auc_score(y_arr[meta_val_idx], meta_oof[meta_val_idx])
        print(
            f"Nested blend meta fold {fold} selected mode={fold_mode}, "
            f"weights={fold_weights_tuple}, train AUC={fold_score:.5f}, "
            f"valid AUC={fold_auc:.5f}"
        )

    nested_score = roc_auc_score(y_arr, meta_oof)

    final_mode, final_weights, selected_full_score, candidate_count = (
        select_three_stream_blend(
            y_arr,
            realmlp_core_oof,
            alt_oof,
            cat_oof,
        )
    )
    full_oof_selected_test = blend_three_streams(
        final_mode, final_weights, realmlp_core_test, alt_test, cat_test
    )
    test_preds = fold_bagged_test

    print(
        f"Nested meta-blend OOF AUC: {nested_score:.5f}; "
        f"test blend uses nested fold-bagged configurations; "
        f"diagnostic full-OOF mode={final_mode}, weights={final_weights}, "
        f"full-OOF selected AUC={selected_full_score:.5f}"
    )
    return (
        meta_oof,
        test_preds,
        {
            "nested_blend_oof_auc": float(nested_score),
            "test_blend_strategy": "nested_fold_bagged",
            "diagnostic_full_oof_test_prediction_mean": float(
                np.mean(full_oof_selected_test)
            ),
            "final_selected_blend_mode": final_mode,
            "final_selected_core_weight": float(final_weights[0]),
            "final_selected_alt_cv_weight": float(final_weights[1]),
            "final_selected_catboost_weight": float(final_weights[2]),
            "full_oof_selected_blend_auc": float(selected_full_score),
            "blend_candidate_count": int(candidate_count),
            "nested_fold_modes": fold_modes,
            "nested_fold_weights": [list(map(float, w)) for w in fold_weights],
        },
    )


def train_and_predict(X, y, X_test, orig, y_orig, combo_names):
    all_oof = []
    all_test = []

    for model_seed in CFG.SEEDS:
        oof_seed, test_seed = train_one_seed(
            X, y, X_test, orig, y_orig, combo_names, model_seed, cv_seed=CFG.SEED
        )
        all_oof.append(oof_seed)
        all_test.append(test_seed)

    realmlp_oof = np.mean(np.vstack(all_oof), axis=0)
    realmlp_test = np.mean(np.vstack(all_test), axis=0)
    realmlp_score = roc_auc_score(y, realmlp_oof)

    alt_oof, alt_test = train_one_seed(
        X,
        y,
        X_test,
        orig,
        y_orig,
        combo_names,
        CFG.ALT_CV_MODEL_SEED,
        cv_seed=CFG.ALT_CV_SEED,
        stream_name=f"Alt-CV RealMLP seed {CFG.ALT_CV_MODEL_SEED}",
    )
    alt_score = roc_auc_score(y, alt_oof)
    alt_corr = float(np.corrcoef(realmlp_oof, alt_oof)[0, 1])

    alt_w = CFG.ALT_CV_BLEND_WEIGHT
    realmlp_diverse_oof = (1.0 - alt_w) * realmlp_oof + alt_w * alt_oof
    realmlp_diverse_test = (1.0 - alt_w) * realmlp_test + alt_w * alt_test
    realmlp_diverse_score = roc_auc_score(y, realmlp_diverse_oof)

    cat_oof, cat_test = train_catboost_expert(X, y, X_test, orig, y_orig, combo_names)
    cat_score = roc_auc_score(y, cat_oof)
    oof_corr = float(np.corrcoef(realmlp_diverse_oof, cat_oof)[0, 1])

    fixed_cat_w = CFG.CAT_BLEND_WEIGHT
    fixed_oof_preds = (1.0 - fixed_cat_w) * realmlp_diverse_oof + fixed_cat_w * cat_oof
    fixed_blend_score = roc_auc_score(y, fixed_oof_preds)

    old_mode, old_weights, old_selected_score, old_candidate_count = (
        select_three_stream_blend(
            y,
            realmlp_oof,
            alt_oof,
            cat_oof,
        )
    )
    old_oof_preds = blend_three_streams(
        old_mode, old_weights, realmlp_oof, alt_oof, cat_oof
    )
    old_blend_score = roc_auc_score(y, old_oof_preds)

    oof_preds, test_preds, nested_stats = nested_select_three_stream_blend(
        y,
        realmlp_oof,
        alt_oof,
        cat_oof,
        realmlp_test,
        alt_test,
        cat_test,
    )
    blend_score = roc_auc_score(y, oof_preds)

    print("\n" + "=" * 24)
    print(f"RealMLP core OOF AUC: {realmlp_score:.5f}")
    print(f"Alt-CV RealMLP OOF AUC: {alt_score:.5f}")
    print(f"RealMLP core/Alt-CV OOF correlation: {alt_corr:.6f}")
    print(
        f"Fixed {alt_w:.2f} Alt-CV RealMLP blend OOF AUC: {realmlp_diverse_score:.5f}"
    )
    print(f"CatBoost expert OOF AUC: {cat_score:.5f}")
    print(f"Diverse RealMLP/CatBoost OOF correlation: {oof_corr:.6f}")
    print(f"Parent fixed CatBoost blend OOF AUC: {fixed_blend_score:.5f}")
    print(
        "Full-OOF selected three-stream blend OOF AUC: "
        f"{old_blend_score:.5f} with mode={old_mode}, weights "
        f"core={old_weights[0]:.3f}, alt_cv={old_weights[1]:.3f}, catboost={old_weights[2]:.3f}"
    )
    print(f"Nested selected three-stream blend OOF AUC: {blend_score:.5f}")
    print("=" * 24)

    stats = {
        "realmlp_n_ens": int(PARAMS["n_ens"]),
        "realmlp_oof_auc": float(realmlp_score),
        "alternate_cv_realmlp_oof_auc": float(alt_score),
        "realmlp_alt_cv_oof_correlation": alt_corr,
        "alternate_cv_realmlp_blend_weight": float(alt_w),
        "diverse_realmlp_oof_auc": float(realmlp_diverse_score),
        "catboost_oof_auc": float(cat_score),
        "realmlp_catboost_oof_correlation": oof_corr,
        "parent_catboost_blend_weight": float(fixed_cat_w),
        "parent_fixed_blend_oof_auc": float(fixed_blend_score),
        "old_selected_blend_mode": old_mode,
        "old_selected_core_weight": float(old_weights[0]),
        "old_selected_alt_cv_weight": float(old_weights[1]),
        "old_selected_catboost_weight": float(old_weights[2]),
        "old_selected_blend_oof_auc": float(old_selected_score),
        "old_blend_oof_auc": float(old_blend_score),
        "old_blend_candidate_count": int(old_candidate_count),
        "blend_oof_auc": float(blend_score),
        **nested_stats,
    }
    return oof_preds, test_preds, stats


def main():
    input_dir = Path(os.environ.get("AIDE_INPUT_DIR", "input"))
    working_dir = Path(os.environ.get("AIDE_WORKING_DIR", "working"))
    working_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(CFG.SEED)
    print("PyTorch  version:", torch.__version__)
    print("PyTabKit version:", version("pytabkit"))
    print("CatBoost version:", version("catboost"))
    print("Model seeds:", CFG.SEEDS)
    print("RealMLP n_ens:", PARAMS["n_ens"])
    print("Alternate RealMLP model seed:", CFG.ALT_CV_MODEL_SEED)
    print("Alternate RealMLP CV seed:", CFG.ALT_CV_SEED)
    print("CatBoost seed:", CFG.CAT_SEED)

    train, test, orig = load_data(input_dir)
    X, y, train_id, X_test, test_id, orig, y_orig, combo_names = prepare_features(
        train, test, orig
    )
    del train, test

    oof_preds, test_preds, model_stats = train_and_predict(
        X, y, X_test, orig, y_orig, combo_names
    )

    oof_score = roc_auc_score(y, oof_preds)
    print("\n" + "=" * 24)
    print(
        f"Overall blended OOF AUC: {Fore.BLACK}{Style.BRIGHT}{oof_score:.5f}{Style.RESET_ALL}"
    )
    print("=" * 24)

    oof_df = pd.DataFrame({ID: train_id, TARGET: oof_preds})
    oof_path = working_dir / "oof_preds.csv"
    oof_df.to_csv(oof_path, index=False)

    required_oof_df = pd.DataFrame(
        {
            "row": np.arange(len(y), dtype=np.int64),
            "target": y.to_numpy(),
            "prediction": oof_preds,
        }
    )
    required_oof_path = working_dir / "oof_predictions.csv.gz"
    required_oof_df.to_csv(required_oof_path, index=False, compression="gzip")

    sub = pd.DataFrame({ID: test_id, TARGET: test_preds})
    sub_path = working_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)

    test_pred_path = working_dir / "test_predictions.csv.gz"
    sub.to_csv(test_pred_path, index=False, compression="gzip")

    print(f"Saved OOF predictions: {oof_path}")
    print(f"Saved compressed OOF predictions: {required_oof_path}")
    print(f"Saved submission: {sub_path}")
    print(f"Saved compressed test predictions: {test_pred_path}")
    print(
        "AIDE_RESULT_JSON: "
        + json.dumps(
            {
                "is_bug": False,
                "summary": "RealMLP/PyTabKit two-seed n_ens=40 5-fold OOF validation completed with alternate-CV RealMLP, CatBoost, nested OOF-selected raw/logit/rankit blending, and nested fold-bagged test blend configurations.",
                "metric": float(oof_score),
                "lower_is_better": False,
                "validity_warning": "The reported primary AUC uses nested meta-OOF blend selection; submission predictions average the nested fold-selected blend configurations instead of relying on a single full-OOF-selected test blend.",
                "run_stats": {
                    "metric_name": "roc_auc",
                    "cv_score": float(oof_score),
                    "submission_path": str(sub_path),
                    "oof_path": str(oof_path),
                    "compressed_oof_path": str(required_oof_path),
                    "test_predictions_path": str(test_pred_path),
                    "original_dataset_file": "f1_strategy_dataset_v4.csv",
                    "model_seeds": CFG.SEEDS,
                    "realmlp_n_ens": PARAMS["n_ens"],
                    "alternate_cv_model_seed": CFG.ALT_CV_MODEL_SEED,
                    "alternate_cv_seed": CFG.ALT_CV_SEED,
                    "catboost_seed": CFG.CAT_SEED,
                    **model_stats,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
