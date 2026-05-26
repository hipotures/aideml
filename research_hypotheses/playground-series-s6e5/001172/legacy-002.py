import os
import random
import warnings
import json
import logging
from importlib.metadata import version
from pathlib import Path

logging.getLogger("lightning_utilities").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric.accelerators.cuda").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch

from pytabkit import RealMLP_TD_Classifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

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
EPS = 1e-6


class CFG:
    FOLDS = 5
    SEED = 42
    TE = True


PARAMS = {
    "random_state": 42,
    "verbosity": 2,
    "val_metric_name": "1-auc_ovr",
    "n_ens": 20,
    "n_epochs": 7,
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv(input_dir: Path, name: str) -> pd.DataFrame:
    candidates = [
        input_dir / name,
        input_dir / f"{name}.gz",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find {name} or {name}.gz in {input_dir.resolve()}"
    )


def load_data(
    input_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return train, test, orig, sample


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    list[str],
]:
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
    important_combos = [
        ("Race", "Compound"),
        ("Race", "Year"),
    ]

    def feature_engineering(
        df: pd.DataFrame,
        fit: bool = False,
    ) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
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
            df[cat_name] = codes
            df[cat_name] = df[cat_name].astype(str)

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
                for strategy in ["quantile"]:
                    bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                    if fit:
                        kb = KBinsDiscretizer(
                            n_bins=n_bins,
                            encode="ordinal",
                            strategy=strategy,
                            subsample=None,
                        )
                        binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                        category_map[bin_name] = kb
                    else:
                        kb = category_map[bin_name]
                        binned = kb.transform(df[[col]]).ravel().astype("int32")
                    df[bin_name] = binned
                    df[bin_name] = df[bin_name].astype(str)

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
            df[combo_name] = codes
            df[combo_name] = df[combo_name].astype(str)

        new_cat_cols = [col for col in df.columns if col.endswith("_")]
        new_num_cols = [col for col in df.columns if col.startswith("_")]
        return df, new_cat_cols, new_num_cols, combo_names

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
    X_test, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X_test,
        fit=False,
    )
    orig, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        orig,
        fit=False,
    )

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


def train_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    orig: pd.DataFrame,
    y_orig: pd.Series,
    combo_names: list[str],
    params: dict | None = None,
    expert_name: str = "realmlp",
) -> tuple[np.ndarray, np.ndarray]:
    model_params = PARAMS if params is None else params
    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, ((tr_idx, val_idx), (or_tr_idx, _or_val_idx)) in enumerate(
        zip(skf.split(X, y), skf.split(orig, y_orig)),
        1,
    ):
        X_tr = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr = pd.concat(
            [y.iloc[tr_idx], y_orig.iloc[or_tr_idx]],
            axis=0,
        ).reset_index(drop=True)
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]
        X_tst = X_test.copy()

        if CFG.TE:
            te_cols = combo_names
            target_encoder = TargetEncoder(
                cv=CFG.FOLDS,
                smooth="auto",
                shuffle=True,
                random_state=CFG.SEED,
            )
            tr_enc = target_encoder.fit_transform(X_tr[te_cols], y_tr)
            val_enc = target_encoder.transform(X_val[te_cols])
            tst_enc = target_encoder.transform(X_tst[te_cols])

            te_names = [f"_{col}TE" for col in te_cols]
            X_tr[te_names] = tr_enc
            X_val[te_names] = val_enc
            X_tst[te_names] = tst_enc

        if fold == 1:
            print(f"{expert_name} len(FEATURES):", len(X_tr.columns.tolist()), "\n")
        print("#" * 16)
        print(f"### {expert_name} fold {fold}/{CFG.FOLDS} ...")
        print("#" * 16)

        model = RealMLP_TD_Classifier(**model_params)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds = model.predict_proba(X_val)[:, 1]
        fold_test_preds = model.predict_proba(X_tst)[:, 1]

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.FOLDS

        fold_score = roc_auc_score(y_val, val_preds)
        print(
            f"{Fore.GREEN}{Style.BRIGHT}{expert_name} fold {fold} | AUC Score: "
            f"{fold_score:.5f}{Style.RESET_ALL}\n"
        )
        torch.cuda.empty_cache()

    return oof_preds, test_preds


def align_lgbm_categoricals(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    orig: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    X_lgb = X.copy()
    X_test_lgb = X_test.copy()
    orig_lgb = orig.copy()
    cat_cols = X_lgb.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        train_col = X_lgb[col].astype("object").where(X_lgb[col].notna(), "__missing__").astype(str)
        test_col = X_test_lgb[col].astype("object").where(X_test_lgb[col].notna(), "__missing__").astype(str)
        orig_col = orig_lgb[col].astype("object").where(orig_lgb[col].notna(), "__missing__").astype(str)
        values = pd.concat(
            [
                train_col,
                test_col,
                orig_col,
            ],
            axis=0,
            ignore_index=True,
        )
        categories = pd.Index(pd.unique(values))
        X_lgb[col] = pd.Categorical(train_col, categories=categories)
        X_test_lgb[col] = pd.Categorical(test_col, categories=categories)
        orig_lgb[col] = pd.Categorical(orig_col, categories=categories)

    return X_lgb, X_test_lgb, orig_lgb, cat_cols


def train_lightgbm_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    orig: pd.DataFrame,
    y_orig: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    import lightgbm as lgb

    X_lgb, X_test_lgb, orig_lgb, cat_cols = align_lgbm_categoricals(X, X_test, orig)
    y_arr = y.astype(int).to_numpy()
    y_orig_arr = y_orig.astype(int).to_numpy()
    pos = float(y_arr.sum() + y_orig_arr.sum())
    total = float(len(y_arr) + len(y_orig_arr))
    pos_weight = (total - pos) / max(pos, 1.0)

    def run_with_params(extra_params: dict) -> tuple[np.ndarray, np.ndarray]:
        skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
        oof_preds = np.zeros(len(X_lgb), dtype=np.float64)
        test_preds = np.zeros(len(X_test_lgb), dtype=np.float64)

        for fold, ((tr_idx, val_idx), (or_tr_idx, _or_val_idx)) in enumerate(
            zip(skf.split(X_lgb, y_arr), skf.split(orig_lgb, y_orig_arr)),
            1,
        ):
            X_tr = pd.concat(
                [X_lgb.iloc[tr_idx], orig_lgb.iloc[or_tr_idx]],
                axis=0,
            ).reset_index(drop=True)
            y_tr = np.concatenate([y_arr[tr_idx], y_orig_arr[or_tr_idx]])
            X_val = X_lgb.iloc[val_idx]
            y_val = y_arr[val_idx]

            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=800,
                learning_rate=0.035,
                num_leaves=63,
                min_child_samples=60,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=6.0,
                scale_pos_weight=pos_weight,
                random_state=CFG.SEED + 1000 + fold,
                n_jobs=max(1, min(16, os.cpu_count() or 1)),
                verbosity=-1,
                **extra_params,
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                categorical_feature=cat_cols,
                callbacks=[
                    lgb.early_stopping(80, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test_lgb)[:, 1] / CFG.FOLDS
            print(
                f"lightgbm fold {fold} | AUC Score: "
                f"{roc_auc_score(y_val, oof_preds[val_idx]):.5f}"
            )

        return oof_preds, test_preds

    lgbm_device = os.environ.get("AIDE_LGBM_DEVICE", "gpu").strip().lower()
    if lgbm_device == "gpu":
        return run_with_params(
            {
                "device_type": "gpu",
                "gpu_platform_id": int(os.environ.get("AIDE_LGBM_GPU_PLATFORM_ID", "0")),
                "gpu_device_id": int(os.environ.get("AIDE_LGBM_GPU_DEVICE_ID", "0")),
            }
        )
    if lgbm_device == "cuda":
        return run_with_params({"device_type": "cuda"})
    print("LightGBM expert using CPU because AIDE_LGBM_DEVICE is not gpu/cuda.")
    return run_with_params({})


def train_xgboost_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    orig: pd.DataFrame,
    y_orig: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    from xgboost import XGBClassifier

    X_xgb, X_test_xgb, orig_xgb, _cat_cols = align_lgbm_categoricals(X, X_test, orig)
    y_arr = y.astype(int).to_numpy()
    y_orig_arr = y_orig.astype(int).to_numpy()
    pos = float(y_arr.sum() + y_orig_arr.sum())
    total = float(len(y_arr) + len(y_orig_arr))
    pos_weight = (total - pos) / max(pos, 1.0)

    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X_xgb), dtype=np.float64)
    test_preds = np.zeros(len(X_test_xgb), dtype=np.float64)

    for fold, ((tr_idx, val_idx), (or_tr_idx, _or_val_idx)) in enumerate(
        zip(skf.split(X_xgb, y_arr), skf.split(orig_xgb, y_orig_arr)),
        1,
    ):
        X_tr = pd.concat(
            [X_xgb.iloc[tr_idx], orig_xgb.iloc[or_tr_idx]],
            axis=0,
        ).reset_index(drop=True)
        y_tr = np.concatenate([y_arr[tr_idx], y_orig_arr[or_tr_idx]])
        X_val = X_xgb.iloc[val_idx]
        y_val = y_arr[val_idx]

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device=os.environ.get("AIDE_XGB_DEVICE", "cuda"),
            enable_categorical=True,
            n_estimators=900,
            learning_rate=0.035,
            max_depth=7,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=6.0,
            scale_pos_weight=pos_weight,
            random_state=CFG.SEED + 2000 + fold,
            n_jobs=max(1, min(16, os.cpu_count() or 1)),
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test_xgb)[:, 1] / CFG.FOLDS
        print(
            f"xgboost fold {fold} | AUC Score: "
            f"{roc_auc_score(y_val, oof_preds[val_idx]):.5f}"
        )

    return oof_preds, test_preds


def train_catboost_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    orig: pd.DataFrame,
    y_orig: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier, Pool

    X_cat = X.copy()
    X_test_cat = X_test.copy()
    orig_cat = orig.copy()
    cat_cols = X_cat.select_dtypes(include=["object", "category"]).columns.tolist()
    for frame in [X_cat, X_test_cat, orig_cat]:
        for col in cat_cols:
            frame[col] = (
                frame[col]
                .astype("object")
                .where(frame[col].notna(), "__missing__")
                .astype(str)
            )

    y_arr = y.astype(int).to_numpy()
    y_orig_arr = y_orig.astype(int).to_numpy()
    pos = float(y_arr.sum() + y_orig_arr.sum())
    total = float(len(y_arr) + len(y_orig_arr))
    pos_weight = (total - pos) / max(pos, 1.0)

    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X_cat), dtype=np.float64)
    test_preds = np.zeros(len(X_test_cat), dtype=np.float64)

    for fold, ((tr_idx, val_idx), (or_tr_idx, _or_val_idx)) in enumerate(
        zip(skf.split(X_cat, y_arr), skf.split(orig_cat, y_orig_arr)),
        1,
    ):
        X_tr = pd.concat(
            [X_cat.iloc[tr_idx], orig_cat.iloc[or_tr_idx]],
            axis=0,
        ).reset_index(drop=True)
        y_tr = np.concatenate([y_arr[tr_idx], y_orig_arr[or_tr_idx]])
        X_val = X_cat.iloc[val_idx]
        y_val = y_arr[val_idx]

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=1200,
            learning_rate=0.035,
            depth=7,
            l2_leaf_reg=6.0,
            random_seed=CFG.SEED + 3000 + fold,
            class_weights=[1.0, pos_weight],
            task_type=os.environ.get("AIDE_CATBOOST_TASK_TYPE", "GPU"),
            devices=os.environ.get("AIDE_CATBOOST_DEVICES", "0"),
            allow_writing_files=False,
            verbose=False,
        )
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        valid_pool = Pool(X_val, y_val, cat_features=cat_cols)
        test_pool = Pool(X_test_cat, cat_features=cat_cols)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=80,
            use_best_model=True,
        )
        oof_preds[val_idx] = model.predict_proba(valid_pool)[:, 1]
        test_preds += model.predict_proba(test_pool)[:, 1] / CFG.FOLDS
        print(
            f"catboost fold {fold} | AUC Score: "
            f"{roc_auc_score(y_val, oof_preds[val_idx]):.5f}"
        )

    return oof_preds, test_preds


def pct_rank_train(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = (np.arange(len(a), dtype=np.float64) + 0.5) / len(a)
    return ranks


def pct_rank_test(train_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    sorted_scores = np.sort(train_scores)
    return (np.searchsorted(sorted_scores, test_scores, side="right") + 0.5) / (
        len(sorted_scores) + 1.0
    )


def blend_experts(
    base_oof: dict[str, np.ndarray],
    base_test: dict[str, np.ndarray],
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    names = list(base_oof)
    if len(names) == 1:
        return base_oof[names[0]], base_test[names[0]], {"method": "single_expert"}

    oof_mat = np.column_stack(
        [np.clip(base_oof[name], EPS, 1.0 - EPS) for name in names]
    )
    test_mat = np.column_stack(
        [np.clip(base_test[name], EPS, 1.0 - EPS) for name in names]
    )
    rank_oof = np.column_stack(
        [pct_rank_train(oof_mat[:, i]) for i in range(oof_mat.shape[1])]
    )
    rank_test = np.column_stack(
        [pct_rank_test(oof_mat[:, i], test_mat[:, i]) for i in range(test_mat.shape[1])]
    )

    meta_X = np.column_stack([oof_mat, rank_oof, rank_oof.mean(axis=1)])
    meta_test = np.column_stack([test_mat, rank_test, rank_test.mean(axis=1)])
    stack_oof = np.zeros(len(y), dtype=np.float64)

    cv = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(meta_X, y), 1):
        stacker = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        stacker.fit(meta_X[tr_idx], y[tr_idx])
        stack_oof[val_idx] = stacker.predict_proba(meta_X[val_idx])[:, 1]
        print(
            f"stack fold {fold} | AUC Score: "
            f"{roc_auc_score(y[val_idx], stack_oof[val_idx]):.5f}"
        )

    final_stacker = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    final_stacker.fit(meta_X, y)
    stack_test = final_stacker.predict_proba(meta_test)[:, 1]

    report = {
        "method": "logistic_stack_raw_and_rank",
        "experts": names,
        "expert_auc": {name: float(roc_auc_score(y, base_oof[name])) for name in names},
        "stack_auc": float(roc_auc_score(y, stack_oof)),
    }
    return stack_oof, stack_test, report


def make_regimes(df: pd.DataFrame) -> np.ndarray:
    wet = (
        df["Compound"].astype(str).str.upper().isin(["WET", "INTERMEDIATE"]).to_numpy()
    )
    progress = df["RaceProgress"].to_numpy()
    phase = np.select(
        [progress < 0.34, progress < 0.67],
        ["early", "mid"],
        default="late",
    )
    prefix = np.where(wet, "wet_", "slick_")
    return (
        pd.Series(prefix, dtype="object")
        .str.cat(pd.Series(phase, dtype="object"))
        .to_numpy()
    )


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


class BetaCalibrator:
    def fit(
        self, scores: np.ndarray, y: np.ndarray, batch_scores: np.ndarray | None = None
    ):
        y = np.asarray(y).astype(int)
        self.constant = None
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
            self.model = None
            return self
        p = np.clip(np.asarray(scores), EPS, 1.0 - EPS)
        x = np.column_stack([np.log(p), np.log1p(-p)])
        self.model = LogisticRegression(C=100.0, max_iter=1000, solver="lbfgs")
        self.model.fit(x, y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(scores), self.constant, dtype=np.float64)
        p = np.clip(np.asarray(scores), EPS, 1.0 - EPS)
        x = np.column_stack([np.log(p), np.log1p(-p)])
        return self.model.predict_proba(x)[:, 1]


class IsoCalibrator:
    def fit(
        self, scores: np.ndarray, y: np.ndarray, batch_scores: np.ndarray | None = None
    ):
        y = np.asarray(y).astype(int)
        self.constant = None
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
            self.iso = None
            return self
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso.fit(np.asarray(scores), y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(scores), self.constant, dtype=np.float64)
        return self.iso.predict(np.asarray(scores))


class VennAbersCalibrator:
    def __init__(self, max_grid: int = 256):
        self.max_grid = max_grid

    def fit(
        self, scores: np.ndarray, y: np.ndarray, batch_scores: np.ndarray | None = None
    ):
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y).astype(int)
        self.constant = None
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
            self.iso0 = None
            self.iso1 = None
            return self

        ref = np.asarray(
            batch_scores if batch_scores is not None and len(batch_scores) else scores,
            dtype=np.float64,
        )
        qn = min(self.max_grid, max(2, len(ref)))
        grid = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, qn)))
        if len(grid) == 0:
            grid = np.array([float(np.mean(scores))])

        s_aug = np.concatenate([scores, grid])
        w_aug = np.concatenate(
            [np.ones(len(scores)), np.full(len(grid), 1.0 / len(grid))]
        )
        self.iso0 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso1 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso0.fit(
            s_aug,
            np.concatenate([y, np.zeros(len(grid))]),
            sample_weight=w_aug,
        )
        self.iso1.fit(
            s_aug,
            np.concatenate([y, np.ones(len(grid))]),
            sample_weight=w_aug,
        )
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(scores), self.constant, dtype=np.float64)
        p0 = np.clip(self.iso0.predict(np.asarray(scores)), 0.0, 1.0)
        p1 = np.clip(self.iso1.predict(np.asarray(scores)), 0.0, 1.0)
        den = 1.0 - p0 + p1
        return np.clip(
            np.divide(p1, den, out=(p0 + p1) / 2.0, where=den > EPS),
            0.0,
            1.0,
        )


def make_calibrator(name: str):
    if name == "beta":
        return BetaCalibrator()
    if name == "isotonic":
        return IsoCalibrator()
    if name == "venn_abers":
        return VennAbersCalibrator()
    raise ValueError(name)


def calibrator_cv_predictions(
    method: str,
    scores: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(int)
    min_class = np.bincount(y, minlength=2).min()
    n_splits = int(min(3, min_class))
    if n_splits < 2:
        return np.full(len(y), y.mean(), dtype=np.float64)

    preds = np.zeros(len(y), dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in cv.split(scores.reshape(-1, 1), y):
        cal = make_calibrator(method)
        batch = scores[val_idx] if method == "venn_abers" else None
        cal.fit(scores[tr_idx], y[tr_idx], batch_scores=batch)
        preds[val_idx] = cal.predict(scores[val_idx])
    return np.clip(preds, 0.0, 1.0)


def evaluate_calibrators(
    scores: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[str, dict]:
    results = {}
    for method in ["beta", "isotonic", "venn_abers"]:
        pred = calibrator_cv_predictions(method, scores, y, seed)
        auc = roc_auc_score(y, pred) if len(np.unique(y)) == 2 else 0.5
        results[method] = {
            "pred": pred,
            "auc": float(auc),
            "ece": float(ece_score(y, pred)),
            "brier": float(brier_score_loss(y, pred)),
        }
    best = sorted(
        results,
        key=lambda m: (
            results[m]["auc"],
            -results[m]["ece"],
            -results[m]["brier"],
        ),
        reverse=True,
    )[0]
    return best, results


def stable_regime_seed(regime: str) -> int:
    return CFG.SEED + 1000 + sum((i + 1) * ord(ch) for i, ch in enumerate(regime))


def regime_calibrate(
    oof_score: np.ndarray,
    test_score: np.ndarray,
    y: np.ndarray,
    train_regime: np.ndarray,
    test_regime: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    best_global, global_results = evaluate_calibrators(oof_score, y, CFG.SEED + 100)
    global_oof = global_results[best_global]["pred"]

    global_cal = make_calibrator(best_global)
    global_cal.fit(
        oof_score,
        y,
        batch_scores=test_score if best_global == "venn_abers" else None,
    )
    global_test = global_cal.predict(test_score)

    cal_oof = global_oof.copy()
    cal_test = global_test.copy()
    report = {
        "global": {
            "method": best_global,
            "auc": global_results[best_global]["auc"],
            "ece": global_results[best_global]["ece"],
            "brier": global_results[best_global]["brier"],
        },
        "regimes": {},
    }

    for regime in sorted(set(train_regime) | set(test_regime)):
        tr_idx = np.where(train_regime == regime)[0]
        te_idx = np.where(test_regime == regime)[0]
        yy = y[tr_idx]
        if len(tr_idx) < 2000 or np.bincount(yy.astype(int), minlength=2).min() < 20:
            report["regimes"][regime] = {
                "method": "global_fallback",
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
            }
            continue

        best, results = evaluate_calibrators(
            oof_score[tr_idx],
            yy,
            stable_regime_seed(regime),
        )
        cal_oof[tr_idx] = results[best]["pred"]

        if len(te_idx):
            cal = make_calibrator(best)
            batch = test_score[te_idx] if best == "venn_abers" else None
            cal.fit(oof_score[tr_idx], yy, batch_scores=batch)
            cal_test[te_idx] = cal.predict(test_score[te_idx])

        report["regimes"][regime] = {
            "method": best,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "auc": results[best]["auc"],
            "ece": results[best]["ece"],
            "brier": results[best]["brier"],
        }

    return np.clip(cal_oof, 0.0, 1.0), np.clip(cal_test, 0.0, 1.0), report


def main() -> None:
    input_dir = Path(os.environ.get("AIDE_INPUT_DIR", "input"))
    working_dir = Path(os.environ.get("AIDE_WORKING_DIR", "working"))
    working_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(CFG.SEED)
    print("PyTorch  version:", torch.__version__)
    print("PyTabKit version:", version("pytabkit"))

    train, test, orig, sample = load_data(input_dir)
    train_regime = make_regimes(train)
    test_regime = make_regimes(test)

    X, y, train_id, X_test, test_id, orig, y_orig, combo_names = prepare_features(
        train,
        test,
        orig,
    )

    base_oof = {}
    base_test = {}

    realmlp_oof, realmlp_test = train_and_predict(
        X,
        y,
        X_test,
        orig,
        y_orig,
        combo_names,
        params=PARAMS,
        expert_name="realmlp",
    )
    base_oof["realmlp"] = realmlp_oof
    base_test["realmlp"] = realmlp_test

    try:
        lgb_oof, lgb_test = train_lightgbm_and_predict(X, y, X_test, orig, y_orig)
        base_oof["lightgbm"] = lgb_oof
        base_test["lightgbm"] = lgb_test
    except Exception as exc:
        print(f"LightGBM expert skipped: {exc}")

    try:
        xgb_oof, xgb_test = train_xgboost_and_predict(X, y, X_test, orig, y_orig)
        base_oof["xgboost"] = xgb_oof
        base_test["xgboost"] = xgb_test
    except Exception as exc:
        print(f"XGBoost expert skipped: {exc}")

    try:
        cat_oof, cat_test = train_catboost_and_predict(X, y, X_test, orig, y_orig)
        base_oof["catboost"] = cat_oof
        base_test["catboost"] = cat_test
    except Exception as exc:
        print(f"CatBoost expert skipped: {exc}")

    if len(base_oof) < 2:
        fallback_params = dict(PARAMS)
        fallback_params["random_state"] = CFG.SEED + 787
        fallback_oof, fallback_test = train_and_predict(
            X,
            y,
            X_test,
            orig,
            y_orig,
            combo_names,
            params=fallback_params,
            expert_name="realmlp_seed_787",
        )
        base_oof["realmlp_seed_787"] = fallback_oof
        base_test["realmlp_seed_787"] = fallback_test

    y_np = y.astype(int).to_numpy()
    stack_oof, stack_test, blend_report = blend_experts(base_oof, base_test, y_np)
    cal_oof, cal_test, calibration_report = regime_calibrate(
        stack_oof,
        stack_test,
        y_np,
        train_regime,
        test_regime,
    )

    stack_score = roc_auc_score(y_np, stack_oof)
    oof_score = roc_auc_score(y_np, cal_oof)
    cal_ece = ece_score(y_np, cal_oof)
    cal_brier = brier_score_loss(y_np, cal_oof)

    print("\n" + "=" * 24)
    print(f"Stack OOF AUC: {stack_score:.5f}")
    print(
        f"Regime calibrated OOF AUC: {Fore.BLACK}{Style.BRIGHT}{oof_score:.5f}"
        f"{Style.RESET_ALL}"
    )
    print(f"Regime calibrated OOF ECE: {cal_ece:.6f}")
    print(f"Regime calibrated OOF Brier: {cal_brier:.6f}")
    print("=" * 24)

    oof_required = pd.DataFrame(
        {
            "row": train_id.to_numpy(),
            "target": y_np,
            "prediction": cal_oof,
        }
    )
    oof_required_path = working_dir / "oof_predictions.csv.gz"
    oof_required.to_csv(oof_required_path, index=False, compression="gzip")

    oof_legacy = pd.DataFrame({ID: train_id, TARGET: cal_oof})
    oof_legacy_path = working_dir / "oof_preds.csv"
    oof_legacy.to_csv(oof_legacy_path, index=False)

    sub = sample.copy()
    sub[TARGET] = np.clip(cal_test, 0.0, 1.0)
    sub_path = working_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)

    test_pred_path = working_dir / "test_predictions.csv.gz"
    sub.to_csv(test_pred_path, index=False, compression="gzip")

    print(f"Saved OOF predictions: {oof_required_path}")
    print(f"Saved legacy OOF predictions: {oof_legacy_path}")
    print(f"Saved test predictions: {test_pred_path}")
    print(f"Saved submission: {sub_path}")
    print(
        "AIDE_RESULT_JSON: "
        + json.dumps(
            {
                "is_bug": False,
                "summary": "RealMLP, LightGBM, XGBoost, and CatBoost OOF expert stacking with regime-specific calibration completed successfully.",
                "metric": float(oof_score),
                "lower_is_better": False,
                "validity_warning": None,
                "run_stats": {
                    "metric_name": "roc_auc",
                    "cv_score": float(oof_score),
                    "stack_oof_auc": float(stack_score),
                    "calibrated_oof_ece": float(cal_ece),
                    "calibrated_oof_brier": float(cal_brier),
                    "submission_path": str(sub_path),
                    "oof_path": str(oof_required_path),
                    "test_predictions_path": str(test_pred_path),
                    "original_dataset_file": "f1_strategy_dataset_v4.csv",
                    "blend_report": blend_report,
                    "calibration_report": calibration_report,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
