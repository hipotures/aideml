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
    SEEDS = [42, 829, 137, 2027]
    CAT_SEED = 1379
    CAT_BLEND_WEIGHT = 0.15
    TE = True


PARAMS = {
    "random_state": 42,
    "verbosity": 2,
    "val_metric_name": "1-auc_ovr",
    "n_ens": 20,
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


def add_fold_target_encoding(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    X_tst: pd.DataFrame,
    combo_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return X_tr, X_val, X_tst


def train_one_seed(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    orig: pd.DataFrame,
    y_orig: pd.Series,
    combo_names: list[str],
    model_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed_everything(model_seed)

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
            X_tr, X_val, X_tst = add_fold_target_encoding(
                X_tr,
                y_tr,
                X_val,
                X_tst,
                combo_names,
            )

        if fold == 1:
            print("len(FEATURES):", len(X_tr.columns.tolist()), "\n")
        print("#" * 16)
        print(f"### Seed {model_seed} | Fold {fold}/{CFG.FOLDS} ...")
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
            f"{Fore.GREEN}{Style.BRIGHT}Seed {model_seed} | Fold {fold} | AUC Score: "
            f"{fold_score:.5f}{Style.RESET_ALL}\n"
        )
        torch.cuda.empty_cache()

    seed_score = roc_auc_score(y, oof_preds)
    print(
        f"{Fore.GREEN}{Style.BRIGHT}Seed {model_seed} | Overall OOF AUC: "
        f"{seed_score:.5f}{Style.RESET_ALL}\n"
    )
    return oof_preds, test_preds


def make_catboost_pool(
    X_df: pd.DataFrame,
    y: pd.Series | None = None,
) -> Pool:
    X_cb = X_df.copy()
    cat_features = X_cb.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_features:
        X_cb[col] = X_cb[col].astype("string").fillna("__MISSING__").astype(str)
    return Pool(X_cb, label=y, cat_features=cat_features)


def fit_catboost_with_fallback(
    params: dict,
    train_pool: Pool,
    val_pool: Pool,
) -> CatBoostClassifier:
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


def train_catboost_expert(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    orig: pd.DataFrame,
    y_orig: pd.Series,
    combo_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    seed_everything(CFG.CAT_SEED)

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
            X_tr, X_val, X_tst = add_fold_target_encoding(
                X_tr,
                y_tr,
                X_val,
                X_tst,
                combo_names,
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
            f"{Fore.GREEN}{Style.BRIGHT}CatBoost Fold {fold} | AUC Score: "
            f"{fold_score:.5f}{Style.RESET_ALL}\n"
        )

    cat_score = roc_auc_score(y, oof_preds)
    print(
        f"{Fore.GREEN}{Style.BRIGHT}CatBoost Overall OOF AUC: "
        f"{cat_score:.5f}{Style.RESET_ALL}\n"
    )
    return oof_preds, test_preds


def train_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    orig: pd.DataFrame,
    y_orig: pd.Series,
    combo_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    all_oof = []
    all_test = []

    for model_seed in CFG.SEEDS:
        oof_seed, test_seed = train_one_seed(
            X,
            y,
            X_test,
            orig,
            y_orig,
            combo_names,
            model_seed,
        )
        all_oof.append(oof_seed)
        all_test.append(test_seed)

    realmlp_oof = np.mean(np.vstack(all_oof), axis=0)
    realmlp_test = np.mean(np.vstack(all_test), axis=0)
    realmlp_score = roc_auc_score(y, realmlp_oof)

    cat_oof, cat_test = train_catboost_expert(
        X,
        y,
        X_test,
        orig,
        y_orig,
        combo_names,
    )
    cat_score = roc_auc_score(y, cat_oof)
    oof_corr = float(np.corrcoef(realmlp_oof, cat_oof)[0, 1])

    blend_w = CFG.CAT_BLEND_WEIGHT
    oof_preds = (1.0 - blend_w) * realmlp_oof + blend_w * cat_oof
    test_preds = (1.0 - blend_w) * realmlp_test + blend_w * cat_test
    blend_score = roc_auc_score(y, oof_preds)

    print("\n" + "=" * 24)
    print(f"RealMLP core OOF AUC: {realmlp_score:.5f}")
    print(f"CatBoost expert OOF AUC: {cat_score:.5f}")
    print(f"RealMLP/CatBoost OOF correlation: {oof_corr:.6f}")
    print(f"Fixed {blend_w:.2f} CatBoost blend OOF AUC: {blend_score:.5f}")
    print("=" * 24)

    stats = {
        "realmlp_oof_auc": float(realmlp_score),
        "catboost_oof_auc": float(cat_score),
        "realmlp_catboost_oof_correlation": oof_corr,
        "catboost_blend_weight": float(blend_w),
        "blend_oof_auc": float(blend_score),
    }
    return oof_preds, test_preds, stats


def main() -> None:
    input_dir = Path(os.environ.get("AIDE_INPUT_DIR", "input"))
    working_dir = Path(os.environ.get("AIDE_WORKING_DIR", "working"))
    working_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(CFG.SEED)
    print("PyTorch  version:", torch.__version__)
    print("PyTabKit version:", version("pytabkit"))
    print("CatBoost version:", version("catboost"))
    print("Model seeds:", CFG.SEEDS)
    print("CatBoost seed:", CFG.CAT_SEED)

    train, test, orig = load_data(input_dir)
    X, y, train_id, X_test, test_id, orig, y_orig, combo_names = prepare_features(
        train,
        test,
        orig,
    )
    del train, test

    oof_preds, test_preds, model_stats = train_and_predict(
        X,
        y,
        X_test,
        orig,
        y_orig,
        combo_names,
    )

    oof_score = roc_auc_score(y, oof_preds)
    print("\n" + "=" * 24)
    print(
        f"Overall blended OOF AUC: {Fore.BLACK}{Style.BRIGHT}{oof_score:.5f}"
        f"{Style.RESET_ALL}"
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
                "summary": "RealMLP/PyTabKit four-seed 5-fold OOF validation completed with a repeated CatBoost stability expert and fixed low-weight blend.",
                "metric": float(oof_score),
                "lower_is_better": False,
                "validity_warning": None,
                "run_stats": {
                    "metric_name": "roc_auc",
                    "cv_score": float(oof_score),
                    "submission_path": str(sub_path),
                    "oof_path": str(oof_path),
                    "compressed_oof_path": str(required_oof_path),
                    "test_predictions_path": str(test_pred_path),
                    "original_dataset_file": "f1_strategy_dataset_v4.csv",
                    "model_seeds": CFG.SEEDS,
                    "catboost_seed": CFG.CAT_SEED,
                    **model_stats,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
