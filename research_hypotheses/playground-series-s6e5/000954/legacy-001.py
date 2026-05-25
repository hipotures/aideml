import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 954
INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")


def sanitize_columns(cols):
    mapping, used = {}, set()
    for col in cols:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_")
        if not base:
            base = "feature"
        if base[0].isdigit():
            base = "f_" + base
        name, i = base, 1
        while name in used:
            i += 1
            name = f"{base}_{i}"
        mapping[col] = name
        used.add(name)
    return mapping


def add_rule_state_features(train_df, test_df):
    train_base = train_df.drop(columns=[TARGET], errors="ignore").copy()
    test_base = test_df.copy()

    train_base["__part"] = "train"
    test_base["__part"] = "test"
    train_base["__order"] = np.arange(len(train_base))
    test_base["__order"] = np.arange(len(test_base))

    all_df = pd.concat([train_base, test_base], axis=0, ignore_index=True)
    all_df["Compound"] = all_df["Compound"].astype(str).str.upper()
    all_df["Race"] = all_df["Race"].astype(str)
    all_df["Driver"] = all_df["Driver"].astype(str)

    s = all_df.sort_values(
        ["Year", "Race", "Driver", "LapNumber", ID_COL], kind="mergesort"
    ).copy()
    group_cols = ["Year", "Race", "Driver"]
    group_keys = [s[c] for c in group_cols]

    race_progress = s["RaceProgress"].astype(float).clip(lower=1e-4)
    est_total = np.rint(s["LapNumber"].astype(float) / race_progress)
    s["Estimated_Total_Laps"] = (
        np.maximum(est_total, s["LapNumber"].astype(float))
        .clip(upper=120)
        .astype("float32")
    )
    s["Laps_Left"] = np.maximum(
        0.0, s["Estimated_Total_Laps"] - s["LapNumber"].astype(float)
    ).astype("float32")

    s["Is_Dry_Compound"] = s["Compound"].isin(DRY_COMPOUNDS).astype("int8")
    s["Is_Wet_Compound"] = s["Compound"].isin(WET_COMPOUNDS).astype("int8")
    s["Monaco_2025_Special_Rule"] = (
        (s["Year"].astype(int) == 2025)
        & s["Race"].str.contains("Monaco", case=False, na=False)
    ).astype("int8")

    s["PitStops_So_Far"] = (
        s.groupby(group_cols, sort=False)["PitStop"].cumsum().astype("int16")
    )
    sets_from_stint = (
        s.groupby(group_cols, sort=False)["Stint"]
        .cummax()
        .fillna(s["Stint"])
        .astype("int16")
    )
    sets_from_pits = (s["PitStops_So_Far"] + 1).astype("int16")
    s["Sets_Used_So_Far"] = (
        np.maximum(sets_from_stint, sets_from_pits).clip(1, 12).astype("int16")
    )

    dry_seen_cols = []
    for comp in DRY_COMPOUNDS:
        col = f"Seen_{comp}_So_Far"
        s[col] = (
            s["Compound"]
            .eq(comp)
            .astype("int8")
            .groupby(group_keys, sort=False)
            .cummax()
            .astype("int8")
        )
        dry_seen_cols.append(col)

    wet_seen_cols = []
    for comp in WET_COMPOUNDS:
        col = f"Seen_{comp}_So_Far"
        s[col] = (
            s["Compound"]
            .eq(comp)
            .astype("int8")
            .groupby(group_keys, sort=False)
            .cummax()
            .astype("int8")
        )
        wet_seen_cols.append(col)

    s["Dry_Compounds_Used_So_Far"] = s[dry_seen_cols].sum(axis=1).astype("int8")
    s["Wet_Compounds_Used_So_Far"] = s[wet_seen_cols].sum(axis=1).astype("int8")
    s["Any_Wet_Compound_Seen_So_Far"] = (s["Wet_Compounds_Used_So_Far"] > 0).astype(
        "int8"
    )
    s["Wet_Exemption_Flag"] = s["Any_Wet_Compound_Seen_So_Far"]

    normal_dry_remaining = np.where(
        s["Wet_Exemption_Flag"].values == 0,
        np.maximum(0, 2 - s["Dry_Compounds_Used_So_Far"].values),
        0,
    )

    monaco_set_remaining = np.where(
        s["Monaco_2025_Special_Rule"].values == 1,
        np.maximum(0, 3 - s["Sets_Used_So_Far"].values),
        0,
    )

    monaco_dry_remaining = np.where(
        (s["Monaco_2025_Special_Rule"].values == 1)
        & (s["Wet_Exemption_Flag"].values == 0),
        np.maximum(0, 2 - s["Dry_Compounds_Used_So_Far"].values),
        0,
    )

    s["Normal_Dry_Requirement_Remaining"] = normal_dry_remaining.astype("int8")
    s["Monaco_2025_Set_Requirement_Remaining"] = monaco_set_remaining.astype("int8")
    s["Monaco_2025_Dry_Requirement_Remaining"] = monaco_dry_remaining.astype("int8")
    s["Monaco_2025_Wet_Rule_Exemption"] = (
        (s["Monaco_2025_Special_Rule"].values == 1)
        & (s["Wet_Exemption_Flag"].values == 1)
    ).astype("int8")

    s["Required_Stops_Remaining"] = np.where(
        s["Monaco_2025_Special_Rule"].values == 1,
        np.maximum(monaco_set_remaining, monaco_dry_remaining),
        normal_dry_remaining,
    ).astype("int8")

    denom = (s["Laps_Left"].astype(float) + 1.0).clip(lower=1.0)
    s["Rule_Pressure_Per_Lap_Left"] = (
        s["Required_Stops_Remaining"].astype(float) / denom
    ).astype("float32")
    s["Dry_Rule_Pressure_Per_Lap_Left"] = (
        s["Normal_Dry_Requirement_Remaining"].astype(float) / denom
    ).astype("float32")
    s["Monaco_Set_Rule_Pressure_Per_Lap_Left"] = (
        s["Monaco_2025_Set_Requirement_Remaining"].astype(float) / denom
    ).astype("float32")
    s["Rule_Pressure_Final_Window"] = (
        (s["Required_Stops_Remaining"].values > 0)
        & (
            s["Laps_Left"].values
            <= 5 * np.maximum(1, s["Required_Stops_Remaining"].values)
        )
    ).astype("int8")

    restored = s.sort_index(kind="mergesort")
    train_features = (
        restored.loc[restored["__part"] == "train"]
        .sort_values("__order")
        .drop(columns=["__part", "__order"])
        .reset_index(drop=True)
    )
    test_features = (
        restored.loc[restored["__part"] == "test"]
        .sort_values("__order")
        .drop(columns=["__part", "__order"])
        .reset_index(drop=True)
    )
    return train_features, test_features


def align_categoricals(train_features, test_features, cat_cols):
    for col in cat_cols:
        train_values = train_features[col].astype(str).fillna("__MISSING__")
        test_values = test_features[col].astype(str).fillna("__MISSING__")
        cats = pd.Index(
            pd.unique(pd.concat([train_values, test_values], ignore_index=True))
        )
        train_features[col] = pd.Categorical(train_values, categories=cats)
        test_features[col] = pd.Categorical(test_values, categories=cats)
    return train_features, test_features


def make_splits(y, groups):
    x_dummy = np.zeros(len(y))
    if HAS_SGK:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(splitter.split(x_dummy, y, groups))
        method = "StratifiedGroupKFold"
    else:
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(x_dummy, y, groups))
        method = "GroupKFold"

    if all(y.iloc[val_idx].nunique() == 2 for _, val_idx in splits):
        return splits, method

    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(x_dummy, y, groups))
    return splits, "GroupKFold"


def predict_proba(model, X):
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None and best_iter > 0:
        return model.predict_proba(X, num_iteration=best_iter)[:, 1]
    return model.predict_proba(X)[:, 1]


def main():
    os.makedirs(WORKING_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
    sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

    y = train[TARGET].astype("int8")
    train_features, test_features = add_rule_state_features(train, test)

    cat_cols = ["Compound", "Driver", "Race"]
    train_features, test_features = align_categoricals(
        train_features, test_features, cat_cols
    )

    feature_cols = [c for c in train_features.columns if c != ID_COL]
    col_map = sanitize_columns(feature_cols)
    X = train_features[feature_cols].rename(columns=col_map)
    X_test = test_features[feature_cols].rename(columns=col_map)
    cat_features = [col_map[c] for c in cat_cols]

    numeric_cols = [c for c in X.columns if c not in cat_features]
    for df in (X, X_test):
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        for col in numeric_cols:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype("float32")
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast="integer")

    groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
    splits, split_method = make_splits(y, groups)

    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []

    n_jobs = max(1, min(8, (os.cpu_count() or 2) - 1))

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        pos = max(1, int(y_tr.sum()))
        neg = max(1, int(len(y_tr) - y_tr.sum()))
        scale_pos_weight = neg / pos

        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=1600,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=90,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED + fold,
            n_jobs=n_jobs,
            verbosity=-1,
            force_col_wise=True,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )

        va_pred = predict_proba(model, X_va)
        oof[va_idx] = va_pred.astype(np.float32)
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)

        test_pred += predict_proba(model, X_test) / len(splits)
        print(f"fold {fold} {split_method} ROC AUC: {fold_auc:.6f}")

    cv_auc = roc_auc_score(y, oof)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = sample.copy()
    submission[TARGET] = test_pred
    submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
    submission.to_csv(
        os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    pd.DataFrame(
        {
            "row": np.arange(len(train)),
            "target": y.astype(int),
            "prediction": oof,
        }
    ).to_csv(
        os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    result = {
        "metric": "roc_auc",
        "cv_auc": float(cv_auc),
        "fold_aucs": [float(v) for v in fold_aucs],
        "cv_split": split_method,
        "research_hypotheses_llm_claimed_used": ["000954"],
    }

    with open(os.path.join(WORKING_DIR, "result_review.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"OOF ROC AUC: {cv_auc:.6f}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
