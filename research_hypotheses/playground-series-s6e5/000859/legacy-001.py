import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "This solution requires lightgbm, which is listed as installed."
    ) from e

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


def safe_names(cols):
    out, seen = [], {}
    for c in cols:
        s = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
        if not s:
            s = "f"
        if s[0].isdigit():
            s = "f_" + s
        base = s
        k = seen.get(base, 0)
        if k:
            s = f"{base}_{k}"
        seen[base] = k + 1
        out.append(s)
    return out


def add_rule_debt_features(train, test):
    train = train.copy()
    test = test.copy()
    train["_is_train"] = 1
    test["_is_train"] = 0
    test["PitNextLap"] = np.nan

    df = pd.concat([train, test], ignore_index=True, sort=False)
    df["_orig_order"] = np.arange(len(df))

    for c in ["Race", "Driver", "Compound"]:
        df[c] = df[c].fillna("NA").astype(str)
    df["Compound"] = df["Compound"].str.upper()

    df = df.sort_values(
        ["Year", "Race", "Driver", "LapNumber", "id"], kind="mergesort"
    ).reset_index(drop=True)
    g = ["Year", "Race", "Driver"]

    dry = ["SOFT", "MEDIUM", "HARD"]
    wet = ["INTERMEDIATE", "WET"]

    df["_current_is_wet"] = df["Compound"].isin(wet).astype(np.int8)
    df["_current_is_dry"] = df["Compound"].isin(dry).astype(np.int8)

    df["pit_count_to_date"] = (
        df.groupby(g, sort=False)["PitStop"].cumsum().astype(np.int16)
    )
    df["pit_count_before_lap"] = (
        (df["pit_count_to_date"] - df["PitStop"].astype(np.int16))
        .clip(lower=0)
        .astype(np.int16)
    )
    df["wet_used_to_date"] = (
        df.groupby(g, sort=False)["_current_is_wet"].cummax().astype(np.int8)
    )

    used_cols = []
    for comp in dry:
        flag = f"_is_{comp.lower()}"
        used = f"used_{comp.lower()}_to_date"
        df[flag] = (df["Compound"] == comp).astype(np.int8)
        df[used] = df.groupby(g, sort=False)[flag].cummax().astype(np.int8)
        used_cols.append(used)

    df["dry_compounds_used_to_date"] = df[used_cols].sum(axis=1).astype(np.int8)
    dry_debt = np.maximum(0, 2 - df["dry_compounds_used_to_date"])
    stop_debt = np.maximum(0, 1 - df["pit_count_to_date"])

    df["mandatory_compound_debt"] = np.where(
        df["wet_used_to_date"] == 1, 0, dry_debt
    ).astype(np.int8)
    df["mandatory_stop_debt"] = np.where(
        df["wet_used_to_date"] == 1, 0, stop_debt
    ).astype(np.int8)

    monaco_2025 = (df["Year"].astype(int) == 2025) & df["Race"].str.contains(
        "Monaco", case=False, na=False
    )
    df["is_monaco_2025_rule"] = monaco_2025.astype(np.int8)
    df["monaco_2025_stop_debt"] = np.where(
        monaco_2025, np.maximum(0, 2 - df["pit_count_to_date"]), 0
    ).astype(np.int8)
    df["total_rule_stop_debt"] = np.maximum(
        df["mandatory_stop_debt"], df["monaco_2025_stop_debt"]
    ).astype(np.int8)
    df["rule_debt_sum"] = (
        df["mandatory_compound_debt"]
        + df["mandatory_stop_debt"]
        + df["monaco_2025_stop_debt"]
    ).astype(np.int8)

    rp = df["RaceProgress"].astype(float).clip(lower=1e-4)
    est_total = np.rint(df["LapNumber"].astype(float) / rp).clip(1, 120)
    df["estimated_total_laps_from_progress"] = est_total.astype(np.float32)
    df["laps_remaining"] = (
        (df["estimated_total_laps_from_progress"] - df["LapNumber"].astype(float))
        .clip(lower=0)
        .astype(np.float32)
    )

    race_max = (
        df.groupby(["Year", "Race"], sort=False)["LapNumber"]
        .transform("max")
        .astype(float)
    )
    df["race_max_lap_in_table"] = race_max.astype(np.float32)
    df["laps_remaining_table"] = (
        (race_max - df["LapNumber"].astype(float)).clip(lower=0).astype(np.float32)
    )

    debt_cols = [
        "mandatory_stop_debt",
        "mandatory_compound_debt",
        "monaco_2025_stop_debt",
        "total_rule_stop_debt",
    ]
    denom = df["laps_remaining"].astype(float) + 1.0

    for d in debt_cols:
        val = df[d].astype(float)
        df[f"{d}_per_lap_remaining"] = (val / denom).astype(np.float32)
        df[f"{d}_x_laps_remaining"] = (val * df["laps_remaining"]).astype(np.float32)
        df[f"{d}_x_laps_remaining_table"] = (val * df["laps_remaining_table"]).astype(
            np.float32
        )
        df[f"{d}_x_stint"] = (val * df["Stint"].astype(float)).astype(np.float32)
        df[f"{d}_x_tyre_life"] = (val * df["TyreLife"].astype(float)).astype(np.float32)
        df[f"{d}_critical"] = (
            (df[d] > 0) & (df["laps_remaining"] <= (df[d].astype(float) + 1.0))
        ).astype(np.int8)

    for comp in dry + wet:
        is_comp = (df["Compound"] == comp).astype(np.int8)
        for d in debt_cols:
            df[f"{d}_on_{comp.lower()}"] = (df[d].astype(np.int8) * is_comp).astype(
                np.int8
            )

    df["rule_debt_pressure"] = (df["rule_debt_sum"].astype(float) / denom).astype(
        np.float32
    )
    df["compound_debt_x_current_is_dry"] = (
        df["mandatory_compound_debt"] * df["_current_is_dry"]
    ).astype(np.int8)
    df["stop_debt_x_current_is_wet"] = (
        df["total_rule_stop_debt"] * df["_current_is_wet"]
    ).astype(np.int8)

    df = df.sort_values("_orig_order", kind="mergesort").reset_index(drop=True)
    train_fe = df[df["_is_train"] == 1].copy()
    test_fe = df[df["_is_train"] == 0].copy().drop(columns=["PitNextLap"])
    return train_fe, test_fe


def make_matrices(train_fe, test_fe):
    drop = {"PitNextLap", "id", "_is_train", "_orig_order"}
    features = [c for c in train_fe.columns if c not in drop and c in test_fe.columns]

    all_x = pd.concat(
        [train_fe[features], test_fe[features]], axis=0, ignore_index=True
    )
    cat_cols = [
        c
        for c in features
        if all_x[c].dtype == "object" or str(all_x[c].dtype) == "category"
    ]

    for c in cat_cols:
        all_x[c] = all_x[c].astype(str).fillna("__NA__").astype("category")

    num_cols = [c for c in features if c not in cat_cols]
    for c in num_cols:
        all_x[c] = pd.to_numeric(all_x[c], errors="coerce")
    all_x[num_cols] = all_x[num_cols].replace([np.inf, -np.inf], np.nan).fillna(-999.0)

    new_cols = safe_names(all_x.columns)
    rename = dict(zip(all_x.columns, new_cols))
    all_x = all_x.rename(columns=rename)
    cat_cols = [rename[c] for c in cat_cols]

    x_train = all_x.iloc[: len(train_fe)].reset_index(drop=True)
    x_test = all_x.iloc[len(train_fe) :].reset_index(drop=True)
    y = train_fe["PitNextLap"].astype(int).values
    return x_train, x_test, y, cat_cols


def make_model(seed, scale_pos_weight, n_estimators=2000):
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=90,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train_fe, test_fe = add_rule_debt_features(train, test)
x_train, x_test, y, cat_cols = make_matrices(train_fe, test_fe)

groups = (
    train_fe["Year"].astype(str)
    + "|"
    + train_fe["Race"].astype(str)
    + "|"
    + train_fe["Driver"].astype(str)
).values

if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        splits = list(splitter.split(x_train, y, groups))
        cv_name = "StratifiedGroupKFold"
    except Exception:
        splits = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x_train, y)
        )
        cv_name = "StratifiedKFold"
else:
    splits = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x_train, y)
    )
    cv_name = "StratifiedKFold"

oof = np.zeros(len(y), dtype=np.float32)
fold_aucs = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    y_tr, y_va = y[tr_idx], y[va_idx]
    pos = max(float(y_tr.sum()), 1.0)
    spw = float((len(y_tr) - y_tr.sum()) / pos)

    model = make_model(seed=1000 + fold, scale_pos_weight=spw)
    model.fit(
        x_train.iloc[tr_idx],
        y_tr,
        eval_set=[(x_train.iloc[va_idx], y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
    )

    pred = model.predict_proba(
        x_train.iloc[va_idx], num_iteration=model.best_iteration_
    )[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y_va, pred)
    fold_aucs.append(float(auc))
    best_iters.append(int(model.best_iteration_ or model.n_estimators))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold {cv_name} ROC AUC: {cv_auc:.6f}")

mean_best_iter = int(np.clip(np.mean(best_iters) * 1.10, 100, 2500))
pos = max(float(y.sum()), 1.0)
spw = float((len(y) - y.sum()) / pos)

final_model = make_model(seed=2026, scale_pos_weight=spw, n_estimators=mean_best_iter)
final_model.fit(x_train, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(x_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission["PitNextLap"] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(y)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000859"],
    "cv_name": cv_name,
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_aucs,
    "final_model_estimators": int(mean_best_iter),
}
for name in ["result.json", "review.json"]:
    with open(os.path.join(WORK_DIR, name), "w") as f:
        json.dump(review, f, indent=2)

print(json.dumps(review, indent=2))
