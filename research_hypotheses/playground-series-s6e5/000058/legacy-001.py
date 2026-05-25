import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESIS_ID = "000058"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x).strip().upper()).strip("_").lower()


def normalize_text(s):
    return s.fillna("UNKNOWN").astype(str).str.strip().str.upper()


def add_compound_switch_features(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()
    tr["_is_train"] = 1
    te["_is_train"] = 0
    if TARGET not in te.columns:
        te[TARGET] = np.nan

    all_df = pd.concat([tr, te], ignore_index=True, sort=False)
    all_df["_orig_order"] = np.arange(len(all_df))
    all_df["_compound_norm"] = normalize_text(all_df["Compound"])
    all_df["_driver_norm"] = normalize_text(all_df["Driver"])
    all_df["_race_norm"] = normalize_text(all_df["Race"])

    group_cols = ["Year", "_race_norm", "_driver_norm"]
    sort_cols = group_cols + ["LapNumber", ID_COL]
    work = all_df.sort_values(sort_cols, kind="mergesort").copy()

    prev = work.groupby(group_cols, sort=False)["_compound_norm"].shift(1)
    work["PrevCompound"] = prev.fillna("START")
    work["compound_has_previous"] = prev.notna().astype(np.int8)
    work["compound_same_as_previous"] = (
        prev.eq(work["_compound_norm"]) & prev.notna()
    ).astype(np.int8)
    work["compound_switch"] = (prev.ne(work["_compound_norm"]) & prev.notna()).astype(
        np.int8
    )

    known = sorted(
        set(work["_compound_norm"].unique()).union(
            {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
        )
    )
    seen_cols = []
    current_seen_before = np.zeros(len(work), dtype=bool)

    for comp in known:
        col = f"compound_seen_{safe_name(comp)}_before"
        flag_col = f"_is_{safe_name(comp)}"
        work[flag_col] = work["_compound_norm"].eq(comp).astype(np.int8)
        seen_count = (
            work.groupby(group_cols, sort=False)[flag_col].cumsum() - work[flag_col]
        )
        work[col] = (seen_count > 0).astype(np.int8)
        seen_cols.append(col)
        current_seen_before |= (work["_compound_norm"].values == comp) & (
            work[col].values == 1
        )
        work.drop(columns=[flag_col], inplace=True)

    dry = {"SOFT", "MEDIUM", "HARD"}
    work["compound_seen_current_before"] = current_seen_before.astype(np.int8)
    work["compound_seen_before_count"] = work[seen_cols].sum(axis=1).astype(np.int16)
    work["compound_dry_seen_before_count"] = (
        work[[c for c in seen_cols if any(x in c for x in ["soft", "medium", "hard"])]]
        .sum(axis=1)
        .astype(np.int16)
    )
    work["compound_is_new_any"] = (~current_seen_before).astype(np.int8)
    work["compound_is_new_slick"] = (
        work["_compound_norm"].isin(dry).values & (~current_seen_before)
    ).astype(np.int8)
    work["compound_return_to_seen"] = (
        current_seen_before & (work["compound_switch"].values == 1)
    ).astype(np.int8)

    current_pit = work["PitStop"].fillna(0).astype(np.int8).values
    work["compound_switch_with_current_pit"] = (
        (work["compound_switch"].values == 1) & (current_pit == 1)
    ).astype(np.int8)
    work["compound_new_slick_with_current_pit"] = (
        (work["compound_is_new_slick"].values == 1) & (current_pit == 1)
    ).astype(np.int8)
    work["compound_return_with_current_pit"] = (
        (work["compound_return_to_seen"].values == 1) & (current_pit == 1)
    ).astype(np.int8)

    state = np.full(len(work), "OTHER_SWITCH", dtype=object)
    state[work["compound_is_new_any"].values == 1] = "NEW_NON_SLICK"
    state[work["compound_is_new_slick"].values == 1] = "NEW_SLICK"
    state[work["compound_return_to_seen"].values == 1] = "RETURN_TO_SEEN"
    state[work["compound_same_as_previous"].values == 1] = "SAME_AS_PREVIOUS"
    state[work["compound_has_previous"].values == 0] = "FIRST_OBSERVED"
    work["CompoundSwitchState"] = state

    work = work.sort_values("_orig_order", kind="mergesort").reset_index(drop=True)
    drop_helpers = [
        "_is_train",
        "_orig_order",
        "_compound_norm",
        "_driver_norm",
        "_race_norm",
    ]
    work.drop(columns=drop_helpers, inplace=True)

    return work.iloc[: len(train_df)].copy(), work.iloc[len(train_df) :].copy()


train_fe, test_fe = add_compound_switch_features(train, test)

drop_cols = [ID_COL, TARGET]
features = [c for c in train_fe.columns if c not in drop_cols]
X = train_fe[features].copy()
X_test = test_fe[features].copy()
y = train_fe[TARGET].astype(int).values

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    tr_col = X[col].fillna("MISSING").astype(str)
    te_col = X_test[col].fillna("MISSING").astype(str)
    cats = pd.Index(pd.concat([tr_col, te_col], ignore_index=True).unique())
    X[col] = pd.Categorical(tr_col, categories=cats)
    X_test[col] = pd.Categorical(te_col, categories=cats)

try:
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(
        cv.split(
            X,
            y,
            groups=(
                train["Year"].astype(str)
                + "|"
                + train["Race"].astype(str)
                + "|"
                + train["Driver"].astype(str)
            ).values,
        )
    )
except Exception:
    cv = GroupKFold(n_splits=5)
    splits = list(
        cv.split(
            X,
            y,
            groups=(
                train["Year"].astype(str)
                + "|"
                + train["Race"].astype(str)
                + "|"
                + train["Driver"].astype(str)
            ).values,
        )
    )

import lightgbm as lgb

pos = max(1, int(y.sum()))
neg = max(1, int(len(y) - y.sum()))
base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.035,
    n_estimators=1400,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    scale_pos_weight=neg / pos,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

oof = np.zeros(len(X), dtype=np.float32)
fold_aucs = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**base_params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    fold_aucs.append(float(auc))
    best_iters.append(int(model.best_iteration_ or base_params["n_estimators"]))
    print(f"fold {fold} roc_auc: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
final_estimators = int(np.median(best_iters)) if best_iters else 700
final_params = dict(base_params)
final_params["n_estimators"] = final_estimators

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_cols)
test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission = sample.copy()
submission[TARGET] = test_pred
submission[[ID_COL, TARGET]].to_csv(
    os.path.join(WORK_DIR, "submission.csv"), index=False
)
submission[[ID_COL, TARGET]].to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_aucs,
    "final_n_estimators": final_estimators,
    "research_hypotheses_llm_claimed_used": [HYPOTHESIS_ID],
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"CV ROC AUC: {cv_auc:.6f}")
print(json.dumps(result, indent=2))
