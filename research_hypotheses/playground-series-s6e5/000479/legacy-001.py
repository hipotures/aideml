import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID = "id"
SEQ_COLS = ["Year", "Race", "Driver"]
N_SPLITS = 5
N_BUCKETS = 7
BLEND_MAIN_WEIGHT = 0.80
RANDOM_STATE = 479


def clean_columns(df):
    seen = {}
    names = []
    for c in df.columns:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", c).strip("_")
        n = seen.get(base, 0)
        seen[base] = n + 1
        names.append(base if n == 0 else f"{base}_{n}")
    out = df.copy()
    out.columns = names
    return out


def add_sequence_features(df):
    df = df.copy()
    order = df.sort_values(SEQ_COLS + ["LapNumber", ID]).index
    s = df.loc[order].copy()
    g = s.groupby(SEQ_COLS, sort=False, observed=True)

    s["seq_lap_index"] = g.cumcount() + 1
    s["pit_count_so_far"] = g["PitStop"].cumsum()
    s["_last_pit_lap"] = s["LapNumber"].where(s["PitStop"].eq(1))
    s["_last_pit_lap"] = g["_last_pit_lap"].ffill()
    s["laps_since_pit_event"] = (s["LapNumber"] - s["_last_pit_lap"]).fillna(
        s["TyreLife"]
    )

    rp = s["RaceProgress"].clip(lower=1e-4)
    tl = s["TyreLife"].clip(lower=1)
    s["estimated_total_laps"] = s["LapNumber"] / rp
    s["estimated_laps_remaining"] = s["estimated_total_laps"] - s["LapNumber"]
    s["degradation_per_tyre_lap"] = s["Cumulative_Degradation"] / tl
    s["tyre_life_x_progress"] = s["TyreLife"] * s["RaceProgress"]
    s["stint_x_progress"] = s["Stint"] * s["RaceProgress"]

    roll_cols = [
        "LapTime_s",
        "LapTime_Delta",
        "Position",
        "Position_Change",
        "TyreLife",
        "Cumulative_Degradation",
    ]
    for col in roll_cols:
        prev = g[col].shift(1)
        roll3 = g[col].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        s[f"{col}_prev"] = prev
        s[f"{col}_diff1"] = s[col] - prev
        s[f"{col}_roll3_mean"] = roll3
        s[f"{col}_vs_roll3"] = s[col] - roll3

    s = s.drop(columns=["_last_pit_lap"])
    new_cols = [c for c in s.columns if c not in df.columns]
    return pd.concat([df, s.loc[df.index, new_cols]], axis=1)


def derive_auxiliary_labels(df):
    labels = pd.DataFrame(index=df.index)
    labels["LapsToNextPit"] = 99.0
    labels["StopWithin3"] = 0
    labels["StopWithin5"] = 0
    labels["CensoredToFinish"] = 1
    labels["HazardBucket"] = 0

    tmp = df.sort_values(SEQ_COLS + ["LapNumber", ID])
    for _, grp in tmp.groupby(SEQ_COLS, sort=False, observed=True):
        idx = grp.index.to_numpy()
        laps = grp["LapNumber"].to_numpy()
        pit_laps = grp.loc[grp["PitStop"].eq(1), "LapNumber"].to_numpy()
        pos = np.searchsorted(pit_laps, laps, side="right")
        has_next = pos < len(pit_laps)

        l2n = grp["LapNumber"].max() - laps + 1.0
        l2n[has_next] = pit_laps[pos[has_next]] - laps[has_next]
        l2n = np.clip(l2n, 1, 99)

        bucket = np.zeros(len(grp), dtype=np.int8)
        bucket[has_next & (l2n == 1)] = 1
        bucket[has_next & (l2n == 2)] = 2
        bucket[has_next & (l2n == 3)] = 3
        bucket[has_next & (l2n >= 4) & (l2n <= 5)] = 4
        bucket[has_next & (l2n >= 6) & (l2n <= 10)] = 5
        bucket[has_next & (l2n > 10)] = 6

        labels.loc[idx, "LapsToNextPit"] = l2n
        labels.loc[idx, "StopWithin3"] = (has_next & (l2n <= 3)).astype(np.int8)
        labels.loc[idx, "StopWithin5"] = (has_next & (l2n <= 5)).astype(np.int8)
        labels.loc[idx, "CensoredToFinish"] = (~has_next).astype(np.int8)
        labels.loc[idx, "HazardBucket"] = bucket

    return labels


def full_bucket_proba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), N_BUCKETS), dtype=np.float32)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, j]
    return out


train = clean_columns(pd.read_csv(INPUT / "train.csv.gz"))
test = clean_columns(pd.read_csv(INPUT / "test.csv.gz"))
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

train = add_sequence_features(train)
test = add_sequence_features(test)
aux = derive_auxiliary_labels(train)

y = train[TARGET].astype(int).to_numpy()
hazard_y = aux["HazardBucket"].astype(int).to_numpy()

features = [c for c in train.columns if c not in [ID, TARGET]]
cat_cols = [c for c in features if train[c].dtype == "object"]
for c in ["Year", "Stint", "Compound", "Race", "Driver"]:
    if c in features and c not in cat_cols:
        cat_cols.append(c)

for c in cat_cols:
    cats = pd.unique(pd.concat([train[c], test[c]], ignore_index=True).astype(str))
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

X = train[features]
X_test = test[features]
groups = (
    train["Year"].astype(str)
    + "|"
    + train["Race"].astype(str)
    + "|"
    + train["Driver"].astype(str)
).to_numpy()

if HAS_SGK:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X, y, groups))
else:
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(X, y, groups))

oof_main = np.zeros(len(train), dtype=np.float32)
oof_aux = np.zeros(len(train), dtype=np.float32)
test_main = np.zeros(len(test), dtype=np.float32)
test_aux = np.zeros(len(test), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    h_tr, h_va = hazard_y[tr_idx], hazard_y[va_idx]

    pos = max(int(y_tr.sum()), 1)
    neg = max(len(y_tr) - pos, 1)

    main_model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=90,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_lambda=2.0,
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE + fold,
        n_jobs=os.cpu_count() or 1,
        verbosity=-1,
    )
    main_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )

    hazard_model = LGBMClassifier(
        objective="multiclass",
        num_class=N_BUCKETS,
        metric="multi_logloss",
        class_weight="balanced",
        n_estimators=650,
        learning_rate=0.045,
        num_leaves=63,
        min_child_samples=90,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_lambda=2.0,
        random_state=RANDOM_STATE + 100 + fold,
        n_jobs=os.cpu_count() or 1,
        verbosity=-1,
    )
    hazard_model.fit(
        X_tr,
        h_tr,
        eval_set=[(X_va, h_va)],
        eval_metric="multi_logloss",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(70, verbose=False), log_evaluation(0)],
    )

    va_main = main_model.predict_proba(X_va)[:, 1]
    va_aux = full_bucket_proba(hazard_model, X_va)[:, 1]

    oof_main[va_idx] = va_main
    oof_aux[va_idx] = va_aux

    test_main += main_model.predict_proba(X_test)[:, 1] / N_SPLITS
    test_aux += full_bucket_proba(hazard_model, X_test)[:, 1] / N_SPLITS

    va_blend = np.clip(
        BLEND_MAIN_WEIGHT * va_main + (1 - BLEND_MAIN_WEIGHT) * va_aux, 0, 1
    )
    print(
        f"fold {fold}: main_auc={roc_auc_score(y_va, va_main):.6f} "
        f"hazard_auc={roc_auc_score(y_va, va_aux):.6f} "
        f"blend_auc={roc_auc_score(y_va, va_blend):.6f}"
    )

oof_pred = np.clip(
    BLEND_MAIN_WEIGHT * oof_main + (1 - BLEND_MAIN_WEIGHT) * oof_aux, 0, 1
)
test_pred = np.clip(
    BLEND_MAIN_WEIGHT * test_main + (1 - BLEND_MAIN_WEIGHT) * test_aux, 0, 1
)

cv_auc = roc_auc_score(y, oof_pred)
main_auc = roc_auc_score(y, oof_main)
aux_auc = roc_auc_score(y, oof_aux)

print(f"CV ROC AUC: {cv_auc:.6f}")
print(f"Direct model ROC AUC: {main_auc:.6f}")
print(f"Auxiliary hazard ROC AUC: {aux_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

test_pred_by_id = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: test_pred})
submission = sample[[ID]].merge(test_pred_by_id, on=ID, how="left")
submission[TARGET] = submission[TARGET].fillna(float(np.mean(oof_pred)))
submission.to_csv(WORK / "submission.csv", index=False)
submission.to_csv(WORK / "test_predictions.csv.gz", index=False, compression="gzip")

review = {
    "cv_roc_auc": float(cv_auc),
    "direct_model_roc_auc": float(main_auc),
    "auxiliary_hazard_roc_auc": float(aux_auc),
    "blend_main_weight": BLEND_MAIN_WEIGHT,
    "research_hypotheses_llm_claimed_used": ["000479"],
    "saved_files": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
        "./working/result_review.json",
    ],
}
with open(WORK / "result_review.json", "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review, indent=2))
