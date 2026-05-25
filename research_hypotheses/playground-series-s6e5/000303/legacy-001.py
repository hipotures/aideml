import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESIS_ID = "000303"


def clean_columns(df):
    out = df.copy()
    out.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c).strip("_") for c in out.columns]
    return out


def add_prev_pit_wave_features(df):
    df = df.copy()
    df["_row_order"] = np.arange(len(df))
    lap_keys = ["Year", "Race", "LapNumber"]

    lap_stats = (
        df.groupby(lap_keys, observed=True)["PitStop"]
        .agg(lap_pit_count="sum", lap_car_count="size", lap_pit_rate="mean")
        .reset_index()
    )
    prev_lap = lap_stats.copy()
    prev_lap["LapNumber"] += 1
    prev_lap = prev_lap.rename(
        columns={
            "lap_pit_count": "prev_lap_pit_count",
            "lap_car_count": "prev_lap_car_count",
            "lap_pit_rate": "prev_lap_pit_rate",
        }
    )
    df = df.merge(prev_lap, on=lap_keys, how="left", sort=False)

    comp_keys = lap_keys + ["Compound"]
    comp_stats = (
        df.groupby(comp_keys, observed=True)["PitStop"]
        .agg(comp_pit_count="sum", comp_car_count="size", comp_pit_rate="mean")
        .reset_index()
    )
    prev_comp = comp_stats.copy()
    prev_comp["LapNumber"] += 1
    prev_comp = prev_comp.rename(
        columns={
            "comp_pit_count": "prev_compound_pit_count",
            "comp_car_count": "prev_compound_car_count",
            "comp_pit_rate": "prev_compound_pit_rate",
        }
    )
    df = df.merge(prev_comp, on=comp_keys, how="left", sort=False)

    df = df.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
    return df


def add_compound_peer_ranks(df):
    df = df.copy()
    keys = ["Year", "Race", "LapNumber", "Compound"]
    for col in ["TyreLife", "Cumulative_Degradation", "LapTime_Delta"]:
        g = df.groupby(keys, observed=True)[col]
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        df[f"{col}_compound_rank_pct"] = g.rank(method="average", pct=True).astype(
            "float32"
        )
        df[f"{col}_compound_centered"] = (df[col] - mean).astype("float32")
        df[f"{col}_compound_z"] = ((df[col] - mean) / std).astype("float32")
    return df


def add_position_pressure_features(df):
    df = df.copy()
    lap_keys = ["Year", "Race", "LapNumber"]
    pieces = []
    names = [
        "nearby_older_tyre_count",
        "ahead_older_tyre_count",
        "behind_older_tyre_count",
        "ahead_slow_older_count",
        "rejoin_traffic_count",
        "rejoin_older_tyre_count",
        "nearby_mean_tyrelife_diff",
        "rejoin_mean_laptime_delta",
        "clean_air_pressure_proxy",
    ]

    for _, g in df.groupby(lap_keys, sort=False, observed=True):
        idx = g.index
        pos = g["Position"].to_numpy(np.float32)
        tyre = g["TyreLife"].to_numpy(np.float32)
        ltd = g["LapTime_Delta"].to_numpy(np.float32)

        dpos = pos[None, :] - pos[:, None]
        rivals = dpos != 0
        nearby = rivals & (np.abs(dpos) <= 3)
        ahead = (dpos < 0) & (dpos >= -3)
        behind = (dpos > 0) & (dpos <= 3)
        rejoin = (dpos > 0) & (dpos <= 6)
        older = tyre[None, :] > tyre[:, None]
        slower = ltd[None, :] > ltd[:, None]

        near_count = nearby.sum(axis=1)
        rejoin_count = rejoin.sum(axis=1)
        tyre_diff = tyre[None, :] - tyre[:, None]

        nearby_mean_tyre = np.divide(
            (tyre_diff * nearby).sum(axis=1),
            near_count,
            out=np.zeros(len(g), dtype=np.float32),
            where=near_count > 0,
        )
        rejoin_mean_ltd = np.divide(
            (ltd[None, :] * rejoin).sum(axis=1),
            rejoin_count,
            out=np.zeros(len(g), dtype=np.float32),
            where=rejoin_count > 0,
        )

        ahead_slow_older = (ahead & older & slower).sum(axis=1).astype(np.float32)
        rejoin_traffic = rejoin_count.astype(np.float32)

        vals = np.vstack(
            [
                (nearby & older).sum(axis=1),
                (ahead & older).sum(axis=1),
                (behind & older).sum(axis=1),
                ahead_slow_older,
                rejoin_traffic,
                (rejoin & older).sum(axis=1),
                nearby_mean_tyre,
                rejoin_mean_ltd,
                ahead_slow_older - rejoin_traffic,
            ]
        ).T.astype("float32")

        pieces.append(pd.DataFrame(vals, index=idx, columns=names))

    return pd.concat([df, pd.concat(pieces).sort_index()], axis=1)


def make_features(train, test):
    n_train = len(train)
    all_df = pd.concat(
        [train.drop(columns=[TARGET]), test], axis=0, ignore_index=True, sort=False
    )

    all_df = add_prev_pit_wave_features(all_df)
    all_df = add_compound_peer_ranks(all_df)
    all_df = add_position_pressure_features(all_df)

    num_cols = all_df.select_dtypes(include=[np.number]).columns
    all_df[num_cols] = all_df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    cat_cols = [c for c in ["Compound", "Driver", "Race"] if c in all_df.columns]
    for c in cat_cols:
        all_df[c] = all_df[c].astype("category")

    features = [c for c in all_df.columns if c != ID_COL]
    return all_df.iloc[:n_train][features], all_df.iloc[n_train:][features], cat_cols


train = clean_columns(pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz")))
test = clean_columns(pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz")))
sample = clean_columns(pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz")))

y = train[TARGET].astype(int).to_numpy()
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

X, X_test, cat_cols = make_features(train, test)

try:
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X, y, groups))
except Exception:
    from sklearn.model_selection import GroupKFold

    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(X, y, groups))

import lightgbm as lgb


def build_model(y_train, n_estimators=1800):
    pos = max(1, int(y_train.sum()))
    neg = max(1, len(y_train) - pos)
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=64,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=3.0,
        scale_pos_weight=neg / pos,
        random_state=42,
        n_jobs=max(1, min(8, os.cpu_count() or 1)),
        verbosity=-1,
    )


oof = np.zeros(len(train), dtype=np.float32)
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = build_model(y[tr_idx])
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    best_iters.append(model.best_iteration_ or model.n_estimators)
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold StratifiedGroup CV ROC AUC: {cv_auc:.6f}")

final_estimators = int(np.clip(np.mean(best_iters), 100, 1800))
final_model = build_model(y, n_estimators=final_estimators)
final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0, 1)

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "cv_roc_auc": float(cv_auc),
    "research_hypotheses_llm_claimed_used": [HYPOTHESIS_ID],
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
with open(os.path.join(WORK_DIR, "review.json"), "w") as f:
    json.dump(result, f, indent=2)
