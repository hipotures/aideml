import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

H = 8
N_SPLITS = 5
SEED = 208

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

y = train[target_col].astype(int).values
train_features = train.drop(columns=[target_col]).copy()
test_features = test.copy()

all_features = pd.concat([train_features, test_features], axis=0, ignore_index=True)
cat_cols = all_features.select_dtypes(include=["object", "category"]).columns.tolist()

for c in cat_cols:
    all_features[c] = all_features[c].astype("category")

n_train = len(train_features)
X_base = all_features.iloc[:n_train].reset_index(drop=True)
X_test_base = all_features.iloc[n_train:].reset_index(drop=True)


def add_sequence_features(df):
    df = df.copy()
    df["_orig_row"] = np.arange(len(df))
    sort_cols = ["Year", "Race", "Driver", "LapNumber", "Stint"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    group_cols = [c for c in ["Year", "Race", "Driver"] if c in df.columns]
    g = (
        df.groupby(group_cols, observed=True, sort=False)
        if group_cols
        else [(None, df)]
    )

    df["seq_lap_index"] = 0
    df["seq_laps_total"] = 0
    df["seq_laps_left"] = 0
    df["pitstops_so_far"] = 0
    df["laps_since_pit"] = 0

    if group_cols:
        df["seq_lap_index"] = df.groupby(group_cols, observed=True).cumcount() + 1
        df["seq_laps_total"] = df.groupby(group_cols, observed=True)[
            "LapNumber"
        ].transform("size")
        df["seq_laps_left"] = df["seq_laps_total"] - df["seq_lap_index"]
        df["pitstops_so_far"] = (
            df.groupby(group_cols, observed=True)["PitStop"].cumsum() - df["PitStop"]
        )

        last_pit_lap = df["LapNumber"].where(df["PitStop"].eq(1))
        last_pit_lap = last_pit_lap.groupby(
            [df[c] for c in group_cols], observed=True
        ).ffill()
        df["laps_since_pit"] = (
            (df["LapNumber"] - last_pit_lap).fillna(df["LapNumber"]).clip(lower=0)
        )
    else:
        df["seq_lap_index"] = np.arange(len(df)) + 1
        df["seq_laps_total"] = len(df)
        df["seq_laps_left"] = len(df) - df["seq_lap_index"]
        df["pitstops_so_far"] = df["PitStop"].cumsum() - df["PitStop"]

    return (
        df.sort_values("_orig_row").drop(columns=["_orig_row"]).reset_index(drop=True)
    )


X_base = add_sequence_features(X_base)
X_test_base = add_sequence_features(X_test_base)


def laps_until_next_pit(df):
    tmp = df[["Year", "Race", "Driver", "LapNumber", "PitStop"]].copy()
    tmp["_orig_row"] = np.arange(len(tmp))
    sort_cols = ["Year", "Race", "Driver", "LapNumber"]
    tmp = tmp.sort_values(sort_cols).reset_index(drop=True)

    out = np.full(len(tmp), np.inf, dtype=float)
    group_cols = ["Year", "Race", "Driver"]

    for _, idx in tmp.groupby(group_cols, observed=True, sort=False).groups.items():
        pos = np.asarray(idx)
        pit_positions = np.where(tmp.loc[pos, "PitStop"].values == 1)[0]
        if len(pit_positions) == 0:
            continue
        for local_i in range(len(pos)):
            future = pit_positions[pit_positions > local_i]
            if len(future):
                out[pos[local_i]] = future[0] - local_i

    tmp["laps_to_event"] = out
    return tmp.sort_values("_orig_row")["laps_to_event"].values


train_lte = laps_until_next_pit(train)


def make_period_frame(base_df, laps_to_event=None, horizons=8):
    parts = []
    labels = []

    for h in range(1, horizons + 1):
        part = base_df.copy()
        part["horizon"] = h
        part["horizon_leq_tyrelife"] = (part["TyreLife"] >= h).astype(int)
        part["future_race_progress"] = (
            part["RaceProgress"]
            + h
            / np.maximum(part["LapNumber"] / np.maximum(part["RaceProgress"], 1e-6), 1)
        ).clip(0, 1.5)
        part["future_tyre_life"] = part["TyreLife"] + h
        part["future_lap_number"] = part["LapNumber"] + h
        parts.append(part)

        if laps_to_event is not None:
            event_now = laps_to_event == h
            at_risk = laps_to_event >= h
            labels.append(event_now[at_risk].astype(int))
            parts[-1] = part.loc[at_risk].copy()

    Xp = pd.concat(parts, axis=0, ignore_index=True)
    if laps_to_event is None:
        return Xp
    yp = np.concatenate(labels)
    return Xp, yp


def horizon1_frame(base_df):
    part = base_df.copy()
    part["horizon"] = 1
    part["horizon_leq_tyrelife"] = (part["TyreLife"] >= 1).astype(int)
    part["future_race_progress"] = (
        part["RaceProgress"]
        + 1 / np.maximum(part["LapNumber"] / np.maximum(part["RaceProgress"], 1e-6), 1)
    ).clip(0, 1.5)
    part["future_tyre_life"] = part["TyreLife"] + 1
    part["future_lap_number"] = part["LapNumber"] + 1
    return part


groups = (
    train["Year"].astype(str)
    + "|"
    + train["Race"].astype(str)
    + "|"
    + train["Driver"].astype(str)
).values

cv = GroupKFold(n_splits=N_SPLITS)
oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

model_params = dict(
    objective="binary",
    n_estimators=900,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

X_test_h1 = horizon1_frame(X_test_base)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_base, y, groups), 1):
    X_tr_base = X_base.iloc[tr_idx].reset_index(drop=True)
    X_va_h1 = horizon1_frame(X_base.iloc[va_idx].reset_index(drop=True))
    lte_tr = train_lte[tr_idx]

    X_tr_period, y_tr_period = make_period_frame(X_tr_base, lte_tr, H)

    model = LGBMClassifier(**model_params)
    model.fit(
        X_tr_period,
        y_tr_period,
        categorical_feature=cat_cols,
        eval_set=[(X_va_h1, y[va_idx])],
        eval_metric="auc",
        callbacks=[],
    )

    va_pred = model.predict_proba(X_va_h1)[:, 1]
    oof[va_idx] = va_pred
    score = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(score)
    print(f"fold {fold} auc: {score:.6f}")

    test_pred += model.predict_proba(X_test_h1)[:, 1] / N_SPLITS

cv_auc = roc_auc_score(y, oof)
print(f"mean fold auc: {np.mean(fold_scores):.6f}")
print(f"oof roc_auc: {cv_auc:.6f}")

submission = sample.copy()
submission[target_col] = np.clip(test_pred, 0, 1)
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

pd.DataFrame(
    {
        id_col: sample[id_col].values,
        target_col: np.clip(test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000208"],
    "validation_metric": "roc_auc",
    "validation_score": float(cv_auc),
    "fold_scores": [float(s) for s in fold_scores],
}
print(json.dumps(review, indent=2))
