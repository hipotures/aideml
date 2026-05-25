import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["_is_train"] = 1
test["_is_train"] = 0
test[TARGET] = np.nan
all_df = pd.concat([train, test], axis=0, ignore_index=True)


def add_legality_state_machine(df):
    df = df.copy()
    sort_cols = ["Year", "Race", "Driver", "LapNumber", ID_COL]
    df["_orig_order"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    used_dry_count = np.zeros(len(df), dtype=np.int16)
    remaining_required = np.zeros(len(df), dtype=np.int16)
    current_already_used = np.zeros(len(df), dtype=np.int8)
    stop_debt_if_dry = np.zeros(len(df), dtype=np.int8)
    wet_exemption_flag = np.zeros(len(df), dtype=np.int8)
    laps_remaining_illegal = np.zeros(len(df), dtype=np.float32)
    is_wet_weather = df["Compound"].isin(WET_COMPOUNDS).astype(np.int8).to_numpy()

    if "RaceProgress" in df.columns:
        laps_remaining_est = (
            (
                (1.0 - df["RaceProgress"].clip(0, 1))
                * df["LapNumber"]
                / df["RaceProgress"].clip(0.01, 1)
            )
            .fillna(0)
            .to_numpy()
        )
    else:
        max_laps = df.groupby(["Year", "Race"])["LapNumber"].transform("max")
        laps_remaining_est = (max_laps - df["LapNumber"]).clip(lower=0).to_numpy()

    for _, idx in df.groupby(["Year", "Race", "Driver"], sort=False).indices.items():
        seen_dry = set()
        seen_wet = False
        for i in idx:
            comp = str(df.at[i, "Compound"])
            dry_before = set(seen_dry)
            wet_now = comp in WET_COMPOUNDS
            dry_now = comp in DRY_COMPOUNDS

            current_already_used[i] = int(dry_now and comp in dry_before)
            if dry_now:
                seen_dry.add(comp)
            if wet_now:
                seen_wet = True

            used_dry_count[i] = len(seen_dry)
            wet_exemption_flag[i] = int(seen_wet)
            remaining_required[i] = 0 if seen_wet else max(0, 2 - len(seen_dry))
            stop_debt_if_dry[i] = int((not seen_wet) and len(seen_dry) < 2)
            laps_remaining_illegal[i] = (
                laps_remaining_est[i] if stop_debt_if_dry[i] else 0.0
            )

    df["Is_WetWeather"] = is_wet_weather
    df["used_dry_set"] = used_dry_count
    df["remaining_required_dry_count"] = remaining_required
    df["current_compound_already_used"] = current_already_used
    df["stop_debt_if_dry"] = stop_debt_if_dry
    df["wet_exemption_flag"] = wet_exemption_flag
    df["laps_remaining_when_still_illegal"] = laps_remaining_illegal
    df["LapsRemaining_Est"] = laps_remaining_est.astype(np.float32)
    df["dry_legality_pressure"] = df["stop_debt_if_dry"] * np.log1p(
        df["LapsRemaining_Est"].clip(lower=0)
    )
    df["illegal_late_race"] = df["stop_debt_if_dry"] * (
        df["LapsRemaining_Est"] <= 8
    ).astype(np.int8)
    df["reused_dry_late"] = df["current_compound_already_used"] * (
        df["LapsRemaining_Est"] <= 15
    ).astype(np.int8)
    df["dry_debt_x_tyre_life"] = df["stop_debt_if_dry"] * df["TyreLife"].astype(float)
    df["dry_debt_x_race_progress"] = df["stop_debt_if_dry"] * df["RaceProgress"].astype(
        float
    )
    df["wet_exempt_x_dry_compound"] = df["wet_exemption_flag"] * df["Compound"].isin(
        DRY_COMPOUNDS
    ).astype(np.int8)

    return (
        df.sort_values("_orig_order")
        .drop(columns=["_orig_order"])
        .reset_index(drop=True)
    )


all_df = add_legality_state_machine(all_df)

train_fe = all_df[all_df["_is_train"] == 1].copy()
test_fe = all_df[all_df["_is_train"] == 0].copy()

base_drop = [TARGET, "_is_train"]
features = [c for c in train_fe.columns if c not in base_drop]

for df in (train_fe, test_fe):
    df["RaceDriver"] = (
        df["Year"].astype(str)
        + "_"
        + df["Race"].astype(str)
        + "_"
        + df["Driver"].astype(str)
    )
    df["RaceYear"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)

features = [c for c in features if c != ID_COL] + ["RaceDriver", "RaceYear"]

X = train_fe[features].copy()
y = train_fe[TARGET].astype(int).to_numpy()
X_test = test_fe[features].copy()

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    combined = pd.concat([X[c], X_test[c]], axis=0).astype("category")
    cats = combined.cat.categories
    X[c] = pd.Categorical(X[c], categories=cats)
    X_test[c] = pd.Categorical(X_test[c], categories=cats)

num_cols = [c for c in X.columns if c not in cat_cols]
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

try:
    from lightgbm import LGBMClassifier

    model_factory = lambda: LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=316,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_kwargs = {"categorical_feature": cat_cols}
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline

    cat_idx = cat_cols
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_idx,
            )
        ],
        remainder="passthrough",
    )
    model_factory = lambda: make_pipeline(
        pre,
        HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=48,
            l2_regularization=0.05,
            random_state=316,
        ),
    )
    fit_kwargs = {}

groups = train_fe["RaceYear"].astype(str).to_numpy()
if len(np.unique(groups)) >= 5:
    splitter = GroupKFold(n_splits=5).split(X, y, groups)
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=316).split(X, y)

oof = np.zeros(len(X), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(splitter, 1):
    model = model_factory()
    if fit_kwargs:
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            **fit_kwargs,
        )
    else:
        model.fit(X.iloc[tr_idx], y[tr_idx])

    val_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = val_pred
    test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32) / 5.0
    score = roc_auc_score(y[va_idx], val_pred)
    fold_scores.append(score)
    print(f"fold {fold} roc_auc={score:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"oof_roc_auc={cv_auc:.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000316"],
            "metric": "roc_auc",
            "oof_roc_auc": float(cv_auc),
        }
    )
)

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: sample[ID_COL].to_numpy(),
        TARGET: np.clip(test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
