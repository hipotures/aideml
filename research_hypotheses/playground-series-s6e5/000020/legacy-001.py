import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESES_USED = ["000020"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_rule_pressure_features(df):
    df = df.copy()

    dry = df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(int)
    wet = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    remaining_laps = (df["LapNumber"] / df["RaceProgress"]).replace(
        [np.inf, -np.inf], np.nan
    ) - df["LapNumber"]
    remaining_laps = remaining_laps.fillna(
        df["LapNumber"].max() - df["LapNumber"]
    ).clip(lower=0)

    tyre_life = df["TyreLife"].clip(lower=1)
    race_progress = df["RaceProgress"].clip(0, 1)

    # Physically finishable proxy: dry tyres generally get risky after a long stint,
    # while wet/inter tyres are treated more permissively because rules differ.
    dry_max_life = df["Compound"].map({"SOFT": 22, "MEDIUM": 32, "HARD": 42}).fillna(38)
    wet_max_life = df["Compound"].map({"INTERMEDIATE": 34, "WET": 30}).fillna(34)
    effective_max_life = np.where(dry.eq(1), dry_max_life, wet_max_life)

    finish_margin = effective_max_life - tyre_life - remaining_laps
    can_finish_current_tyre = (finish_margin >= 0).astype(int)

    # Legal-pressure proxies after the current row.
    # In dry races, a car usually needs at least two dry compounds or a stop.
    wet_exemption = wet.astype(int)
    observed_stop_debt = (
        (df["PitStop"] == 0) & (df["Stint"] <= 1) & (race_progress > 0.18)
    ).astype(int)
    dry_compound_debt = ((dry == 1) & (df["Stint"] <= 1) & (wet_exemption == 0)).astype(
        int
    )

    can_finish_but_owes_dry = (
        (can_finish_current_tyre == 1) & (dry_compound_debt == 1)
    ).astype(int)
    can_finish_but_owes_stop = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1)
    ).astype(int)
    late_race_legal_pressure = (
        (can_finish_current_tyre == 1)
        & ((dry_compound_debt == 1) | (observed_stop_debt == 1))
        & (race_progress >= 0.65)
    ).astype(int)

    df["RemainingLaps_est"] = remaining_laps
    df["TyreFinishLifeLimit_est"] = effective_max_life
    df["CurrentTyreFinishMargin"] = finish_margin
    df["CanFinishCurrentTyre"] = can_finish_current_tyre
    df["WetRuleExemptionProxy"] = wet_exemption
    df["DryCompoundDebtProxy"] = dry_compound_debt
    df["ObservedStopDebtProxy"] = observed_stop_debt
    df["CanFinishButOwesDryCompound"] = can_finish_but_owes_dry
    df["CanFinishButOwesStop"] = can_finish_but_owes_stop
    df["LateRaceLegalPressure"] = late_race_legal_pressure
    df["LegalPressureScore"] = (
        1.0 * can_finish_but_owes_dry
        + 0.8 * can_finish_but_owes_stop
        + 1.3 * late_race_legal_pressure
        - 0.4 * wet_exemption
    )
    df["FinishMargin_x_LegalPressure"] = finish_margin * df["LegalPressureScore"]
    df["LateRace_x_DryDebt"] = (race_progress >= 0.65).astype(int) * dry_compound_debt
    df["RemainingLaps_x_Debt"] = remaining_laps * (
        dry_compound_debt + observed_stop_debt
    )
    return df


full = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
full_fe = add_rule_pressure_features(full)

X = full_fe.iloc[: len(train)].copy()
X_test = full_fe.iloc[len(train) :].copy()
y = train[TARGET].astype(int).values

drop_cols = [ID_COL]
features = [c for c in X.columns if c not in drop_cols]

cat_cols = [c for c in features if X[c].dtype == "object"]
for c in cat_cols:
    combined = pd.concat([X[c], X_test[c]], axis=0).astype("category")
    X[c] = pd.Categorical(X[c], categories=combined.cat.categories)
    X_test[c] = pd.Categorical(X_test[c], categories=combined.cat.categories)

groups = (train["Year"].astype(str) + "_" + train["Race"].astype(str)).values
n_group_classes = pd.Series(groups).nunique()
if n_group_classes >= 5:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = splitter.split(X, y, groups)
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = splitter.split(X, y)

try:
    from lightgbm import LGBMClassifier

    model_factory = lambda: LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=max(1, os.cpu_count() or 1),
        verbose=-1,
    )
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder

    num_cols = [c for c in features if c not in cat_cols]
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ]
    )
    model_factory = lambda: make_pipeline(
        pre,
        HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=42,
        ),
    )

oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = model_factory()
    if "lightgbm" in str(type(model)).lower():
        model.fit(
            X.iloc[tr_idx][features],
            y[tr_idx],
            categorical_feature=cat_cols,
            eval_set=[(X.iloc[va_idx][features], y[va_idx])],
            eval_metric="auc",
        )
    else:
        model.fit(X.iloc[tr_idx][features], y[tr_idx])

    va_pred = model.predict_proba(X.iloc[va_idx][features])[:, 1]
    te_pred = model.predict_proba(X_test[features])[:, 1]

    oof[va_idx] = va_pred
    test_pred += te_pred / 5.0

    auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(auc)
    print(f"fold {fold} roc_auc: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"mean_fold_roc_auc: {np.mean(fold_scores):.6f}")
print(f"oof_roc_auc: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: sample[ID_COL].values,
        TARGET: np.clip(test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "oof_roc_auc": float(cv_auc),
            "mean_fold_roc_auc": float(np.mean(fold_scores)),
            "research_hypotheses_llm_claimed_used": HYPOTHESES_USED,
            "files_written": [
                "./working/submission.csv",
                "./working/oof_predictions.csv.gz",
                "./working/test_predictions.csv.gz",
            ],
        },
        indent=2,
    )
)
