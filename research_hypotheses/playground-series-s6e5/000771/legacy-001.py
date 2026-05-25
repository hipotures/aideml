import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise RuntimeError("lightgbm is required for this script") from e

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESIS_ID = "000771"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values
test_ids = sample[ID_COL].values


def add_features(df):
    out = df.copy()
    race_str = out["Race"].astype(str)
    comp_str = out["Compound"].astype(str)

    out["is_monaco_2025"] = (
        (out["Year"].astype(int) == 2025)
        & race_str.str.contains("Monaco", case=False, na=False)
    ).astype(int)
    out["year_race"] = out["Year"].astype(str) + "_" + race_str

    out["tyre_x_progress"] = out["TyreLife"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["degradation_per_tyre"] = out["Cumulative_Degradation"] / (
        out["TyreLife"] + 1.0
    )
    out["late_race"] = (out["RaceProgress"] >= 0.70).astype(int)
    out["old_tyre"] = (out["TyreLife"] >= 20).astype(int)
    out["wet_or_inter"] = comp_str.isin(["WET", "INTERMEDIATE"]).astype(int)

    out["m25_position"] = out["is_monaco_2025"] * out["Position"]
    out["m25_stint"] = out["is_monaco_2025"] * out["Stint"]
    out["m25_progress"] = out["is_monaco_2025"] * out["RaceProgress"]
    out["m25_tyre_life"] = out["is_monaco_2025"] * out["TyreLife"]
    out["m25_tyre_pressure"] = out["is_monaco_2025"] * out["tyre_x_progress"]
    out["m25_soft"] = out["is_monaco_2025"] * (comp_str == "SOFT").astype(int)
    out["m25_medium"] = out["is_monaco_2025"] * (comp_str == "MEDIUM").astype(int)
    out["m25_hard"] = out["is_monaco_2025"] * (comp_str == "HARD").astype(int)
    return out


train_fe = add_features(train.drop(columns=[TARGET]))
test_fe = add_features(test)

cat_cols = ["Driver", "Race", "Compound"]
num_cols = [c for c in train_fe.columns if c not in [ID_COL, "year_race"] + cat_cols]
feature_cols = num_cols + cat_cols

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
combined_cat = pd.concat([train_fe[cat_cols], test_fe[cat_cols]], axis=0).astype(str)
enc.fit(combined_cat)

train_x = train_fe[feature_cols].copy()
test_x = test_fe[feature_cols].copy()
train_x[cat_cols] = enc.transform(train_x[cat_cols].astype(str))
test_x[cat_cols] = enc.transform(test_x[cat_cols].astype(str))

groups = train_fe["year_race"].values
gkf = GroupKFold(n_splits=5)

oof = np.zeros(len(train_x), dtype=float)
test_pred = np.zeros(len(test_x), dtype=float)

special_cols = [
    "Position",
    "Stint",
    "RaceProgress",
    "TyreLife",
    "Cumulative_Degradation",
    "degradation_per_tyre",
    "tyre_x_progress",
    "position_x_progress",
    "m25_position",
    "m25_stint",
    "m25_progress",
    "m25_tyre_life",
    "m25_tyre_pressure",
    "m25_soft",
    "m25_medium",
    "m25_hard",
]
m25_test_mask = test_fe["is_monaco_2025"].values.astype(bool)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_x, y, groups), 1):
    x_tr, x_va = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    base = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.6,
        reg_lambda=3.0,
        random_state=2026 + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    base.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[],
    )

    va_pred = base.predict_proba(x_va)[:, 1]
    te_pred = base.predict_proba(test_x)[:, 1]

    m25_tr_mask = train_fe.iloc[tr_idx]["is_monaco_2025"].values.astype(bool)
    m25_va_mask = train_fe.iloc[va_idx]["is_monaco_2025"].values.astype(bool)

    if m25_tr_mask.sum() >= 20 and len(np.unique(y_tr[m25_tr_mask])) == 2:
        specialist = LogisticRegression(
            C=0.25,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=2026 + fold,
        )
        specialist.fit(x_tr.loc[m25_tr_mask, special_cols], y_tr[m25_tr_mask])

        if m25_va_mask.any():
            sp_va = specialist.predict_proba(x_va.loc[m25_va_mask, special_cols])[:, 1]
            va_pred[m25_va_mask] = 0.75 * va_pred[m25_va_mask] + 0.25 * sp_va

        if m25_test_mask.any():
            sp_te = specialist.predict_proba(test_x.loc[m25_test_mask, special_cols])[
                :, 1
            ]
            te_pred[m25_test_mask] = 0.75 * te_pred[m25_test_mask] + 0.25 * sp_te

    oof[va_idx] = va_pred
    test_pred += te_pred / gkf.n_splits
    print(f"fold {fold} auc: {roc_auc_score(y_va, va_pred):.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"GroupKFold Year_Race ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(oof, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: np.clip(test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

with open(os.path.join(WORK_DIR, "review.json"), "w") as f:
    json.dump(
        {
            "research_hypotheses_llm_claimed_used": [HYPOTHESIS_ID],
            "metric": "roc_auc",
            "validation": "5-fold GroupKFold grouped by Year_Race",
            "cv_auc": float(cv_auc),
        },
        f,
        indent=2,
    )
