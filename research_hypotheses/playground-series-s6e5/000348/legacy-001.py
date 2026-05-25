import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]
GROUP_COLS = ["Year", "Race", "Driver"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values


def add_time_to_next_pit(df, is_train=True):
    df = df.copy()
    df["_orig_row"] = np.arange(len(df))
    df = df.sort_values(GROUP_COLS + ["LapNumber", ID_COL]).reset_index(drop=True)

    lower = np.zeros(len(df), dtype=float)
    upper = np.zeros(len(df), dtype=float)

    for _, idx in df.groupby(GROUP_COLS, sort=False).groups.items():
        idx = np.asarray(list(idx))
        laps = df.loc[idx, "LapNumber"].to_numpy()
        pit = df.loc[idx, "PitStop"].to_numpy()
        max_lap = laps.max()
        pit_laps = laps[pit == 1]

        for j, row_idx in enumerate(idx):
            future = pit_laps[pit_laps > laps[j]]
            if len(future):
                t = float(future[0] - laps[j])
                lower[row_idx] = max(t, 1.0)
                upper[row_idx] = max(t, 1.0)
            else:
                censor = float(max(max_lap - laps[j] + 1, 1))
                lower[row_idx] = censor
                upper[row_idx] = np.inf

    df["aft_lower"] = lower
    df["aft_upper"] = upper
    return (
        df.sort_values("_orig_row").drop(columns=["_orig_row"]).reset_index(drop=True)
    )


def make_features(train_df, test_df):
    full = pd.concat(
        [train_df.drop(columns=[TARGET], errors="ignore"), test_df],
        axis=0,
        ignore_index=True,
    )

    full["race_driver"] = full["Race"].astype(str) + "_" + full["Driver"].astype(str)
    full["year_race"] = full["Year"].astype(str) + "_" + full["Race"].astype(str)
    full["tyre_frac"] = full["TyreLife"] / (full["LapNumber"].clip(lower=1))
    full["laps_remaining_est"] = (
        full["LapNumber"] / full["RaceProgress"].clip(0.01, 1.0)
    ) - full["LapNumber"]
    full["degradation_per_lap"] = full["Cumulative_Degradation"] / full[
        "TyreLife"
    ].clip(lower=1)
    full["is_wet_compound"] = full["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    full["recent_pit_context"] = (
        (full["PitStop"] == 1) | (full["TyreLife"] <= 2)
    ).astype(int)

    cat_cols = CAT_COLS + ["race_driver", "year_race"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    full[cat_cols] = enc.fit_transform(full[cat_cols].astype(str))

    drop_cols = [ID_COL]
    features = [c for c in full.columns if c not in drop_cols]
    full[features] = full[features].replace([np.inf, -np.inf], np.nan).fillna(-999)

    X_train = full.iloc[: len(train_df)][features].reset_index(drop=True)
    X_test = full.iloc[len(train_df) :][features].reset_index(drop=True)
    return X_train, X_test, features


train_aft = add_time_to_next_pit(train)
test_aft = add_time_to_next_pit(test, is_train=False)

X, X_test, features = make_features(train, test)
groups = train[GROUP_COLS].astype(str).agg("|".join, axis=1).values

oof_cls = np.zeros(len(train))
oof_aft = np.zeros(len(train))
test_cls = np.zeros(len(test))
test_aft = np.zeros(len(test))

gkf = GroupKFold(n_splits=5)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_depth=5,
        learning_rate=0.045,
        n_estimators=900,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_lambda=2.0,
        random_state=2026 + fold,
        n_jobs=max(1, os.cpu_count() or 1),
        early_stopping_rounds=60,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_cls[va_idx] = clf.predict_proba(X_va)[:, 1]
    test_cls += clf.predict_proba(X_test)[:, 1] / 5

    aft = XGBRegressor(
        objective="survival:aft",
        eval_metric="aft-nloglik",
        tree_method="hist",
        aft_loss_distribution="normal",
        aft_loss_distribution_scale=1.4,
        max_depth=4,
        learning_rate=0.04,
        n_estimators=750,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=12,
        reg_lambda=3.0,
        random_state=4026 + fold,
        n_jobs=max(1, os.cpu_count() or 1),
    )
    aft.fit(
        X_tr,
        train_aft.loc[tr_idx, "aft_lower"].values,
        sample_weight=np.ones(len(tr_idx)),
        xgb_model=None,
        verbose=False,
        y_lower_bound=train_aft.loc[tr_idx, "aft_lower"].values,
        y_upper_bound=train_aft.loc[tr_idx, "aft_upper"].values,
    )

    pred_time_va = np.clip(aft.predict(X_va), 0.25, 100)
    pred_time_te = np.clip(aft.predict(X_test), 0.25, 100)

    # Exponential survival approximation: next-lap hazard = 1 - S(t+1)/S(t).
    aft_prob_va = np.clip(1.0 - np.exp(-1.0 / pred_time_va), 0.0005, 0.9995)
    aft_prob_te = np.clip(1.0 - np.exp(-1.0 / pred_time_te), 0.0005, 0.9995)

    oof_aft[va_idx] = aft_prob_va
    test_aft += aft_prob_te / 5

    fold_blend = 0.75 * oof_cls[va_idx] + 0.25 * oof_aft[va_idx]
    print(f"fold {fold} roc_auc={roc_auc_score(y_va, fold_blend):.6f}")

blend_weights = np.linspace(0.0, 1.0, 21)
scores = []
for w in blend_weights:
    pred = w * oof_cls + (1.0 - w) * oof_aft
    scores.append(roc_auc_score(y, pred))

best_i = int(np.argmax(scores))
best_w = float(blend_weights[best_i])
best_auc = float(scores[best_i])

oof_pred = best_w * oof_cls + (1.0 - best_w) * oof_aft
test_pred = best_w * test_cls + (1.0 - best_w) * test_aft
test_pred = np.clip(test_pred, 0.0005, 0.9995)

print(f"cv_roc_auc={best_auc:.6f}")
print(f"best_classifier_weight={best_w:.2f}")

submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred})
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: test_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000348"],
    "metric": "roc_auc",
    "cv_roc_auc": best_auc,
    "best_classifier_weight": best_w,
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)
