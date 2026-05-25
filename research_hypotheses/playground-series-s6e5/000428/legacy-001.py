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
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

y = train[target_col].astype(int).values
train_ids = train[id_col].values
test_ids = sample[id_col].values

features = [c for c in train.columns if c not in [target_col, id_col]]
cat_cols = [c for c in features if train[c].dtype == "object"]

for c in cat_cols:
    vals = pd.concat([train[c], test[c]], axis=0).astype("category").cat.categories
    dtype = pd.CategoricalDtype(categories=vals)
    train[c] = train[c].astype(dtype)
    test[c] = test[c].astype(dtype)


def add_features(df):
    out = df.copy()
    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["TyreLife_x_Stint"] = out["TyreLife"] * out["Stint"]
    out["LateRace"] = (out["RaceProgress"] > 0.65).astype(int)
    out["LongTyreLife"] = (out["TyreLife"] >= 18).astype(int)
    out["WetOrInter"] = (
        out["Compound"].astype(str).isin(["WET", "INTERMEDIATE"]).astype(int)
    )
    out["RecentPitLap"] = out["PitStop"].astype(int)
    out["AbsPositionChange"] = out["Position_Change"].abs()
    out["LapTimeDeltaAbs"] = out["LapTime_Delta"].abs()
    out["DegradationPerTyreLap"] = out["Cumulative_Degradation"] / np.maximum(
        out["TyreLife"], 1
    )
    return out


train_fe = add_features(train[features])
test_fe = add_features(test[features])
model_features = list(train_fe.columns)
cat_model_cols = [c for c in cat_cols if c in model_features]

groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
n_splits = 5
cv = GroupKFold(n_splits=n_splits)

oof_stage1 = np.zeros(len(train))
oof_stage2 = np.zeros(len(train))
oof_final = np.zeros(len(train))
test_stage1 = np.zeros(len(test))
test_stage2_sum = np.zeros(len(test))
test_stage2_count = np.zeros(len(test))

base_params = dict(
    objective="binary",
    n_estimators=900,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=428,
    n_jobs=-1,
    verbose=-1,
)

stage2_params = dict(
    objective="binary",
    n_estimators=700,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=35,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=10428,
    n_jobs=-1,
    verbose=-1,
)


def candidate_window(df, stage1_pred=None, train_mode=False):
    tyre = df["TyreLife"].values
    prog = df["RaceProgress"].values
    stint = df["Stint"].values
    wet = df["Compound"].astype(str).isin(["WET", "INTERMEDIATE"]).values
    base = (
        ((tyre >= 8) & (prog >= 0.10))
        | ((tyre >= 14) & (stint >= 1))
        | ((tyre >= 5) & wet)
        | ((prog >= 0.55) & (tyre >= 6))
    )
    if stage1_pred is not None:
        if train_mode:
            hard = stage1_pred >= np.quantile(stage1_pred, 0.78)
        else:
            hard = stage1_pred >= np.quantile(stage1_pred, 0.75)
        base = base | hard
    return base


for fold, (tr_idx, va_idx) in enumerate(cv.split(train_fe, y, groups), 1):
    X_tr, X_va = train_fe.iloc[tr_idx], train_fe.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    pos_weight = max(1.0, (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
    m1 = LGBMClassifier(**base_params, scale_pos_weight=pos_weight)
    m1.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_model_cols,
        callbacks=[],
    )

    p1_va = m1.predict_proba(X_va)[:, 1]
    p1_tr = m1.predict_proba(X_tr)[:, 1]
    p1_te = m1.predict_proba(test_fe)[:, 1]

    oof_stage1[va_idx] = p1_va
    test_stage1 += p1_te / n_splits

    cand_tr = candidate_window(X_tr, p1_tr, train_mode=True)
    cand_va = candidate_window(X_va, p1_va, train_mode=False)
    cand_te = candidate_window(test_fe, p1_te, train_mode=False)

    cand_tr = cand_tr | (y_tr == 1)
    hard_negative = (y_tr == 0) & (p1_tr >= np.quantile(p1_tr[y_tr == 0], 0.85))
    stage2_rows = cand_tr | hard_negative

    X2_tr = X_tr.loc[stage2_rows]
    y2_tr = y_tr[stage2_rows]

    if y2_tr.sum() > 0 and len(np.unique(y2_tr)) == 2:
        stage2_weight = max(1.0, 2.5 * (len(y2_tr) - y2_tr.sum()) / max(1, y2_tr.sum()))
        m2 = LGBMClassifier(**stage2_params, scale_pos_weight=stage2_weight)
        m2.fit(
            X2_tr,
            y2_tr,
            eval_set=(
                [(X_va.loc[cand_va], y_va[cand_va])]
                if cand_va.sum() > 20 and len(np.unique(y_va[cand_va])) == 2
                else None
            ),
            eval_metric="auc",
            categorical_feature=cat_model_cols,
            callbacks=[],
        )

        p2_va = p1_va.copy()
        p2_te = p1_te.copy()
        p2_va[cand_va] = m2.predict_proba(X_va.loc[cand_va])[:, 1]
        p2_te[cand_te] = m2.predict_proba(test_fe.loc[cand_te])[:, 1]

        oof_stage2[va_idx] = p2_va
        test_stage2_sum += p2_te
        test_stage2_count += 1
    else:
        oof_stage2[va_idx] = p1_va
        test_stage2_sum += p1_te
        test_stage2_count += 1

    oof_final[va_idx] = p1_va
    oof_final[va_idx][cand_va] = (
        0.55 * p1_va[cand_va] + 0.45 * oof_stage2[va_idx][cand_va]
    )

    fold_auc = roc_auc_score(y_va, oof_final[va_idx])
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

test_stage2 = test_stage2_sum / np.maximum(test_stage2_count, 1)
test_candidate = candidate_window(test_fe, test_stage1, train_mode=False)
test_pred = test_stage1.copy()
test_pred[test_candidate] = (
    0.55 * test_stage1[test_candidate] + 0.45 * test_stage2[test_candidate]
)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

cv_auc = roc_auc_score(y, oof_final)
print(f"OOF ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {"row": np.arange(len(train)), "target": y, "prediction": oof_final}
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame({id_col: test_ids, target_col: test_pred}).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "validation_score": float(cv_auc),
            "research_hypotheses_llm_claimed_used": ["000428"],
            "files": {
                "submission": os.path.join(WORKING_DIR, "submission.csv"),
                "oof_predictions": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
                "test_predictions": os.path.join(
                    WORKING_DIR, "test_predictions.csv.gz"
                ),
            },
        }
    )
)
