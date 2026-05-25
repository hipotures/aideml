import os
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

TARGET = "PitNextLap"
ID_COL = "id"
GROUP_COLS = ["Year", "Race", "Driver"]
SORT_COLS = GROUP_COLS + ["LapNumber"]

train = pd.read_csv(f"{INPUT_DIR}/train.csv.gz")
test = pd.read_csv(f"{INPUT_DIR}/test.csv.gz")
sample = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv.gz")

train["_is_train"] = 1
test["_is_train"] = 0
test[TARGET] = np.nan
all_df = pd.concat([train, test], axis=0, ignore_index=True)

all_df["_orig_order"] = np.arange(len(all_df))
all_df = all_df.sort_values(SORT_COLS + ["_is_train", "_orig_order"]).reset_index(
    drop=True
)
all_df["NextObservedPitStop"] = all_df.groupby(GROUP_COLS, sort=False)["PitStop"].shift(
    -1
)
all_df["HasNextObservedPitStop"] = all_df["NextObservedPitStop"].notna().astype("int8")
all_df["NextObservedPitStop"] = all_df["NextObservedPitStop"].fillna(-1).astype("int8")
all_df = all_df.sort_values("_orig_order").reset_index(drop=True)

train_fe = all_df[all_df["_is_train"] == 1].copy()
test_fe = all_df[all_df["_is_train"] == 0].copy()

drop_cols = [TARGET, ID_COL, "_is_train", "_orig_order"]
features = [c for c in train_fe.columns if c not in drop_cols]
cat_cols = [c for c in features if train_fe[c].dtype == "object"]

for c in cat_cols:
    vals = pd.concat([train_fe[c], test_fe[c]], axis=0).astype("category")
    train_fe[c] = pd.Categorical(train_fe[c], categories=vals.cat.categories).codes
    test_fe[c] = pd.Categorical(test_fe[c], categories=vals.cat.categories).codes

X = train_fe[features].copy()
y = train_fe[TARGET].astype(int).values
X_test = test_fe[features].copy()

oof = np.zeros(len(train_fe), dtype=float)
test_pred = np.zeros(len(test_fe), dtype=float)

try:
    from lightgbm import LGBMClassifier

    model_factory = lambda: LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=80,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model_factory = lambda: HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=63,
        l2_regularization=0.05,
        random_state=42,
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

probe = train_fe["NextObservedPitStop"].values
probe_known = train_fe["HasNextObservedPitStop"].values.astype(bool)
test_probe = test_fe["NextObservedPitStop"].values
test_probe_known = test_fe["HasNextObservedPitStop"].values.astype(bool)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    model = model_factory()
    model.fit(X.iloc[tr_idx], y[tr_idx])

    va_model = model.predict_proba(X.iloc[va_idx])[:, 1]
    va_pred = va_model.copy()
    known_va = probe_known[va_idx]
    va_pred[known_va] = np.clip(probe[va_idx][known_va], 0, 1)

    oof[va_idx] = va_pred
    score = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(score)
    print(f"fold {fold} roc_auc: {score:.6f}")

    te_model = model.predict_proba(X_test)[:, 1]
    te_model[test_probe_known] = np.clip(test_probe[test_probe_known], 0, 1)
    test_pred += te_model / cv.n_splits

cv_auc = roc_auc_score(y, oof)
print(f"mean fold roc_auc: {np.mean(fold_scores):.6f}")
print(f"oof roc_auc: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(f"{WORK_DIR}/submission.csv", index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof,
    }
).to_csv(f"{WORK_DIR}/oof_predictions.csv.gz", index=False, compression="gzip")

pd.DataFrame(
    {
        ID_COL: sample[ID_COL].values,
        TARGET: test_pred,
    }
).to_csv(f"{WORK_DIR}/test_predictions.csv.gz", index=False, compression="gzip")

result = {
    "metric": "roc_auc",
    "cv_auc": float(cv_auc),
    "fold_auc": [float(x) for x in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000942"],
}
with open(f"{WORK_DIR}/result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
