import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

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


def add_features(df):
    df = df.copy()
    for c in ["Driver", "Race", "Compound"]:
        df[c] = df[c].astype(str).fillna("NA")

    df["Driver_x_Compound"] = df["Driver"] + "__" + df["Compound"]
    df["Race_x_Compound"] = df["Race"] + "__" + df["Compound"]
    df["Race_x_Stint"] = df["Race"] + "__" + df["Stint"].astype(str)
    df["RaceProgressBin"] = pd.cut(
        df["RaceProgress"],
        bins=[-0.001, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.001],
        labels=False,
        include_lowest=True,
    ).astype(str)
    df["Race_x_ProgressBin"] = df["Race"] + "__" + df["RaceProgressBin"]

    df["TyreLife_x_Progress"] = df["TyreLife"] * df["RaceProgress"]
    df["LapNumber_x_Progress"] = df["LapNumber"] * df["RaceProgress"]
    df["IsEarlyLap"] = (df["LapNumber"] <= 3).astype(int)
    df["IsWetCompound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    return df


train_fe = add_features(train.drop(columns=[target_col]))
test_fe = add_features(test)

cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "Driver_x_Compound",
    "Race_x_Compound",
    "Race_x_Stint",
    "RaceProgressBin",
    "Race_x_ProgressBin",
]
num_cols = [
    "Year",
    "LapNumber",
    "LapTime (s)",
    "LapTime_Delta",
    "Position",
    "Position_Change",
    "PitStop",
    "Stint",
    "TyreLife",
    "RaceProgress",
    "Cumulative_Degradation",
    "TyreLife_x_Progress",
    "LapNumber_x_Progress",
    "IsEarlyLap",
    "IsWetCompound",
]
feature_cols = cat_cols + num_cols

X = train_fe[feature_cols].copy()
X_test = test_fe[feature_cols].copy()

for c in cat_cols:
    X[c] = X[c].astype(str).fillna("NA")
    X_test[c] = X_test[c].astype(str).fillna("NA")
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(X[c].median())

groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)
cv = GroupKFold(n_splits=5)

cat_oof = np.zeros(len(train))
cat_test = np.zeros(len(test))
lgb_oof = np.zeros(len(train))
lgb_test = np.zeros(len(test))

cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

from catboost import CatBoostClassifier, Pool

try:
    import lightgbm as lgb

    has_lgb = True
except Exception:
    has_lgb = False

if has_lgb:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_lgb = X.copy()
    X_test_lgb = X_test.copy()
    X_lgb[cat_cols] = enc.fit_transform(X_lgb[cat_cols])
    X_test_lgb[cat_cols] = enc.transform(X_test_lgb[cat_cols])
else:
    X_lgb = X_test_lgb = None

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    cat_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=8.0,
        random_seed=2026 + fold,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        verbose=False,
    )
    cat_model.fit(
        Pool(X_tr, y_tr, cat_features=cat_feature_indices),
        eval_set=Pool(X_va, y_va, cat_features=cat_feature_indices),
        use_best_model=True,
    )
    cat_oof[va_idx] = cat_model.predict_proba(
        Pool(X_va, cat_features=cat_feature_indices)
    )[:, 1]
    cat_test += (
        cat_model.predict_proba(Pool(X_test, cat_features=cat_feature_indices))[:, 1]
        / cv.n_splits
    )

    if has_lgb:
        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=80,
            reg_lambda=5.0,
            random_state=4040 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_lgb.iloc[tr_idx],
            y_tr,
            eval_set=[(X_lgb.iloc[va_idx], y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        lgb_oof[va_idx] = lgb_model.predict_proba(X_lgb.iloc[va_idx])[:, 1]
        lgb_test += lgb_model.predict_proba(X_test_lgb)[:, 1] / cv.n_splits

    fold_auc = roc_auc_score(y_va, cat_oof[va_idx])
    print(f"Fold {fold} CatBoost ROC AUC: {fold_auc:.6f}")

cat_auc = roc_auc_score(y, cat_oof)
print(f"CatBoost grouped 5-fold ROC AUC: {cat_auc:.6f}")

final_oof = cat_oof
final_test = cat_test

if has_lgb:
    lgb_auc = roc_auc_score(y, lgb_oof)
    blend_oof = 0.5 * cat_oof + 0.5 * lgb_oof
    blend_test = 0.5 * cat_test + 0.5 * lgb_test
    blend_auc = roc_auc_score(y, blend_oof)
    print(f"LightGBM grouped 5-fold ROC AUC: {lgb_auc:.6f}")
    print(f"50/50 blend grouped 5-fold ROC AUC: {blend_auc:.6f}")
    if blend_auc >= cat_auc:
        final_oof = blend_oof
        final_test = blend_test
        final_auc = blend_auc
        model_used = "catboost_lgbm_50_50_blend"
    else:
        final_auc = cat_auc
        model_used = "catboost"
else:
    print("LightGBM unavailable; using standalone CatBoost predictions.")
    final_auc = cat_auc
    model_used = "catboost"

submission = sample[[id_col]].copy()
submission[target_col] = np.clip(final_test, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(final_oof, 0, 1),
    }
)
oof.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred = sample[[id_col]].copy()
test_pred[target_col] = np.clip(final_test, 0, 1)
test_pred.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "validation_score": float(final_auc),
    "model_used_for_submission": model_used,
    "research_hypotheses_llm_claimed_used": ["000533"],
}
with open(os.path.join(WORKING_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
