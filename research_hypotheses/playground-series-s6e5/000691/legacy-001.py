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

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
cat_cols = ["Driver", "Race", "Compound"]
feature_cols = [c for c in train.columns if c not in [target_col, id_col]]

y = train[target_col].astype(int).values
groups = (train["Year"].astype(str) + "_" + train["Race"].astype(str)).values

all_df = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
for c in cat_cols:
    all_df[c] = all_df[c].astype(str).fillna("missing")

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
all_df[cat_cols] = enc.fit_transform(all_df[cat_cols])

X = all_df.iloc[: len(train)].copy()
X_test = all_df.iloc[len(train) :].copy()

for c in cat_cols:
    X[c] = X[c].astype("int32")
    X_test[c] = X_test[c].astype("int32")

num_cols = [c for c in feature_cols if c not in cat_cols]
X[num_cols] = X[num_cols].astype("float32")
X_test[num_cols] = X_test[num_cols].astype("float32")

n_splits = 5
cv = GroupKFold(n_splits=n_splits)

oof_rank = np.zeros(len(train), dtype=np.float32)
oof_lgb = np.zeros(len(train), dtype=np.float32)
oof_cat = np.zeros(len(train), dtype=np.float32)

test_rank = np.zeros(len(test), dtype=np.float32)
test_lgb = np.zeros(len(test), dtype=np.float32)
test_cat = np.zeros(len(test), dtype=np.float32)

try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("lightgbm is required for hypothesis 000691") from e

try:
    from catboost import CatBoostClassifier, Pool

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


def rank_group_sizes(df):
    tmp = df[["Year", "Race", "LapNumber"]].copy()
    tmp["_order"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["Year", "Race", "LapNumber", "_order"])
    sizes = tmp.groupby(["Year", "Race", "LapNumber"], sort=False).size().values
    return tmp["_order"].values, sizes


rank_test_order, rank_test_groups = rank_group_sizes(test)
X_test_rank_sorted = X_test.iloc[rank_test_order]

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    print(f"Fold {fold}/{n_splits}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    train_part = train.iloc[tr_idx]
    valid_part = train.iloc[va_idx]

    tr_order, tr_rank_groups = rank_group_sizes(train_part)
    va_order, va_rank_groups = rank_group_sizes(valid_part)

    X_tr_rank = X_tr.iloc[tr_order]
    y_tr_rank = y_tr[tr_order]
    X_va_rank = X_va.iloc[va_order]

    ranker = lgb.LGBMRanker(
        objective="rank_xendcg",
        metric="auc",
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=2026 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    ranker.fit(
        X_tr_rank,
        y_tr_rank,
        group=tr_rank_groups,
        eval_set=[(X_va_rank, y_va[va_order])],
        eval_group=[va_rank_groups],
        eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    va_rank_pred_sorted = ranker.predict(
        X_va_rank, num_iteration=ranker.best_iteration_
    )
    va_rank_pred = np.empty(len(va_idx), dtype=np.float32)
    va_rank_pred[va_order] = va_rank_pred_sorted
    oof_rank[va_idx] = va_rank_pred
    test_rank_sorted = ranker.predict(
        X_test_rank_sorted, num_iteration=ranker.best_iteration_
    )
    tmp_test_rank = np.empty(len(test), dtype=np.float32)
    tmp_test_rank[rank_test_order] = test_rank_sorted
    test_rank += tmp_test_rank / n_splits

    clf = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1200,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.2,
        random_state=4026 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    oof_lgb[va_idx] = clf.predict_proba(X_va)[:, 1]
    test_lgb += clf.predict_proba(X_test)[:, 1] / n_splits

    if HAS_CATBOOST:
        cat_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=900,
            learning_rate=0.035,
            depth=7,
            l2_leaf_reg=5.0,
            random_seed=6026 + fold,
            od_type="Iter",
            od_wait=80,
            verbose=False,
            allow_writing_files=False,
        )
        cat_features = [feature_cols.index(c) for c in cat_cols]
        cat_model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features),
            eval_set=Pool(X_va, y_va, cat_features=cat_features),
            use_best_model=True,
        )
        oof_cat[va_idx] = cat_model.predict_proba(
            Pool(X_va, cat_features=cat_features)
        )[:, 1]
        test_cat += (
            cat_model.predict_proba(Pool(X_test, cat_features=cat_features))[:, 1]
            / n_splits
        )
    else:
        oof_cat[va_idx] = oof_lgb[va_idx]
        test_cat += clf.predict_proba(X_test)[:, 1] / n_splits


def minmax(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.min(a), np.max(a)
    return (a - lo) / (hi - lo + 1e-12)


rank_o = minmax(oof_rank)
lgb_o = np.clip(oof_lgb, 1e-6, 1 - 1e-6)
cat_o = np.clip(oof_cat, 1e-6, 1 - 1e-6)

rank_t = minmax(test_rank)
lgb_t = np.clip(test_lgb, 1e-6, 1 - 1e-6)
cat_t = np.clip(test_cat, 1e-6, 1 - 1e-6)

best_auc = -1.0
best_w = None
for wr in np.linspace(0.05, 0.50, 10):
    for wl in np.linspace(0.20, 0.75, 12):
        wc = 1.0 - wr - wl
        if wc < 0.05 or wc > 0.75:
            continue
        pred = wr * rank_o + wl * lgb_o + wc * cat_o
        auc = roc_auc_score(y, pred)
        if auc > best_auc:
            best_auc = auc
            best_w = (wr, wl, wc)

wr, wl, wc = best_w
blend_oof = wr * rank_o + wl * lgb_o + wc * cat_o
blend_test = wr * rank_t + wl * lgb_t + wc * cat_t

calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
calibrator.fit(blend_oof.reshape(-1, 1), y)
cal_oof = calibrator.predict_proba(blend_oof.reshape(-1, 1))[:, 1]
cal_test = calibrator.predict_proba(blend_test.reshape(-1, 1))[:, 1]

auc_raw = roc_auc_score(y, blend_oof)
auc_cal = roc_auc_score(y, cal_oof)

print(f"OOF ROC AUC raw blend: {auc_raw:.6f}")
print(f"OOF ROC AUC calibrated blend: {auc_cal:.6f}")
print(f"Best blend weights rank/lgb/cat: {wr:.3f}/{wl:.3f}/{wc:.3f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": cal_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample[[id_col]].copy()
test_pred_df[target_col] = np.clip(cal_test, 1e-6, 1 - 1e-6)
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
test_pred_df.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(
        {
            "metric": "roc_auc",
            "validation_score": float(auc_cal),
            "raw_blend_score": float(auc_raw),
            "research_hypotheses_llm_claimed_used": ["000691"],
            "blend_weights": {
                "rank_xendcg": float(wr),
                "lightgbm_gbdt": float(wl),
                "catboost": float(wc),
            },
        },
        f,
        indent=2,
    )
