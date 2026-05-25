import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, CatBoostRanker, Pool

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
GROUP_COLS = ["Race", "Year", "Driver", "Stint"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
features = [c for c in train.columns if c not in [TARGET, ID_COL]]
cat_features = [features.index(c) for c in CAT_COLS if c in features]

for c in CAT_COLS:
    train[c] = train[c].astype(str)
    test[c] = test[c].astype(str)


def add_features(df):
    out = df.copy()
    out["StintProgress"] = out["TyreLife"] / out.groupby(GROUP_COLS)[
        "TyreLife"
    ].transform("max").clip(lower=1)
    out["LapInStintRank"] = out.groupby(GROUP_COLS)["LapNumber"].rank(method="first")
    out["StintLapCount"] = out.groupby(GROUP_COLS)["LapNumber"].transform("count")
    out["RemainingRace"] = 1.0 - out["RaceProgress"]
    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["Deg_per_TyreLife"] = out["Cumulative_Degradation"] / out["TyreLife"].clip(
        lower=1
    )
    return out


train_fe = add_features(train)
test_fe = add_features(test)
features = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
cat_features = [features.index(c) for c in CAT_COLS if c in features]


def make_rank_labels(df):
    g = df.groupby(GROUP_COLS, sort=False)
    max_lap = g["LapNumber"].transform("max")
    min_lap = g["LapNumber"].transform("min")
    span = (max_lap - min_lap).replace(0, 1)
    rel_pos = (df["LapNumber"] - min_lap) / span
    return (rel_pos + 4.0 * df[TARGET].astype(float)).astype(float).values


rank_y = make_rank_labels(train_fe)
fold_groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)

oof_bin = np.zeros(len(train_fe))
oof_rank = np.zeros(len(train_fe))
test_bin_folds = []
test_rank_folds = []

gkf = GroupKFold(n_splits=5)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_fe, y, groups=fold_groups), 1):
    X_tr = train_fe.iloc[tr_idx][features]
    X_va = train_fe.iloc[va_idx][features]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=6,
        random_seed=2026 + fold,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
        thread_count=max(1, os.cpu_count() or 1),
    )
    clf.fit(
        Pool(X_tr, y_tr, cat_features=cat_features),
        eval_set=Pool(X_va, y_va, cat_features=cat_features),
        use_best_model=True,
    )
    oof_bin[va_idx] = clf.predict_proba(X_va)[:, 1]
    test_bin_folds.append(clf.predict_proba(test_fe[features])[:, 1])

    rank_tr = train_fe.iloc[tr_idx].copy()
    rank_va = train_fe.iloc[va_idx].copy()
    rank_tr["_rank_target"] = rank_y[tr_idx]
    rank_va["_orig_idx"] = va_idx

    sort_cols = GROUP_COLS + ["LapNumber", ID_COL]
    rank_tr = rank_tr.sort_values(sort_cols).reset_index(drop=True)
    rank_va = rank_va.sort_values(sort_cols).reset_index(drop=True)
    test_rank = test_fe.copy().sort_values(sort_cols).reset_index()

    ranker = CatBoostRanker(
        loss_function="YetiRankPairwise",
        iterations=450,
        learning_rate=0.06,
        depth=5,
        l2_leaf_reg=8,
        random_seed=4026 + fold,
        verbose=False,
        allow_writing_files=False,
        thread_count=max(1, os.cpu_count() or 1),
    )
    ranker.fit(
        Pool(
            rank_tr[features],
            label=rank_tr["_rank_target"],
            group_id=rank_tr[GROUP_COLS].astype(str).agg("|".join, axis=1),
            cat_features=cat_features,
        )
    )

    va_score_sorted = ranker.predict(rank_va[features])
    oof_rank[rank_va["_orig_idx"].values] = va_score_sorted

    te_score_sorted = ranker.predict(test_rank[features])
    te_score = np.zeros(len(test_fe))
    te_score[test_rank["index"].values] = te_score_sorted
    test_rank_folds.append(te_score)

    print(
        f"fold {fold} binary_auc={roc_auc_score(y_va, oof_bin[va_idx]):.6f} rank_raw_auc={roc_auc_score(y_va, oof_rank[va_idx]):.6f}"
    )

cal = LogisticRegression(max_iter=1000, solver="lbfgs")
cal.fit(oof_rank.reshape(-1, 1), y)
oof_rank_cal = cal.predict_proba(oof_rank.reshape(-1, 1))[:, 1]

test_bin = np.mean(test_bin_folds, axis=0)
test_rank_raw = np.mean(test_rank_folds, axis=0)
test_rank_cal = cal.predict_proba(test_rank_raw.reshape(-1, 1))[:, 1]

best_w, best_auc = 0.0, -1.0
for w in np.linspace(0, 1, 21):
    pred = (1 - w) * oof_bin + w * oof_rank_cal
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_w, best_auc = float(w), float(auc)

oof_pred = (1 - best_w) * oof_bin + best_w * oof_rank_cal
test_pred = (1 - best_w) * test_bin + best_w * test_rank_cal
test_pred = np.clip(test_pred, 0.0, 1.0)

print(f"OOF ROC AUC: {roc_auc_score(y, oof_pred):.6f}")
print(f"Selected ranker blend weight: {best_w:.2f}")

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: sample[ID_COL].values,
        TARGET: test_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(
        {
            "metric": "roc_auc",
            "oof_roc_auc": float(roc_auc_score(y, oof_pred)),
            "research_hypotheses_llm_claimed_used": ["000304"],
        },
        f,
        indent=2,
    )
