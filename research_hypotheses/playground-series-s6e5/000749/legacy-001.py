import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from lightgbm import LGBMClassifier
from catboost import CatBoostRanker, Pool

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
SEED = 42
BLEND_POINTWISE = 0.65

train_raw = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test_raw = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_features(df):
    df = df.copy()
    df["Year_Race"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)
    df["query_id"] = df["Year_Race"] + "_lap_" + df["LapNumber"].astype(str)

    q = df.groupby("query_id", sort=False)
    df["query_size"] = q["LapNumber"].transform("size").astype("float32")

    for col in [
        "TyreLife",
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "Position",
        "RaceProgress",
    ]:
        pct_col = (
            col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            + "_q_pct"
        )
        mean_col = (
            col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            + "_q_center"
        )
        df[pct_col] = q[col].rank(pct=True).astype("float32")
        df[mean_col] = (df[col] - q[col].transform("mean")).astype("float32")

    df["front_position_pressure_q_pct"] = (
        q["Position"].rank(pct=True, ascending=False).astype("float32")
    )
    df["tyre_life_x_progress"] = (df["TyreLife"] * df["RaceProgress"]).astype("float32")
    df["degradation_per_tyre_lap"] = (
        df["Cumulative_Degradation"] / (df["TyreLife"] + 1.0)
    ).astype("float32")
    df["lap_delta_abs"] = df["LapTime_Delta"].abs().astype("float32")
    df["is_current_pit_lap"] = df["PitStop"].astype("int8")
    return df


train = add_features(train_raw)
test = add_features(test_raw)

y = train[TARGET].astype(int).values
groups = train["Year_Race"].values

drop_cols = [ID_COL, TARGET, "Year_Race", "query_id"]
features = [c for c in train.columns if c not in drop_cols]
cat_features = [c for c in CAT_COLS if c in features]

for c in cat_features:
    cats = pd.Index(pd.concat([train[c], test[c]], axis=0).astype(str).unique())
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

point_oof = np.zeros(len(train), dtype=np.float32)
rank_oof_raw = np.zeros(len(train), dtype=np.float32)
test_point_sum = np.zeros(len(test), dtype=np.float64)
test_rank_sum = np.zeros(len(test), dtype=np.float64)

logo = LeaveOneGroupOut()
splits = list(logo.split(train, y, groups))
n_splits = len(splits)
print(
    f"Running LeaveOneGroupOut on Year_Race with {n_splits} folds for hypothesis 000749"
)

lgb_params = dict(
    objective="binary",
    n_estimators=180,
    learning_rate=0.045,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=max(1, os.cpu_count() or 1),
    verbosity=-1,
)

rank_params = dict(
    loss_function="QueryCrossEntropy",
    iterations=80,
    learning_rate=0.08,
    depth=5,
    l2_leaf_reg=6.0,
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
    thread_count=max(1, os.cpu_count() or 1),
)


def query_percentile(values, query_ids):
    s = pd.Series(values)
    q = pd.Series(query_ids)
    pct = s.groupby(q, sort=False).rank(pct=True).astype("float32")
    return pct.values


for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr = train.iloc[tr_idx][features]
    X_va = train.iloc[va_idx][features]
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    clf = LGBMClassifier(**lgb_params)
    clf.fit(
        X_tr,
        y_tr,
        categorical_feature=cat_features,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
    )
    point_oof[va_idx] = clf.predict_proba(X_va)[:, 1].astype(np.float32)
    test_point_sum += clf.predict_proba(test[features])[:, 1] / n_splits

    rank_tr = train.iloc[tr_idx].copy()
    rank_va = train.iloc[va_idx].copy()

    rank_tr["_target"] = y_tr
    rank_va["_target"] = y_va

    rank_tr = rank_tr.sort_values("query_id", kind="mergesort")
    rank_va = rank_va.sort_values("query_id", kind="mergesort")

    ranker = CatBoostRanker(**rank_params)
    ranker.fit(
        Pool(
            rank_tr[features],
            label=rank_tr["_target"],
            group_id=rank_tr["query_id"],
            cat_features=cat_features,
        )
    )

    va_pred_sorted = ranker.predict(
        Pool(rank_va[features], group_id=rank_va["query_id"], cat_features=cat_features)
    )
    rank_oof_raw[rank_va.index.values] = va_pred_sorted.astype(np.float32)

    test_sorted = test.sort_values("query_id", kind="mergesort")
    test_pred_sorted = ranker.predict(
        Pool(
            test_sorted[features],
            group_id=test_sorted["query_id"],
            cat_features=cat_features,
        )
    )
    tmp = pd.Series(test_pred_sorted, index=test_sorted.index).sort_index().values
    test_rank_sum += tmp / n_splits

    if fold % 10 == 0 or fold == n_splits:
        current_blend = BLEND_POINTWISE * point_oof[point_oof > 0] if False else None
        print(f"Completed fold {fold}/{n_splits}")

rank_oof = query_percentile(rank_oof_raw, train["query_id"].values)
test_rank = query_percentile(test_rank_sum, test["query_id"].values)

blend_oof = BLEND_POINTWISE * point_oof + (1.0 - BLEND_POINTWISE) * rank_oof
point_auc = roc_auc_score(y, point_oof)
rank_auc = roc_auc_score(y, rank_oof)
blend_auc = roc_auc_score(y, blend_oof)

print(f"OOF ROC AUC pointwise: {point_auc:.6f}")
print(f"OOF ROC AUC query_ranker: {rank_auc:.6f}")
print(f"OOF ROC AUC blended: {blend_auc:.6f}")

full_point = LGBMClassifier(**lgb_params)
full_point.fit(train[features], y, categorical_feature=cat_features)
full_point_test = full_point.predict_proba(test[features])[:, 1]

full_rank_train = train.copy().sort_values("query_id", kind="mergesort")
full_ranker = CatBoostRanker(**{**rank_params, "iterations": 120})
full_ranker.fit(
    Pool(
        full_rank_train[features],
        label=full_rank_train[TARGET].astype(int),
        group_id=full_rank_train["query_id"],
        cat_features=cat_features,
    )
)

test_sorted = test.sort_values("query_id", kind="mergesort")
full_rank_test_sorted = full_ranker.predict(
    Pool(
        test_sorted[features],
        group_id=test_sorted["query_id"],
        cat_features=cat_features,
    )
)
full_rank_test_raw = (
    pd.Series(full_rank_test_sorted, index=test_sorted.index).sort_index().values
)
full_rank_test = query_percentile(full_rank_test_raw, test["query_id"].values)

test_pred = BLEND_POINTWISE * full_point_test + (1.0 - BLEND_POINTWISE) * full_rank_test
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": blend_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_predictions = sample.copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "validation_score": float(blend_auc),
    "pointwise_oof_auc": float(point_auc),
    "query_ranker_oof_auc": float(rank_auc),
    "research_hypotheses_llm_claimed_used": ["000749"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print(json.dumps(result, indent=2))
