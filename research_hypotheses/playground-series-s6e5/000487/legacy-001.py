import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
y = train[target_col].astype(int).values


def add_features(df):
    out = df.copy()
    out["Race_Year"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)
    out["Race_Year_Driver"] = out["Race_Year"] + "_" + out["Driver"].astype(str)
    out["GroupKey"] = out["Race_Year_Driver"] + "_S" + out["Stint"].astype(str)

    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["TyreLife_per_Stint"] = out["TyreLife"] / (out["Stint"] + 1.0)
    out["LapNumber_per_RaceProgress"] = out["LapNumber"] / (out["RaceProgress"] + 0.01)
    out["Degradation_per_TyreLife"] = out["Cumulative_Degradation"] / (
        out["TyreLife"] + 1.0
    )
    out["Abs_LapTime_Delta"] = out["LapTime_Delta"].abs()
    out["Abs_Position_Change"] = out["Position_Change"].abs()
    out["LateRace_TyreLife"] = out["TyreLife"] * (out["RaceProgress"] > 0.55).astype(
        int
    )
    out["CurrentPit_x_TyreLife"] = out["PitStop"] * out["TyreLife"]
    return out


train_fe = add_features(train.drop(columns=[target_col]))
test_fe = add_features(test)

features = [c for c in train_fe.columns if c != id_col]
cat_cols = ["Compound", "Driver", "Race", "Race_Year", "Race_Year_Driver", "GroupKey"]
for c in cat_cols:
    all_vals = pd.concat([train_fe[c], test_fe[c]], axis=0).astype("category")
    cats = all_vals.cat.categories
    train_fe[c] = pd.Categorical(train_fe[c], categories=cats)
    test_fe[c] = pd.Categorical(test_fe[c], categories=cats)

X = train_fe[features]
X_test = test_fe[features]

clf_oof = np.zeros(len(train))
rank_oof_raw = np.zeros(len(train))
clf_test = np.zeros(len(test))
rank_test_raw = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=487)

clf_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.045,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.5,
    n_estimators=2500,
    random_state=487,
    n_jobs=-1,
    verbosity=-1,
)

rank_params = dict(
    objective="lambdarank",
    metric="auc",
    learning_rate=0.04,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.5,
    n_estimators=1800,
    random_state=1487,
    n_jobs=-1,
    verbosity=-1,
)


def sorted_group_sizes(group_values):
    return (
        pd.Series(group_values)
        .value_counts(sort=False)
        .loc[pd.Series(group_values).drop_duplicates()]
        .values
    )


def prepare_rank_frame(X_part, y_part=None):
    tmp = X_part.copy()
    tmp["_row_order"] = np.arange(len(tmp))
    tmp["_group_sort"] = tmp["GroupKey"].cat.codes.values
    if y_part is not None:
        tmp["_target"] = y_part
    tmp = tmp.sort_values(["_group_sort", "_row_order"], kind="mergesort")
    group_sizes = sorted_group_sizes(tmp["_group_sort"].values)
    order = tmp["_row_order"].values
    if y_part is None:
        return tmp.drop(columns=["_row_order", "_group_sort"]), group_sizes, order, None
    labels = tmp["_target"].values
    return (
        tmp.drop(columns=["_row_order", "_group_sort", "_target"]),
        group_sizes,
        order,
        labels,
    )


for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    clf_oof[va_idx] = clf.predict_proba(X_va)[:, 1]
    clf_test += clf.predict_proba(X_test)[:, 1] / skf.n_splits

    Xr_tr, group_tr, _, yr_tr = prepare_rank_frame(X_tr, y_tr)
    Xr_va, _, va_order, _ = prepare_rank_frame(X_va)

    ranker = lgb.LGBMRanker(**rank_params)
    ranker.fit(
        Xr_tr,
        yr_tr,
        group=group_tr,
        categorical_feature=cat_cols,
        callbacks=[lgb.log_evaluation(0)],
    )
    va_pred_sorted = ranker.predict(Xr_va, num_iteration=ranker.best_iteration_)
    rank_oof_raw[va_idx[va_order]] = va_pred_sorted
    rank_test_raw += (
        ranker.predict(X_test, num_iteration=ranker.best_iteration_) / skf.n_splits
    )

    print(
        f"fold {fold}: classifier_auc={roc_auc_score(y_va, clf_oof[va_idx]):.6f}, ranker_raw_auc={roc_auc_score(y_va, rank_oof_raw[va_idx]):.6f}"
    )

qt = QuantileTransformer(
    n_quantiles=min(1000, len(train)), output_distribution="uniform", random_state=487
)
rank_oof = qt.fit_transform(rank_oof_raw.reshape(-1, 1)).ravel()
rank_test = qt.transform(rank_test_raw.reshape(-1, 1)).ravel()

best_auc = -1.0
best_w = 0.0
for w in np.linspace(0, 1, 51):
    pred = (1 - w) * clf_oof + w * rank_oof
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_auc = auc
        best_w = float(w)

test_pred = (1 - best_w) * clf_test + best_w * rank_test
test_pred = np.clip(test_pred, 0, 1)

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": (1 - best_w) * clf_oof + best_w * rank_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame({id_col: sample[id_col].values, target_col: test_pred}).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000487"],
    "cv_auc": float(best_auc),
    "classifier_oof_auc": float(roc_auc_score(y, clf_oof)),
    "ranker_oof_auc": float(roc_auc_score(y, rank_oof)),
    "best_ranker_blend_weight": best_w,
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
