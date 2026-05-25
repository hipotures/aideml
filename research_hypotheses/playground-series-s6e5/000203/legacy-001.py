import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
RANDOM_STATE = 42
N_SPLITS = 5
os.makedirs(WORK_DIR, exist_ok=True)


def safe_name(x):
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(x)).strip("_")
    if not s:
        s = "x"
    if s[0].isdigit():
        s = "f_" + s
    return s


def add_pit_wave_features(df):
    df = df.copy()
    keys = ["Race", "Year", "LapNumber"]
    state_cols = ["TyreLife", "Cumulative_Degradation", "LapTime_Delta"]

    lap_stats = (
        df.groupby(keys, sort=False)
        .agg(
            field_size=("id", "size"),
            pit_count=("PitStop", "sum"),
            pit_frac=("PitStop", "mean"),
        )
        .reset_index()
    )
    df = df.merge(lap_stats[keys + ["field_size"]], on=keys, how="left")

    for lag in (1, 2):
        prev = lap_stats.copy()
        prev["LapNumber"] = prev["LapNumber"] + lag
        prev = prev.rename(
            columns={
                "field_size": f"field_size_prev{lag}",
                "pit_count": f"pit_count_prev{lag}",
                "pit_frac": f"pit_frac_prev{lag}",
            }
        )
        lag_cols = [
            f"field_size_prev{lag}",
            f"pit_count_prev{lag}",
            f"pit_frac_prev{lag}",
        ]
        df = df.merge(prev[keys + lag_cols], on=keys, how="left")
        df[lag_cols] = df[lag_cols].fillna(0.0)

    denom = df["field_size_prev1"] + df["field_size_prev2"]
    df["pit_count_prev12"] = df["pit_count_prev1"] + df["pit_count_prev2"]
    df["pit_frac_prev12"] = np.where(denom > 0, df["pit_count_prev12"] / denom, 0.0)

    q = df.groupby(keys, sort=False)[state_cols].quantile([0.25, 0.50, 0.75]).unstack()
    q.columns = [
        f"{c}_field_p{int(round(v * 100)):02d}" for c, v in q.columns.to_flat_index()
    ]
    df = df.merge(q.reset_index(), on=keys, how="left")

    agg = df.groupby(keys, sort=False)[state_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_field_{s}" for c, s in agg.columns.to_flat_index()]
    df = df.merge(agg.reset_index(), on=keys, how="left")

    for col in state_cols:
        df[f"{col}_field_rank_pct"] = df.groupby(keys, sort=False)[col].rank(
            method="average", pct=True
        )
        q25, q50, q75 = f"{col}_field_p25", f"{col}_field_p50", f"{col}_field_p75"
        df[f"{col}_field_iqr"] = df[q75] - df[q25]
        df[f"{col}_dev_p25"] = df[col] - df[q25]
        df[f"{col}_dev_p50"] = df[col] - df[q50]
        df[f"{col}_dev_p75"] = df[col] - df[q75]
        df[f"{col}_dev_mean"] = df[col] - df[f"{col}_field_mean"]
        df[f"{col}_dev_p50_iqr"] = df[f"{col}_dev_p50"] / df[
            f"{col}_field_iqr"
        ].replace(0, np.nan)

    df["inv_position"] = 1.0 / df["Position"].clip(lower=1)
    wave_cols = ["pit_frac_prev1", "pit_frac_prev2", "pit_frac_prev12"]
    dev_cols = [f"{c}_dev_p50_iqr" for c in state_cols]

    for col in wave_cols:
        df[f"{col}_x_position"] = df[col] * df["Position"]
        df[f"{col}_x_inv_position"] = df[col] * df["inv_position"]

    for col in dev_cols:
        df[f"{col}_x_position"] = df[col] * df["Position"]

    compound_str = df["Compound"].astype(str)
    for compound in sorted(compound_str.unique()):
        mask = (compound_str == compound).astype("float32")
        suffix = safe_name(compound)
        for col in wave_cols + dev_cols:
            df[f"{col}_x_compound_{suffix}"] = df[col] * mask

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    for col in num_cols:
        if col == "id":
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def unique_clean_mapping(cols):
    used_counts, used_names, mapping = {}, set(), {}
    for col in cols:
        base = safe_name(col)
        i = used_counts.get(base, 0)
        name = base if i == 0 else f"{base}_{i}"
        while name in used_names:
            i += 1
            name = f"{base}_{i}"
        used_counts[base] = i + 1
        used_names.add(name)
        mapping[col] = name
    return mapping


def make_model(n_estimators, scale_pos_weight):
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=int(n_estimators),
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=float(scale_pos_weight),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def pos_weight(y):
    pos = float(np.sum(y))
    neg = float(len(y) - pos)
    return neg / max(pos, 1.0)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

n_train = len(train)
test[TARGET] = np.nan
train["_is_train"] = 1
test["_is_train"] = 0
all_df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
all_df = add_pit_wave_features(all_df)

feature_cols = [c for c in all_df.columns if c not in {"id", TARGET, "_is_train"}]
cat_original = [
    c
    for c in feature_cols
    if pd.api.types.is_object_dtype(all_df[c])
    or pd.api.types.is_categorical_dtype(all_df[c])
]
rename_map = unique_clean_mapping(feature_cols)

X_all = all_df[feature_cols].rename(columns=rename_map)
cat_cols = [rename_map[c] for c in cat_original]
for col in cat_cols:
    X_all[col] = X_all[col].astype("category")

X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)
y = all_df.loc[: n_train - 1, TARGET].astype("int8").to_numpy()
groups = (
    all_df.loc[: n_train - 1, "Race"].astype(str)
    + "_"
    + all_df.loc[: n_train - 1, "Year"].astype(str)
).to_numpy()

if StratifiedGroupKFold is not None and len(np.unique(groups)) >= N_SPLITS:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X_train, y, groups))
    cv_name = "5-fold StratifiedGroupKFold by Race-Year"
    if any(len(np.unique(y[val_idx])) < 2 for _, val_idx in splits):
        splitter = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        splits = list(splitter.split(X_train, y))
        cv_name = "5-fold StratifiedKFold"
else:
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X_train, y))
    cv_name = "5-fold StratifiedKFold"

oof = np.zeros(n_train, dtype="float32")
fold_scores, best_iters = [], []

for fold, (tr_idx, val_idx) in enumerate(splits, 1):
    model = make_model(n_estimators=2000, scale_pos_weight=pos_weight(y[tr_idx]))
    model.fit(
        X_train.iloc[tr_idx],
        y[tr_idx],
        categorical_feature=cat_cols,
        eval_set=[(X_train.iloc[val_idx], y[val_idx])],
        eval_metric="auc",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    pred = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    oof[val_idx] = pred.astype("float32")
    auc = roc_auc_score(y[val_idx], pred)
    best_iter = getattr(model, "best_iteration_", None) or model.n_estimators
    fold_scores.append(float(auc))
    best_iters.append(int(best_iter))
    print(f"Fold {fold} ROC AUC: {auc:.6f} best_iter={best_iter}")

cv_auc = roc_auc_score(y, oof)
print(f"{cv_name} ROC AUC: {cv_auc:.6f}")

final_iters = int(np.clip(round(np.mean(best_iters) * 1.10), 100, 2000))
final_model = make_model(n_estimators=final_iters, scale_pos_weight=pos_weight(y))
final_model.fit(X_train, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

pd.DataFrame(
    {
        "row": np.arange(n_train, dtype=np.int32),
        "target": y.astype(int),
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_by_id = pd.DataFrame({"id": test["id"].to_numpy(), TARGET: test_pred})
submission = sample[["id"]].merge(test_pred_by_id, on="id", how="left")
if submission[TARGET].isna().any():
    raise RuntimeError("Some sample_submission ids did not receive predictions.")

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

result = {
    "metric": "roc_auc",
    "cv_name": cv_name,
    "cv_auc": float(cv_auc),
    "fold_auc": fold_scores,
    "final_iterations": final_iters,
    "research_hypotheses_llm_claimed_used": ["000203"],
}
for name in ["result.json", "review.json"]:
    with open(os.path.join(WORK_DIR, name), "w") as f:
        json.dump(result, f, indent=2)

print(json.dumps(result, sort_keys=True))
