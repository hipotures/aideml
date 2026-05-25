import os
import re
import gc
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
SEED = 2026
os.makedirs(WORK_DIR, exist_ok=True)


def sanitize_columns(df):
    new_cols, seen = [], {}
    for col in df.columns:
        name = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
        if not name:
            name = "feature"
        if name[0].isdigit():
            name = "f_" + name
        base, k = name, 1
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen[name] = 1
        new_cols.append(name)
    out = df.copy()
    out.columns = new_cols
    return out, dict(zip(df.columns, new_cols))


def make_base_features(df):
    out = df.copy()
    for col in ["Race", "Driver", "Compound"]:
        out[col] = out[col].fillna("missing").astype(str)

    out["Year"] = out["Year"].astype(int)
    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["WetDry"] = np.where(
        out["Compound"].isin(["INTERMEDIATE", "WET"]), "WET", "DRY"
    )

    dry_rank_map = {"HARD": 0.0, "MEDIUM": 1.0, "SOFT": 2.0}
    out["DryRank"] = out["Compound"].map(dry_rank_map)
    dry_mask = out["DryRank"].notna()

    # Data-only absolute compound proxy: infer each race-year allocation as C1-C3,
    # C2-C4, or C3-C5 from dry tyre-life/degradation context.
    event_stats = (
        out.loc[dry_mask]
        .groupby("Race_Year", sort=False)
        .agg(
            med_tyre_life=("TyreLife", "median"),
            med_degradation=("Cumulative_Degradation", "median"),
        )
    )
    if len(event_stats):
        score = 0.70 * event_stats["med_tyre_life"].rank(pct=True) - 0.30 * event_stats[
            "med_degradation"
        ].rank(pct=True)
        try:
            alloc_base = pd.qcut(
                score.rank(method="first"), 3, labels=[1, 2, 3]
            ).astype(int)
        except ValueError:
            alloc_base = pd.Series(2, index=event_stats.index)
        base_map = alloc_base.to_dict()
    else:
        base_map = {}

    out["AllocBase"] = out["Race_Year"].map(base_map).fillna(2).astype(int)
    out["AbsCompound"] = out["Compound"]
    abs_num = (
        (out.loc[dry_mask, "AllocBase"] + out.loc[dry_mask, "DryRank"])
        .clip(1, 5)
        .astype(int)
    )
    out.loc[dry_mask, "AbsCompound"] = "C" + abs_num.astype(str)

    max_lap = (
        out.groupby("Race_Year", sort=False)["LapNumber"].transform("max").astype(float)
    )
    out["LapsRemaining"] = (max_lap - out["LapNumber"].astype(float)).clip(lower=0)
    out["LapFractionFromMax"] = out["LapNumber"].astype(float) / max_lap.clip(lower=1)
    out["TyreLifeFraction"] = out["TyreLife"].astype(float) / max_lap.clip(lower=1)
    out["DegPerTyreLap"] = out["Cumulative_Degradation"] / out["TyreLife"].clip(lower=1)
    out["LapTimeDeltaAbs"] = out["LapTime_Delta"].abs()
    out["IsWet"] = (out["WetDry"] == "WET").astype(int)

    bins = [-0.1, 0, 2, 5, 8, 12, 18, 25, 35, 50, 80, 200]
    labels = [
        "0",
        "1_2",
        "3_5",
        "6_8",
        "9_12",
        "13_18",
        "19_25",
        "26_35",
        "36_50",
        "51_80",
        "81p",
    ]
    out["LapBin"] = pd.cut(
        out["LapsRemaining"], bins=bins, labels=labels, include_lowest=True
    ).astype(str)

    out = out.drop(columns=[ID_COL], errors="ignore")
    cat_cols = [
        "Race",
        "Driver",
        "Compound",
        "Race_Year",
        "WetDry",
        "AbsCompound",
        "LapBin",
    ]
    for col in cat_cols:
        out[col] = out[col].astype("category")

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    for col in num_cols:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype(np.float32)

    out, name_map = sanitize_columns(out)
    cat_cols = [name_map[c] for c in cat_cols]
    return out, cat_cols, name_map


class HierarchicalPriorEncoder:
    def __init__(self, levels):
        self.levels = levels
        self.key_cols = {name: cols for name, cols, _, _ in levels}
        self.maps = {}
        self.global_mean = None

    @staticmethod
    def _keys_frame(X, cols):
        return X.loc[:, cols].astype(str).fillna("__NA__")

    def _lookup(self, X, cols, map_df):
        left = self._keys_frame(X, cols)
        left["__ord__"] = np.arange(len(left))
        merged = left.merge(map_df, on=cols, how="left", sort=False)
        merged = merged.sort_values("__ord__")
        return merged["posterior"].to_numpy(dtype=float)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.global_mean = float(np.mean(y))

        for name, cols, parents, alpha in self.levels:
            keys = self._keys_frame(X, cols)
            keys["_target_"] = y
            agg = (
                keys.groupby(cols, sort=False)["_target_"]
                .agg(["sum", "count"])
                .reset_index()
            )

            if parents is None:
                parent_prior = np.full(len(agg), self.global_mean, dtype=float)
            else:
                if isinstance(parents, str):
                    parents = [parents]
                parent_vals = []
                for parent in parents:
                    vals = self._lookup(agg, self.key_cols[parent], self.maps[parent])
                    parent_vals.append(vals)
                parent_prior = np.nanmean(np.vstack(parent_vals), axis=0)
                parent_prior = np.where(
                    np.isfinite(parent_prior), parent_prior, self.global_mean
                )

            posterior = (agg["sum"].to_numpy(dtype=float) + alpha * parent_prior) / (
                agg["count"].to_numpy(dtype=float) + alpha
            )
            map_df = agg[cols].copy()
            map_df["posterior"] = posterior
            self.maps[name] = map_df

        return self

    def transform(self, X):
        out = pd.DataFrame(index=np.arange(len(X)))
        for name, cols, parents, _ in self.levels:
            if parents is None:
                parent_prior = np.full(len(X), self.global_mean, dtype=float)
            else:
                if isinstance(parents, str):
                    parents = [parents]
                parent_prior = np.mean(
                    [out[f"prior_{p}"].to_numpy(dtype=float) for p in parents], axis=0
                )

            raw = self._lookup(X, cols, self.maps[name])
            known = np.isfinite(raw)
            vals = np.where(known, raw, parent_prior)

            out[f"prior_{name}"] = vals.astype(np.float32)
            out[f"prior_{name}_gap_parent"] = (vals - parent_prior).astype(np.float32)
            out[f"prior_{name}_gap_global"] = (vals - self.global_mean).astype(
                np.float32
            )
            out[f"prior_{name}_known"] = known.astype(np.int8)

        return out


def make_splits(y, n_splits=5, seed=SEED):
    counts = np.bincount(np.asarray(y, dtype=int))
    usable = min(n_splits, int(counts.min()))
    if usable < 2:
        raise ValueError("Not enough examples of each class for stratified validation.")
    return list(
        StratifiedKFold(n_splits=usable, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )


def build_oof_priors(X, y, splits, levels):
    oof = None
    y = np.asarray(y)
    for tr_idx, va_idx in splits:
        enc = HierarchicalPriorEncoder(levels).fit(X.iloc[tr_idx], y[tr_idx])
        part = enc.transform(X.iloc[va_idx])
        if oof is None:
            oof = pd.DataFrame(np.nan, index=np.arange(len(X)), columns=part.columns)
        oof.iloc[va_idx] = part.to_numpy()
    return oof.astype(np.float32)


def join_features(base, priors):
    return pd.concat(
        [base.reset_index(drop=True), priors.reset_index(drop=True)], axis=1
    )


def make_model(y_train, seed, n_estimators=1600):
    pos = max(float(np.sum(y_train)), 1.0)
    neg = max(float(len(y_train) - np.sum(y_train)), 1.0)
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.045,
        n_estimators=int(n_estimators),
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=3.0,
        scale_pos_weight=neg / pos,
        random_state=seed,
        n_jobs=max(1, min(8, os.cpu_count() or 1)),
        verbose=-1,
    )


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
combined = pd.concat(
    [train.drop(columns=[TARGET]), test], ignore_index=True, sort=False
)
features_all, cat_cols, name_map = make_base_features(combined)

X_train_base = features_all.iloc[: len(train)].reset_index(drop=True)
X_test_base = features_all.iloc[len(train) :].reset_index(drop=True)

ry = name_map["Race_Year"]
abs_comp = name_map["AbsCompound"]
wetdry = name_map["WetDry"]
lapbin = name_map["LapBin"]

levels = [
    ("abs_stint", [abs_comp, "Stint", wetdry], None, 80.0),
    ("ry_abs", [ry, abs_comp, wetdry], None, 100.0),
    ("ry_abs_stint", [ry, abs_comp, "Stint", wetdry], ["ry_abs", "abs_stint"], 60.0),
    ("ry_abs_stint_lap", [ry, abs_comp, "Stint", lapbin, wetdry], "ry_abs_stint", 35.0),
]

outer_splits = make_splits(y, n_splits=5, seed=SEED)
oof_pred = np.zeros(len(y), dtype=np.float32)
oof_prior_full = None
fold_scores, best_iters = [], []

for fold, (tr_idx, va_idx) in enumerate(outer_splits, start=1):
    X_tr_base = X_train_base.iloc[tr_idx].reset_index(drop=True)
    X_va_base = X_train_base.iloc[va_idx].reset_index(drop=True)
    y_tr, y_va = y[tr_idx], y[va_idx]

    inner_splits = make_splits(y_tr, n_splits=4, seed=SEED + fold)
    tr_priors = build_oof_priors(X_tr_base, y_tr, inner_splits, levels)

    fold_encoder = HierarchicalPriorEncoder(levels).fit(X_tr_base, y_tr)
    va_priors = fold_encoder.transform(X_va_base)

    if oof_prior_full is None:
        oof_prior_full = pd.DataFrame(
            np.nan, index=np.arange(len(y)), columns=va_priors.columns
        )
    oof_prior_full.iloc[va_idx] = va_priors.to_numpy()

    X_tr = join_features(X_tr_base, tr_priors)
    X_va = join_features(X_va_base, va_priors)

    model = make_model(y_tr, seed=SEED + fold, n_estimators=1800)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    pred = model.predict_proba(X_va)[:, 1]
    oof_pred[va_idx] = pred.astype(np.float32)

    fold_auc = roc_auc_score(y_va, pred)
    fold_scores.append(fold_auc)
    best_iters.append(int(getattr(model, "best_iteration_", 0) or model.n_estimators))
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

    del X_tr_base, X_va_base, tr_priors, va_priors, X_tr, X_va, model, fold_encoder
    gc.collect()

cv_auc = roc_auc_score(y, oof_pred)
print(f"OOF ROC AUC: {cv_auc:.6f}")

oof_prior_full = oof_prior_full.astype(np.float32)
oof_df = pd.DataFrame({"row": np.arange(len(y)), "target": y, "prediction": oof_pred})
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

valid_best = [b for b in best_iters if b > 0]
final_estimators = (
    int(np.clip(np.median(valid_best) * 1.10, 200, 1800)) if valid_best else 800
)

full_encoder = HierarchicalPriorEncoder(levels).fit(X_train_base, y)
test_priors = full_encoder.transform(X_test_base)
X_full = join_features(X_train_base, oof_prior_full)
X_test = join_features(X_test_base, test_priors)

final_model = make_model(y, seed=SEED + 999, n_estimators=final_estimators)
final_model.fit(X_full, y, categorical_feature=cat_cols)

test_pred = np.clip(final_model.predict_proba(X_test)[:, 1], 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

test_predictions = sample.copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(x) for x in fold_scores],
    "final_estimators": int(final_estimators),
    "research_hypotheses_llm_claimed_used": ["000186"],
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review))
