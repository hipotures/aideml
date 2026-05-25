import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

SEED = 2026
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"

PRESSURE_FEATURES = [
    "service_pressure",
    "laps_since_pit_pressure",
    "tyre_age_overage",
    "next_lap_service_pressure",
    "finishability_pressure",
    "degradation_pressure",
    "lap_slowdown_pressure",
    "service_safety_margin",
    "finish_safety_margin",
    "current_pit_safety",
]
MONOTONE_CONSTRAINTS = [1, 1, 1, 1, 1, 1, 1, -1, -1, -1]


def clean_columns(cols):
    out, seen = [], {}
    for c in cols:
        s = re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_")
        s = s or "col"
        if s in seen:
            seen[s] += 1
            s = f"{s}_{seen[s]}"
        else:
            seen[s] = 0
        out.append(s)
    return out


def rank01(x):
    return pd.Series(x).rank(method="average").to_numpy(dtype=np.float64) / (
        len(x) + 1.0
    )


def safe_quantile(s, q, default):
    v = pd.to_numeric(s, errors="coerce").quantile(q)
    return float(v) if np.isfinite(v) else float(default)


class PressureFeatureBuilder:
    def fit(self, df):
        stops = df[
            (pd.to_numeric(df["PitStop"], errors="coerce").fillna(0) > 0)
            & (pd.to_numeric(df["TyreLife"], errors="coerce").fillna(0) > 1)
        ]
        if len(stops) < 50:
            stops = df[pd.to_numeric(df["TyreLife"], errors="coerce").fillna(0) > 1]

        self.global_service_life = safe_quantile(stops["TyreLife"], 0.55, 22.0)
        self.global_max_life = max(
            self.global_service_life + 2.0,
            safe_quantile(df["TyreLife"], 0.90, self.global_service_life * 1.4),
        )

        long_run = df[pd.to_numeric(df["TyreLife"], errors="coerce").fillna(0) > 1]
        self.service_stats = {
            ("Year", "Race", "Compound"): stops.groupby(
                ["Year", "Race", "Compound"], observed=True
            )["TyreLife"].quantile(0.55),
            ("Race", "Compound"): stops.groupby(["Race", "Compound"], observed=True)[
                "TyreLife"
            ].quantile(0.55),
            ("Compound",): stops.groupby(["Compound"], observed=True)[
                "TyreLife"
            ].quantile(0.55),
        }
        self.max_stats = {
            ("Race", "Compound"): long_run.groupby(["Race", "Compound"], observed=True)[
                "TyreLife"
            ].quantile(0.90),
            ("Compound",): long_run.groupby(["Compound"], observed=True)[
                "TyreLife"
            ].quantile(0.90),
        }
        return self

    def _map_stat(self, df, stats, keys):
        left = df[list(keys)].astype(str).reset_index(drop=True)
        right = stats.rename("__value__").reset_index()
        for k in keys:
            right[k] = right[k].astype(str)
        return left.merge(right, on=list(keys), how="left", sort=False)[
            "__value__"
        ].astype(float)

    def transform(self, df):
        expected = pd.Series(np.nan, index=np.arange(len(df)), dtype=float)
        for keys in [("Year", "Race", "Compound"), ("Race", "Compound"), ("Compound",)]:
            expected = expected.fillna(
                self._map_stat(df, self.service_stats[keys], keys)
            )
        expected = (
            expected.fillna(self.global_service_life)
            .clip(3.0, 90.0)
            .to_numpy(dtype=np.float32)
        )

        max_life = pd.Series(np.nan, index=np.arange(len(df)), dtype=float)
        for keys in [("Race", "Compound"), ("Compound",)]:
            max_life = max_life.fillna(self._map_stat(df, self.max_stats[keys], keys))
        max_life = (
            max_life.fillna(self.global_max_life)
            .clip(4.0, 110.0)
            .to_numpy(dtype=np.float32)
        )
        max_life = np.maximum(max_life, expected + 2.0)

        tyre = (
            pd.to_numeric(df["TyreLife"], errors="coerce")
            .fillna(0)
            .clip(0, 120)
            .to_numpy(dtype=np.float32)
        )
        lap = pd.to_numeric(df["LapNumber"], errors="coerce").fillna(0).clip(0, 120)
        progress = (
            pd.to_numeric(df["RaceProgress"], errors="coerce")
            .fillna(0.5)
            .clip(0.01, 1.0)
        )
        remaining = (lap / progress - lap).clip(0, 120).to_numpy(dtype=np.float32)

        deg = (
            pd.to_numeric(df["Cumulative_Degradation"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        lap_delta = (
            pd.to_numeric(df["LapTime_Delta"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        lap_time = (
            pd.to_numeric(df["LapTime_s"], errors="coerce")
            .fillna(90)
            .clip(1, 3000)
            .to_numpy(dtype=np.float32)
        )
        pit_now = (
            pd.to_numeric(df["PitStop"], errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .to_numpy(dtype=np.float32)
        )

        eps = 1e-3
        feats = pd.DataFrame(
            {
                "service_pressure": np.clip(
                    (tyre - expected) / (expected + eps), -2.0, 3.0
                ),
                "laps_since_pit_pressure": np.clip(tyre / (expected + eps), 0.0, 4.0),
                "tyre_age_overage": np.clip(
                    np.maximum(tyre - expected, 0) / (expected + eps), 0.0, 4.0
                ),
                "next_lap_service_pressure": np.clip(
                    (tyre + 1.0 - expected) / (expected + eps), -2.0, 3.0
                ),
                "finishability_pressure": np.clip(
                    np.maximum(tyre + remaining - max_life, 0) / (max_life + eps),
                    0.0,
                    5.0,
                ),
                "degradation_pressure": np.clip(deg / (expected + eps), -5.0, 12.0),
                "lap_slowdown_pressure": np.clip(
                    np.maximum(lap_delta, 0) / lap_time, 0.0, 5.0
                ),
                "service_safety_margin": np.clip(
                    (expected - tyre) / (expected + eps), -3.0, 3.0
                ),
                "finish_safety_margin": np.clip(
                    (max_life - tyre - remaining) / (max_life + eps), -5.0, 5.0
                ),
                "current_pit_safety": pit_now,
            }
        )
        return feats.astype(np.float32)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

rename_map = dict(zip(train.columns, clean_columns(train.columns)))
train = train.rename(columns=rename_map)
test = test.rename(
    columns={c: rename_map.get(c, clean_columns([c])[0]) for c in test.columns}
)

y = train[TARGET].astype(int).to_numpy()
base_features = [c for c in test.columns if c != ID_COL]

cat_cols = [
    c for c in base_features if train[c].dtype == "object" or test[c].dtype == "object"
]
for c in cat_cols:
    cats = pd.Index(
        pd.concat([train[c], test[c]], ignore_index=True)
        .astype(str)
        .fillna("__MISSING__")
        .unique()
    )
    train[c] = pd.Categorical(
        train[c].astype(str).fillna("__MISSING__"), categories=cats
    )
    test[c] = pd.Categorical(test[c].astype(str).fillna("__MISSING__"), categories=cats)

num_cols = [c for c in base_features if c not in cat_cols]
for c in num_cols:
    med = pd.to_numeric(train[c], errors="coerce").median()
    train[c] = pd.to_numeric(train[c], errors="coerce").fillna(med).astype(np.float32)
    test[c] = pd.to_numeric(test[c], errors="coerce").fillna(med).astype(np.float32)

oof_uncon = np.zeros(len(train), dtype=np.float64)
oof_mono = np.zeros(len(train), dtype=np.float64)
test_uncon = np.zeros(len(test), dtype=np.float64)
test_mono = np.zeros(len(test), dtype=np.float64)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
    builder = PressureFeatureBuilder().fit(train.iloc[tr_idx])

    p_tr = builder.transform(train.iloc[tr_idx])
    p_va = builder.transform(train.iloc[va_idx])
    p_te = builder.transform(test)

    x_tr = pd.concat(
        [
            train.iloc[tr_idx][base_features].reset_index(drop=True),
            p_tr.reset_index(drop=True),
        ],
        axis=1,
    )
    x_va = pd.concat(
        [
            train.iloc[va_idx][base_features].reset_index(drop=True),
            p_va.reset_index(drop=True),
        ],
        axis=1,
    )
    x_te = pd.concat(
        [test[base_features].reset_index(drop=True), p_te.reset_index(drop=True)],
        axis=1,
    )

    y_tr, y_va = y[tr_idx], y[va_idx]

    uncon = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1200,
        learning_rate=0.045,
        num_leaves=96,
        min_child_samples=90,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=4.0,
        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    uncon.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="auc",
        categorical_feature=[c for c in cat_cols if c in x_tr.columns],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )

    mono = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=140,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.95,
        reg_alpha=0.1,
        reg_lambda=6.0,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=SEED + 100 + fold,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    mono.fit(
        p_tr[PRESSURE_FEATURES],
        y_tr,
        eval_set=[(p_va[PRESSURE_FEATURES], y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )

    va_un = uncon.predict_proba(x_va)[:, 1]
    va_mo = mono.predict_proba(p_va[PRESSURE_FEATURES])[:, 1]
    oof_uncon[va_idx] = va_un
    oof_mono[va_idx] = va_mo

    test_uncon += uncon.predict_proba(x_te)[:, 1] / N_SPLITS
    test_mono += mono.predict_proba(p_te[PRESSURE_FEATURES])[:, 1] / N_SPLITS

    print(
        f"fold {fold}: uncon_auc={roc_auc_score(y_va, va_un):.6f} "
        f"mono_auc={roc_auc_score(y_va, va_mo):.6f} "
        f"prob_blend_auc={roc_auc_score(y_va, 0.80 * va_un + 0.20 * va_mo):.6f}"
    )

oof_pred = np.clip(0.80 * oof_uncon + 0.20 * rank01(oof_mono), 0, 1)
test_pred = np.clip(0.80 * test_uncon + 0.20 * rank01(test_mono), 0, 1)

auc_uncon = roc_auc_score(y, oof_uncon)
auc_mono = roc_auc_score(y, oof_mono)
auc_blend = roc_auc_score(y, oof_pred)

print(f"OOF ROC AUC unconstrained: {auc_uncon:.6f}")
print(f"OOF ROC AUC monotone specialist: {auc_mono:.6f}")
print(f"OOF ROC AUC blended: {auc_blend:.6f}")

submission = sample.copy()
submission[TARGET] = test_pred
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

test_predictions = sample.copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv": f"{N_SPLITS}-fold StratifiedKFold",
    "oof_roc_auc": float(auc_blend),
    "unconstrained_oof_roc_auc": float(auc_uncon),
    "monotone_specialist_oof_roc_auc": float(auc_mono),
    "research_hypotheses_llm_claimed_used": ["000769"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
