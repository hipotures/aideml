import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

RANDOM_STATE = 2026
TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

for c in CAT_COLS:
    cats = pd.Index(
        pd.concat([train[c], test[c]], ignore_index=True).astype(str).unique()
    )
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

y = train[TARGET].astype(int).to_numpy()
base_features = [c for c in train.columns if c not in [ID_COL, TARGET]]


class HierarchicalHazardPriors:
    def __init__(self):
        self.alpha_comp = 300.0
        self.alpha_yc = 120.0
        self.alpha_rc = 120.0
        self.alpha_cs = 70.0
        self.alpha_csl = 25.0

    def _life_bin(self, s):
        return np.clip(np.rint(s).astype(np.int16), 1, 90)

    def _agg(self, d, keys, prefix):
        out = (
            d.groupby(keys, observed=True, sort=False)["_target"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": f"{prefix}_sum", "count": f"{prefix}_count"})
        )
        return out

    @staticmethod
    def _post(df, prefix, prior, alpha):
        sums = df[f"{prefix}_sum"].fillna(0.0).astype(np.float32)
        counts = df[f"{prefix}_count"].fillna(0.0).astype(np.float32)
        return (sums + alpha * prior) / (counts + alpha)

    def fit(self, df, target):
        d = df[["Year", "Race", "Compound", "Stint", "TyreLife"]].copy()
        d["_life_bin"] = self._life_bin(d["TyreLife"])
        d["_target"] = target.astype(np.float32)
        self.global_prior = float(np.clip(d["_target"].mean(), 1e-5, 1 - 1e-5))

        self.comp = self._agg(d, ["Compound"], "comp")
        self.yc = self._agg(d, ["Year", "Compound"], "yc")
        self.rc = self._agg(d, ["Race", "Compound"], "rc")
        self.cs = self._agg(d, ["Compound", "Stint"], "cs")
        self.csl = self._agg(d, ["Compound", "Stint", "_life_bin"], "csl")
        return self

    def transform(self, df):
        out = df[
            [
                "Year",
                "Race",
                "Compound",
                "Stint",
                "TyreLife",
                "LapNumber",
                "RaceProgress",
                "PitStop",
                "Cumulative_Degradation",
            ]
        ].copy()
        out["_life_bin"] = self._life_bin(out["TyreLife"])
        out["_ord"] = np.arange(len(out), dtype=np.int32)

        out = out.merge(self.comp, on=["Compound"], how="left", sort=False)
        out["hazard_compound"] = self._post(
            out, "comp", self.global_prior, self.alpha_comp
        )

        out = out.merge(self.yc, on=["Year", "Compound"], how="left", sort=False)
        out["hazard_year_compound"] = self._post(
            out, "yc", out["hazard_compound"], self.alpha_yc
        )

        out = out.merge(self.rc, on=["Race", "Compound"], how="left", sort=False)
        out["hazard_race_compound"] = self._post(
            out, "rc", out["hazard_compound"], self.alpha_rc
        )

        out["hazard_backoff"] = (
            0.55 * out["hazard_race_compound"] + 0.45 * out["hazard_year_compound"]
        )

        out = out.merge(self.cs, on=["Compound", "Stint"], how="left", sort=False)
        out["hazard_compound_stint"] = self._post(
            out, "cs", out["hazard_backoff"], self.alpha_cs
        )

        out = out.merge(
            self.csl, on=["Compound", "Stint", "_life_bin"], how="left", sort=False
        )
        local_prior = 0.65 * out["hazard_compound_stint"] + 0.35 * out["hazard_backoff"]
        out["hazard_prior"] = self._post(out, "csl", local_prior, self.alpha_csl).clip(
            1e-5, 1 - 1e-5
        )

        out = out.sort_values("_ord")
        h = out["hazard_prior"].astype(np.float32)
        est_total_laps = (out["LapNumber"] / out["RaceProgress"].clip(lower=1e-4)).clip(
            1, 120
        )
        laps_left = (est_total_laps - out["LapNumber"]).clip(0, 120).astype(np.float32)
        unit_cumhaz = -np.log1p(-h)

        feats = pd.DataFrame(index=df.index)
        for c in [
            "hazard_compound",
            "hazard_year_compound",
            "hazard_race_compound",
            "hazard_compound_stint",
            "hazard_backoff",
            "hazard_prior",
        ]:
            feats[c] = out[c].astype(np.float32).to_numpy()

        feats["hazard_logit"] = np.log(h / (1.0 - h)).astype(np.float32)
        feats["prior_cumhaz_to_finish"] = (
            (unit_cumhaz * laps_left).clip(0, 50).astype(np.float32)
        )
        feats["prior_survival_to_finish"] = np.exp(
            -feats["prior_cumhaz_to_finish"]
        ).astype(np.float32)
        feats["prior_stopprob_to_finish"] = (
            1.0 - feats["prior_survival_to_finish"]
        ).astype(np.float32)

        for k in [3, 5, 10]:
            horizon = np.minimum(laps_left, k)
            feats[f"prior_stopprob_next{k}"] = (
                1.0 - np.exp(-(unit_cumhaz * horizon))
            ).astype(np.float32)

        feats["hazard_life_count_log"] = np.log1p(out["csl_count"].fillna(0.0)).astype(
            np.float32
        )
        feats["hazard_race_comp_count_log"] = np.log1p(
            out["rc_count"].fillna(0.0)
        ).astype(np.float32)
        feats["hazard_year_comp_count_log"] = np.log1p(
            out["yc_count"].fillna(0.0)
        ).astype(np.float32)
        feats["pitstop_minus_prior"] = (out["PitStop"].astype(np.float32) - h).astype(
            np.float32
        )
        feats["tyrelife_times_prior"] = (out["TyreLife"].astype(np.float32) * h).astype(
            np.float32
        )
        feats["degradation_per_tyre_lap"] = (
            out["Cumulative_Degradation"].astype(np.float32)
            / (out["TyreLife"].astype(np.float32) + 1.0)
        ).astype(np.float32)
        feats["laps_left_estimate"] = laps_left.astype(np.float32)

        feats = feats.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
        return feats


def make_features(df, prior_model):
    x = df[base_features].copy()
    for c in x.columns:
        if c not in CAT_COLS:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype(np.float32)
    hp = prior_model.transform(df).reset_index(drop=True)
    x = x.reset_index(drop=True)
    return pd.concat([x, hp], axis=1)


def model_params(scale_pos_weight, n_estimators=1800):
    return dict(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        metric="auc",
    )


groups = (
    train["Year"].astype(str)
    + "|"
    + train["Race"].astype(str)
    + "|"
    + train["Driver"].astype(str)
).to_numpy()

try:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(train, y, groups))
except Exception:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(train, y))

oof = np.zeros(len(train), dtype=np.float32)
fold_scores = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    y_tr, y_va = y[tr_idx], y[va_idx]
    prior = HierarchicalHazardPriors().fit(train.iloc[tr_idx], y_tr)

    X_tr = make_features(train.iloc[tr_idx], prior)
    X_va = make_features(train.iloc[va_idx], prior)

    spw = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    model = lgb.LGBMClassifier(**model_params(spw))

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
    oof[va_idx] = pred.astype(np.float32)
    auc = roc_auc_score(y_va, pred)
    fold_scores.append(float(auc))
    best_iters.append(int(model.best_iteration_ or model.n_estimators))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

    del X_tr, X_va, model, prior

overall_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {overall_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int32),
        "target": y.astype(np.int8),
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_prior = HierarchicalHazardPriors().fit(train, y)
X_all = make_features(train, final_prior)
X_test = make_features(test, final_prior)

full_spw = float((len(y) - y.sum()) / max(y.sum(), 1))
final_estimators = max(100, int(np.median(best_iters)))
final_model = lgb.LGBMClassifier(
    **model_params(full_spw, n_estimators=final_estimators)
)
final_model.fit(X_all, y, categorical_feature=CAT_COLS)

test_pred = final_model.predict_proba(X_test)[:, 1].clip(0, 1)

target_col = [c for c in sample.columns if c != ID_COL][0]
sample[target_col] = test_pred
sample.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
sample.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000128"],
    "metric": "roc_auc",
    "cv_folds": len(splits),
    "fold_auc": fold_scores,
    "oof_roc_auc": float(overall_auc),
    "final_n_estimators": int(final_estimators),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review, sort_keys=True))
