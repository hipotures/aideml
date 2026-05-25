import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42
os.makedirs(WORKING_DIR, exist_ok=True)


class CounterfactualStopNowFeatures:
    def fit(self, df):
        lap_col = "LapTime (s)"
        d = df.copy()
        race = d["Race"].astype(str)
        year = d["Year"].astype(str)
        ry = race + "|" + year

        nonpit = d[d["PitStop"].eq(0)]
        if len(nonpit):
            nrace = nonpit["Race"].astype(str)
            nry = nrace + "|" + nonpit["Year"].astype(str)
            self.nonpit_lap_global = float(nonpit[lap_col].median())
            self.nonpit_lap_by_ry = nonpit.groupby(nry)[lap_col].median().to_dict()
            self.nonpit_lap_by_race = nonpit.groupby(nrace)[lap_col].median().to_dict()
        else:
            self.nonpit_lap_global = float(d[lap_col].median())
            self.nonpit_lap_by_ry = {}
            self.nonpit_lap_by_race = {}

        pit = d[d["PitStop"].eq(1)]
        if len(pit):
            prace = pit["Race"].astype(str)
            pry = prace + "|" + pit["Year"].astype(str)
            base = pry.map(self.nonpit_lap_by_ry)
            base = base.fillna(prace.map(self.nonpit_lap_by_race)).fillna(
                self.nonpit_lap_global
            )
            loss = (pit[lap_col].astype(float) - base.astype(float)).replace(
                [np.inf, -np.inf], np.nan
            )
            valid = loss.dropna()
            if len(valid):
                lo, hi = valid.quantile([0.02, 0.98])
                loss = loss.clip(lower=max(0.0, float(lo)), upper=max(1.0, float(hi)))
                global_loss = float(np.nanmedian(loss))
                if not np.isfinite(global_loss) or global_loss <= 0:
                    global_loss = 20.0
            else:
                global_loss = 20.0
            loss = loss.fillna(global_loss)
            self.global_pit_loss = global_loss

            tmp = pd.DataFrame(
                {"race": prace.values, "ry": pry.values, "loss": loss.values}
            )
            self.pit_loss_by_ry = tmp.groupby("ry")["loss"].median().to_dict()
            self.pit_loss_by_race = tmp.groupby("race")["loss"].median().to_dict()

            low_bar = float(tmp["loss"].quantile(0.25))
            tmp["low_loss"] = tmp["loss"] <= low_bar
            self.low_loss_rate_by_race = (
                tmp.groupby("race")["low_loss"].mean().to_dict()
            )
            self.global_low_loss_rate = float(tmp["low_loss"].mean())
        else:
            self.global_pit_loss = 20.0
            self.pit_loss_by_ry = {}
            self.pit_loss_by_race = {}
            self.low_loss_rate_by_race = {}
            self.global_low_loss_rate = 0.25

        usable = d[d["PitStop"].eq(0)]
        if len(usable) == 0:
            usable = d
        comp = usable["Compound"].astype(str)
        age = usable["TyreLife"].astype(float).clip(lower=1.0)
        raw = (usable["Cumulative_Degradation"].astype(float) / age).replace(
            [np.inf, -np.inf], np.nan
        )

        if raw.notna().any():
            lo, hi = raw.quantile([0.02, 0.98])
            raw = raw.clip(lower=float(lo), upper=float(hi))
            pos = raw.clip(lower=0.0)
            fallback = pos[pos > 0].median()
            if not np.isfinite(fallback):
                fallback = raw.abs().median()
            self.global_deg_slope = max(
                0.001, float(fallback) if np.isfinite(fallback) else 0.05
            )
            comp_slope = pos.groupby(comp).median()
            self.compound_slope = {
                str(k): (
                    max(0.001, float(v))
                    if np.isfinite(v) and v > 0
                    else self.global_deg_slope
                )
                for k, v in comp_slope.items()
            }
        else:
            self.global_deg_slope = 0.05
            self.compound_slope = {}

        life = usable["TyreLife"].astype(float).clip(lower=1.0)
        self.global_life_p90 = float(life.quantile(0.90)) if len(life) else 35.0
        self.life_p90_by_compound = life.groupby(comp).quantile(0.90).to_dict()

        dry = ["SOFT", "MEDIUM", "HARD"]
        wet = ["INTERMEDIATE", "WET"]
        all_slopes = list(self.compound_slope.values()) or [self.global_deg_slope]
        dry_slopes = [self.compound_slope.get(c, np.nan) for c in dry]
        wet_slopes = [self.compound_slope.get(c, np.nan) for c in wet]
        self.best_any_slope = float(np.nanmin(all_slopes))
        self.best_dry_slope = (
            float(np.nanmin(dry_slopes))
            if np.isfinite(np.nanmin(dry_slopes))
            else self.best_any_slope
        )
        self.best_wet_slope = (
            float(np.nanmin(wet_slopes))
            if np.isfinite(np.nanmin(wet_slopes))
            else self.best_any_slope
        )
        return self

    def transform(self, df):
        out = df.copy()
        race = out["Race"].astype(str)
        year = out["Year"].astype(str)
        ry = race + "|" + year
        compound = out["Compound"].astype(str)

        pit_loss = ry.map(self.pit_loss_by_ry)
        pit_loss = (
            pit_loss.fillna(race.map(self.pit_loss_by_race))
            .fillna(self.global_pit_loss)
            .astype(float)
        )

        slope = (
            compound.map(self.compound_slope)
            .fillna(self.global_deg_slope)
            .astype(float)
        )
        life90 = (
            compound.map(self.life_p90_by_compound)
            .fillna(self.global_life_p90)
            .astype(float)
        )
        low_loss_rate = (
            race.map(self.low_loss_rate_by_race)
            .fillna(self.global_low_loss_rate)
            .astype(float)
        )

        lap = out["LapNumber"].astype(float)
        progress = out["RaceProgress"].astype(float).clip(lower=0.005, upper=1.0)
        expected_total_laps = (
            (lap / progress).replace([np.inf, -np.inf], np.nan).fillna(lap)
        )
        laps_remaining = (expected_total_laps - lap).clip(lower=0.0, upper=90.0)

        age = out["TyreLife"].astype(float).clip(lower=0.0).to_numpy()
        s = slope.to_numpy()
        p = pit_loss.to_numpy()
        lr = low_loss_rate.to_numpy()
        l90 = life90.to_numpy()
        rem = laps_remaining.to_numpy()
        horizon = np.minimum(rem, 20.0)

        is_wet = compound.isin(["INTERMEDIATE", "WET"]).to_numpy()
        best_next_slope = np.where(
            is_wet, min(self.best_any_slope, self.best_wet_slope), self.best_dry_slope
        )

        current1 = s * (age + 1.0)
        current2 = s * (age + 2.0)
        fresh1 = best_next_slope * 1.0
        fresh2 = best_next_slope * 2.0

        best_stop_now_advantage = (current1 + current2) - (p + fresh1 + fresh2)
        expected_wait_discount = lr * p * 0.20
        regret_if_wait_1lap = current1 - fresh2 - expected_wait_discount

        current_horizon_cost = s * (age + 0.5 * horizon) * horizon
        fresh_horizon_cost = p + best_next_slope * (0.5 * horizon) * horizon
        best_next_compound_margin = current_horizon_cost - fresh_horizon_cost

        healthy_tyre = np.clip((l90 - age) / (l90 + 1.0), 0.0, 1.0)
        yellow_flag_option_value_proxy = (
            lr * p * np.sqrt(rem + 1.0) / 10.0 * healthy_tyre
        )
        finishability_pressure = np.clip((age + rem - l90) / (l90 + 1.0), 0.0, None)

        out["cf_pit_loss_baseline"] = pit_loss
        out["cf_compound_degradation_slope"] = slope
        out["cf_laps_remaining_proxy"] = laps_remaining
        out["cf_tyre_life_to_template_p90"] = out["TyreLife"].astype(float) / (
            life90 + 1.0
        )
        out["cf_finishability_pressure"] = finishability_pressure
        out["best_stop_now_advantage"] = best_stop_now_advantage
        out["regret_if_wait_1lap"] = regret_if_wait_1lap
        out["best_next_compound_margin"] = best_next_compound_margin
        out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
        out["cf_stop_now_advantage_net_option"] = (
            best_stop_now_advantage - yellow_flag_option_value_proxy
        )
        out["cf_current_vs_best_compound_slope_margin"] = slope - best_next_slope
        return out


def make_model(n_estimators):
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=max(1, min(8, os.cpu_count() or 1)),
        verbose=-1,
    )


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int)
train_x = train.drop(columns=[TARGET]).copy()
test_x = test.copy()

cat_cols = [c for c in ["Driver", "Race", "Compound"] if c in train_x.columns]
for c in cat_cols:
    cats = pd.Index(
        pd.concat([train_x[c], test_x[c]], ignore_index=True)
        .astype(str)
        .fillna("missing")
        .unique()
    )
    train_x[c] = pd.Categorical(
        train_x[c].astype(str).fillna("missing"), categories=cats
    )
    test_x[c] = pd.Categorical(test_x[c].astype(str).fillna("missing"), categories=cats)

groups = train_x["Year"].astype(str) + "|" + train_x["Race"].astype(str)
if StratifiedGroupKFold is not None:
    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(train_x, y, groups))
    except Exception:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(train_x, y))
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(train_x, y))

oof = np.zeros(len(train_x), dtype=float)
fold_aucs = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    fe = CounterfactualStopNowFeatures().fit(train_x.iloc[tr_idx])
    X_tr = fe.transform(train_x.iloc[tr_idx]).drop(columns=[ID_COL])
    X_va = fe.transform(train_x.iloc[va_idx]).drop(columns=[ID_COL])

    model = make_model(2200)
    model.fit(
        X_tr,
        y.iloc[tr_idx],
        eval_set=[(X_va, y.iloc[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )

    pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y.iloc[va_idx], pred)
    fold_aucs.append(float(auc))
    best_iterations.append(int(model.best_iteration_ or 2200))
    print(f"fold {fold} roc_auc: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"cv roc_auc: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train_x)),
        "target": y.values,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_rounds = int(np.median(best_iterations)) if best_iterations else 1000
final_rounds = max(100, final_rounds)

full_fe = CounterfactualStopNowFeatures().fit(train_x)
X_full = full_fe.transform(train_x).drop(columns=[ID_COL])
X_test = full_fe.transform(test_x).drop(columns=[ID_COL])

final_model = make_model(final_rounds)
final_model.fit(X_full, y, categorical_feature=cat_cols)

test_pred = np.clip(final_model.predict_proba(X_test)[:, 1], 0.0, 1.0)
submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": fold_aucs,
            "final_n_estimators": final_rounds,
            "research_hypotheses_llm_claimed_used": ["000365"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        },
        indent=2,
    )
)
