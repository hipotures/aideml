import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
NUM_BASE_COLS = [
    "Year",
    "LapNumber",
    "LapTime (s)",
    "LapTime_Delta",
    "PitStop",
    "Position",
    "Position_Change",
    "RaceProgress",
    "Stint",
    "TyreLife",
    "Cumulative_Degradation",
]
FRESH_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def robust_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 20 or np.nanstd(x) < 1e-9:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(np.clip(slope, -0.75, 1.75))


class CounterfactualPitFeatures:
    def fit(self, df):
        d = df.copy()
        d["race_year"] = d["Race"].astype(str) + "_" + d["Year"].astype(str)
        nonpit = d[d["PitStop"] == 0].copy()
        pit = d[d["PitStop"] == 1].copy()

        self.global_lap_median_ = float(nonpit["LapTime (s)"].median())
        self.global_pit_loss_ = 22.0

        if len(pit) and len(nonpit):
            normal_by_ry = nonpit.groupby("race_year")["LapTime (s)"].median()
            pit_tmp = pit.join(normal_by_ry.rename("normal_lap"), on="race_year")
            losses = pit_tmp["LapTime (s)"] - pit_tmp["normal_lap"]
            losses = losses.replace([np.inf, -np.inf], np.nan).dropna()
            losses = losses[(losses > 5) & (losses < 80)]
            if len(losses):
                self.global_pit_loss_ = float(losses.median())
            self.pit_loss_by_ry_ = (
                losses.groupby(pit_tmp.loc[losses.index, "race_year"])
                .median()
                .clip(8, 60)
                .to_dict()
                if len(losses)
                else {}
            )
        else:
            self.pit_loss_by_ry_ = {}

        self.median_lap_by_ry_ = (
            nonpit.groupby("race_year")["LapTime (s)"].median().to_dict()
        )
        self.median_lap_by_compound_ = (
            nonpit.groupby("Compound")["LapTime (s)"].median().to_dict()
        )

        self.decay_by_compound_ = {}
        for comp, g in nonpit.groupby("Compound"):
            self.decay_by_compound_[comp] = robust_slope(
                g["TyreLife"], g["LapTime (s)"]
            )
        self.global_decay_ = robust_slope(nonpit["TyreLife"], nonpit["LapTime (s)"])
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        race_year = df["Race"].astype(str) + "_" + df["Year"].astype(str)

        pit_loss = (
            race_year.map(self.pit_loss_by_ry_)
            .fillna(self.global_pit_loss_)
            .astype(float)
        )
        ry_base = (
            race_year.map(self.median_lap_by_ry_)
            .fillna(self.global_lap_median_)
            .astype(float)
        )
        comp_base = (
            df["Compound"]
            .map(self.median_lap_by_compound_)
            .fillna(self.global_lap_median_)
            .astype(float)
        )
        curr_decay = (
            df["Compound"]
            .map(self.decay_by_compound_)
            .fillna(self.global_decay_)
            .astype(float)
        )

        current_life = df["TyreLife"].astype(float)
        current_lap = df["LapTime (s)"].astype(float)
        current_continue = current_lap + curr_decay.clip(-0.5, 1.5)
        stay_then_pit_cost = current_continue + pit_loss

        fresh_costs = []
        fresh_continuations = []
        dry_track = ~df["Compound"].isin(["INTERMEDIATE", "WET"])
        for comp in FRESH_COMPOUNDS:
            decay = self.decay_by_compound_.get(comp, self.global_decay_)
            base = self.median_lap_by_compound_.get(comp, self.global_lap_median_)
            fresh_lap = 0.55 * ry_base + 0.45 * base + decay
            if comp in ["INTERMEDIATE", "WET"]:
                feasible = ~dry_track
            else:
                feasible = dry_track
            fresh_lap = np.where(feasible, fresh_lap, fresh_lap + 30.0)
            fresh_continuations.append(fresh_lap)
            fresh_costs.append(fresh_lap + pit_loss)

        fresh_costs = np.vstack(fresh_costs)
        fresh_continuations = np.vstack(fresh_continuations)
        best_stop_now = np.min(fresh_costs, axis=0)
        worst_stop_now = np.max(fresh_costs, axis=0)
        best_fresh_next = np.min(fresh_continuations, axis=0)

        out["cf_pit_loss"] = pit_loss
        out["cf_current_decay"] = curr_decay
        out["cf_current_continue_lap"] = current_continue
        out["cf_best_stop_now_cost"] = best_stop_now
        out["cf_worst_stop_now_cost"] = worst_stop_now
        out["cf_wait_then_pit_cost"] = stay_then_pit_cost
        out["cf_best_stop_now_advantage"] = stay_then_pit_cost - best_stop_now
        out["cf_worst_delay_penalty"] = stay_then_pit_cost - worst_stop_now
        out["cf_current_vs_best_fresh_gap"] = current_continue - best_fresh_next
        out["cf_decay_x_tyre_life"] = curr_decay * current_life
        out["cf_remaining_race_laps_est"] = (
            df["LapNumber"]
            * (1.0 - df["RaceProgress"])
            / np.maximum(df["RaceProgress"], 0.01)
        )
        out["cf_can_finish_current_stint"] = (
            out["cf_remaining_race_laps_est"] <= np.maximum(1.0, 70.0 - current_life)
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_basic_features(df):
    x = df.copy()
    x["lap_progress_ratio"] = x["LapNumber"] / np.maximum(
        x["RaceProgress"] * 100.0, 1.0
    )
    x["tyre_life_progress"] = x["TyreLife"] * x["RaceProgress"]
    x["deg_per_tyre_lap"] = x["Cumulative_Degradation"] / np.maximum(x["TyreLife"], 1.0)
    x["abs_position_change"] = x["Position_Change"].abs()
    x["is_wet_compound"] = x["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    return x


def make_xy(train_part, valid_part, test_part=None):
    cf = CounterfactualPitFeatures().fit(train_part)
    parts = [train_part, valid_part] + ([] if test_part is None else [test_part])
    made = []
    for p in parts:
        b = add_basic_features(p.drop(columns=[TARGET], errors="ignore"))
        c = cf.transform(p.drop(columns=[TARGET], errors="ignore"))
        z = pd.concat([b, c], axis=1)
        for col in CAT_COLS:
            z[col] = z[col].astype("category")
        made.append(z)
    return made


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=711)
oof = np.zeros(len(train), dtype=float)
test_fold_preds = np.zeros(len(test), dtype=float)
scores = []

params = dict(
    objective="binary",
    n_estimators=1400,
    learning_rate=0.035,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=711,
    n_jobs=-1,
    verbose=-1,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
    tr, va = train.iloc[tr_idx].copy(), train.iloc[va_idx].copy()
    x_tr, x_va, x_te = make_xy(tr, va, test)
    drop_cols = [ID_COL]
    features = [c for c in x_tr.columns if c not in drop_cols]

    model = LGBMClassifier(**params)
    model.fit(
        x_tr[features],
        tr[TARGET].astype(int),
        eval_set=[(x_va[features], va[TARGET].astype(int))],
        eval_metric="auc",
        categorical_feature=[c for c in CAT_COLS if c in features],
        callbacks=[],
    )

    pred = model.predict_proba(x_va[features])[:, 1]
    oof[va_idx] = pred
    test_fold_preds += model.predict_proba(x_te[features])[:, 1] / skf.n_splits
    auc = roc_auc_score(va[TARGET].astype(int), pred)
    scores.append(auc)
    print(f"fold {fold} roc_auc={auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold CV ROC AUC: {cv_auc:.6f}")
print(
    json.dumps(
        {"research_hypotheses_llm_claimed_used": ["000711"], "cv_roc_auc": cv_auc}
    )
)

pd.DataFrame({"row": np.arange(len(train)), "target": y, "prediction": oof}).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

sample[TARGET] = np.clip(test_fold_preds, 0, 1)
sample.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
sample.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
print(f"saved {os.path.join(WORK_DIR, 'submission.csv')}")
