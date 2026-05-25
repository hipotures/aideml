import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def race_archetypes(race):
    r = str(race).lower()
    pit_loss = "medium"
    deg = "medium"
    overtake = "medium"
    wet = "medium"

    low_pit = [
        "united states",
        "austrian",
        "bahrain",
        "belgian",
        "saudi",
        "italian",
        "qatar",
    ]
    high_pit = [
        "russian",
        "monaco",
        "singapore",
        "abu dhabi",
        "dutch",
        "hungarian",
        "las vegas",
    ]
    high_deg = [
        "bahrain",
        "spanish",
        "hungarian",
        "qatar",
        "united states",
        "mexico",
        "japanese",
        "emilia",
    ]
    low_deg = ["monaco", "russian", "abu dhabi", "australian", "canadian", "las vegas"]
    hard_overtake = ["monaco", "hungarian", "dutch", "singapore", "emilia", "abu dhabi"]
    easy_overtake = [
        "bahrain",
        "belgian",
        "italian",
        "saudi",
        "united states",
        "austrian",
        "las vegas",
    ]
    wet_sensitive = [
        "belgian",
        "british",
        "dutch",
        "japanese",
        "canadian",
        "brazil",
        "sao paulo",
        "são paulo",
    ]

    if any(x in r for x in low_pit):
        pit_loss = "low"
    if any(x in r for x in high_pit):
        pit_loss = "high"
    if "monaco" in r:
        pit_loss = "very_high"

    if any(x in r for x in high_deg):
        deg = "high"
    if any(x in r for x in low_deg):
        deg = "low"
    if any(x in r for x in ["intermediate", "wet", "pre-season"]):
        deg = "variable"

    if any(x in r for x in hard_overtake):
        overtake = "hard"
    if any(x in r for x in easy_overtake):
        overtake = "easy"
    if "monaco" in r:
        overtake = "very_hard"

    if any(x in r for x in wet_sensitive):
        wet = "high"
    if any(x in r for x in ["bahrain", "qatar", "abu dhabi", "saudi", "las vegas"]):
        wet = "low"

    return pit_loss, deg, overtake, wet


PIT_SCORE = {"low": 0.7, "medium": 1.0, "high": 1.25, "very_high": 1.5}
DEG_SCORE = {"low": 0.7, "medium": 1.0, "high": 1.35, "variable": 1.15}
OVERTAKE_SCORE = {"easy": 0.75, "medium": 1.0, "hard": 1.25, "very_hard": 1.55}
WET_SCORE = {"low": 0.7, "medium": 1.0, "high": 1.3}


def add_manual_features(df):
    out = df.copy()
    arch = out["Race"].map(race_archetypes)
    out["pit_loss_band"] = [x[0] for x in arch]
    out["degradation_band"] = [x[1] for x in arch]
    out["overtake_difficulty"] = [x[2] for x in arch]
    out["wet_sensitivity"] = [x[3] for x in arch]

    out["pit_loss_score"] = out["pit_loss_band"].map(PIT_SCORE).astype(float)
    out["degradation_score"] = out["degradation_band"].map(DEG_SCORE).astype(float)
    out["overtake_score"] = out["overtake_difficulty"].map(OVERTAKE_SCORE).astype(float)
    out["wet_score"] = out["wet_sensitivity"].map(WET_SCORE).astype(float)

    estimated_total = out["LapNumber"] / out["RaceProgress"].clip(0.01, 1.0)
    estimated_total = estimated_total.replace([np.inf, -np.inf], np.nan)
    out["Estimated_Total_Laps"] = estimated_total.fillna(out["LapNumber"].max()).clip(
        out["LapNumber"], 90
    )
    out["Estimated_Laps_Remaining"] = (
        out["Estimated_Total_Laps"] - out["LapNumber"]
    ).clip(0, 90)

    out["deg_per_tyre_lap"] = out["Cumulative_Degradation"] / out["TyreLife"].clip(1)
    out["abs_laptime_delta"] = out["LapTime_Delta"].abs()
    out["is_wet_compound"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)

    out["compound_pit_loss"] = (
        out["Compound"].astype(str) + "_" + out["pit_loss_band"].astype(str)
    )
    out["compound_degradation"] = (
        out["Compound"].astype(str) + "_" + out["degradation_band"].astype(str)
    )
    out["compound_wet_sensitivity"] = (
        out["Compound"].astype(str) + "_" + out["wet_sensitivity"].astype(str)
    )

    out["stint_x_pit_loss"] = out["Stint"] * out["pit_loss_score"]
    out["tyrelife_x_degradation"] = out["TyreLife"] * out["degradation_score"]
    out["remaining_x_overtake"] = (
        out["Estimated_Laps_Remaining"] * out["overtake_score"]
    )
    out["wet_compound_x_sensitivity"] = out["is_wet_compound"] * out["wet_score"]
    return out


train_fe = add_manual_features(train)
test_fe = add_manual_features(test)


def fit_expected_life(df):
    stop_rows = df[df["PitStop"] == 1].copy()
    global_life = (
        float(stop_rows["TyreLife"].median())
        if len(stop_rows)
        else float(df["TyreLife"].median())
    )
    keys = ["Compound", "degradation_band", "Stint"]
    maps = {
        "compound_deg_stint": stop_rows.groupby(keys)["TyreLife"].median(),
        "compound_deg": stop_rows.groupby(["Compound", "degradation_band"])[
            "TyreLife"
        ].median(),
        "compound": stop_rows.groupby(["Compound"])["TyreLife"].median(),
        "global": global_life,
    }
    return maps


def apply_expected_life(df, maps):
    idx1 = pd.MultiIndex.from_frame(df[["Compound", "degradation_band", "Stint"]])
    idx2 = pd.MultiIndex.from_frame(df[["Compound", "degradation_band"]])
    expected = pd.Series(
        idx1.map(maps["compound_deg_stint"]), index=df.index, dtype="float64"
    )
    expected = expected.fillna(
        pd.Series(idx2.map(maps["compound_deg"]), index=df.index, dtype="float64")
    )
    expected = expected.fillna(df["Compound"].map(maps["compound"]))
    expected = expected.fillna(maps["global"]).clip(1)
    out = df.copy()
    out["Expected_TyreLife_At_Stop"] = expected
    out["TyreLife_ExpectedLife_Ratio"] = (
        out["TyreLife"] / out["Expected_TyreLife_At_Stop"]
    )
    out["ratio_x_degradation"] = (
        out["TyreLife_ExpectedLife_Ratio"] * out["degradation_score"]
    )
    out["ratio_x_pit_loss"] = out["TyreLife_ExpectedLife_Ratio"] * out["pit_loss_score"]
    out["ratio_x_overtake"] = out["TyreLife_ExpectedLife_Ratio"] * out["overtake_score"]
    out["stint_x_ratio"] = out["Stint"] * out["TyreLife_ExpectedLife_Ratio"]
    return out


def make_prior_map(df, y):
    tmp = df[["Year", "Race"]].copy()
    tmp["target"] = np.asarray(y, dtype=float)
    global_mean = float(tmp["target"].mean())
    stats = tmp.groupby(["Year", "Race"])["target"].agg(["sum", "count"])
    alpha = 20.0
    prior = (stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)
    return prior, global_mean


def apply_race_year_prior(df, prior, global_mean):
    idx = pd.MultiIndex.from_frame(df[["Year", "Race"]])
    out = df.copy()
    out["race_year_pitnext_prior_oof"] = pd.Series(
        idx.map(prior), index=df.index, dtype="float64"
    ).fillna(global_mean)
    return out


def build_train_oof_priors(df, y, n_splits=5):
    priors = pd.Series(index=df.index, dtype="float64")
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + 100
    )
    for tr_idx, va_idx in skf.split(df, y):
        prior_map, global_mean = make_prior_map(df.iloc[tr_idx], y.iloc[tr_idx])
        priors.iloc[va_idx] = apply_race_year_prior(
            df.iloc[va_idx], prior_map, global_mean
        )["race_year_pitnext_prior_oof"].values
    out = df.copy()
    out["race_year_pitnext_prior_oof"] = priors.fillna(float(y.mean()))
    return out


drop_cols = [ID_COL, TARGET, "Race"]
base_cat_cols = [
    "Compound",
    "Driver",
    "pit_loss_band",
    "degradation_band",
    "overtake_difficulty",
    "wet_sensitivity",
    "compound_pit_loss",
    "compound_degradation",
    "compound_wet_sensitivity",
]


def prepare_xy(train_part, valid_part, y_part):
    life_maps = fit_expected_life(train_part)
    tr = apply_expected_life(train_part, life_maps)
    va = apply_expected_life(valid_part, life_maps)

    tr = build_train_oof_priors(tr, y_part.reset_index(drop=True), n_splits=5)
    prior_map, global_mean = make_prior_map(train_part, y_part)
    va = apply_race_year_prior(va, prior_map, global_mean)
    return tr, va


def encode_fit_transform(X_train, X_valid, X_test=None):
    cat_cols = [c for c in base_cat_cols if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
    )
    Xtr_cat = pd.DataFrame(
        enc.fit_transform(X_train[cat_cols].astype(str)),
        columns=cat_cols,
        index=X_train.index,
    ).astype("int32")
    Xva_cat = pd.DataFrame(
        enc.transform(X_valid[cat_cols].astype(str)),
        columns=cat_cols,
        index=X_valid.index,
    ).astype("int32")

    Xtr = pd.concat([X_train[num_cols].astype("float32"), Xtr_cat], axis=1)
    Xva = pd.concat([X_valid[num_cols].astype("float32"), Xva_cat], axis=1)

    if X_test is None:
        return Xtr, Xva, None, cat_cols

    Xte_cat = pd.DataFrame(
        enc.transform(X_test[cat_cols].astype(str)),
        columns=cat_cols,
        index=X_test.index,
    ).astype("int32")
    Xte = pd.concat([X_test[num_cols].astype("float32"), Xte_cat], axis=1)
    return Xtr, Xva, Xte, cat_cols


try:
    from lightgbm import LGBMClassifier

    model_name = "lightgbm"
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier

    LGBMClassifier = None
    model_name = "hist_gradient_boosting"

y = train[TARGET].astype(int)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(train), dtype=float)
test_preds = np.zeros(len(test), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_fe, y), 1):
    fold_train = train_fe.iloc[tr_idx].reset_index(drop=True)
    fold_valid = train_fe.iloc[va_idx].reset_index(drop=True)
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    y_va = y.iloc[va_idx].reset_index(drop=True)

    tr_aug, va_aug = prepare_xy(fold_train, fold_valid, y_tr)

    feature_cols = [c for c in tr_aug.columns if c not in drop_cols]
    X_tr_raw = tr_aug[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_va_raw = va_aug[feature_cols].replace([np.inf, -np.inf], np.nan)

    X_tr, X_va, _, cat_cols = encode_fit_transform(X_tr_raw, X_va_raw)

    if model_name == "lightgbm":
        clf = LGBMClassifier(
            objective="binary",
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_cols,
        )
        pred = clf.predict_proba(X_va)[:, 1]
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=350,
            max_leaf_nodes=63,
            l2_regularization=0.05,
            random_state=RANDOM_STATE + fold,
        )
        clf.fit(X_tr.fillna(-999), y_tr)
        pred = clf.predict_proba(X_va.fillna(-999))[:, 1]

    oof[va_idx] = pred
    score = roc_auc_score(y_va, pred)
    fold_scores.append(score)
    print(f"fold {fold} roc_auc={score:.6f}")

oof_auc = roc_auc_score(y, oof)
print(f"oof_roc_auc={oof_auc:.6f}")

pd.DataFrame(
    {"row": np.arange(len(train)), "target": y.values, "prediction": oof}
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_life_maps = fit_expected_life(train_fe)
final_train = apply_expected_life(train_fe, final_life_maps)
final_test = apply_expected_life(test_fe, final_life_maps)

final_train = build_train_oof_priors(final_train, y.reset_index(drop=True), n_splits=5)
final_prior_map, final_global_mean = make_prior_map(train_fe, y)
final_test = apply_race_year_prior(final_test, final_prior_map, final_global_mean)

feature_cols = [c for c in final_train.columns if c not in drop_cols]
X_all_raw = final_train[feature_cols].replace([np.inf, -np.inf], np.nan)
X_test_raw = final_test[feature_cols].replace([np.inf, -np.inf], np.nan)

X_all, X_test_enc, _, cat_cols = encode_fit_transform(X_all_raw, X_test_raw)

if model_name == "lightgbm":
    final_model = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    final_model.fit(X_all, y, categorical_feature=cat_cols)
    test_pred = final_model.predict_proba(X_test_enc)[:, 1]
else:
    final_model = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=350,
        max_leaf_nodes=63,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_all.fillna(-999), y)
    test_pred = final_model.predict_proba(X_test_enc.fillna(-999))[:, 1]

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

test_predictions = submission[[ID_COL, TARGET]].copy()
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000956"],
    "validation_metric": "roc_auc",
    "fold_auc": [float(x) for x in fold_scores],
    "oof_auc": float(oof_auc),
    "model": model_name,
}
with open(os.path.join(WORK_DIR, "review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review))
