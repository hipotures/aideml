import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise RuntimeError(
        "This solution requires lightgbm, which is listed as installed."
    ) from e

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
HYPOTHESES_USED = ["000324"]


def add_features(df):
    df = df.copy()
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["IsTesting"] = (df["Race"].astype(str) == "Pre-Season Testing").astype(int)
    df["IsWet"] = df["Compound"].isin(WET_COMPOUNDS).astype(int)
    df["IsDryRace"] = ((df["IsTesting"] == 0) & (df["IsWet"] == 0)).astype(int)
    df["TyreLife_x_Progress"] = df["TyreLife"] * df["RaceProgress"]
    df["TyreLife_per_Stint"] = df["TyreLife"] / np.maximum(df["Stint"], 1)
    df["Lap_frac"] = df["LapNumber"] / np.maximum(df["LapNumber"].max(), 1)
    df["Deg_per_TyreLife"] = df["Cumulative_Degradation"] / np.maximum(
        df["TyreLife"], 1
    )
    df["Abs_Position_Change"] = df["Position_Change"].abs()
    df["RecentPitOrOutlap"] = ((df["PitStop"] == 1) | (df["TyreLife"] <= 2)).astype(int)
    return df


def regime_mask(df, regime):
    if regime == "testing":
        return df["IsTesting"].values == 1
    if regime == "wet":
        return (df["IsTesting"].values == 0) & (df["IsWet"].values == 1)
    if regime == "dry":
        return (df["IsTesting"].values == 0) & (df["IsWet"].values == 0)
    raise ValueError(regime)


def make_model(seed, n_estimators=700):
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_model(
    train_x, train_y, valid_x, test_x, seed, sample_weight=None, n_estimators=700
):
    model = make_model(seed, n_estimators=n_estimators)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(train_x, train_y, **fit_kwargs)
    valid_pred = model.predict_proba(valid_x)[:, 1] if len(valid_x) else np.array([])
    test_pred = model.predict_proba(test_x)[:, 1] if len(test_x) else np.array([])
    return valid_pred, test_pred


def train_shift_weights(x_train, x_test, seed):
    all_x = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    y_domain = np.r_[np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)]
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(all_x, y_domain)
    p_test_like = np.clip(clf.predict_proba(x_train)[:, 1], 0.02, 0.98)
    prior_train = len(x_train) / (len(x_train) + len(x_test))
    prior_test = len(x_test) / (len(x_train) + len(x_test))
    weights = (p_test_like / (1.0 - p_test_like)) * (prior_train / prior_test)
    return np.clip(weights / np.mean(weights), 0.25, 4.0)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values
test_ids = sample[ID_COL].values

train_fe = add_features(train.drop(columns=[TARGET]))
test_fe = add_features(test)

features = [c for c in train_fe.columns if c != ID_COL]
combined = pd.concat([train_fe[features], test_fe[features]], axis=0, ignore_index=True)

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
combined[CAT_COLS] = enc.fit_transform(combined[CAT_COLS].astype(str))

for c in combined.columns:
    if combined[c].dtype == "object":
        combined[c] = combined[c].astype("category").cat.codes
combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)

x = combined.iloc[: len(train_fe)].reset_index(drop=True)
x_test = combined.iloc[len(train_fe) :].reset_index(drop=True)

groups = train_fe["Year"].values
logo = LeaveOneGroupOut()
oof = np.zeros(len(train_fe), dtype=float)
test_fold_preds = []

for fold, (tr_idx, va_idx) in enumerate(logo.split(x, y, groups), 1):
    x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    seed = 324 + fold

    weights = train_shift_weights(x_tr, x_test, seed)
    pooled_va, pooled_test = fit_predict_model(x_tr, y_tr, x_va, x_test, seed, weights)

    fold_va = pooled_va.copy()
    fold_test = pooled_test.copy()

    for regime in ["dry", "wet", "testing"]:
        tr_mask_all = regime_mask(train_fe.iloc[tr_idx], regime)
        va_mask = regime_mask(train_fe.iloc[va_idx], regime)
        te_mask = regime_mask(test_fe, regime)

        if tr_mask_all.sum() >= 1500 and np.unique(y_tr[tr_mask_all]).size == 2:
            reg_va, reg_test = fit_predict_model(
                x_tr.iloc[tr_mask_all],
                y_tr[tr_mask_all],
                x_va.iloc[va_mask],
                x_test.iloc[te_mask],
                seed + 100,
                None,
                n_estimators=500,
            )
            fold_va[va_mask] = 0.75 * reg_va + 0.25 * pooled_va[va_mask]
            fold_test[te_mask] = 0.75 * reg_test + 0.25 * pooled_test[te_mask]

    oof[va_idx] = np.clip(fold_va, 0, 1)
    test_fold_preds.append(np.clip(fold_test, 0, 1))
    print(
        f"Fold {fold} held-out Year={sorted(set(groups[va_idx]))} AUC: {roc_auc_score(y_va, oof[va_idx]):.6f}"
    )

cv_auc = roc_auc_score(y, oof)
test_pred = np.mean(test_fold_preds, axis=0)

print(f"LeaveOneGroupOut Year ROC AUC: {cv_auc:.6f}")
print({"research_hypotheses_llm_claimed_used": HYPOTHESES_USED})

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_prediction_df = sample.copy()
test_prediction_df[TARGET] = np.clip(test_pred, 0, 1)
test_prediction_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
test_prediction_df.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
