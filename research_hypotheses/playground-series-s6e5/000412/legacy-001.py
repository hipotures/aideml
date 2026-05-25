import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
cat_cols = ["Compound", "Driver", "Race"]


def add_features(df):
    df = df.copy()
    df["is_wet_or_intermediate"] = (
        df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    )
    df["is_testing"] = (df["Race"] == "Pre-Season Testing").astype(int)
    df["is_fallback_regime"] = (
        (df["is_wet_or_intermediate"] == 1) | (df["is_testing"] == 1)
    ).astype(int)
    df["is_slick"] = df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(int)
    df["tyre_life_x_progress"] = df["TyreLife"] * df["RaceProgress"]
    df["degradation_per_lap"] = df["Cumulative_Degradation"] / np.maximum(
        df["TyreLife"], 1
    )
    df["lap_remaining_frac"] = 1.0 - df["RaceProgress"]
    df["stint_lap_ratio"] = df["TyreLife"] / np.maximum(df["LapNumber"], 1)
    return df


train_fe = add_features(train)
test_fe = add_features(test)

features = [c for c in train_fe.columns if c not in [target_col, id_col]]
for c in cat_cols:
    all_vals = pd.concat([train_fe[c], test_fe[c]], axis=0).astype(str)
    cats = pd.Categorical(all_vals).categories
    train_fe[c] = pd.Categorical(train_fe[c].astype(str), categories=cats)
    test_fe[c] = pd.Categorical(test_fe[c].astype(str), categories=cats)

y = train_fe[target_col].astype(int).values
groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)

try:
    from lightgbm import LGBMClassifier

    model_kind = "lightgbm"
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder

    model_kind = "sklearn_hgb"


def make_lgbm(seed, fallback=False):
    if fallback:
        return LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=6,
            min_child_samples=25,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            class_weight="balanced",
        )
    return LGBMClassifier(
        objective="binary",
        n_estimators=650,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )


def fit_predict(train_df, val_df, train_idx, val_idx, seed):
    fallback_train = train_df["is_fallback_regime"].values == 1
    fallback_val = val_df["is_fallback_regime"].values == 1
    preds = np.zeros(len(val_df), dtype=float)

    if model_kind == "lightgbm":
        dry_mask = ~fallback_train
        if dry_mask.sum() > 0 and train_df.loc[dry_mask, target_col].nunique() == 2:
            dry_model = make_lgbm(seed, fallback=False)
            dry_model.fit(
                train_df.loc[dry_mask, features],
                train_df.loc[dry_mask, target_col].astype(int),
                categorical_feature=cat_cols,
            )
            if (~fallback_val).sum() > 0:
                preds[~fallback_val] = dry_model.predict_proba(
                    val_df.loc[~fallback_val, features]
                )[:, 1]

        fb_model = make_lgbm(seed + 1000, fallback=True)
        if (
            fallback_train.sum() >= 50
            and train_df.loc[fallback_train, target_col].nunique() == 2
        ):
            fb_fit_mask = fallback_train
        else:
            fb_fit_mask = np.ones(len(train_df), dtype=bool)
        fb_model.fit(
            train_df.loc[fb_fit_mask, features],
            train_df.loc[fb_fit_mask, target_col].astype(int),
            categorical_feature=cat_cols,
        )
        if fallback_val.sum() > 0:
            preds[fallback_val] = fb_model.predict_proba(
                val_df.loc[fallback_val, features]
            )[:, 1]
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr = train_df[features].copy()
        Xva = val_df[features].copy()
        Xtr[cat_cols] = enc.fit_transform(Xtr[cat_cols].astype(str))
        Xva[cat_cols] = enc.transform(Xva[cat_cols].astype(str))

        dry_mask = ~fallback_train
        if dry_mask.sum() > 0 and train_df.loc[dry_mask, target_col].nunique() == 2:
            dry_model = HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.04,
                l2_regularization=0.05,
                random_state=seed,
            )
            dry_model.fit(
                Xtr.loc[dry_mask], train_df.loc[dry_mask, target_col].astype(int)
            )
            if (~fallback_val).sum() > 0:
                preds[~fallback_val] = dry_model.predict_proba(Xva.loc[~fallback_val])[
                    :, 1
                ]

        fb_fit_mask = (
            fallback_train
            if fallback_train.sum() >= 50
            and train_df.loc[fallback_train, target_col].nunique() == 2
            else np.ones(len(train_df), dtype=bool)
        )
        fb_model = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.04,
            l2_regularization=0.1,
            random_state=seed + 1000,
        )
        fb_model.fit(
            Xtr.loc[fb_fit_mask], train_df.loc[fb_fit_mask, target_col].astype(int)
        )
        if fallback_val.sum() > 0:
            preds[fallback_val] = fb_model.predict_proba(Xva.loc[fallback_val])[:, 1]

    fill = train_df[target_col].mean()
    preds = np.where(preds == 0, fill, preds)
    return np.clip(preds, 1e-6, 1 - 1e-6)


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_fe), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(train_fe, y, groups), 1):
    tr_df = train_fe.iloc[tr_idx].reset_index(drop=True)
    va_df = train_fe.iloc[va_idx].reset_index(drop=True)
    preds = fit_predict(tr_df, va_df, tr_idx, va_idx, seed=42 + fold)
    oof[va_idx] = preds
    auc = roc_auc_score(y[va_idx], preds)
    fold_scores.append(auc)
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

full_train = train_fe.reset_index(drop=True)
full_test = test_fe.reset_index(drop=True)
fallback_train = full_train["is_fallback_regime"].values == 1
fallback_test = full_test["is_fallback_regime"].values == 1
test_pred = np.zeros(len(full_test), dtype=float)

if model_kind == "lightgbm":
    dry_model = make_lgbm(2026, fallback=False)
    dry_model.fit(
        full_train.loc[~fallback_train, features],
        full_train.loc[~fallback_train, target_col].astype(int),
        categorical_feature=cat_cols,
    )
    if (~fallback_test).sum() > 0:
        test_pred[~fallback_test] = dry_model.predict_proba(
            full_test.loc[~fallback_test, features]
        )[:, 1]

    fb_model = make_lgbm(3026, fallback=True)
    fb_fit_mask = (
        fallback_train
        if fallback_train.sum() >= 50
        and full_train.loc[fallback_train, target_col].nunique() == 2
        else np.ones(len(full_train), dtype=bool)
    )
    fb_model.fit(
        full_train.loc[fb_fit_mask, features],
        full_train.loc[fb_fit_mask, target_col].astype(int),
        categorical_feature=cat_cols,
    )
    if fallback_test.sum() > 0:
        test_pred[fallback_test] = fb_model.predict_proba(
            full_test.loc[fallback_test, features]
        )[:, 1]
else:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xfull = full_train[features].copy()
    Xtest = full_test[features].copy()
    Xfull[cat_cols] = enc.fit_transform(Xfull[cat_cols].astype(str))
    Xtest[cat_cols] = enc.transform(Xtest[cat_cols].astype(str))

    dry_model = HistGradientBoostingClassifier(
        max_iter=350, learning_rate=0.04, l2_regularization=0.05, random_state=2026
    )
    dry_model.fit(
        Xfull.loc[~fallback_train],
        full_train.loc[~fallback_train, target_col].astype(int),
    )
    if (~fallback_test).sum() > 0:
        test_pred[~fallback_test] = dry_model.predict_proba(Xtest.loc[~fallback_test])[
            :, 1
        ]

    fb_fit_mask = (
        fallback_train
        if fallback_train.sum() >= 50
        and full_train.loc[fallback_train, target_col].nunique() == 2
        else np.ones(len(full_train), dtype=bool)
    )
    fb_model = HistGradientBoostingClassifier(
        max_iter=220, learning_rate=0.04, l2_regularization=0.1, random_state=3026
    )
    fb_model.fit(
        Xfull.loc[fb_fit_mask], full_train.loc[fb_fit_mask, target_col].astype(int)
    )
    if fallback_test.sum() > 0:
        test_pred[fallback_test] = fb_model.predict_proba(Xtest.loc[fallback_test])[
            :, 1
        ]

test_pred = np.where(test_pred == 0, full_train[target_col].mean(), test_pred)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "research_hypotheses_llm_claimed_used": ["000412"],
    "metric": "roc_auc",
    "cv": "5-fold StratifiedGroupKFold grouped by Year_Race",
    "oof_roc_auc": float(cv_auc),
    "mean_fold_roc_auc": float(np.mean(fold_scores)),
    "model": model_kind,
}
with open(os.path.join(WORKING_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review, indent=2))
