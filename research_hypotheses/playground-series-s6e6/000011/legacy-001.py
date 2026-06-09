from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from aide_solution_helpers import (
    load_competition_data,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)

from lightgbm import LGBMClassifier

RANDOM_SEED = 42
N_SPLITS = 5
CLASS_ORDER = ["GALAXY", "STAR", "QSO"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    for col in ["u", "g", "r", "i", "z", "redshift"]:
        feats[col] = df[col].astype(np.float32)

    # Color-engineered features derived only from covariates.
    feats["u_g"] = feats["u"] - feats["g"]
    feats["g_r"] = feats["g"] - feats["r"]
    feats["r_i"] = feats["r"] - feats["i"]
    feats["i_z"] = feats["i"] - feats["z"]
    feats["u_r"] = feats["u"] - feats["r"]
    feats["g_i"] = feats["g"] - feats["i"]
    feats["r_z"] = feats["r"] - feats["z"]
    feats["u_z"] = feats["u"] - feats["z"]

    z = feats["redshift"]
    feats["z_abs"] = z.abs()
    feats["z_sq"] = z * z
    feats["z_log1p_abs"] = np.log1p(z.abs())
    feats["z_pos"] = z.clip(lower=0)
    feats["z_neg"] = (-z).clip(lower=0)
    feats["z_lt_0p2"] = (z < 0.2).astype(np.float32)
    feats["z_0p2_0p6"] = ((z >= 0.2) & (z < 0.6)).astype(np.float32)
    feats["z_ge_0p6"] = (z >= 0.6).astype(np.float32)
    feats["z_x_u_g"] = z * feats["u_g"]
    feats["z_x_g_r"] = z * feats["g_r"]
    feats["z_x_r_i"] = z * feats["r_i"]
    feats["z_x_i_z"] = z * feats["i_z"]

    return feats


def make_binary_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def make_multiclass_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=2,
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    train, test, sample_sub = load_competition_data()

    y = train["class"].astype(str)
    y_star = (y == "STAR").astype(int).to_numpy()

    # Use train+test only for covariate feature construction; no target column is present.
    combined = pd.concat(
        [train.drop(columns=["class"]), test], axis=0, ignore_index=True
    )
    combined_features = build_features(combined)
    x_train = combined_features.iloc[: len(train)].reset_index(drop=True)
    x_test = combined_features.iloc[len(train) :].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    oof_proba = np.zeros((len(train), 3), dtype=np.float64)
    test_proba = np.zeros((len(test), 3), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        x_tr = x_train.iloc[tr_idx]
        x_va = x_train.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        star_model = make_binary_model()
        star_model.fit(
            x_tr,
            (y_tr == "STAR").astype(int),
            eval_set=[(x_va, (y_va == "STAR").astype(int))],
            eval_metric="binary_logloss",
            callbacks=[],
        )

        p_star_va = star_model.predict_proba(x_va)[:, 1]
        p_star_te = star_model.predict_proba(x_test)[:, 1]

        nonstar_mask = y_tr != "STAR"
        x_tr_nonstar = x_tr.loc[nonstar_mask]
        y_tr_nonstar = (y_tr.loc[nonstar_mask] == "QSO").astype(int)

        x_va_nonstar = x_va
        y_va_nonstar = (y_va != "STAR").astype(int)

        nonstar_model = make_multiclass_model()
        nonstar_model.fit(
            x_tr_nonstar,
            y_tr_nonstar,
            eval_set=[(x_va_nonstar, y_va_nonstar)],
            eval_metric="multi_logloss",
            callbacks=[],
        )

        p_nonstar_va = nonstar_model.predict_proba(x_va)
        p_nonstar_te = nonstar_model.predict_proba(x_test)

        # Hierarchical combination: first STAR vs non-STAR, then GALAXY vs QSO.
        oof_proba[va_idx, 1] = p_star_va
        oof_proba[va_idx, 0] = (1.0 - p_star_va) * p_nonstar_va[:, 0]
        oof_proba[va_idx, 2] = (1.0 - p_star_va) * p_nonstar_va[:, 1]

        test_proba[:, 1] += p_star_te / N_SPLITS
        test_proba[:, 0] += ((1.0 - p_star_te) * p_nonstar_te[:, 0]) / N_SPLITS
        test_proba[:, 2] += ((1.0 - p_star_te) * p_nonstar_te[:, 1]) / N_SPLITS

        fold_pred = np.array(CLASS_ORDER)[np.argmax(oof_proba[va_idx], axis=1)]
        fold_score = balanced_accuracy_score(y_va, fold_pred)
        print(f"Fold {fold}: balanced_accuracy={fold_score:.6f}")

    oof_pred = np.array(CLASS_ORDER)[np.argmax(oof_proba, axis=1)]
    cv_score = balanced_accuracy_score(y, oof_pred)
    print(f"OOF balanced_accuracy={cv_score:.6f}")

    oof_frame = pd.DataFrame(
        {
            "row": train["id"].to_numpy(),
            "target": y.to_numpy(),
            "prediction": oof_pred,
        }
    )
    write_oof_predictions(oof_frame)

    submission = pd.DataFrame(
        {
            "id": sample_sub["id"],
            "class": np.array(CLASS_ORDER)[np.argmax(test_proba, axis=1)],
        }
    )
    write_submission(submission)

    test_pred_frame = pd.DataFrame({"id": sample_sub["id"]})
    for idx, cls in enumerate(CLASS_ORDER):
        test_pred_frame[cls] = test_proba[:, idx]
    write_test_predictions(test_pred_frame)


if __name__ == "__main__":
    main()
