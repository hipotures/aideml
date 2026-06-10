from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from aide_solution_helpers import (
    load_competition_data,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)

RANDOM_SEED = 42
N_SPLITS = 5


@dataclass
class FeatureBundle:
    train: pd.DataFrame
    test: pd.DataFrame


def signed_log1p(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))


def build_photometric_features(df: pd.DataFrame) -> pd.DataFrame:
    mags = df[["u", "g", "r", "i", "z"]].astype(np.float64).copy()
    feats = mags.copy()

    feats["u_g"] = mags["u"] - mags["g"]
    feats["g_r"] = mags["g"] - mags["r"]
    feats["r_i"] = mags["r"] - mags["i"]
    feats["i_z"] = mags["i"] - mags["z"]
    feats["u_r"] = mags["u"] - mags["r"]
    feats["g_i"] = mags["g"] - mags["i"]
    feats["r_z"] = mags["r"] - mags["z"]
    feats["u_z"] = mags["u"] - mags["z"]
    feats["g_z"] = mags["g"] - mags["z"]

    feats["ug_grad"] = feats["u_g"] - feats["g_r"]
    feats["gr_grad"] = feats["g_r"] - feats["r_i"]
    feats["ri_grad"] = feats["r_i"] - feats["i_z"]

    feats["mag_mean"] = mags.mean(axis=1)
    feats["mag_min"] = mags.min(axis=1)
    feats["mag_max"] = mags.max(axis=1)
    feats["mag_std"] = mags.std(axis=1)
    feats["mag_range"] = feats["mag_max"] - feats["mag_min"]

    redshift = df["redshift"].astype(np.float64)
    feats["redshift"] = redshift
    feats["redshift_log1p"] = np.log1p(np.clip(redshift, a_min=0, a_max=None))
    feats["redshift_sq"] = redshift**2
    feats["redshift_sqrt"] = np.sqrt(np.clip(redshift, a_min=0, a_max=None))
    feats["redshift_signed_log1p"] = signed_log1p(redshift)

    for col in ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z", "u_z"]:
        feats[f"{col}_logabs"] = signed_log1p(feats[col])
        feats[f"{col}_sq"] = feats[col] ** 2

    return feats


def make_feature_bundle(train: pd.DataFrame, test: pd.DataFrame) -> FeatureBundle:
    train_feats = build_photometric_features(train)
    test_feats = build_photometric_features(test)
    return FeatureBundle(train=train_feats, test=test_feats)


def main() -> None:
    train, test, sample_sub = load_competition_data()

    bundle = make_feature_bundle(train, test)
    feature_cols = list(bundle.train.columns)

    X = bundle.train[feature_cols].to_numpy(dtype=np.float64)
    X_test = bundle.test[feature_cols].to_numpy(dtype=np.float64)
    y_raw = train["class"].astype(str).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = list(encoder.classes_)

    oof_proba = np.zeros((len(train), len(classes)), dtype=np.float64)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_test)

        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            auto_class_weights="Balanced",
            iterations=800,
            depth=8,
            learning_rate=0.08,
            l2_leaf_reg=4.0,
            random_seed=RANDOM_SEED + fold,
            od_type="Iter",
            od_wait=80,
            allow_writing_files=False,
            verbose=False,
            thread_count=os.cpu_count() or 1,
        )

        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        oof_proba[va_idx] = model.predict_proba(X_va)
        test_proba += model.predict_proba(X_te) / N_SPLITS

    oof_pred = oof_proba.argmax(axis=1)
    oof_metric = balanced_accuracy_score(y, oof_pred)
    print(f"OOF balanced accuracy: {oof_metric:.6f}")

    oof_frame = pd.DataFrame(
        {
            "row": train["id"].to_numpy(),
            "target": encoder.inverse_transform(y),
            "prediction": encoder.inverse_transform(oof_pred),
        }
    )
    write_oof_predictions(oof_frame)

    test_pred_labels = encoder.inverse_transform(test_proba.argmax(axis=1))
    test_frame = pd.DataFrame({"id": sample_sub["id"].to_numpy()})
    for idx, class_name in enumerate(classes):
        test_frame[class_name] = test_proba[:, idx]
    write_test_predictions(test_frame)

    submission = pd.DataFrame(
        {"id": sample_sub["id"].to_numpy(), "class": test_pred_labels}
    )
    write_submission(submission)


if __name__ == "__main__":
    main()
