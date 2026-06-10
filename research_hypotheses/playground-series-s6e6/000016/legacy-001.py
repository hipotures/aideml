from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from aide_solution_helpers import (
    load_competition_data,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)

RANDOM_STATE = 42
N_SPLITS = 5
TARGET_COL = "class"
ID_COL = "id"
LABELS = ["GALAXY", "STAR", "QSO"]
PHOTOMETRIC_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR_COLS = [
    "u_g_color",
    "g_r_color",
    "r_i_color",
    "i_z_color",
    "u_r_color",
    "g_i_color",
    "r_z_color",
]


def make_color_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["u_g_color"] = df["u"] - df["g"]
    out["g_r_color"] = df["g"] - df["r"]
    out["r_i_color"] = df["r"] - df["i"]
    out["i_z_color"] = df["i"] - df["z"]
    out["u_r_color"] = df["u"] - df["r"]
    out["g_i_color"] = df["g"] - df["i"]
    out["r_z_color"] = df["r"] - df["z"]
    return out


def clean_aux_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PHOTOMETRIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in ["u", "g", "r", "i", "z", "redshift"]:
        out.loc[out[col] < -90, col] = np.nan
    return out


def fill_with_medians(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return df.fillna(medians)


def fit_aux_transform(
    aux: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame | pd.Series], pd.Series, StandardScaler]:
    aux = clean_aux_frame(aux)
    aux_base = pd.concat([aux[PHOTOMETRIC_COLS], make_color_features(aux)], axis=1)
    medians = aux_base.median()

    aux_filled = fill_with_medians(aux_base, medians)
    scaler = StandardScaler()
    scaler.fit(aux_filled)

    aux_scaled = pd.DataFrame(
        scaler.transform(aux_filled), columns=aux_base.columns, index=aux.index
    )
    labels = aux[TARGET_COL].astype(str)

    means = {}
    stds = {}
    counts = {}
    prior_stats = {}
    prior_cols = ["redshift", "u_g_color", "g_r_color", "r_i_color", "i_z_color"]

    for label in LABELS:
        mask = labels == label
        counts[label] = float(mask.sum())
        block = aux_scaled.loc[mask]
        means[label] = block.mean(axis=0)
        std = block.std(axis=0).replace(0.0, 1e-6).fillna(1e-6)
        stds[label] = std

        prior_block = aux_filled.loc[mask, prior_cols]
        prior_stats[label] = {
            "mean": prior_block.mean(axis=0),
            "std": prior_block.std(axis=0).replace(0.0, 1e-6).fillna(1e-6),
        }

    proto = {
        "means": pd.DataFrame(means),
        "stds": pd.DataFrame(stds),
        "counts": pd.Series(counts),
        "priors": prior_stats,
        "prior_cols": prior_cols,
    }
    return proto, medians, scaler


def build_aux_features(
    df: pd.DataFrame, medians: pd.Series, scaler: StandardScaler, proto: dict
) -> pd.DataFrame:
    base = pd.concat([df[PHOTOMETRIC_COLS], make_color_features(df)], axis=1)
    base = base.replace([np.inf, -np.inf], np.nan)
    filled = fill_with_medians(base, medians)
    scaled = pd.DataFrame(
        scaler.transform(filled), columns=base.columns, index=df.index
    )

    means = proto["means"]
    stds = proto["stds"]
    counts = proto["counts"]
    priors = proto["priors"]
    prior_cols = proto["prior_cols"]

    rows = []
    total_count = counts.sum()

    for idx in range(len(df)):
        x = scaled.iloc[idx].to_numpy()
        row = {}

        dist_values = []
        prior_values = []

        for label in LABELS:
            mu = means[label].to_numpy()
            sd = stds[label].to_numpy()
            z = (x - mu) / sd
            dist = float(np.linalg.norm(z))
            manhattan = float(np.abs(z).mean())
            logprior = float(np.log(counts[label] / total_count))

            row[f"aux_dist_{label.lower()}"] = dist
            row[f"aux_manhattan_{label.lower()}"] = manhattan
            row[f"aux_logprior_{label.lower()}"] = logprior

            dist_values.append(dist)

            prior_row = filled.iloc[idx][prior_cols]
            prior_mean = priors[label]["mean"]
            prior_std = priors[label]["std"]
            prior_z = (prior_row - prior_mean) / prior_std
            prior_score = float(-(prior_z**2).sum())
            row[f"aux_prior_score_{label.lower()}"] = prior_score
            prior_values.append(prior_score)

        dist_sorted = sorted(dist_values)
        prior_sorted = sorted(prior_values)
        row["aux_best_dist"] = float(dist_sorted[0])
        row["aux_dist_gap"] = float(dist_sorted[1] - dist_sorted[0])
        row["aux_prior_gap"] = float(prior_sorted[-1] - prior_sorted[-2])

        rows.append(row)

    return pd.DataFrame(rows, index=df.index)


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame, aux: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proto, medians, scaler = fit_aux_transform(aux)

    train_base = pd.concat(
        [train[PHOTOMETRIC_COLS], make_color_features(train)], axis=1
    )
    test_base = pd.concat([test[PHOTOMETRIC_COLS], make_color_features(test)], axis=1)

    train_base = fill_with_medians(
        train_base.replace([np.inf, -np.inf], np.nan), medians
    )
    test_base = fill_with_medians(test_base.replace([np.inf, -np.inf], np.nan), medians)

    train_scaled = pd.DataFrame(
        scaler.transform(train_base),
        columns=train_base.columns,
        index=train.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_base),
        columns=test_base.columns,
        index=test.index,
    )

    train_aux = build_aux_features(train, medians, scaler, proto)
    test_aux = build_aux_features(test, medians, scaler, proto)

    train_features = pd.concat([train_scaled, train_aux], axis=1)
    test_features = pd.concat([test_scaled, test_aux], axis=1)
    return train_features, test_features


def main() -> None:
    train, test, sample_sub = load_competition_data()
    aux = pd.read_csv(Path("./input/original_sdss17/star_classification.csv"))

    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    y = train[TARGET_COL].astype(str)
    y_idx = y.map(label_to_idx).to_numpy()

    train_features, test_features = prepare_features(train, test, aux)

    oof_proba = np.zeros((len(train), len(LABELS)), dtype=np.float64)
    test_proba = np.zeros((len(test), len(LABELS)), dtype=np.float64)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(train_features, y_idx), start=1):
        X_tr = train_features.iloc[tr_idx]
        X_va = train_features.iloc[va_idx]
        y_tr = y_idx[tr_idx]
        y_va = y_idx[va_idx]

        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr)

        va_proba = model.predict_proba(X_va)
        oof_proba[va_idx] = va_proba
        va_pred = va_proba.argmax(axis=1)
        score = balanced_accuracy_score(y_va, va_pred)
        fold_scores.append(score)
        print(f"Fold {fold} balanced_accuracy: {score:.6f}")

        test_proba += model.predict_proba(test_features) / N_SPLITS

    oof_pred_idx = oof_proba.argmax(axis=1)
    oof_score = balanced_accuracy_score(y_idx, oof_pred_idx)
    print(f"OOF balanced_accuracy: {oof_score:.6f}")
    print(
        f"Fold mean balanced_accuracy: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}"
    )

    oof_frame = pd.DataFrame(
        {
            "row": train[ID_COL].to_numpy(),
            "target": train[TARGET_COL].to_numpy(),
            "prediction": [LABELS[i] for i in oof_pred_idx],
        }
    )
    write_oof_predictions(oof_frame)

    test_pred_idx = test_proba.argmax(axis=1)
    test_labels = [LABELS[i] for i in test_pred_idx]
    test_pred_frame = pd.DataFrame(
        {ID_COL: sample_sub[ID_COL].to_numpy(), TARGET_COL: test_labels}
    )
    write_test_predictions(test_pred_frame)

    submission = pd.DataFrame(
        {ID_COL: sample_sub[ID_COL].to_numpy(), TARGET_COL: test_labels}
    )
    write_submission(submission)


if __name__ == "__main__":
    main()
