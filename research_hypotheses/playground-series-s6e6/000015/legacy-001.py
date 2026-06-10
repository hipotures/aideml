from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aide_solution_helpers import (
    load_competition_data,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)

RANDOM_STATE = 42
N_SPLITS = 5
TARGET_CLASSES = ["GALAXY", "STAR", "QSO"]


def make_color_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["u_g"] = frame["u"] - frame["g"]
    out["g_r"] = frame["g"] - frame["r"]
    out["r_i"] = frame["r"] - frame["i"]
    out["i_z"] = frame["i"] - frame["z"]
    out["u_r"] = frame["u"] - frame["r"]
    out["g_i"] = frame["g"] - frame["i"]
    out["r_z"] = frame["r"] - frame["z"]
    return out


def build_bin_edges(values: pd.Series, q: int = 8) -> np.ndarray:
    try:
        _, edges = pd.qcut(values, q=q, retbins=True, duplicates="drop")
    except ValueError:
        edges = np.array([values.min() - 1e-6, values.max() + 1e-6], dtype=float)
    if len(edges) < 2:
        edges = np.array([values.min() - 1e-6, values.max() + 1e-6], dtype=float)
    edges = np.asarray(edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    if len(edges) <= 2:
        return pd.Series(["bin_0"] * len(values), index=values.index)
    binned = pd.cut(values, bins=edges, include_lowest=True)
    return binned.astype(str).fillna("bin_nan")


def smoothed_target_encoding(
    train_col: pd.Series,
    y_train: pd.Series,
    valid_col: pd.Series,
    test_col: pd.Series,
    class_names: list[str],
    smoothing: float = 20.0,
):
    global_prior = y_train.value_counts(normalize=True).reindex(
        class_names, fill_value=0.0
    )
    train_parts = []
    valid_parts = []
    test_parts = []

    counts = train_col.value_counts()

    for cls in class_names:
        cls_mask = (y_train == cls).astype(float)
        stats = (
            pd.DataFrame({"cat": train_col, "y": cls_mask})
            .groupby("cat")["y"]
            .agg(["mean", "count"])
        )
        enc_map = {}
        for key, row in stats.iterrows():
            n = float(row["count"])
            enc_map[key] = (row["mean"] * n + global_prior[cls] * smoothing) / (
                n + smoothing
            )

        train_parts.append(
            train_col.map(enc_map)
            .fillna(global_prior[cls])
            .astype(float)
            .rename(f"{cls}_te")
        )
        valid_parts.append(
            valid_col.map(enc_map)
            .fillna(global_prior[cls])
            .astype(float)
            .rename(f"{cls}_te")
        )
        test_parts.append(
            test_col.map(enc_map)
            .fillna(global_prior[cls])
            .astype(float)
            .rename(f"{cls}_te")
        )

    def freq_encode(series: pd.Series) -> pd.Series:
        freq = counts / len(train_col)
        return series.map(freq).fillna(0.0).astype(float)

    train_out = pd.concat(train_parts + [freq_encode(train_col).rename("freq")], axis=1)
    valid_out = pd.concat(valid_parts + [freq_encode(valid_col).rename("freq")], axis=1)
    test_out = pd.concat(test_parts + [freq_encode(test_col).rename("freq")], axis=1)
    return train_out, valid_out, test_out


def one_hot_frame(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    cols: list[str],
):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    X_train = encoder.fit_transform(train_frame[cols].astype(str))
    X_valid = encoder.transform(valid_frame[cols].astype(str))
    X_test = encoder.transform(test_frame[cols].astype(str))
    return X_train, X_valid, X_test


def numeric_matrix(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[cols].astype(float))
    X_valid = scaler.transform(valid_df[cols].astype(float))
    X_test = scaler.transform(test_df[cols].astype(float))
    return (
        sparse.csr_matrix(X_train),
        sparse.csr_matrix(X_valid),
        sparse.csr_matrix(X_test),
    )


def align_proba(proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(TARGET_CLASSES)), dtype=float)
    for j, cls in enumerate(model_classes):
        aligned[:, TARGET_CLASSES.index(cls)] = proba[:, j]
    return aligned


def build_fold_features(train_df, valid_df, test_df, y_train):
    numeric_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    color_cols = ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z"]
    base_cats = ["spectral_type", "galaxy_population"]

    train_colors = make_color_features(train_df)
    valid_colors = make_color_features(valid_df)
    test_colors = make_color_features(test_df)

    redshift_edges = build_bin_edges(train_df["redshift"])
    train_redshift_bin = apply_bins(train_df["redshift"], redshift_edges)
    valid_redshift_bin = apply_bins(valid_df["redshift"], redshift_edges)
    test_redshift_bin = apply_bins(test_df["redshift"], redshift_edges)

    color_bin_edges = build_bin_edges(train_colors["g_r"])
    train_color_bin = apply_bins(train_colors["g_r"], color_bin_edges)
    valid_color_bin = apply_bins(valid_colors["g_r"], color_bin_edges)
    test_color_bin = apply_bins(test_colors["g_r"], color_bin_edges)

    train_cats = train_df[base_cats].astype(str).copy()
    valid_cats = valid_df[base_cats].astype(str).copy()
    test_cats = test_df[base_cats].astype(str).copy()

    train_cats["spectral_type_x_redshift_bin"] = (
        train_cats["spectral_type"] + "__" + train_redshift_bin.astype(str)
    )
    valid_cats["spectral_type_x_redshift_bin"] = (
        valid_cats["spectral_type"] + "__" + valid_redshift_bin.astype(str)
    )
    test_cats["spectral_type_x_redshift_bin"] = (
        test_cats["spectral_type"] + "__" + test_redshift_bin.astype(str)
    )

    train_cats["galaxy_population_x_color_bin"] = (
        train_cats["galaxy_population"] + "__" + train_color_bin.astype(str)
    )
    valid_cats["galaxy_population_x_color_bin"] = (
        valid_cats["galaxy_population"] + "__" + valid_color_bin.astype(str)
    )
    test_cats["galaxy_population_x_color_bin"] = (
        test_cats["galaxy_population"] + "__" + test_color_bin.astype(str)
    )

    fold_target_enc_train, fold_target_enc_valid, fold_target_enc_test = (
        smoothed_target_encoding(
            train_df["spectral_type"].astype(str),
            y_train,
            valid_df["spectral_type"].astype(str),
            test_df["spectral_type"].astype(str),
            TARGET_CLASSES,
        )
    )
    pop_target_enc_train, pop_target_enc_valid, pop_target_enc_test = (
        smoothed_target_encoding(
            train_df["galaxy_population"].astype(str),
            y_train,
            valid_df["galaxy_population"].astype(str),
            test_df["galaxy_population"].astype(str),
            TARGET_CLASSES,
        )
    )

    train_num = pd.concat(
        [
            train_df[numeric_cols].reset_index(drop=True),
            train_colors.reset_index(drop=True),
        ],
        axis=1,
    )
    valid_num = pd.concat(
        [
            valid_df[numeric_cols].reset_index(drop=True),
            valid_colors.reset_index(drop=True),
        ],
        axis=1,
    )
    test_num = pd.concat(
        [
            test_df[numeric_cols].reset_index(drop=True),
            test_colors.reset_index(drop=True),
        ],
        axis=1,
    )

    X_train_num, X_valid_num, X_test_num = numeric_matrix(
        train_num, valid_num, test_num, numeric_cols + color_cols
    )
    X_train_cat, X_valid_cat, X_test_cat = one_hot_frame(
        train_cats, valid_cats, test_cats, list(train_cats.columns)
    )

    X_train_extra = sparse.csr_matrix(
        np.hstack([fold_target_enc_train.values, pop_target_enc_train.values])
    )
    X_valid_extra = sparse.csr_matrix(
        np.hstack([fold_target_enc_valid.values, pop_target_enc_valid.values])
    )
    X_test_extra = sparse.csr_matrix(
        np.hstack([fold_target_enc_test.values, pop_target_enc_test.values])
    )

    X_train = sparse.hstack([X_train_num, X_train_cat, X_train_extra], format="csr")
    X_valid = sparse.hstack([X_valid_num, X_valid_cat, X_valid_extra], format="csr")
    X_test = sparse.hstack([X_test_num, X_test_cat, X_test_extra], format="csr")
    return X_train, X_valid, X_test


def main():
    train, test, sample_sub = load_competition_data()
    y = train["class"].astype(str)
    test_ids = sample_sub["id"].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros((len(train), len(TARGET_CLASSES)), dtype=float)
    test_pred = np.zeros((len(test), len(TARGET_CLASSES)), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        y_va = y.iloc[va_idx].reset_index(drop=True)

        X_tr, X_va, X_te = build_fold_features(
            tr_df, va_df, test.reset_index(drop=True), y_tr
        )

        model = LogisticRegression(
            max_iter=2000,
            solver="saga",
            multi_class="multinomial",
            class_weight="balanced",
            n_jobs=min(16, os.cpu_count() or 1),
            random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr)

        va_proba = align_proba(model.predict_proba(X_va), model.classes_)
        te_proba = align_proba(model.predict_proba(X_te), model.classes_)

        oof_pred[va_idx] = va_proba
        test_pred += te_proba / N_SPLITS

        fold_score = balanced_accuracy_score(
            y_va, np.array(TARGET_CLASSES)[np.argmax(va_proba, axis=1)]
        )
        print(f"Fold {fold}: balanced_accuracy={fold_score:.6f}")

    oof_label = np.array(TARGET_CLASSES)[np.argmax(oof_pred, axis=1)]
    cv_score = balanced_accuracy_score(y, oof_label)
    print(f"CV balanced_accuracy={cv_score:.6f}")

    oof_df = pd.DataFrame(
        {
            "row": train["id"].values,
            "target": y.values,
            "prediction": oof_label,
        }
    )
    write_oof_predictions(oof_df)

    test_label = np.array(TARGET_CLASSES)[np.argmax(test_pred, axis=1)]
    submission = pd.DataFrame({"id": test_ids, "class": test_label})
    write_submission(submission)

    test_pred_df = pd.DataFrame({"id": test_ids})
    for i, cls in enumerate(TARGET_CLASSES):
        test_pred_df[cls] = test_pred[:, i]
    write_test_predictions(test_pred_df)


if __name__ == "__main__":
    main()
