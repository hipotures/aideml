import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostClassifier

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)

RAW_NUMERIC_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CAT_COLS = ["spectral_type", "galaxy_population"]
CAT_LEVELS = {
    "spectral_type": ["A/F", "G/K", "M", "O/B"],
    "galaxy_population": ["Blue_Cloud", "Red_Sequence"],
}

# Deterministic SDSS-inspired regime rules (0/1 flag + signed margin)
RULE_DEFS = {
    "qso_uv1": [("u_g", "<", 0.60)],
    "qso_uv2": [("u_g", "<", 0.85), ("g_r", "<", 0.70)],
    "qso_uv3": [("u_g", "<", 0.40), ("r_i", "<", 0.25)],
    "highz_dropout1": [("u_g", ">", 1.20)],
    "highz_dropout2": [("u_g", ">", 1.10), ("g_r", "<", 0.55)],
    "highz_dropout3": [("g_r", ">", 0.35), ("i_z", ">", 0.20)],
    "red_broad1": [("u_r", ">", 2.00)],
    "red_broad2": [("g_i", ">", 1.10)],
    "red_broad3": [("r_z", ">", 0.45), ("u_r", ">", 1.70)],
    "greenpea1": [("u_r", ">", 1.45), ("r_i", "<", 0.30), ("r_z", "<", 0.15)],
    "greenpea2": [("g_r", ">", 0.20), ("r_i", ">", -0.20), ("r_z", "<", 0.45)],
    "greenpea3": [("g_i", ">", 0.95), ("u_g", ">", 0.70)],
}

RULE_FAMILIES = {
    "uv_qso": ["qso_uv1", "qso_uv2", "qso_uv3"],
    "highz_dropout": ["highz_dropout1", "highz_dropout2", "highz_dropout3"],
    "red_broad": ["red_broad1", "red_broad2", "red_broad3"],
    "green_pea": ["greenpea1", "greenpea2", "greenpea3"],
}


def term_margin(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    if op in ("<", "<="):
        return float(threshold) - values
    if op in (">", ">="):
        return values - float(threshold)
    raise ValueError(f"Unsupported rule operator: {op}")


def add_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic rule flags/margins and family summaries."""
    feat = df.copy()

    # Basic ugriz color differences
    feat["u_g"] = feat["u"] - feat["g"]
    feat["g_r"] = feat["g"] - feat["r"]
    feat["r_i"] = feat["r"] - feat["i"]
    feat["i_z"] = feat["i"] - feat["z"]
    feat["u_i"] = feat["u"] - feat["i"]
    feat["u_r"] = feat["u"] - feat["r"]
    feat["u_z"] = feat["u"] - feat["z"]
    feat["r_z"] = feat["r"] - feat["z"]
    feat["g_i"] = feat["g"] - feat["i"]
    feat["g_z"] = feat["g"] - feat["z"]
    feat["u_mag2g"] = feat["u"] - 2.0 * feat["g"]

    feature_df = feat[
        RAW_NUMERIC_COLS
        + [
            "u_g",
            "g_r",
            "r_i",
            "i_z",
            "u_i",
            "u_r",
            "u_z",
            "r_z",
            "g_i",
            "g_z",
            "u_mag2g",
        ]
    ].copy()

    # One-hot categorical encoding with fixed level order for train/test alignment
    for cat in CAT_COLS:
        for level in CAT_LEVELS[cat]:
            feature_df[f"{cat}={level}"] = (
                feat[cat].astype("category") == level
            ).astype(np.int8)

    flag_cols = []
    margin_cols = []
    for rule_name, terms in RULE_DEFS.items():
        margins = []
        for col, op, th in terms:
            margins.append(term_margin(feat[col].to_numpy(dtype=np.float64), op, th))
        stacked = np.vstack(margins)
        rule_margin = np.min(stacked, axis=0)
        rule_flag = (rule_margin > 0).astype(np.int8)

        feature_df[f"{rule_name}_margin"] = rule_margin
        feature_df[f"{rule_name}_flag"] = rule_flag
        margin_cols.append(f"{rule_name}_margin")
        flag_cols.append(f"{rule_name}_flag")

    # Regime-family summaries
    fam_total_cols = []
    for fam, members in RULE_FAMILIES.items():
        fam_flag_cols = [f"{m}_flag" for m in members]
        fam_margin_cols = [f"{m}_margin" for m in members]
        total = feature_df[fam_flag_cols].sum(axis=1).astype(np.int16)
        max_margin = feature_df[fam_margin_cols].max(axis=1)
        feature_df[f"{fam}_count"] = total
        feature_df[f"{fam}_max_margin"] = max_margin
        fam_total_cols.append(f"{fam}_count")

    # Rule-count and margin summaries
    feature_df["rule_flag_count"] = feature_df[flag_cols].sum(axis=1).astype(np.int16)
    feature_df["rule_margin_max"] = feature_df[margin_cols].max(axis=1)
    feature_df["rule_margin_min"] = feature_df[margin_cols].min(axis=1)
    feature_df["rule_margin_sum"] = feature_df[margin_cols].sum(axis=1)

    # Coarse observed redshift bands
    z = feat["redshift"].to_numpy()
    z_bands = [
        ("z00_01", -np.inf, 0.01),
        ("z01_20", 0.01, 0.20),
        ("z20_50", 0.20, 0.50),
        ("z50_90", 0.50, 0.90),
        ("z90_20", 0.90, 2.00),
        ("z20p", 2.00, np.inf),
    ]
    z_band_cols = []
    for name, lo, hi in z_bands:
        if np.isinf(hi):
            zcol = (z >= lo).astype(np.int8)
        else:
            zcol = ((z >= lo) & (z < hi)).astype(np.int8)
        feature_df[f"z_band_{name}"] = zcol
        z_band_cols.append(f"z_band_{name}")

    # Interactions with redshift bands and categorical one-hots
    for fam in RULE_FAMILIES.keys():
        fam_total = feature_df[f"{fam}_count"]
        for z_col in z_band_cols:
            feature_df[f"{fam}_x_{z_col}"] = fam_total * feature_df[z_col]

    for cat in CAT_COLS:
        for level in CAT_LEVELS[cat]:
            ccol = f"{cat}={level}"
            for fam in RULE_FAMILIES.keys():
                feature_df[f"{ccol}_x_{fam}_count"] = (
                    feature_df[ccol] * feature_df[f"{fam}_count"]
                )

    return feature_df


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    X_train = add_rule_features(train)
    X_test = add_rule_features(test)
    return X_train, X_test


def build_model(iterations=450, depth=10, lr=0.12, random_state=2026, task_type="GPU"):
    base = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": lr,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": random_state,
        "verbose": False,
        "auto_class_weights": "Balanced",
        "thread_count": -1,
    }
    if task_type == "GPU":
        base.update({"task_type": "GPU", "devices": "0", "gpu_ram_part": 0.8})
    else:
        base.update({"task_type": "CPU"})
    return CatBoostClassifier(**base)


def fit_predict_oof(train: pd.DataFrame, test: pd.DataFrame):
    y = train["class"].to_numpy()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    X_train, X_test = build_features(train, test)

    with aide_stage("make_folds_stage"):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X_train, y_enc))

    oof_pred = np.zeros((len(train),), dtype=np.int64)
    test_pred = np.zeros((len(test), n_classes), dtype=np.float64)

    with aide_stage("fit_predict_fold_stage"):
        for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
            log_stage(f"Fold {fold_idx}/5: training CatBoostClassifier on GPU")
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_tr = y_enc[tr_idx]
            y_va = y_enc[va_idx]

            fold_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
            model = build_model(random_state=2000 + fold_idx, task_type="GPU")
            try:
                model.fit(
                    X_tr,
                    y_tr,
                    sample_weight=fold_weights,
                    eval_set=(X_va, y_va),
                )
            except Exception as exc:
                print(
                    f"Fold {fold_idx}: GPU CatBoost failed ({type(exc).__name__}: {exc}). "
                    "Falling back to CPU for this fold."
                )
                model = build_model(random_state=2000 + fold_idx, task_type="CPU")
                model.fit(X_tr, y_tr, sample_weight=fold_weights, eval_set=(X_va, y_va))

            va_pred = model.predict(X_va).astype(np.int64).ravel()
            oof_pred[va_idx] = va_pred
            test_pred += model.predict_proba(X_test).astype(np.float64)

            fold_score = balanced_accuracy_score(y_va, va_pred)
            print(
                f"Fold {fold_idx} OOF balanced accuracy: {fold_score:.6f}", flush=True
            )

    oof_labels = le.inverse_transform(oof_pred)
    test_pred /= float(len(splits))
    test_labels = le.inverse_transform(np.argmax(test_pred, axis=1))

    return oof_labels, test_pred, test_labels, le, X_train, X_test


def main():
    train, test, _sample_sub = load_competition_data()

    with aide_stage("build_features_stage"):
        # Auxiliary star_classification.csv is not used here; it has a different schema and no id-level
        # key to safely and directly merge with train/test rows in this task.
        oof_pred, test_prob, test_pred, le, _x_train, _x_test = fit_predict_oof(
            train, test
        )

    with aide_stage("score_stage"):
        cv_score = balanced_accuracy_score(train["class"], oof_pred)
        print(
            f"OOF balanced accuracy (5-fold stratified CV): {cv_score:.6f}", flush=True
        )

    with aide_stage("write_outputs_stage"):
        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train)),
                "target": train["class"].to_numpy(),
                "prediction": oof_pred,
            }
        )
        write_oof_predictions(oof_df)

        test_prob_df = pd.DataFrame(test_prob, columns=le.classes_)
        test_prob_df.insert(0, "id", test["id"].to_numpy())
        write_test_predictions(test_prob_df)

        submission = pd.DataFrame({"id": test["id"].to_numpy(), "class": test_pred})
        write_submission(submission)

        # Explicitly ensure required final file exists for grading/evaluation path.
        # write_submission is the required writer-backed artifact writer.


if __name__ == "__main__":
    with aide_stage("main_stage"):
        main()
