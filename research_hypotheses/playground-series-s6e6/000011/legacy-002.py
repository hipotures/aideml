import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from aide_solution_helpers import (
    aide_stage,
    load_competition_data,
    log_stage,
    working_dir,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
    write_validation_predictions,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
CLASS_NAMES = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
CAT_COLS = ["spectral_type", "galaxy_population"]
MAG_COLS = ["u", "g", "r", "i", "z"]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    for col in ["alpha", "delta", "redshift"] + MAG_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in MAG_COLS:
        df.loc[df[col] <= -999, col] = np.nan

    for col in CAT_COLS:
        df[col] = df[col].fillna("missing").astype(str)

    color_pairs = [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("u", "r"),
        ("g", "i"),
        ("r", "z"),
        ("u", "z"),
    ]
    for left, right in color_pairs:
        df[f"color_{left}_{right}"] = df[left] - df[right]

    df["color_curve_blue"] = df["color_u_g"] - df["color_g_r"]
    df["color_curve_red"] = df["color_r_i"] - df["color_i_z"]
    df["mag_mean"] = df[MAG_COLS].mean(axis=1)
    df["mag_std"] = df[MAG_COLS].std(axis=1, ddof=0)
    df["mag_range"] = df[MAG_COLS].max(axis=1) - df[MAG_COLS].min(axis=1)
    df["missing_mag_count"] = df[MAG_COLS].isna().sum(axis=1)

    redshift = df["redshift"].fillna(0.0)
    redshift_abs = redshift.abs()
    df["redshift_abs"] = redshift_abs
    df["redshift_sq"] = redshift * redshift
    df["redshift_log1p_abs"] = np.log1p(redshift_abs)
    df["redshift_sqrt_abs"] = np.sqrt(redshift_abs)
    df["redshift_pos"] = np.clip(redshift, 0.0, None)
    df["redshift_neg"] = np.clip(-redshift, 0.0, None)
    for knot in (0.2, 0.5, 1.0, 2.0):
        suffix = str(knot).replace(".", "_")
        df[f"redshift_relu_{suffix}"] = np.clip(redshift - knot, 0.0, None)

    df["redshift_x_color_u_g"] = redshift * df["color_u_g"].fillna(0.0)
    df["redshift_x_color_g_r"] = redshift * df["color_g_r"].fillna(0.0)
    df["redshift_x_color_r_i"] = redshift * df["color_r_i"].fillna(0.0)
    df["redshift_x_color_i_z"] = redshift * df["color_i_z"].fillna(0.0)

    return df


def fit_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    model_name: str,
) -> CatBoostClassifier:
    base_params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "random_seed": RANDOM_STATE,
        "auto_class_weights": "Balanced",
        "od_type": "Iter",
        "od_wait": 100,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": False,
    }

    last_error = None
    for use_gpu in (True, False):
        params = base_params.copy()
        if use_gpu:
            params.update(
                {
                    "task_type": "GPU",
                    "devices": "0",
                    "gpu_ram_part": 0.8,
                }
            )
        else:
            log_stage(f"event=info|model={model_name}|device=cpu_fallback")

        model = CatBoostClassifier(**params)
        try:
            model.fit(
                x_train,
                y_train,
                cat_features=cat_features,
                eval_set=(x_valid, y_valid),
                use_best_model=True,
                verbose=False,
            )
            return model
        except Exception as exc:
            last_error = exc
            if use_gpu:
                log_stage(
                    f"event=warning|model={model_name}|device=gpu_failed|error_type={exc.__class__.__name__}"
                )
            else:
                raise

    raise last_error


def positive_class_probability(
    model: CatBoostClassifier, x: pd.DataFrame
) -> np.ndarray:
    class_list = list(model.classes_)
    positive_index = class_list.index(1)
    return model.predict_proba(x)[:, positive_index]


def hard_route_predictions(p_star: np.ndarray, p_qso_cond: np.ndarray) -> np.ndarray:
    predictions = np.full(p_star.shape[0], "GALAXY", dtype=object)
    qso_mask = (p_star < 0.5) & (p_qso_cond >= 0.5)
    star_mask = p_star >= 0.5
    predictions[qso_mask] = "QSO"
    predictions[star_mask] = "STAR"
    return predictions


def hierarchical_probabilities(
    p_star: np.ndarray, p_qso_cond: np.ndarray
) -> np.ndarray:
    p_nonstar = 1.0 - p_star
    probs = np.column_stack(
        [
            p_nonstar * (1.0 - p_qso_cond),
            p_nonstar * p_qso_cond,
            p_star,
        ]
    )
    row_sums = np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs / row_sums


def main() -> None:
    _ = working_dir()
    _ = write_validation_predictions

    train, test, sample_sub = load_competition_data()

    with aide_stage("build_features_stage"):
        y = train["class"].astype(str).reset_index(drop=True)

        train_features = build_features(train.drop(columns=["class"])).reset_index(
            drop=True
        )
        test_features = build_features(test).reset_index(drop=True)

        feature_cols = [col for col in train_features.columns if col != "id"]
        x = train_features[feature_cols]
        x_test = test_features[feature_cols]

    with aide_stage("make_folds_stage"):
        splitter = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        folds = list(splitter.split(x, y))

    n_train = len(x)
    n_test = len(x_test)

    oof_star_prob = np.zeros(n_train, dtype=np.float32)
    oof_qso_cond_prob = np.zeros(n_train, dtype=np.float32)
    test_star_prob = np.zeros(n_test, dtype=np.float32)
    test_qso_cond_prob = np.zeros(n_test, dtype=np.float32)
    fold_scores = []

    with aide_stage("fit_predict_fold_stage"):
        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
            x_train = x.iloc[train_idx]
            x_valid = x.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            y_stage1_train = (y_train == "STAR").astype(int)
            y_stage1_valid = (y_valid == "STAR").astype(int)

            log_stage(f"event=info|fold={fold}|model=catboost_stage1_star_vs_nonstar")
            model_stage1 = fit_catboost(
                x_train=x_train,
                y_train=y_stage1_train,
                x_valid=x_valid,
                y_valid=y_stage1_valid,
                cat_features=CAT_COLS,
                model_name=f"fold{fold}_stage1",
            )

            p_star_valid = positive_class_probability(model_stage1, x_valid)
            p_star_test = positive_class_probability(model_stage1, x_test)

            nonstar_train_mask = y_train != "STAR"
            nonstar_valid_mask = y_valid != "STAR"

            x_train_stage2 = x_train.loc[nonstar_train_mask]
            y_train_stage2 = (y_train.loc[nonstar_train_mask] == "QSO").astype(int)
            x_valid_stage2 = x_valid.loc[nonstar_valid_mask]
            y_valid_stage2 = (y_valid.loc[nonstar_valid_mask] == "QSO").astype(int)

            log_stage(f"event=info|fold={fold}|model=catboost_stage2_galaxy_vs_qso")
            model_stage2 = fit_catboost(
                x_train=x_train_stage2,
                y_train=y_train_stage2,
                x_valid=x_valid_stage2,
                y_valid=y_valid_stage2,
                cat_features=CAT_COLS,
                model_name=f"fold{fold}_stage2",
            )

            p_qso_cond_valid = positive_class_probability(model_stage2, x_valid)
            p_qso_cond_test = positive_class_probability(model_stage2, x_test)

            oof_star_prob[valid_idx] = p_star_valid.astype(np.float32)
            oof_qso_cond_prob[valid_idx] = p_qso_cond_valid.astype(np.float32)
            test_star_prob += p_star_test.astype(np.float32) / N_SPLITS
            test_qso_cond_prob += p_qso_cond_test.astype(np.float32) / N_SPLITS

            fold_pred = hard_route_predictions(p_star_valid, p_qso_cond_valid)
            fold_score = balanced_accuracy_score(y_valid, fold_pred)
            fold_scores.append(fold_score)
            print(f"Fold {fold} balanced_accuracy: {fold_score:.6f}", flush=True)

    with aide_stage("score_stage"):
        oof_pred = hard_route_predictions(oof_star_prob, oof_qso_cond_prob)
        oof_score = balanced_accuracy_score(y, oof_pred)
        print(f"OOF balanced_accuracy: {oof_score:.6f}", flush=True)
        print(f"Mean fold balanced_accuracy: {np.mean(fold_scores):.6f}", flush=True)

        oof_proba = hierarchical_probabilities(oof_star_prob, oof_qso_cond_prob)
        test_proba = hierarchical_probabilities(test_star_prob, test_qso_cond_prob)
        test_pred = hard_route_predictions(test_star_prob, test_qso_cond_prob)

    with aide_stage("write_outputs_stage"):
        submission = sample_sub[["id"]].copy()
        submission["class"] = test_pred
        write_submission(submission)

        oof_frame = pd.DataFrame(
            {
                "row": np.arange(n_train),
                "target": y.to_numpy(),
                "prediction": oof_pred,
            }
        )
        write_oof_predictions(oof_frame)

        test_pred_frame = pd.DataFrame(
            {
                "id": sample_sub["id"].to_numpy(),
                "GALAXY": test_proba[:, 0],
                "QSO": test_proba[:, 1],
                "STAR": test_proba[:, 2],
            }
        )
        write_test_predictions(test_pred_frame)


if __name__ == "__main__":
    main()
