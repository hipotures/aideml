import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(f"{INPUT_DIR}/train.csv.gz")
test = pd.read_csv(f"{INPUT_DIR}/test.csv.gz")
sample = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv.gz")

TARGET = "PitNextLap"
ID = "id"
y = train[TARGET].astype(int).values
n_train = len(train)
n_test = len(test)


def add_features(train_df, test_df):
    base_train = train_df.drop(columns=[TARGET]).copy()
    base_test = test_df.copy()
    all_df = pd.concat([base_train, base_test], axis=0, ignore_index=True)

    all_df["WetDry"] = all_df["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")
    all_df["TyreLifeOverLap"] = all_df["TyreLife"] / np.maximum(all_df["LapNumber"], 1)
    all_df["DegPerTyreLife"] = all_df["Cumulative_Degradation"] / np.maximum(
        all_df["TyreLife"], 1
    )
    all_df["LapTime_Delta_Abs"] = all_df["LapTime_Delta"].abs()
    all_df["RaceProgressLeft"] = 1.0 - all_df["RaceProgress"]
    all_df["LateRaceTyreLife"] = all_df["RaceProgress"] * all_df["TyreLife"]
    all_df["PitStop_x_TyreLife"] = all_df["PitStop"] * all_df["TyreLife"]
    all_df["TyreLife_bucket"] = np.floor(all_df["TyreLife"] / 3).astype("int16")
    all_df["Lap_bucket"] = np.floor(all_df["LapNumber"] / 5).astype("int16")
    all_df["Progress_bucket"] = (
        np.floor(all_df["RaceProgress"] * 10).clip(0, 10).astype("int16")
    )

    all_df["Race_Compound"] = (
        all_df["Race"].astype(str) + "|" + all_df["Compound"].astype(str)
    )
    all_df["Driver_Compound"] = (
        all_df["Driver"].astype(str) + "|" + all_df["Compound"].astype(str)
    )
    all_df["Race_Stint"] = (
        all_df["Race"].astype(str) + "|S" + all_df["Stint"].astype(str)
    )

    qcols = ["Year", "Race", "LapNumber"]
    all_df["query_size"] = (
        all_df.groupby(qcols)["LapNumber"].transform("size").astype("float32")
    )

    all_df = all_df.replace([np.inf, -np.inf], np.nan)
    for c in all_df.columns:
        if all_df[c].dtype == "object":
            all_df[c] = all_df[c].fillna("missing")
        else:
            all_df[c] = all_df[c].fillna(all_df[c].median())

    return all_df.iloc[:n_train].reset_index(drop=True), all_df.iloc[
        n_train:
    ].reset_index(drop=True)


train_fe, test_fe = add_features(train, test)
feature_cols = [c for c in train_fe.columns if c != ID]
cat_cols = [c for c in feature_cols if train_fe[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

try:
    from sklearn.model_selection import StratifiedGroupKFold

    groups = train_fe["Year"].astype(str) + "|" + train_fe["Race"].astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(train_fe, y, groups))
except Exception:
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(train_fe, y))


def encode_for_tree(tr_df, va_df, te_df):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr_cat = (
        enc.fit_transform(tr_df[cat_cols].astype(str)).astype("float32")
        if cat_cols
        else np.empty((len(tr_df), 0), dtype="float32")
    )
    xva_cat = (
        enc.transform(va_df[cat_cols].astype(str)).astype("float32")
        if cat_cols
        else np.empty((len(va_df), 0), dtype="float32")
    )
    xte_cat = (
        enc.transform(te_df[cat_cols].astype(str)).astype("float32")
        if cat_cols
        else np.empty((len(te_df), 0), dtype="float32")
    )

    med = tr_df[num_cols].median()
    xtr_num = tr_df[num_cols].fillna(med).astype("float32").values
    xva_num = va_df[num_cols].fillna(med).astype("float32").values
    xte_num = te_df[num_cols].fillna(med).astype("float32").values

    return (
        np.hstack([xtr_num, xtr_cat]),
        np.hstack([xva_num, xva_cat]),
        np.hstack([xte_num, xte_cat]),
    )


def group_rank_score(raw, qkey):
    s = pd.Series(raw)
    q = pd.Series(qkey).astype(str)
    pct = s.groupby(q).rank(pct=True, method="average").values
    z = (raw - np.nanmedian(raw)) / (
        np.nanpercentile(raw, 75) - np.nanpercentile(raw, 25) + 1e-6
    )
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
    return np.clip(0.5 * pct + 0.5 * sig, 0, 1)


def smooth_te_fit_apply(tr_df, tr_y, app_df, keys, prior, m=60):
    tmp = tr_df[keys].copy()
    tmp["_y"] = tr_y
    stats = tmp.groupby(keys)["_y"].agg(["sum", "count"])
    enc = (stats["sum"] + prior * m) / (stats["count"] + m)
    idx = pd.MultiIndex.from_frame(app_df[keys])
    vals = idx.map(enc).astype("float64")
    return np.where(pd.isna(vals), prior, vals)


def choice_expert(tr_df, tr_y, app_df):
    prior = float(np.mean(tr_y))
    key_sets = [
        ["Compound", "Stint", "TyreLife_bucket"],
        ["Race", "Compound", "Stint"],
        ["Driver", "Compound"],
        ["Race", "LapNumber"],
        ["Year", "Race", "Stint"],
        ["WetDry", "Position", "Stint"],
        ["Progress_bucket", "TyreLife_bucket", "Compound"],
    ]
    encs = [smooth_te_fit_apply(tr_df, tr_y, app_df, k, prior) for k in key_sets]
    score = np.mean(encs, axis=0)

    qkey = (
        app_df["Year"].astype(str)
        + "|"
        + app_df["Race"].astype(str)
        + "|"
        + app_df["LapNumber"].astype(str)
    )
    s = pd.Series(score)
    denom = s.groupby(qkey).transform("sum").values + 1e-9
    expected = np.minimum(app_df["query_size"].values * prior, 1.5)
    allocated = score / denom * expected
    rank_boost = s.groupby(qkey).rank(pct=True).values * min(0.2, prior * 3.0)

    return np.clip(0.70 * score + 0.20 * allocated + 0.10 * rank_boost, 0, 1)


def adversarial_shift_scores(train_df, test_df):
    all_df = pd.concat(
        [train_df[feature_cols], test_df[feature_cols]], ignore_index=True
    )
    labels = np.r_[np.zeros(len(train_df), dtype=int), np.ones(len(test_df), dtype=int)]
    scores = np.zeros(len(all_df), dtype="float32")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 99)

    for tr_idx, va_idx in skf.split(all_df, labels):
        tr_part = all_df.iloc[tr_idx]
        va_part = all_df.iloc[va_idx]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr_cat = (
            enc.fit_transform(tr_part[cat_cols].astype(str)).astype("float32")
            if cat_cols
            else np.empty((len(tr_part), 0), dtype="float32")
        )
        va_cat = (
            enc.transform(va_part[cat_cols].astype(str)).astype("float32")
            if cat_cols
            else np.empty((len(va_part), 0), dtype="float32")
        )
        med = tr_part[num_cols].median()
        tr_num = tr_part[num_cols].fillna(med).astype("float32").values
        va_num = va_part[num_cols].fillna(med).astype("float32").values

        model = HistGradientBoostingClassifier(
            max_iter=60,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
        model.fit(np.hstack([tr_num, tr_cat]), labels[tr_idx])
        scores[va_idx] = model.predict_proba(np.hstack([va_num, va_cat]))[:, 1]

    return scores[: len(train_df)], scores[len(train_df) :]


print("Building fold-safe adversarial-shift meta-feature...")
adv_train, adv_test = adversarial_shift_scores(train_fe, test_fe)

from xgboost import XGBClassifier, XGBRanker
from catboost import CatBoostClassifier

base_names = ["xgb", "catboost", "ranking_expert", "choice_expert"]
oof_base = np.zeros((n_train, len(base_names)), dtype="float32")
test_base = np.zeros((n_test, len(base_names)), dtype="float32")
fold_aucs = []
threads = min(8, os.cpu_count() or 1)

train_qkey = (
    train_fe["Year"].astype(str)
    + "|"
    + train_fe["Race"].astype(str)
    + "|"
    + train_fe["LapNumber"].astype(str)
)
test_qkey = (
    test_fe["Year"].astype(str)
    + "|"
    + test_fe["Race"].astype(str)
    + "|"
    + test_fe["LapNumber"].astype(str)
)
cat_feature_indices = [feature_cols.index(c) for c in cat_cols]

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    print(f"Training base fold {fold}/{N_SPLITS}...")
    tr_df = train_fe.iloc[tr_idx].reset_index(drop=True)
    va_df = train_fe.iloc[va_idx].reset_index(drop=True)
    y_tr, y_va = y[tr_idx], y[va_idx]
    pos = max(1, int(y_tr.sum()))
    scale_pos_weight = float((len(y_tr) - pos) / pos)

    xtr, xva, xte = encode_for_tree(tr_df, va_df, test_fe)

    xgb = XGBClassifier(
        n_estimators=320,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE + fold,
        n_jobs=threads,
        scale_pos_weight=scale_pos_weight,
    )
    xgb.fit(xtr, y_tr, verbose=False)
    oof_base[va_idx, 0] = xgb.predict_proba(xva)[:, 1]
    test_base[:, 0] += xgb.predict_proba(xte)[:, 1] / N_SPLITS

    cb_tr = tr_df[feature_cols].copy()
    cb_va = va_df[feature_cols].copy()
    cb_te = test_fe[feature_cols].copy()
    for c in cat_cols:
        cb_tr[c] = cb_tr[c].astype(str)
        cb_va[c] = cb_va[c].astype(str)
        cb_te[c] = cb_te[c].astype(str)

    cat = CatBoostClassifier(
        iterations=450,
        depth=6,
        learning_rate=0.045,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1.0, scale_pos_weight],
        l2_leaf_reg=5.0,
        random_strength=0.8,
        random_seed=RANDOM_STATE + fold,
        allow_writing_files=False,
        verbose=False,
        thread_count=threads,
    )
    cat.fit(cb_tr, y_tr, cat_features=cat_feature_indices)
    oof_base[va_idx, 1] = cat.predict_proba(cb_va)[:, 1]
    test_base[:, 1] += cat.predict_proba(cb_te)[:, 1] / N_SPLITS

    tr_q = train_qkey.iloc[tr_idx].reset_index(drop=True).values
    order = np.argsort(tr_q)
    _, group_sizes = np.unique(tr_q[order], return_counts=True)
    ranker = XGBRanker(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.055,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        objective="rank:pairwise",
        tree_method="hist",
        random_state=RANDOM_STATE + 100 + fold,
        n_jobs=threads,
    )
    ranker.fit(xtr[order], y_tr[order], group=group_sizes, verbose=False)
    oof_base[va_idx, 2] = group_rank_score(
        ranker.predict(xva), train_qkey.iloc[va_idx].values
    )
    test_base[:, 2] += (
        group_rank_score(ranker.predict(xte), test_qkey.values) / N_SPLITS
    )

    oof_base[va_idx, 3] = choice_expert(tr_df, y_tr, va_df)
    test_base[:, 3] += choice_expert(tr_df, y_tr, test_fe) / N_SPLITS

    fold_auc = {
        name: roc_auc_score(y_va, oof_base[va_idx, i])
        for i, name in enumerate(base_names)
    }
    fold_aucs.append(fold_auc)
    print("Fold", fold, "base AUCs:", {k: round(v, 6) for k, v in fold_auc.items()})


def build_meta_matrix(base_pred, fe_df, adv_score):
    max_q = max(
        float(train_fe["query_size"].max()), float(test_fe["query_size"].max()), 1.0
    )
    sorted_base = np.sort(base_pred, axis=1)
    ambiguity_margin = sorted_base[:, -1] - sorted_base[:, -2]
    return np.column_stack(
        [
            base_pred,
            adv_score.astype("float32"),
            np.clip(fe_df["query_size"].values / max_q, 0, 1).astype("float32"),
            fe_df["WetDry"].values.astype("float32"),
            ambiguity_margin.astype("float32"),
        ]
    ).astype("float32")


meta_feature_names = base_names + [
    "adversarial_shift",
    "query_size",
    "wetdry",
    "ambiguity_margin",
]
X_meta = build_meta_matrix(oof_base, train_fe, adv_train)
X_test_meta = build_meta_matrix(test_base, test_fe, adv_test)


def fit_auc_superlearner(X, y, seed=42, steps=900, batch_size=4096):
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))

    X = np.nan_to_num(X.astype("float32"), nan=0.0, posinf=1.0, neginf=0.0)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.ones(X.shape[1], dtype="float32") / X.shape[1]

    rng = np.random.default_rng(seed)
    xt = torch.from_numpy(X)
    theta = torch.zeros(X.shape[1], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.06, weight_decay=1e-4)

    for _ in range(steps):
        p = rng.choice(pos_idx, size=batch_size, replace=len(pos_idx) < batch_size)
        n = rng.choice(neg_idx, size=batch_size, replace=len(neg_idx) < batch_size)
        p = torch.as_tensor(p, dtype=torch.long)
        n = torch.as_tensor(n, dtype=torch.long)
        w = torch.softmax(theta, dim=0)
        margin = xt[p].matmul(w) - xt[n].matmul(w)
        loss = torch.nn.functional.softplus(-8.0 * margin).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = torch.softmax(theta, dim=0).cpu().numpy().astype("float32")
    return w / np.sum(w)


print("Training AUC-optimized non-negative Super Learner...")
meta_oof = np.zeros(n_train, dtype="float32")
meta_weights = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    w = fit_auc_superlearner(X_meta[tr_idx], y[tr_idx], seed=RANDOM_STATE + 500 + fold)
    meta_weights.append(w)
    meta_oof[va_idx] = np.clip(X_meta[va_idx].dot(w), 0, 1)
    print(
        f"Stacker fold {fold} AUC: {roc_auc_score(y[va_idx], meta_oof[va_idx]):.6f}",
        dict(zip(meta_feature_names, np.round(w, 4))),
    )

cv_auc = roc_auc_score(y, meta_oof)
final_w = fit_auc_superlearner(X_meta, y, seed=RANDOM_STATE + 999, steps=1200)
test_pred = np.clip(X_test_meta.dot(final_w), 0, 1)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(f"{WORK_DIR}/submission.csv", index=False)

pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": meta_oof,
    }
).to_csv(f"{WORK_DIR}/oof_predictions.csv.gz", index=False, compression="gzip")

test_pred_df = sample[[ID]].copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    f"{WORK_DIR}/test_predictions.csv.gz", index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["001140"],
    "metric": "roc_auc",
    "cv_folds": N_SPLITS,
    "oof_auc": float(cv_auc),
    "base_oof_auc": {
        name: float(roc_auc_score(y, oof_base[:, i]))
        for i, name in enumerate(base_names)
    },
    "final_superlearner_weights": dict(zip(meta_feature_names, map(float, final_w))),
    "submission_path": f"{WORK_DIR}/submission.csv",
    "oof_path": f"{WORK_DIR}/oof_predictions.csv.gz",
    "test_predictions_path": f"{WORK_DIR}/test_predictions.csv.gz",
}

with open(f"{WORK_DIR}/result_review.json", "w") as f:
    json.dump(review, f, indent=2)

print(f"5-fold OOF ROC AUC: {cv_auc:.6f}")
print(json.dumps(review, indent=2))
