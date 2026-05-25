import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from catboost import CatBoostClassifier, CatBoostRanker, Pool

warnings.filterwarnings("ignore")

INPUT = "./input"
WORKING = "./working"
os.makedirs(WORKING, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5
THREADS = min(8, os.cpu_count() or 1)

train = pd.read_csv(os.path.join(INPUT, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT, "sample_submission.csv.gz"))

train = train.rename(columns={"LapTime (s)": "LapTime_s"})
test = test.rename(columns={"LapTime (s)": "LapTime_s"})

y = train[TARGET].astype(int).values
n_train = len(train)


def add_features(df):
    df = df.copy()
    for c in ["Race", "Driver", "Compound"]:
        df[c] = df[c].astype(str).fillna("missing")

    df["race_year"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)
    df["race_lap"] = df["race_year"] + "_" + df["LapNumber"].astype(str)

    df["TyreLife_sq"] = df["TyreLife"] ** 2
    df["LapNumber_sq"] = df["LapNumber"] ** 2
    df["TyreLife_x_RaceProgress"] = df["TyreLife"] * df["RaceProgress"]
    df["Stint_x_TyreLife"] = df["Stint"] * df["TyreLife"]
    df["Position_x_RaceProgress"] = df["Position"] * df["RaceProgress"]

    sort_cols = ["Year", "Race", "LapNumber", "Position"]
    sdf = df.sort_values(sort_cols).copy()
    g = sdf.groupby("race_lap", sort=False)

    local_cols = [
        "TyreLife",
        "LapTime_s",
        "Cumulative_Degradation",
        "Position_Change",
        "PitStop",
        "Stint",
    ]
    for col in local_cols:
        safe = col.replace(" ", "_")
        ahead = g[col].shift(1)
        behind = g[col].shift(-1)
        sdf[f"{safe}_ahead_delta"] = sdf[col] - ahead
        sdf[f"{safe}_behind_delta"] = sdf[col] - behind
        sdf[f"{safe}_nearest_abs_delta"] = np.minimum(
            np.abs(sdf[f"{safe}_ahead_delta"].fillna(np.inf)),
            np.abs(sdf[f"{safe}_behind_delta"].fillna(np.inf)),
        ).replace(np.inf, np.nan)

    for col in ["LapTime_s", "TyreLife", "Cumulative_Degradation", "Position_Change"]:
        mean = g[col].transform("mean")
        std = g[col].transform("std").replace(0, np.nan)
        sdf[f"{col}_lap_mean"] = mean
        sdf[f"{col}_vs_lap_mean"] = sdf[col] - mean
        sdf[f"{col}_lap_z"] = (sdf[col] - mean) / std
        sdf[f"{col}_lap_rank_pct"] = g[col].rank(method="average", pct=True)

    sdf["race_lap_size"] = g["Position"].transform("count")
    sdf["position_pct_in_lap"] = sdf["Position"] / sdf["race_lap_size"].clip(lower=1)

    close_laptime = sdf["LapTime_s_nearest_abs_delta"] <= np.maximum(
        2.5, 0.60 * sdf["LapTime_s"].groupby(sdf["race_lap"]).transform("std").fillna(0)
    )
    close_tyre = sdf["TyreLife_nearest_abs_delta"] <= 8
    has_neighbor = sdf["LapTime_s_nearest_abs_delta"].notna()
    sdf["local_contest"] = (
        has_neighbor & (close_laptime | close_tyre) & (sdf["PitStop"] == 0)
    ).astype(int)

    return sdf.sort_index()


all_df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
all_df = add_features(all_df)
trn = all_df.iloc[:n_train].reset_index(drop=True)
tst = all_df.iloc[n_train:].reset_index(drop=True)

drop_cols = {ID_COL, TARGET, "race_year", "race_lap"}
features = [c for c in trn.columns if c not in drop_cols]
cat_cols = [c for c in ["Race", "Driver", "Compound"] if c in features]
cat_idx = [features.index(c) for c in cat_cols]
num_cols = [c for c in features if c not in cat_cols]

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

hazard_features = [c for c in features if c in cat_cols or c not in ["Driver"]]
hazard_cat = [c for c in cat_cols if c in hazard_features]
hazard_num = [c for c in hazard_features if c not in hazard_cat]


def make_hazard_model():
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                hazard_num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imp",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        ("ohe", ohe),
                    ]
                ),
                hazard_cat,
            ),
        ],
        sparse_threshold=0.3,
    )
    return Pipeline(
        [
            ("prep", prep),
            (
                "lr",
                LogisticRegression(
                    solver="saga",
                    max_iter=220,
                    C=1.0,
                    class_weight="balanced",
                    n_jobs=THREADS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def pct_rank(a):
    return (
        pd.Series(np.asarray(a)).rank(method="average", pct=True).to_numpy(dtype=float)
    )


def build_pairs(df, labels, max_pairs=350000):
    tmp = df[
        ["race_lap", "Position", "LapTime_s", "TyreLife", "local_contest", "PitStop"]
    ].copy()
    tmp["_y"] = labels
    tmp = tmp[(tmp["local_contest"] == 1) & (tmp["PitStop"] == 0)]

    pairs = []
    for _, g in tmp.groupby("race_lap", sort=False):
        if len(g) < 2 or g["_y"].nunique() < 2:
            continue
        std = g["LapTime_s"].std()
        gap_limit = max(2.5, 0.60 * (0.0 if pd.isna(std) else std))
        pos_rows = g[g["_y"] == 1]
        neg_rows = g[g["_y"] == 0]
        for pi, prow in pos_rows.iterrows():
            cand = neg_rows[
                (np.abs(neg_rows["Position"] - prow["Position"]) <= 3)
                | (np.abs(neg_rows["LapTime_s"] - prow["LapTime_s"]) <= gap_limit)
                | (np.abs(neg_rows["TyreLife"] - prow["TyreLife"]) <= 8)
            ].copy()
            if len(cand) == 0:
                continue
            cand["_dist"] = np.abs(cand["Position"] - prow["Position"]) + 0.15 * np.abs(
                cand["TyreLife"] - prow["TyreLife"]
            )
            for ni in cand.sort_values("_dist").head(6).index:
                pairs.append((pi, ni))

    if len(pairs) > max_pairs:
        rng = np.random.default_rng(RANDOM_STATE)
        keep = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in keep]
    return pairs


def make_rank_pool(df, labels):
    pairs_orig = build_pairs(df, labels)
    if len(pairs_orig) < 500:
        return None, None

    used = sorted(set(i for p in pairs_orig for i in p))
    rank_df = df.loc[used].sort_values(["race_lap", "Position"]).copy()
    idx_map = {idx: i for i, idx in enumerate(rank_df.index)}
    pairs = np.array(
        [
            [idx_map[w], idx_map[l]]
            for w, l in pairs_orig
            if w in idx_map and l in idx_map
        ],
        dtype=np.int32,
    )
    if len(pairs) < 500:
        return None, None

    pool = Pool(
        rank_df[features],
        label=labels[rank_df.index.to_numpy()],
        cat_features=cat_idx,
        group_id=rank_df["race_lap"].astype(str).values,
        pairs=pairs,
    )
    return pool, len(pairs)


clf_params = dict(
    iterations=650,
    learning_rate=0.045,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    l2_leaf_reg=7.0,
    random_strength=0.7,
    od_type="Iter",
    od_wait=60,
    allow_writing_files=False,
    verbose=False,
    thread_count=THREADS,
)

rank_params = dict(
    iterations=280,
    learning_rate=0.055,
    depth=6,
    loss_function="PairLogitPairwise",
    l2_leaf_reg=8.0,
    random_seed=RANDOM_STATE,
    allow_writing_files=False,
    verbose=False,
    thread_count=THREADS,
)

groups = trn["race_year"].values
if StratifiedGroupKFold is not None:
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(cv.split(trn[features], y, groups))
else:
    cv = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(trn[features], y, groups))

oof_main = np.zeros(n_train)
oof_hazard = np.zeros(n_train)
oof_rank = np.zeros(n_train)
best_iters = []

for fold, (fit_idx, val_idx) in enumerate(splits, 1):
    X_fit = trn.iloc[fit_idx]
    X_val = trn.iloc[val_idx]
    y_fit = y[fit_idx]
    y_val = y[val_idx]

    fit_pool = Pool(X_fit[features], y_fit, cat_features=cat_idx)
    val_pool = Pool(X_val[features], y_val, cat_features=cat_idx)

    clf = CatBoostClassifier(**{**clf_params, "random_seed": RANDOM_STATE + fold})
    clf.fit(fit_pool, eval_set=val_pool, use_best_model=True, verbose=False)
    oof_main[val_idx] = clf.predict_proba(val_pool)[:, 1]
    bi = clf.get_best_iteration()
    if bi is not None and bi > 0:
        best_iters.append(bi + 1)

    hazard = make_hazard_model()
    hazard.fit(X_fit[hazard_features], y_fit)
    oof_hazard[val_idx] = hazard.predict_proba(X_val[hazard_features])[:, 1]

    rank_pool, pair_count = make_rank_pool(X_fit, y_fit)
    if rank_pool is not None:
        ranker = CatBoostRanker(
            **{**rank_params, "random_seed": RANDOM_STATE + 100 + fold}
        )
        ranker.fit(rank_pool, verbose=False)
        oof_rank[val_idx] = ranker.predict(Pool(X_val[features], cat_features=cat_idx))
    else:
        oof_rank[val_idx] = oof_main[val_idx]
        pair_count = 0

    fold_blend = (
        0.55 * pct_rank(oof_main[val_idx])
        + 0.25 * pct_rank(oof_rank[val_idx])
        + 0.20 * pct_rank(oof_hazard[val_idx])
    )
    print(
        f"fold {fold} auc={roc_auc_score(y_val, fold_blend):.6f} rank_pairs={pair_count}"
    )

main_r = pct_rank(oof_main)
hazard_r = pct_rank(oof_hazard)
rank_r = pct_rank(oof_rank)

candidates = []
for wm in np.arange(0.45, 0.76, 0.05):
    for wr in np.arange(0.10, 0.41, 0.05):
        wh = 1.0 - wm - wr
        if 0.05 <= wh <= 0.35:
            candidates.append((float(wm), float(wr), float(wh)))

best_auc = -1.0
best_w = (0.55, 0.25, 0.20)
for wm, wr, wh in candidates:
    pred = wm * main_r + wr * rank_r + wh * hazard_r
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_auc = auc
        best_w = (wm, wr, wh)

oof_pred = best_w[0] * main_r + best_w[1] * rank_r + best_w[2] * hazard_r
print(f"5-fold grouped ROC AUC: {roc_auc_score(y, oof_pred):.6f}")

pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORKING, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_iters = int(np.median(best_iters)) if best_iters else clf_params["iterations"]
final_iters = max(150, min(final_iters, clf_params["iterations"]))

final_clf = CatBoostClassifier(
    **{
        **clf_params,
        "iterations": final_iters,
        "random_seed": RANDOM_STATE + 999,
        "od_type": None,
    }
)
final_clf.fit(Pool(trn[features], y, cat_features=cat_idx), verbose=False)
test_main = final_clf.predict_proba(Pool(tst[features], cat_features=cat_idx))[:, 1]

final_hazard = make_hazard_model()
final_hazard.fit(trn[hazard_features], y)
test_hazard = final_hazard.predict_proba(tst[hazard_features])[:, 1]

final_rank_pool, final_pair_count = make_rank_pool(trn, y)
if final_rank_pool is not None:
    final_ranker = CatBoostRanker(**{**rank_params, "random_seed": RANDOM_STATE + 1999})
    final_ranker.fit(final_rank_pool, verbose=False)
    test_rank = final_ranker.predict(Pool(tst[features], cat_features=cat_idx))
else:
    test_rank = test_main
    final_pair_count = 0

test_pred = (
    best_w[0] * pct_rank(test_main)
    + best_w[1] * pct_rank(test_rank)
    + best_w[2] * pct_rank(test_hazard)
)
test_pred = np.clip(test_pred, 0.0, 1.0)

sub = sample.copy()
sub[TARGET] = test_pred
sub.to_csv(os.path.join(WORKING, "submission.csv"), index=False)
sub.to_csv(
    os.path.join(WORKING, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000718"],
            "metric": "roc_auc",
            "cv_auc": float(roc_auc_score(y, oof_pred)),
            "main_oof_auc": float(roc_auc_score(y, oof_main)),
            "rank_oof_auc": float(roc_auc_score(y, oof_rank)),
            "hazard_oof_auc": float(roc_auc_score(y, oof_hazard)),
            "blend_weights_main_rank_hazard": list(best_w),
            "final_rank_pairs": int(final_pair_count),
            "submission_path": os.path.join(WORKING, "submission.csv"),
        },
        indent=2,
    )
)
