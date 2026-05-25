import os
import json
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
os.makedirs(WORK_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


def add_features(df):
    df = df.copy()
    df["RaceYear"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)
    rp = df["RaceProgress"].clip(lower=0.01)
    total_laps = (df["LapNumber"] / rp).replace([np.inf, -np.inf], np.nan)
    df["EstimatedTotalLaps"] = total_laps.fillna(df["LapNumber"])
    df["LapsRemainingEst"] = (df["EstimatedTotalLaps"] - df["LapNumber"]).clip(lower=0)
    df["TyreLifeFracRace"] = df["TyreLife"] / df["EstimatedTotalLaps"].clip(lower=1)
    df["DegPerTyreLap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1)
    df["LapTimeDeltaAbs"] = df["LapTime_Delta"].abs()
    df["PositionGain"] = (-df["Position_Change"]).clip(lower=0)
    df["PositionLoss"] = df["Position_Change"].clip(lower=0)
    return df.replace([np.inf, -np.inf], np.nan)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = add_features(train)
test = add_features(test)
y = train[TARGET].astype(int).to_numpy()

cat_cols = ["Compound", "Driver", "Race", "Year"]
group_cols = ["RaceYear", "LapNumber"]
drop_cols = {ID_COL, TARGET, "RaceYear"}
num_cols = [c for c in train.columns if c not in drop_cols and c not in cat_cols]
feature_cols = num_cols + cat_cols

groups = train["RaceYear"].astype(str).to_numpy()
try:
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(train, y, groups))
except Exception:
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(train, y, groups))


def make_group_matrix(df):
    gid = df.groupby(group_cols, sort=False).ngroup().to_numpy(np.int64)
    n_groups = int(gid.max()) + 1
    counts = np.bincount(gid, minlength=n_groups)
    max_len = int(counts.max())
    members = np.zeros((n_groups, max_len), dtype=np.int64)
    mask = np.zeros((n_groups, max_len), dtype=bool)
    pos = np.zeros(n_groups, dtype=np.int64)
    for i, g in enumerate(gid):
        j = pos[g]
        members[g, j] = i
        mask[g, j] = True
        pos[g] += 1
    return gid, members, mask


def build_cat_maps(df):
    maps, cards = {}, []
    for c in cat_cols:
        vals = pd.Series(df[c].astype(str).fillna("__NA__")).unique()
        maps[c] = {v: i + 1 for i, v in enumerate(vals)}
        cards.append(len(vals) + 1)
    return maps, cards


def cat_codes(df, maps):
    arrs = []
    for c in cat_cols:
        arrs.append(
            df[c]
            .astype(str)
            .fillna("__NA__")
            .map(maps[c])
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )
    return np.stack(arrs, axis=1)


def run_torch_deepsets(train_df, test_df, y, splits):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        return None, None, "torch_unavailable"

    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DeepSetsModel(nn.Module):
        def __init__(self, n_num, cat_cards):
            super().__init__()
            emb_dims = [min(16, max(3, int(np.sqrt(card) + 1))) for card in cat_cards]
            self.embs = nn.ModuleList(
                [nn.Embedding(card, dim) for card, dim in zip(cat_cards, emb_dims)]
            )
            in_dim = n_num + sum(emb_dims)
            self.row_net = nn.Sequential(
                nn.Linear(in_dim, 96),
                nn.ReLU(),
                nn.BatchNorm1d(96),
                nn.Dropout(0.10),
                nn.Linear(96, 64),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(64 * 3, 96),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(96, 1),
            )

        def encode(self, nums, cats):
            emb = [layer(cats[:, i]) for i, layer in enumerate(self.embs)]
            return self.row_net(torch.cat([nums] + emb, dim=1))

        def forward(self, nums, cats, row_idx, member_idx, member_mask):
            focal = self.encode(nums[row_idx], cats[row_idx])
            flat_members = member_idx.reshape(-1)
            member_emb = self.encode(nums[flat_members], cats[flat_members]).reshape(
                member_idx.shape[0], member_idx.shape[1], -1
            )
            mask_f = member_mask.unsqueeze(-1).float()
            pooled_mean = (member_emb * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
            pooled_max = (
                member_emb.masked_fill(~member_mask.unsqueeze(-1), -1e4).max(1).values
            )
            return self.head(
                torch.cat([focal, pooled_mean, pooled_max], dim=1)
            ).squeeze(1)

    def context(df, scaler, maps, fit=False):
        df = df.reset_index(drop=True).copy()
        nums = df[num_cols].fillna(0)
        if fit:
            scaler.fit(nums)
        num_arr = scaler.transform(nums).astype(np.float32)
        cat_arr = cat_codes(df, maps)
        gid, members, mask = make_group_matrix(df)
        return {
            "num": torch.tensor(num_arr, dtype=torch.float32, device=device),
            "cat": torch.tensor(cat_arr, dtype=torch.long, device=device),
            "gid": torch.tensor(gid, dtype=torch.long, device=device),
            "members": torch.tensor(members, dtype=torch.long, device=device),
            "mask": torch.tensor(mask, dtype=torch.bool, device=device),
        }

    def predict(model, ctx, batch_size=4096):
        model.eval()
        n = ctx["num"].shape[0]
        out = np.zeros(n, dtype=np.float32)
        loader = DataLoader(
            TensorDataset(torch.arange(n, dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for (idx_cpu,) in loader:
                idx = idx_cpu.to(device)
                gid = ctx["gid"][idx]
                logits = model(
                    ctx["num"], ctx["cat"], idx, ctx["members"][gid], ctx["mask"][gid]
                )
                out[idx_cpu.numpy()] = torch.sigmoid(logits).detach().cpu().numpy()
        return out

    oof = np.zeros(len(train_df), dtype=np.float32)
    test_fold_preds = []

    for fold, (trn_idx, val_idx) in enumerate(splits, 1):
        trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        tst_df = test_df.reset_index(drop=True)

        maps, cards = build_cat_maps(trn_df)
        scaler = StandardScaler()
        trn_ctx = context(trn_df, scaler, maps, fit=True)
        val_ctx = context(val_df, scaler, maps, fit=False)
        tst_ctx = context(tst_df, scaler, maps, fit=False)

        model = DeepSetsModel(len(num_cols), cards).to(device)
        pos = max(float(y[trn_idx].sum()), 1.0)
        neg = max(float(len(trn_idx) - y[trn_idx].sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(min(50.0, neg / pos), device=device)
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        labels = torch.tensor(y[trn_idx].astype(np.float32), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(torch.arange(len(trn_idx), dtype=torch.long), labels),
            batch_size=2048,
            shuffle=True,
            drop_last=False,
        )

        model.train()
        for epoch in range(4):
            for idx_cpu, lab_cpu in loader:
                idx = idx_cpu.to(device)
                lab = lab_cpu.to(device)
                gid = trn_ctx["gid"][idx]
                logits = model(
                    trn_ctx["num"],
                    trn_ctx["cat"],
                    idx,
                    trn_ctx["members"][gid],
                    trn_ctx["mask"][gid],
                )
                loss = loss_fn(logits, lab)
                opt.zero_grad()
                loss.backward()
                opt.step()

        oof[val_idx] = predict(model, val_ctx)
        test_fold_preds.append(predict(model, tst_ctx))
        print(
            f"DeepSets fold {fold} snapshot AUC: {roc_auc_score(y[val_idx], oof[val_idx]):.6f}"
        )

        del model, trn_ctx, val_ctx, tst_ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return oof, np.mean(test_fold_preds, axis=0), "torch_deepsets"


def run_fallback_snapshot_specialist(train_df, test_df, y, splits):
    from sklearn.linear_model import SGDClassifier

    def pooled_frame(df, maps=None, fit_maps=False):
        df = df.reset_index(drop=True).copy()
        if fit_maps:
            maps, _ = build_cat_maps(df)
        cats = pd.DataFrame(
            cat_codes(df, maps), columns=[f"{c}_code" for c in cat_cols]
        )
        nums = df[num_cols].fillna(0).reset_index(drop=True)
        pooled_mean = (
            df.groupby(group_cols, sort=False)[num_cols]
            .transform("mean")
            .add_prefix("set_mean_")
            .reset_index(drop=True)
        )
        pooled_max = (
            df.groupby(group_cols, sort=False)[num_cols]
            .transform("max")
            .add_prefix("set_max_")
            .reset_index(drop=True)
        )
        return pd.concat([nums, cats, pooled_mean, pooled_max], axis=1).fillna(0), maps

    oof = np.zeros(len(train_df), dtype=np.float32)
    test_fold_preds = []

    for fold, (trn_idx, val_idx) in enumerate(splits, 1):
        x_tr, maps = pooled_frame(train_df.iloc[trn_idx], fit_maps=True)
        x_val, _ = pooled_frame(train_df.iloc[val_idx], maps=maps)
        x_test, _ = pooled_frame(test_df, maps=maps)

        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        clf = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.05,
            class_weight="balanced",
            max_iter=20,
            tol=1e-3,
            random_state=SEED + fold,
        )
        clf.fit(x_tr, y[trn_idx])
        oof[val_idx] = clf.predict_proba(x_val)[:, 1]
        test_fold_preds.append(clf.predict_proba(x_test)[:, 1])
        print(
            f"Fallback pooled-snapshot fold {fold} AUC: {roc_auc_score(y[val_idx], oof[val_idx]):.6f}"
        )

    return oof, np.mean(test_fold_preds, axis=0), "fallback_pooled_snapshot"


deep_oof, deep_test, snapshot_model = run_torch_deepsets(train, test, y, splits)
if deep_oof is None:
    deep_oof, deep_test, snapshot_model = run_fallback_snapshot_specialist(
        train, test, y, splits
    )

deep_auc = roc_auc_score(y, deep_oof)
print(f"Snapshot encoder OOF ROC AUC: {deep_auc:.6f} ({snapshot_model})")

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

train_lgb = train[feature_cols].copy()
test_lgb = test[feature_cols].copy()
for c in cat_cols:
    cats = pd.Index(
        pd.concat([train_lgb[c], test_lgb[c]], axis=0)
        .astype(str)
        .fillna("__NA__")
        .unique()
    )
    dtype = pd.CategoricalDtype(categories=cats)
    train_lgb[c] = train_lgb[c].astype(str).fillna("__NA__").astype(dtype)
    test_lgb[c] = test_lgb[c].astype(str).fillna("__NA__").astype(dtype)

train_lgb["deepsets_pitnextlap"] = deep_oof
test_lgb["deepsets_pitnextlap"] = deep_test
lgb_features = feature_cols + ["deepsets_pitnextlap"]

oof = np.zeros(len(train_lgb), dtype=np.float32)
test_preds = np.zeros(len(test_lgb), dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(splits, 1):
    model = LGBMClassifier(
        objective="binary",
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        class_weight="balanced",
        random_state=SEED + fold,
        n_jobs=min(8, os.cpu_count() or 1),
        verbosity=-1,
    )
    model.fit(
        train_lgb.iloc[trn_idx][lgb_features],
        y[trn_idx],
        eval_set=[(train_lgb.iloc[val_idx][lgb_features], y[val_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(100)],
    )
    oof[val_idx] = model.predict_proba(train_lgb.iloc[val_idx][lgb_features])[:, 1]
    test_preds += model.predict_proba(test_lgb[lgb_features])[:, 1] / len(splits)
    print(
        f"LightGBM stacked fold {fold} AUC: {roc_auc_score(y[val_idx], oof[val_idx]):.6f}"
    )

cv_auc = roc_auc_score(y, oof)
print(f"5-fold RaceYear-grouped ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_preds, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(oof, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(deep_oof, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "snapshot_oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "research_hypotheses_llm_claimed_used": ["000390"],
    "metric": "roc_auc",
    "cv_scheme": "5-fold StratifiedGroupKFold by RaceYear",
    "snapshot_model": snapshot_model,
    "snapshot_oof_auc": float(deep_auc),
    "stacked_oof_auc": float(cv_auc),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
