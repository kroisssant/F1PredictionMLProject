import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .config import (
    TARGET_COL, FLAT_FEATURE_COLS, HIST_CAT_COLS,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    GRU_CHECKPOINT_PATH, GRU_ARTIFACTS_PATH,
)
from .utils import compute_metrics, year_cv_folds

GRU_CAT_COLS = [c for c in HIST_CAT_COLS if c != "pit_lap"]
GRU_NUM_COLS = [c for c in FLAT_FEATURE_COLS if c not in set(GRU_CAT_COLS)]


def _make_sequences(sub_df, feature_cols, f_sc, t_sc, sequence_length):
    seqs, targets = [], []
    for _, grp in sub_df.groupby(["year", "round_number", "driver"]):
        if len(grp) < sequence_length + 1:
            continue
        grp = grp.sort_values("number").reset_index(drop=True)
        feats = f_sc.transform(grp[feature_cols].values.astype(np.float32))
        tgts = t_sc.transform(grp[[TARGET_COL]].values.astype(np.float32)).flatten()
        for i in range(len(grp) - sequence_length):
            seqs.append(feats[i: i + sequence_length])
            targets.append(tgts[i + sequence_length])
    if not seqs:
        return None, None
    X = torch.tensor(np.array(seqs, dtype=np.float32))
    y = torch.tensor(np.array(targets, dtype=np.float32).reshape(-1, 1))
    return X, y


class _GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


class GRUModel:
    def __init__(self, hyperparams: dict) -> None:
        self.hyperparams = hyperparams

    def _prepare_data(self, df: pd.DataFrame, sequence_length: int):
        df = df[df[TARGET_COL].notna()].copy()

        cat_cols = [c for c in GRU_CAT_COLS if c in df.columns]
        num_cols = [c for c in GRU_NUM_COLS if c in df.columns]

        for col in num_cols:
            df[col] = df.groupby(["year", "round_number", "driver"])[col].transform(
                lambda x: x.fillna(x.median())
            )
        df = df.fillna(0)

        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        feature_cols = cat_cols + num_cols

        train_mask = df["year"].isin(TRAIN_YEARS)

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        feature_scaler.fit(df.loc[train_mask, feature_cols].values.astype(np.float32))
        target_scaler.fit(df.loc[train_mask, [TARGET_COL]].values.astype(np.float32))

        splits = {}
        for name, years in [("train", TRAIN_YEARS), ("val", VAL_YEARS), ("test", TEST_YEARS)]:
            sub = df[df["year"].isin(years)]
            X, y = _make_sequences(sub, feature_cols, feature_scaler, target_scaler, sequence_length)
            splits[name] = (X, y)

        input_size = len(feature_cols)
        return splits, feature_scaler, target_scaler, label_encoders, input_size, feature_cols

    def train(self, df: pd.DataFrame, use_cross_validation: bool = False) -> dict:
        hidden_size = self.hyperparams.get("hidden_size", 64)
        num_layers = self.hyperparams.get("num_layers", 2)
        lr = self.hyperparams.get("lr", 1e-3)
        batch_size = self.hyperparams.get("batch_size", 32)
        num_epochs = self.hyperparams.get("num_epochs", 50)
        dropout = self.hyperparams.get("dropout", 0.2)
        sequence_length = self.hyperparams.get("sequence_length", 10)
        patience = self.hyperparams.get("patience", 10)
        seed = self.hyperparams.get("seed", None)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = nn.MSELoss()

        splits, feature_scaler, target_scaler, label_encoders, input_size, _ = \
            self._prepare_data(df, sequence_length)

        result = {}

        if use_cross_validation:
            clean_df = df[df[TARGET_COL].notna()].reset_index(drop=True)
            folds = year_cv_folds(clean_df)
            cv_folds = []
            for fold_i, (tr_idx, val_idx, val_year) in enumerate(folds):
                fold_df = clean_df.copy()
                cat_cols = [c for c in GRU_CAT_COLS if c in fold_df.columns]
                num_cols = [c for c in GRU_NUM_COLS if c in fold_df.columns]
                for col in num_cols:
                    fold_df[col] = fold_df.groupby(["year", "round_number", "driver"])[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                fold_df = fold_df.fillna(0)
                for col in cat_cols:
                    le = LabelEncoder()
                    fold_df[col] = le.fit_transform(fold_df[col].astype(str))
                fc = cat_cols + num_cols
                f_sc = StandardScaler()
                t_sc = StandardScaler()
                f_sc.fit(fold_df.iloc[tr_idx][fc].values.astype(np.float32))
                t_sc.fit(fold_df.iloc[tr_idx][[TARGET_COL]].values.astype(np.float32))

                Xtr, ytr = _make_sequences(fold_df.iloc[tr_idx], fc, f_sc, t_sc, sequence_length)
                Xvl, yvl = _make_sequences(fold_df.iloc[val_idx], fc, f_sc, t_sc, sequence_length)
                if Xtr is None or Xvl is None:
                    continue

                fold_net = _GRUNet(len(fc), hidden_size, num_layers, 1, dropout).to(device)
                fold_opt = torch.optim.Adam(fold_net.parameters(), lr=lr)
                tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
                vl_loader = DataLoader(TensorDataset(Xvl, yvl), batch_size=batch_size, shuffle=False)

                best_vl, pat_cnt = float("inf"), 0
                for _ in range(num_epochs):
                    fold_net.train()
                    for bx, by in tr_loader:
                        bx, by = bx.to(device), by.to(device)
                        fold_opt.zero_grad()
                        loss_fn(fold_net(bx), by).backward()
                        fold_opt.step()
                    fold_net.eval()
                    with torch.no_grad():
                        vl = sum(loss_fn(fold_net(bx.to(device)), by.to(device)).item()
                                 for bx, by in vl_loader) / len(vl_loader)
                    if vl < best_vl:
                        best_vl, pat_cnt = vl, 0
                    else:
                        pat_cnt += 1
                        if pat_cnt >= patience:
                            break

                fold_net.eval()
                preds, actuals = [], []
                with torch.no_grad():
                    for bx, by in vl_loader:
                        preds.append(fold_net(bx.to(device)).cpu().numpy())
                        actuals.append(by.numpy())
                p = t_sc.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
                a = t_sc.inverse_transform(np.concatenate(actuals).reshape(-1, 1)).flatten()
                m = compute_metrics(a, p)
                cv_folds.append({"fold": fold_i + 1, "val_year": val_year,
                                  "val_rmse": m["rmse"], "val_mae": m["mae"], "val_r2": m["r2"]})
            result["cv_folds"] = cv_folds

        Xtr, ytr = splits["train"]
        Xvl, yvl = splits["val"]

        net = _GRUNet(input_size, hidden_size, num_layers, 1, dropout).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
        vl_loader = DataLoader(TensorDataset(Xvl, yvl), batch_size=batch_size, shuffle=False)

        best_val, pat_cnt, best_state = float("inf"), 0, None
        for _ in range(num_epochs):
            net.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss_fn(net(bx), by).backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                vl = sum(loss_fn(net(bx.to(device)), by.to(device)).item()
                         for bx, by in vl_loader) / len(vl_loader)
            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                pat_cnt = 0
            else:
                pat_cnt += 1
                if pat_cnt >= patience:
                    break

        if best_state:
            net.load_state_dict(best_state)

        def _eval_split(loader):
            net.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for bx, by in loader:
                    preds.append(net(bx.to(device)).cpu().numpy())
                    actuals.append(by.numpy())
            p = target_scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
            a = target_scaler.inverse_transform(np.concatenate(actuals).reshape(-1, 1)).flatten()
            return compute_metrics(a, p)

        result["train"] = _eval_split(tr_loader)
        result["val"]   = _eval_split(vl_loader)

        Xte, yte = splits["test"]
        if Xte is not None:
            te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)
            result["test"] = _eval_split(te_loader)
        else:
            result["test"] = None

        os.makedirs(os.path.dirname(GRU_CHECKPOINT_PATH), exist_ok=True)
        torch.save(net.state_dict(), GRU_CHECKPOINT_PATH)
        with open(GRU_ARTIFACTS_PATH, "wb") as f:
            pickle.dump({
                "feature_scaler": feature_scaler,
                "target_scaler": target_scaler,
                "label_encoders": label_encoders,
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "sequence_length": sequence_length,
            }, f)

        return result

    @classmethod
    def load(cls, checkpoint_path=None, artifacts_path=None):
        checkpoint_path = checkpoint_path or GRU_CHECKPOINT_PATH
        artifacts_path = artifacts_path or GRU_ARTIFACTS_PATH
        with open(artifacts_path, "rb") as f:
            arts = pickle.load(f)
        net = _GRUNet(arts["input_size"], arts["hidden_size"], arts["num_layers"],
                      1, arts["dropout"])
        net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        net.eval()
        instance = cls({})
        instance._net = net
        instance._artifacts = arts
        return instance

    def predict(self, df: pd.DataFrame):
        arts = self._artifacts
        sequence_length = arts["sequence_length"]
        cat_cols = [c for c in GRU_CAT_COLS if c in df.columns]
        num_cols = [c for c in GRU_NUM_COLS if c in df.columns]
        feature_cols = cat_cols + num_cols

        df = df.copy()
        for col in num_cols:
            df[col] = df.groupby(["year", "round_number", "driver"])[col].transform(
                lambda x: x.fillna(x.median())
            )
        df = df.fillna(0)
        for col in cat_cols:
            df[col] = arts["label_encoders"][col].transform(df[col].astype(str))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net.to(device)
        self._net.eval()

        seqs = []
        for _, grp in df.groupby(["year", "round_number", "driver"]):
            if len(grp) < sequence_length + 1:
                continue
            grp = grp.sort_values("number").reset_index(drop=True)
            feats = arts["feature_scaler"].transform(grp[feature_cols].values.astype(np.float32))
            for i in range(len(grp) - sequence_length):
                seqs.append(feats[i: i + sequence_length])

        if not seqs:
            return np.array([])

        X = torch.tensor(np.array(seqs, dtype=np.float32)).to(device)
        with torch.no_grad():
            raw = self._net(X).cpu().numpy()
        return arts["target_scaler"].inverse_transform(raw.reshape(-1, 1)).flatten()
