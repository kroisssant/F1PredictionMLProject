import os
import pickle
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from omegaconf import OmegaConf
from tft_torch.tft import TemporalFusionTransformer
import tft_torch.loss as tft_loss

from .config import (
    TFT_CHECKPOINT_PATH, TFT_ARTIFACTS_PATH,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    TARGET_COL, GROUP_COLS,
    STATIC_NUM_COLS,
    HIST_NUM_COLS, HIST_CAT_COLS,
    FUTURE_NUM_COLS,
    CLIP_LO, CLIP_HI,
)
from .utils import compute_metrics, year_cv_folds

_BATCH_INFER = 2048

def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(GROUP_COLS + ["number"]).reset_index(drop=True)
    for c in HIST_CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)
    # pit_lap is 0/1/2 in practice - anything higher is noise (data issue)
    df["pit_lap"] = (
        pd.to_numeric(df["pit_lap"], errors="coerce")
        .fillna(0).clip(0, 2).astype(int).astype(str)
    )
    for c in set(HIST_NUM_COLS + FUTURE_NUM_COLS):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df.groupby(GROUP_COLS)[c].transform(lambda x: x.ffill().bfill())
    for c in STATIC_NUM_COLS + HIST_NUM_COLS + FUTURE_NUM_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df


def _encode_categoricals(df: pd.DataFrame):
    encoders, cardinalities = {}, {}
    for c in HIST_CAT_COLS:
        if c not in df.columns:
            continue
        le = LabelEncoder()
        df[f"{c}_enc"] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
        cardinalities[c] = int(df[f"{c}_enc"].max()) + 1
    return df, encoders, cardinalities


def _scale_numerics(df: pd.DataFrame, train_mask: pd.Series):
    scalers, clip_bounds = {}, {}
    for c in set(STATIC_NUM_COLS + HIST_NUM_COLS + FUTURE_NUM_COLS):
        if c not in df.columns:
            continue
        train_vals = df.loc[train_mask, c].values
        # percentile clip on train only some gap cols go huge on safety car laps
        lo = float(np.nanpercentile(train_vals, CLIP_LO * 100))
        hi = float(np.nanpercentile(train_vals, CLIP_HI * 100))
        clip_bounds[c] = (lo, hi)
        clipped = df[c].clip(lower=lo, upper=hi).values
        sc = StandardScaler()
        sc.fit(clipped[train_mask].reshape(-1, 1))
        df[f"{c}_sc"] = sc.transform(clipped.reshape(-1, 1)).flatten()
        scalers[c] = sc
    return df, scalers, clip_bounds


def _build_windows(df: pd.DataFrame, history_len: int, future_len: int) -> dict:
    total = history_len + future_len
    sn_cols = [f"{c}_sc" for c in STATIC_NUM_COLS]
    sc_cols = []
    hn_cols = [f"{c}_sc" for c in HIST_NUM_COLS]
    hc_cols = [f"{c}_enc" for c in HIST_CAT_COLS]
    fn_cols = [f"{c}_sc" for c in FUTURE_NUM_COLS]
    t_col = f"{TARGET_COL}_sc"

    arrays = {
        "static_feats_numeric": [],
        "static_feats_categorical": [],
        "historical_ts_numeric": [],
        "historical_ts_categorical": [],
        "future_ts_numeric": [],
        "target": [],
    }

    for _, grp in df.groupby(GROUP_COLS, sort=False):
        if len(grp) < total:
            continue
        for start in range(len(grp) - total + 1):
            w = grp.iloc[start: start + total]
            h = w.iloc[:history_len]
            f = w.iloc[history_len:]
            arrays["static_feats_numeric"].append(h.iloc[0][sn_cols].values.astype(np.float32))
            arrays["static_feats_categorical"].append(h.iloc[0][sc_cols].values.astype(np.int32))
            arrays["historical_ts_numeric"].append(h[hn_cols].values.astype(np.float32))
            arrays["historical_ts_categorical"].append(h[hc_cols].values.astype(np.int32))
            arrays["future_ts_numeric"].append(f[fn_cols].values.astype(np.float32))
            arrays["target"].append(f[t_col].values.astype(np.float32))

    return {k: np.stack(v) for k, v in arrays.items() if v}


class _DictDataset(Dataset):
    def __init__(self, arrays: dict):
        self.keys = list(arrays.keys())
        for k, v in arrays.items():
            t = torch.FloatTensor(v) if np.issubdtype(v.dtype, np.floating) else torch.LongTensor(v)
            setattr(self, k, t)

    def __len__(self):
        return getattr(self, self.keys[0]).shape[0]

    def __getitem__(self, idx):
        return {k: getattr(self, k)[idx] for k in self.keys}


def _build_cfg(cardinalities: dict, state_size: int, lstm_layers: int,
               attention_heads: int, dropout: float) -> OmegaConf:
    return OmegaConf.create({
        "task_type": "regression",
        "target_window_start": None,
        "model": {
            "dropout": dropout,
            "state_size": state_size,
            "output_quantiles": [0.1, 0.5, 0.9],
            "lstm_layers": lstm_layers,
            "attention_heads": attention_heads,
        },
        "data_props": {
            "num_historical_numeric": len(HIST_NUM_COLS),
            "num_historical_categorical": len(HIST_CAT_COLS),
            "historical_categorical_cardinalities": [cardinalities[c] for c in HIST_CAT_COLS],
            "num_static_numeric": len(STATIC_NUM_COLS),
            "num_static_categorical": 0,
            "static_categorical_cardinalities": [],
            "num_future_numeric": len(FUTURE_NUM_COLS),
            "num_future_categorical": 0,
            "future_categorical_cardinalities": [],
        },
    })


def _train_loop(cfg, splits: dict, device, lr: float, batch_size: int,
                max_epochs: int, patience: int, epoch_iters: int,
                max_grad_norm: float, checkpoint_path: str = None):
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for p in m.parameters():
                (nn.init.orthogonal_ if p.dim() >= 2 else nn.init.zeros_)(p)

    model = TemporalFusionTransformer(config=cfg)
    model.apply(weight_init)
    model.to(device)

    train_ds = _DictDataset(splits["train"])
    val_ds = _DictDataset(splits["val"])
    train_iter = cycle(DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True))
    val_loader = DataLoader(val_ds, batch_size=_BATCH_INFER, shuffle=False)

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    quantiles_t = torch.tensor([0.1, 0.5, 0.9], device=device)
    best_val = float("inf")
    best_state = None
    pat_cnt = 0

    for _ in range(1, max_epochs + 1):
        model.train()
        for _ in range(epoch_iters):
            batch = next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(batch)
            loss, _, _ = tft_loss.get_quantiles_loss_and_q_risk(
                outputs=out["predicted_quantiles"],
                targets=batch["target"],
                desired_quantiles=quantiles_t,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

        model.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch)
                l, _, _ = tft_loss.get_quantiles_loss_and_q_risk(
                    outputs=out["predicted_quantiles"],
                    targets=batch["target"],
                    desired_quantiles=quantiles_t,
                )
                v_losses.append(l.item())

        val_loss = float(np.mean(v_losses))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _evaluate_split(loader, model, scaler, device) -> dict:
    preds, actuals = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            preds.append(out["predicted_quantiles"][:, :, 1].cpu().numpy())
            actuals.append(batch["target"].cpu().numpy())

    pred_s = scaler.inverse_transform(
        np.concatenate(preds, axis=0).reshape(-1, 1)).flatten()
    act_s = scaler.inverse_transform(
        np.concatenate(actuals, axis=0).reshape(-1, 1)).flatten()

    return compute_metrics(act_s, pred_s)


class TFTModel:
    def __init__(self, hyperparams: dict) -> None:
        self.hyperparams = hyperparams

    def train(self, df: pd.DataFrame, use_cross_validation: bool = False) -> dict:
        hp = self.hyperparams
        state_size = hp.get("state_size", 64)
        lstm_layers = hp.get("lstm_layers", 1)
        attention_heads = hp.get("attention_heads", 4)
        dropout = hp.get("dropout", 0.1)
        lr = hp.get("lr", 1e-3)
        batch_size = hp.get("batch_size", 256)
        history_len = hp.get("history_len", 20)
        future_len = hp.get("future_len", 5)
        max_epochs = hp.get("max_epochs", 300)
        patience = hp.get("patience", 15)
        epoch_iters = hp.get("epoch_iters", 100)
        max_grad_norm = hp.get("max_grad_norm", 1.0)
        seed = hp.get("seed", None)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        clean_df = _prepare_data(df)
        train_mask = clean_df["year"].isin(TRAIN_YEARS)
        clean_df, encoders, cardinalities = _encode_categoricals(clean_df)
        clean_df, scalers, clip_bounds = _scale_numerics(clean_df, train_mask)

        splits = {}
        for name, years in [("train", TRAIN_YEARS), ("val", VAL_YEARS), ("test", TEST_YEARS)]:
            sub = clean_df[clean_df["year"].isin(years)].reset_index(drop=True)
            splits[name] = _build_windows(sub, history_len, future_len)

        cfg = _build_cfg(cardinalities, state_size, lstm_layers, attention_heads, dropout)
        result = {}

        if use_cross_validation:
            folds = year_cv_folds(clean_df)
            cv_folds = []
            for fold_i, (tr_idx, val_idx, val_year) in enumerate(folds):
                fold_df = clean_df.copy()
                fold_train_mask = pd.Series(False, index=fold_df.index)
                fold_train_mask.iloc[tr_idx] = True
                fold_df, fold_sc, _ = _scale_numerics(fold_df, fold_train_mask)
                fold_cfg = _build_cfg(cardinalities, state_size, lstm_layers, attention_heads, dropout)

                tr_windows = _build_windows(fold_df.iloc[tr_idx].reset_index(drop=True), history_len, future_len)
                vl_windows = _build_windows(fold_df.iloc[val_idx].reset_index(drop=True), history_len, future_len)
                if not tr_windows or not vl_windows:
                    continue

                # capped at 50 epochs for CV - full training comes after
                fold_model = _train_loop(fold_cfg, {"train": tr_windows, "val": vl_windows},
                                         device, lr=lr, batch_size=batch_size,
                                         max_epochs=min(max_epochs, 50), patience=patience,
                                         epoch_iters=epoch_iters, max_grad_norm=max_grad_norm)
                fold_model.eval()
                vl_loader = DataLoader(_DictDataset(vl_windows), batch_size=_BATCH_INFER, shuffle=False)
                m = _evaluate_split(vl_loader, fold_model, fold_sc[TARGET_COL], device)
                cv_folds.append({"fold": fold_i + 1, "val_year": val_year,
                                  "val_rmse": m["rmse"], "val_mae": m["mae"], "val_r2": m["r2"]})
            result["cv_folds"] = cv_folds

        os.makedirs(os.path.dirname(TFT_CHECKPOINT_PATH), exist_ok=True)
        model = _train_loop(cfg, splits, device, lr=lr, batch_size=batch_size,
                            max_epochs=max_epochs, patience=patience,
                            epoch_iters=epoch_iters, max_grad_norm=max_grad_norm,
                            checkpoint_path=TFT_CHECKPOINT_PATH)
        model.eval()
        target_scaler = scalers[TARGET_COL]

        for split_name in ("train", "val"):
            loader = DataLoader(_DictDataset(splits[split_name]), batch_size=_BATCH_INFER, shuffle=False)
            result[split_name] = _evaluate_split(loader, model, target_scaler, device)

        test_windows = splits.get("test", {})
        if len(test_windows.get("target", [])) > 0:
            loader = DataLoader(_DictDataset(test_windows), batch_size=_BATCH_INFER, shuffle=False)
            result["test"] = _evaluate_split(loader, model, target_scaler, device)
        else:
            result["test"] = None

        with open(TFT_ARTIFACTS_PATH, "wb") as fh:
            pickle.dump({
                "encoders": encoders,
                "scalers": scalers,
                "clip_bounds": clip_bounds,
                "cardinalities": cardinalities,
                "config": OmegaConf.to_container(cfg),
                "hyperparams": hp,
            }, fh)

        return result

    def load(cls, checkpoint_path=None, artifacts_path=None):
        checkpoint_path = checkpoint_path or TFT_CHECKPOINT_PATH
        artifacts_path = artifacts_path or TFT_ARTIFACTS_PATH
        with open(artifacts_path, "rb") as f:
            arts = pickle.load(f)
        cfg = OmegaConf.create(arts["config"])
        net = TemporalFusionTransformer(config=cfg)
        net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        net.eval()
        instance = cls({})
        instance._net = net
        instance._artifacts = arts
        return instance

    def predict(self, df: pd.DataFrame):
        arts = self._artifacts
        hp = arts["hyperparams"]
        history_len = hp.get("history_len", 20)
        future_len = hp.get("future_len", 5)

        df = _prepare_data(df)
        for c, le in arts["encoders"].items():
            if c in df.columns:
                df[f"{c}_enc"] = le.transform(df[c].astype(str))
        for c, sc in arts["scalers"].items():
            if c in df.columns:
                lo, hi = arts["clip_bounds"][c]
                clipped = df[c].clip(lower=lo, upper=hi).values
                df[f"{c}_sc"] = sc.transform(clipped.reshape(-1, 1)).flatten()

        arrays = _build_windows(df, history_len, future_len)
        if not arrays:
            return np.array([])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net.to(device)
        self._net.eval()

        loader = DataLoader(_DictDataset(arrays), batch_size=_BATCH_INFER, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self._net(batch)
                preds.append(out["predicted_quantiles"][:, :, 1].cpu().numpy())

        raw = np.concatenate(preds, axis=0).reshape(-1, 1)
        return arts["scalers"][TARGET_COL].inverse_transform(raw).flatten()
