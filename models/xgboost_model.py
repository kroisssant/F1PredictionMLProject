import pickle
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from xgboost import XGBRegressor

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

from .config import (
    TARGET_COL, FLAT_FEATURE_COLS,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XG_CHECKPOINT_PATH, XG_ARTIFACTS_PATH,
)
from .utils import compute_metrics, year_cv_folds


class XGBoostModel:
    def __init__(self, hyperparams: dict) -> None:
        self.hyperparams = hyperparams

    def _prepare_data(self, df: pd.DataFrame):
        df = df.dropna(subset=[TARGET_COL]).copy()
        available = [c for c in FLAT_FEATURE_COLS if c in df.columns]
        X = pd.get_dummies(df[available], drop_first=True)
        y = df[TARGET_COL].values

        train_mask = df["year"].isin(TRAIN_YEARS).values
        val_mask = df["year"].isin(VAL_YEARS).values
        test_mask = df["year"].isin(TEST_YEARS).values

        scaler = StandardScaler()
        X_arr = X.values.astype(np.float32)
        X_arr[train_mask] = scaler.fit_transform(X_arr[train_mask])
        X_arr[~train_mask] = scaler.transform(X_arr[~train_mask])

        return X_arr, y, train_mask, val_mask, test_mask, list(X.columns), scaler

    def train(self, df: pd.DataFrame, use_cross_validation: bool = False) -> dict:
        X, y, train_mask, val_mask, test_mask, feature_cols, scaler = self._prepare_data(df)

        result = {}

        if use_cross_validation:
            clean_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

            available_cv = [c for c in FLAT_FEATURE_COLS if c in clean_df.columns]
            X_raw = pd.get_dummies(clean_df[available_cv], drop_first=True).values.astype(np.float32)
            y_cv = clean_df[TARGET_COL].values

            folds = year_cv_folds(clean_df)
            cv_folds = []
            for i, (tr_idx, val_idx, val_year) in enumerate(folds):
                fold_scaler = StandardScaler()
                X_tr = fold_scaler.fit_transform(X_raw[tr_idx])
                X_vl = fold_scaler.transform(X_raw[val_idx])

                hp_cv = {**self.hyperparams, "objective": "reg:squarederror", "n_jobs": -1,
                         "device": DEVICE, "tree_method": "hist"}
                m = XGBRegressor(**hp_cv)
                m.fit(X_tr, y_cv[tr_idx])
                fold_metrics = compute_metrics(y_cv[val_idx], m.predict(X_vl))
                cv_folds.append({"fold": i + 1, "val_year": val_year,
                                  "val_rmse": fold_metrics["rmse"],
                                  "val_mae": fold_metrics["mae"],
                                  "val_r2": fold_metrics["r2"]})
            result["cv_folds"] = cv_folds

        hp = {**self.hyperparams, "objective": "reg:squarederror", "n_jobs": -1,
              "device": DEVICE, "tree_method": "hist"}
        model = XGBRegressor(**hp)
        model.fit(X[train_mask], y[train_mask])

        result["train"] = compute_metrics(y[train_mask], model.predict(X[train_mask]))
        result["val"] = compute_metrics(y[val_mask], model.predict(X[val_mask]))

        if test_mask.any():
            result["test"] = compute_metrics(y[test_mask], model.predict(X[test_mask]))
        else:
            result["test"] = None

        with open(XG_CHECKPOINT_PATH, "wb") as f:
            pickle.dump(model, f)
        artifacts = {"scaler": scaler, "feature_cols": feature_cols}
        with open(XG_ARTIFACTS_PATH, "wb") as f:
            pickle.dump(artifacts, f)

        self._model = model
        self._artifacts = artifacts
        return result

    def load(cls, checkpoint_path=None, artifacts_path=None):
        checkpoint_path = checkpoint_path or XG_CHECKPOINT_PATH
        artifacts_path = artifacts_path or XG_ARTIFACTS_PATH
        with open(checkpoint_path, "rb") as f:
            model = pickle.load(f)
        with open(artifacts_path, "rb") as f:
            arts = pickle.load(f)
        instance = cls({})
        instance._model = model
        instance._artifacts = arts
        return instance

    def predict(self, df: pd.DataFrame):
        available = [c for c in FLAT_FEATURE_COLS if c in df.columns]
        X = pd.get_dummies(df[available].copy(), drop_first=True)
        X = X.reindex(columns=self._artifacts["feature_cols"], fill_value=0)
        X_arr = X.values.astype(np.float32)
        if self._artifacts["scaler"] is not None:
            X_arr = self._artifacts["scaler"].transform(X_arr)
        return self._model.predict(X_arr)
