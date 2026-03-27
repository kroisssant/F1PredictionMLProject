import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from .config import (
    TARGET_COL, FLAT_FEATURE_COLS,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    LR_CHECKPOINT_PATH, LR_ARTIFACTS_PATH,
)
from .utils import compute_metrics, year_cv_folds


class LinearRegressionModel:
    def __init__(self, hyperparams: dict) -> None:
        self.hyperparams = hyperparams

    def _prepare_data(self, df: pd.DataFrame):
        df = df.dropna(subset=[TARGET_COL]).copy()
        available = [c for c in FLAT_FEATURE_COLS if c in df.columns]
        X = pd.get_dummies(df[available])
        y = df[TARGET_COL].values

        train_mask = df["year"].isin(TRAIN_YEARS).values
        val_mask = df["year"].isin(VAL_YEARS).values
        test_mask = df["year"].isin(TEST_YEARS).values

        train_medians = X.iloc[train_mask].median()
        X = X.fillna(train_medians)

        return X, y, train_mask, val_mask, test_mask, list(X.columns), train_medians

    def train(self, df: pd.DataFrame, use_cross_validation: bool = False) -> dict:
        seed = self.hyperparams.get("seed", None)
        if seed is not None:
            np.random.seed(seed)

        X, y, train_mask, val_mask, test_mask, feature_cols, train_medians = self._prepare_data(df)

        result = {}

        if use_cross_validation:
            folds = year_cv_folds(df.dropna(subset=[TARGET_COL]).reset_index(drop=True))
            cv_folds = []
            for i, (tr_idx, val_idx, val_year) in enumerate(folds):
                m = LinearRegression()
                m.fit(X.iloc[tr_idx], y[tr_idx])
                preds = m.predict(X.iloc[val_idx])
                metrics = compute_metrics(y[val_idx], preds)
                cv_folds.append({"fold": i + 1, "val_year": val_year,
                                  "val_rmse": metrics["rmse"],
                                  "val_mae": metrics["mae"], "val_r2": metrics["r2"]})
            result["cv_folds"] = cv_folds

        model = LinearRegression()
        model.fit(X.iloc[train_mask], y[train_mask])

        result["train"] = compute_metrics(y[train_mask], model.predict(X.iloc[train_mask]))
        result["val"] = compute_metrics(y[val_mask], model.predict(X.iloc[val_mask]))

        if test_mask.any():
            result["test"] = compute_metrics(y[test_mask], model.predict(X.iloc[test_mask]))
        else:
            result["test"] = None

        with open(LR_CHECKPOINT_PATH, "wb") as f:
            pickle.dump(model, f)
        artifacts = {"feature_cols": feature_cols, "train_medians": train_medians}
        with open(LR_ARTIFACTS_PATH, "wb") as f:
            pickle.dump(artifacts, f)

        self._model = model
        self._artifacts = artifacts
        return result
    @classmethod
    def load(cls, checkpoint_path=None, artifacts_path=None):
        checkpoint_path = checkpoint_path or LR_CHECKPOINT_PATH
        artifacts_path = artifacts_path or LR_ARTIFACTS_PATH
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
        X = pd.get_dummies(df[available].copy())
        X = X.reindex(columns=self._artifacts["feature_cols"], fill_value=0)
        X = X.fillna(self._artifacts["train_medians"])
        return self._model.predict(X)
