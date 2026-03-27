import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import TRAIN_YEARS, VAL_YEARS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def year_cv_folds(df: pd.DataFrame, year_col: str = "year") -> list:
    all_years = TRAIN_YEARS + VAL_YEARS
    folds = []
    for k in range(1, len(all_years)):
        train_years = all_years[:k]
        val_year = all_years[k]
        train_idx = np.where(df[year_col].isin(train_years))[0]
        val_idx = np.where(df[year_col] == val_year)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx, val_year))
    return folds
