import os
import random
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from models.config import DATA_PATH, IMAGE_PATH, TRAIN_YEARS, VAL_YEARS, TARGET_COL
from models import LinearRegressionModel, XGBoostModel, GRUModel, TFTModel

optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs(IMAGE_PATH, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Loading data...")

df = pd.read_csv(DATA_PATH)

N_TRIALS = 10


def make_lr_objective(df):
    def objective(trial):
        result = LinearRegressionModel({}).train(df, use_cross_validation=True)
        return float(np.mean([f["val_rmse"] for f in result["cv_folds"]]))
    return objective

def make_xgb_objective(df):
    def objective(trial):
        hp = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "seed": SEED,
        }
        result = XGBoostModel(hp).train(df, use_cross_validation=True)
        return float(np.mean([f["val_rmse"] for f in result["cv_folds"]]))
    return objective


def make_gru_objective(df):
    def objective(trial):
        hp = {
            "hidden_size": trial.suggest_int("hidden_size", 32, 128, step=32),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "sequence_length": trial.suggest_int("sequence_length", 5, 15),
            "num_epochs": 20, "patience": 5, "seed": SEED,
        }
        result = GRUModel(hp).train(df, use_cross_validation=True)
        return float(np.mean([f["val_rmse"] for f in result["cv_folds"]]))
    return objective


def make_tft_objective(df):
    def objective(trial):
        hp = {
            "state_size": trial.suggest_categorical("state_size", [32, 64, 96]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
            "attention_heads": trial.suggest_categorical("attention_heads", [2, 4]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
            "history_len": trial.suggest_int("history_len", 10, 25),
            "future_len": trial.suggest_int("future_len", 3, 7),
            "max_epochs": 30, "patience": 5, "epoch_iters": 50, "seed": SEED,
        }
        result = TFTModel(hp).train(df, use_cross_validation=True)
        return float(np.mean([f["val_rmse"] for f in result["cv_folds"]]))
    return objective

FINAL_OVERRIDES = {
    "GRU": {"num_epochs": 100, "patience": 10},
    "TFT": {"max_epochs": 300, "patience": 15, "epoch_iters": 100},
}
True
configs = [
    ("LinearRegression", LinearRegressionModel, make_lr_objective(df)),
    ("XGBoost", XGBoostModel, make_xgb_objective(df)),
    ("GRU", GRUModel, make_gru_objective(df)),
    ("TFT", TFTModel, make_tft_objective(df)),
]

results = {}

for name, ModelClass, objective in configs:
    print(f"Tuning {name} ({N_TRIALS} trials)...")

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS,
                   show_progress_bar=True, catch=(Exception,))

    best_hp = {"seed": SEED, **study.best_params, **FINAL_OVERRIDES.get(name, {})}
    print(f"Best val RMSE (tuning): {study.best_value:.4f}s")
    print(f"Best params: {study.best_params}")
    print(f"Final training")

    model = ModelClass(best_hp)
    result = model.train(df, use_cross_validation=True)

    print(f"{'split':<6} {'rmse':>8} {'mae':>8} {'r2':>7}")
    for split in ("train", "val", "test"):
        m = result.get(split)
        if m:
            print(f"{split:<6} {m['rmse']:>7.3f}s {m['mae']:>7.3f}s {m['r2']:>7.3f}")

    results[name] = {"model": model, "result": result, "study": study}

# Plot 1: Optuna convergence + CV RMSE by year
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle("Hyperparameter Tuning & Year-by-Year CV Error", fontsize=13)

for col, name in enumerate(["LinearRegression", "XGBoost", "GRU", "TFT"]):
    study = results[name]["study"]
    cv_folds = results[name]["result"].get("cv_folds", [])

    # Row 0, optimization history
    ax = axes[0][col]
    vals = [t.value for t in study.trials if t.value is not None]
    best_curve = np.minimum.accumulate(vals) if vals else []
    ax.plot(vals, "o", alpha=0.45, color="#4c72b0", markersize=4, label="trial")
    ax.plot(best_curve, "-", color="#c44e52", linewidth=1.8, label="best so far")
    ax.set_title(f"{name} -tuning")
    ax.set_xlabel("trial")
    ax.set_ylabel("val RMSE (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Row 1, CV error by year
    ax = axes[1][col]
    if cv_folds:
        years = [f["val_year"] for f in cv_folds]
        rmses = [f["val_rmse"] for f in cv_folds]
        # highlight the official val year in red
        colors = ["#c44e52" if y in VAL_YEARS else "#4c72b0" for y in years]
        ax.bar(years, rmses, color=colors, alpha=0.75, edgecolor="white", width=0.6)
        ax.set_title(f"{name} -CV RMSE by year")
        ax.set_xlabel("validation year")
        ax.set_ylabel("RMSE (s)")
        ax.set_xticks(years)
        ax.grid(True, axis="y", alpha=0.25)

fig.tight_layout()
out1 = os.path.join(IMAGE_PATH, "tuning_and_cv.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")