# Data paths
DATA_PATH = "./export/f1_model_dataset.csv"
TFT_CHECKPOINT_PATH = "./models/checkpoints/tft_checkpoint.pt"
TFT_ARTIFACTS_PATH = "./models/checkpoints/tft_artifacts.pkl"

XG_CHECKPOINT_PATH = "./models/checkpoints/XG_checkpoint.pt"
XG_ARTIFACTS_PATH = "./models/checkpoints/XG_artifacts.pkl"

LR_CHECKPOINT_PATH = "./models/checkpoints/LR_checkpoint.pt"
LR_ARTIFACTS_PATH = "./models/checkpoints/LR_artifacts.pkl"

GRU_CHECKPOINT_PATH = "./models/checkpoints/GRU_checkpoint.pt"
GRU_ARTIFACTS_PATH = "./models/checkpoints/GRU_artifacts.pkl"

IMAGE_PATH = "./images/"


# Train / val / test splits
TRAIN_YEARS = list(range(2018, 2024))
VAL_YEARS = [2024]
TEST_YEARS = [2025]

# Target 
TARGET_COL = "delta_fastest" # seconds behind fastest car on the same lap

GROUP_COLS = ["year", "round_number", "driver"]
ID_COLS = ["year", "round_number", "number", "driver", "constructor", "engine"]

# TFT feature layout
STATIC_CAT_COLS = []
STATIC_NUM_COLS = [
"ctor_elo_mu", "eng_elo_mu", "driver_elo_mu",
"corners", "slow_corners", "medium_corners", "high_speed_corners",
]

HIST_NUM_COLS = [
TARGET_COL,
"position", "gap_from_leader", "gap_from_ahead", "gap_from_behind",
"tyre_age", "rolling_lap_time_3", "laps_since_flag_change",
"air_temp", "track_temp", "wind_speed", "wind_direction",
]
HIST_CAT_COLS = ["compound", "flag", "pit_lap"]

# Future: known in advance for laps being predicted
FUTURE_CAT_COLS = []
FUTURE_NUM_COLS = ["number"]

# Flat feature vector (linear regression / XGBoost)
FLAT_FEATURE_COLS = (
    STATIC_NUM_COLS
    + HIST_NUM_COLS[1:]           # exclude TARGET_COL itself
    + HIST_CAT_COLS
    + FUTURE_NUM_COLS
    + ["position", "pit_lap", "cumulative_time_s",
       "gap_from_leader", "gap_from_ahead", "gap_from_behind"]
)
FLAT_FEATURE_COLS = list(dict.fromkeys(FLAT_FEATURE_COLS))

# Winsorize bounds (applied to numeric cols before scaling)
CLIP_LO = 0.01
CLIP_HI = 0.99
