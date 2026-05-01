import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Specific File Paths
RAW_DATA_PATH = DATA_DIR / "raw" / "household_power_consumption.txt"
CLEANED_DATA_PATH = DATA_DIR / "processed" / "cleaned.csv"
FEATURED_DATA_PATH = DATA_DIR / "processed" / "featured.csv"

# Model Paths
XGB_MODEL_PATH = MODELS_DIR / "xgboost_v1.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Feature Definitions
TARGET = "Global_active_power"

RAW_COLUMNS = [
    "Date", "Time", "Global_active_power", "Global_reactive_power", 
    "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
]

ENGINEERED_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", 
    "lag_1h", "lag_24h", "lag_48h", "rolling_mean_6h", "rolling_mean_24h", "unmetered_energy"
]

# Hyperparameters (Derived from 06_hyperparameter_tuning.ipynb)
XGB_BEST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": 42
}

# Training Settings
TRAIN_SPLIT_RATIO = 0.8
