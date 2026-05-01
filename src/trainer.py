import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from typing import Dict, Any

from src.utils import get_logger, timer

logger = get_logger(__name__)

class ModelTrainer:
    """
    Handles the initialization, training, and saving of machine learning models.
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @timer
    def train_baseline(self, X_train, y_train, alpha: float = 100.0) -> Ridge:
        """
        Trains a Ridge Regression baseline model.
        """
        logger.info(f"Training Baseline (Ridge Regression, alpha={alpha})...")
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Baseline training complete.")
        return model

    @timer
    def train_xgboost(self, X_train, y_train, params: Dict[str, Any]) -> XGBRegressor:
        """
        Trains the champion XGBoost model using the provided hyperparameter dictionary.
        """
        logger.info(f"Training Advanced Model (XGBoost) with params: {params}...")
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("XGBoost training complete.")
        return model

    def save_model(self, model: Any, filename: str) -> None:
        """
        Serializes and saves a trained model to the models directory.
        """
        file_path = self.models_dir / filename
        logger.info(f"Saving model to {file_path}...")
        joblib.dump(model, file_path)
        logger.info(f"Model saved successfully.")
