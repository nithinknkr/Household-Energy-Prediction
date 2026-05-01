import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """
    Evaluates trained models and generates performance visualizations.
    """
    
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.figures_dir = self.reports_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting style
        sns.set_palette('husl')
        plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})

    def evaluate(self, model: Any, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
        """
        Generates predictions and prints MAE, RMSE, and R2 scores.
        """
        logger.info(f"Evaluating {model_name} on Test Set...")
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        logger.info(f"--- {model_name} Results ---")
        logger.info(f"RMSE : {rmse:.4f} kW")
        logger.info(f"MAE  : {mae:.4f} kW")
        logger.info(f"R²   : {r2:.4f}")
        
        return pd.Series(preds, index=y_test.index, name='Predicted')

    def plot_actual_vs_predicted(self, y_test: pd.Series, preds: pd.Series, window_size: int = 336):
        """
        Plots a time-series slice of Actual vs Predicted values.
        Default window is 336 hours (2 weeks).
        """
        logger.info("Generating Actual vs Predicted TS Plot...")
        slice_idx = min(window_size, len(y_test))
        y_slice = y_test.iloc[:slice_idx]
        p_slice = preds.iloc[:slice_idx]
        
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(y_slice.index, y_slice, label='Actual (kW)', color='black', alpha=0.7, linewidth=1.5)
        ax.plot(p_slice.index, p_slice, label='Predicted (kW)', color='red', alpha=0.8, linewidth=1)
        
        ax.set_title('Household Energy Prediction (2-Week Slice)', fontsize=14, pad=15)
        ax.set_ylabel('Global Active Power (kW)')
        ax.legend()
        plt.tight_layout()
        
        save_path = self.figures_dir / "actual_vs_predicted_ts.png"
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved figure to {save_path}")

    def plot_residuals(self, y_test: pd.Series, preds: pd.Series):
        """
        Plots the residual distribution and scatter plot.
        """
        logger.info("Generating Residual Analysis Plot...")
        residuals = y_test - preds
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        sns.histplot(residuals, bins=50, kde=True, ax=ax1, color='purple')
        ax1.set_title('Residual Distribution', fontsize=12)
        ax1.set_xlabel('Error (Actual - Predicted) kW')
        ax1.axvline(x=0, color='red', linestyle='--')
        
        # Scatter
        sns.scatterplot(x=preds, y=residuals, alpha=0.2, ax=ax2, color='teal')
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title('Residuals vs Predicted Values', fontsize=12)
        ax2.set_xlabel('Predicted Global Active Power (kW)')
        ax2.set_ylabel('Residual (kW)')
        
        plt.tight_layout()
        save_path = self.figures_dir / "residual_analysis.png"
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved figure to {save_path}")

    def plot_feature_importance(self, model: Any, features: list):
        """
        Extracts and plots feature importance if the model supports it.
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Provided model does not support feature_importances_. Skipping plot.")
            return
            
        logger.info("Generating Feature Importance Plot...")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance, x='Importance', y='Feature', palette='magma', ax=ax)
        ax.set_title('XGBoost Feature Importance', fontsize=14)
        plt.tight_layout()
        
        save_path = self.figures_dir / "feature_importance.png"
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved figure to {save_path}")
