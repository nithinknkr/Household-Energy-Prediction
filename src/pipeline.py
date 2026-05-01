import sys
from pathlib import Path

# Ensure the root project directory is in the PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import config
from src.utils import get_logger, timer
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.splitter import DataSplitter
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator

logger = get_logger(__name__)

@timer
def main():
    logger.info("=== Starting Household Energy Prediction Pipeline ===")
    
    # 1. Load Raw Data
    loader = DataLoader(file_path=config.RAW_DATA_PATH)
    raw_df = loader.load_raw_data()
    
    # 2. Preprocess Data
    preprocessor = DataPreprocessor(output_path=config.CLEANED_DATA_PATH)
    clean_df = preprocessor.preprocess(raw_df)
    
    # 3. Engineer Features
    engineer = FeatureEngineer(output_path=config.FEATURED_DATA_PATH)
    featured_df = engineer.engineer_features(clean_df)
    
    # 4. Train-Test Split
    splitter = DataSplitter(
        target_col=config.TARGET,
        feature_cols=config.ENGINEERED_FEATURES,
        train_ratio=config.TRAIN_SPLIT_RATIO
    )
    X_train, X_test, y_train, y_test = splitter.split(featured_df)
    
    # 5. Model Training
    trainer = ModelTrainer(models_dir=config.MODELS_DIR)
    
    # Train Baseline
    baseline_model = trainer.train_baseline(X_train, y_train)
    
    # Train Champion (XGBoost)
    xgb_model = trainer.train_xgboost(X_train, y_train, params=config.XGB_BEST_PARAMS)
    trainer.save_model(xgb_model, filename="best_model.pkl")
    
    # 6. Evaluation & Visualization
    evaluator = ModelEvaluator(reports_dir=config.REPORTS_DIR)
    
    logger.info("Evaluating Baseline...")
    evaluator.evaluate(baseline_model, "Ridge Baseline", X_test, y_test)
    
    logger.info("Evaluating Champion XGBoost...")
    preds = evaluator.evaluate(xgb_model, "XGBoost Champion", X_test, y_test)
    
    # Generate artifacts
    evaluator.plot_actual_vs_predicted(y_test, preds)
    evaluator.plot_residuals(y_test, preds)
    evaluator.plot_feature_importance(xgb_model, features=config.ENGINEERED_FEATURES)
    
    logger.info("=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()
