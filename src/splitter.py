import pandas as pd
from typing import Tuple, List
from src.utils import get_logger

logger = get_logger(__name__)

class DataSplitter:
    """
    Handles strict temporal splitting of time-series datasets to 
    prevent data leakage.
    """
    
    def __init__(self, target_col: str, feature_cols: List[str], train_ratio: float = 0.8):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.train_ratio = train_ratio

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset sequentially based on the train_ratio.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Performing strict Time-Based split (Train: {self.train_ratio*100:.0f}%, Test: {(1-self.train_ratio)*100:.0f}%)...")
        
        split_idx = int(len(df) * self.train_ratio)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Ensure requested features exist in the dataframe
        missing_features = [f for f in self.feature_cols if f not in df.columns]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
            
        X_train = train_df[self.feature_cols]
        y_train = train_df[self.target_col]
        
        X_test = test_df[self.feature_cols]
        y_test = test_df[self.target_col]
        
        logger.info(f"Split complete. Train records: {len(X_train):,}, Test records: {len(X_test):,}")
        return X_train, X_test, y_train, y_test
