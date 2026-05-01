import pandas as pd
from pathlib import Path
from src.utils import get_logger, timer

logger = get_logger(__name__)

class DataPreprocessor:
    """
    Handles data cleaning, datetime parsing, and missing value imputation.
    """
    
    def __init__(self, output_path: Path = None):
        self.output_path = output_path

    @timer
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the preprocessing pipeline on the raw DataFrame.
        """
        logger.info("Starting preprocessing pipeline...")
        df = self._parse_datetime(df)
        df = self._handle_missing_values(df)
        df = self._drop_collinear_features(df)
        
        if self.output_path:
            # Ensure the directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving cleaned dataset to {self.output_path}...")
            df.to_csv(self.output_path)
            
        logger.info("Preprocessing complete.")
        return df

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges 'Date' and 'Time' object columns into a proper DatetimeIndex.
        """
        logger.info("Parsing Date and Time into DatetimeIndex...")
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['Date', 'Time'], inplace=True)
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values. Because this is time-series data with block-level
        missingness (power outages), we use forward fill.
        """
        missing_count = df.isnull().any(axis=1).sum()
        logger.info(f"Found {missing_count:,} rows with missing values. Applying forward-fill...")
        df.ffill(inplace=True)
        return df

    def _drop_collinear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops 'Global_intensity' due to its near-perfect mathematical 
        correlation with 'Global_active_power'.
        """
        if 'Global_intensity' in df.columns:
            logger.info("Dropping 'Global_intensity' to prevent multicollinearity...")
            df.drop(columns=['Global_intensity'], inplace=True)
        return df
