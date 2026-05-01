import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import get_logger, timer

logger = get_logger(__name__)

class FeatureEngineer:
    """
    Transforms the cleaned baseline dataset into a machine learning-ready
    feature set by engineering cyclical time features, rolling statistics, and lags.
    """
    
    def __init__(self, output_path: Path = None):
        self.output_path = output_path

    @timer
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering pipeline.
        """
        logger.info("Starting feature engineering pipeline...")
        
        df = self._resample_to_hourly(df)
        df = self._calculate_unmetered_energy(df)
        df = self._extract_time_features(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        
        # Drop the NaN rows caused by lagging and rolling windows
        initial_length = len(df)
        df.dropna(inplace=True)
        dropped = initial_length - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows due to NaN values introduced by lags/rolling windows.")
        
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving feature-engineered dataset to {self.output_path}...")
            df.to_csv(self.output_path)
            
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def _resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downsamples 1-minute data to 1-hour intervals to reduce noise
        and speed up ML training.
        """
        logger.info("Resampling dataset from 1-minute to Hourly frequency...")
        
        # We take the mean of instantaneous rates (power, voltage), 
        # and the sum of accumulated volume (energy in Wh).
        df_hourly = df.resample('H').agg({
            'Global_active_power': 'mean',
            'Global_reactive_power': 'mean',
            'Voltage': 'mean',
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum'
        })
        return df_hourly

    def _calculate_unmetered_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the baseline energy consumption not captured by the 3 sub-meters.
        """
        logger.info("Calculating physics-based feature: 'unmetered_energy'...")
        
        # Convert Active Power (kW mean) to Total Energy (Wh) for the hour
        df['Total_energy_Wh'] = df['Global_active_power'] * 1000
        
        sub_total = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
        df['unmetered_energy'] = df['Total_energy_Wh'] - sub_total
        
        # Clip negative math artifacts to 0
        df['unmetered_energy'] = df['unmetered_energy'].clip(lower=0)
        
        # Drop intermediate column
        df.drop(columns=['Total_energy_Wh'], inplace=True)
        return df

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal properties and applies Cyclical Encoding (Sine/Cosine).
        """
        logger.info("Applying Cyclical Encoding to temporal features...")
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Sine/Cosine for Hour (0-23)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Sine/Cosine for Month (1-12)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # We keep the raw temporal columns temporarily but they are 
        # usually not passed to the final ML model.
        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates historical lags of the target variable to act as model "memory".
        """
        logger.info("Creating Lag features (T-1h, T-24h, T-48h)...")
        df['lag_1h'] = df['Global_active_power'].shift(1)
        df['lag_24h'] = df['Global_active_power'].shift(24)
        df['lag_48h'] = df['Global_active_power'].shift(48)
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates moving averages to capture momentum and smooth volatility.
        """
        logger.info("Creating Rolling Window statistics (6h, 24h)...")
        df['rolling_mean_6h'] = df['Global_active_power'].rolling(window=6).mean()
        df['rolling_mean_24h'] = df['Global_active_power'].rolling(window=24).mean()
        return df
