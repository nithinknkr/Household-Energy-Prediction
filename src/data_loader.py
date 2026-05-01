import pandas as pd
from pathlib import Path
from src.utils import get_logger, timer

logger = get_logger(__name__)

class DataLoader:
    """
    Handles the ingestion of raw data from the disk.
    """
    
    def __init__(self, file_path: Path):
        self.file_path = file_path

    @timer
    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads the semicolon-delimited text file into a pandas DataFrame.
        Treats '?' as missing values (NaN) correctly.
        """
        if not self.file_path.exists():
            logger.error(f"Data file not found at {self.file_path}")
            raise FileNotFoundError(f"Missing dataset at {self.file_path}")
            
        logger.info(f"Loading raw data from {self.file_path}...")
        
        # low_memory=False prevents mixed-type inference warnings on large files
        df = pd.read_csv(self.file_path, sep=';', na_values='?', low_memory=False)
        
        logger.info(f"Successfully loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
        return df
