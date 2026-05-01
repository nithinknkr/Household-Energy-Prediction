import logging
import time
from functools import wraps
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance with a standard formatter.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def timer(func):
    """
    Decorator to measure the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__name__)
        logger.info(f"Starting execution of '{func.__name__}'...")
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Finished '{func.__name__}' in {elapsed_time:.2f} seconds.")
        return result
    return wrapper
