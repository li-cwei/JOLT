import logging
import os
from datetime import datetime


def setup_logger(dataset_name, log_dir="logs"):
    """Create a dataset-specific logger."""
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(log_dir, f"{dataset_name}_{date_str}.log")
    logger = logging.getLogger("Logger")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] - %(message)s', datefmt="%H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
