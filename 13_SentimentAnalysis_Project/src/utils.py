import os
import sys
import logging
import torch
import platform
from pathlib import Path

def setup_logger(name: str = "SentimentAnalysis"):
    """
    Sets up a logger that outputs to console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def get_device():
    """
    Automatically detects the best available device.
    Priority: CUDA (Colab) > MPS (Mac M1/M2) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def is_colab():
    """
    Checks if the code is running in Google Colab.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_data_dir():
    """
    Returns the data directory path based on the environment.
    Local: ./data
    Colab: /content/data (assumed uploaded or mounted)
    """
    if is_colab():
        return Path("/content/data")
    else:
        # Assuming script is run from project root
        return Path("./data")

def check_system_info(logger):
    """
    Logs system information for debugging context.
    """
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {sys.version.split()[0]}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    device = get_device()
    logger.info(f"Selected Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == 'mps':
        logger.info("Running on Apple Silicon (MPS)")
    
    return device
