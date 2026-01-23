from src.utils import setup_logger, get_data_dir
from src.data_loader import prepare_data
from pathlib import Path

logger = setup_logger()

def main():
    logger.info("Starting Data Loading Verification...")
    
    # Force local data dir for verification
    data_dir = Path("./data/쇼핑몰/01. 패션")
    
    try:
        # Sample limit 1000 for quick debugging
        dataset = prepare_data(data_dir, sample_limit=2000, seed=42)
        
        logger.info("Dataset created successfully!")
        logger.info(f"Train samples: {len(dataset['train'])}")
        logger.info(f"Test samples: {len(dataset['test'])}")
        
        # Verify Labels
        logger.info(f"Sample Train Item: {dataset['train'][0]}")
        
    except Exception as e:
        logger.error(f"Verification Failed: {e}")
        raise e

if __name__ == "__main__":
    main()
