# main.py
import argparse
import logging
import os
from config.settings import DATA_DIR, MODELS_DIR
from etl.etl_pipeline import XrayETLPipeline
from training.train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='XrayGPT MLOps Pipeline')
    parser.add_argument('--mode', choices=['etl', 'train', 'all'], default='all',
                      help='Mode to run: etl, train, or all')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Batch size for data processing')
    parser.add_argument('--num-batches', type=int, default=10,
                      help='Number of batches to process in ETL')
    parser.add_argument('--model-type', default='bio-gpt',
                      help='Type of model to train (bio-gpt, radbert, etc.)')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                      help='Learning rate for training')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if args.mode in ['etl', 'all']:
        logger.info("Starting ETL pipeline")
        etl_pipeline = XrayETLPipeline()
        etl_pipeline.run_pipeline(
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            val_split=0.2
        )
    
    if args.mode in ['train', 'all']:
        logger.info("Starting training pipeline")
        train_data_dir = os.path.join(DATA_DIR, "raw/")
        val_data_dir = os.path.join(DATA_DIR, "raw/") 
        print(f"Training data directory: {train_data_dir}")
        print(f"Validation data directory: {val_data_dir}")
        train_model(
            train_data_dir=train_data_dir,
            val_data_dir=val_data_dir,
            model_type=args.model_type,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()