import os
import sys
import logging
from config import Config
from src.dataset_loader import DatasetLoader
from src.trainer import ModelTrainer
from src.evaluator import HTREvaluator

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=========================================")
    logger.info("   Starting End-to-End Training Pipeline ")
    logger.info("=========================================")

    # 1. Initialize Dataset Loader
    logger.info("Initializing Dataset Loader...")
    loader = DatasetLoader()
    
    # 2. Export Vocabulary
    vocab_path = os.path.join(Config.MODEL_DIR, 'char_map.json')
    loader.export_vocabulary(vocab_path)
    
    # 3. Load Datasets
    train_split_path = os.path.join(Config.DATA_DIR, 'processed', 'splits', 'train_split.json')
    val_split_path = os.path.join(Config.DATA_DIR, 'processed', 'splits', 'val_split.json')
    test_split_path = os.path.join(Config.DATA_DIR, 'processed', 'splits', 'test_split.json')
    
    train_dataset = loader.load_split(train_split_path, is_training=True)
    val_dataset = loader.load_split(val_split_path, is_training=False)
    
    if not train_dataset or not val_dataset:
        logger.error("Failed to load training/validation datasets. Check JSON split paths.")
        return

    # 4. Initialize Trainer
    logger.info("Initializing Model Trainer...")
    trainer = ModelTrainer(resume_training=True)
    
    # 5. Train Model
    logger.info("Starting Training...")
    # Running 1 epoch for dry-run if configured, otherwise Config.EPOCHS
    success = trainer.train(train_dataset, val_dataset, epochs=Config.EPOCHS)
    
    if not success:
        logger.error("Training aborted due to a critical error. Skipping model export and evaluation.")
        return
        
    # 6. Save Inference Model
    logger.info("Saving Standalone Inference Model...")
    trainer.save_inference_model()
    
    # 7. Evaluate on Test Set
    logger.info("=========================================")
    logger.info("   Running Automated Test Set Evaluation ")
    logger.info("=========================================")
    evaluator = HTREvaluator()
    # Ensure evaluator has the correct paths
    evaluator.evaluate()
    
    logger.info("=========================================")
    logger.info("       Training Pipeline Complete        ")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
