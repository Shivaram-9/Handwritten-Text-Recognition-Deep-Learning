import os
import sys
import logging
import traceback
import tensorflow as tf
from tensorflow.keras.callbacks import (
    TensorBoard, 
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau, 
    CSVLogger
)
import matplotlib.pyplot as plt

# Add project root to sys.path to allow importing from config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.model import HTRModel

# Ensure log directories exist
os.makedirs(os.path.join(Config.BASE_DIR, 'logs'), exist_ok=True)

# Configure primary training logger
log_file = os.path.join(Config.BASE_DIR, 'logs', 'training_pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Production-ready Training Pipeline for the HTR Model.
    Handles Callbacks, Mixed Precision, GPU detection, Checkpointing, and Exception Management.
    """
    def __init__(self, resume_training=False):
        self.resume_training = resume_training
        
        # Directory configurations
        self.checkpoint_dir = os.path.join(Config.MODEL_DIR, 'checkpoints')
        self.tensorboard_logs = os.path.join(Config.BASE_DIR, 'logs', 'tensorboard')
        self.csv_log_path = os.path.join(Config.BASE_DIR, 'logs', 'training_history.csv')
        
        # Checkpoint Paths
        self.best_model_path = os.path.join(Config.MODEL_DIR, 'best_htr_model.weights.h5')
        self.latest_model_path = os.path.join(self.checkpoint_dir, 'latest_model.weights.h5')
        self.inference_model_path = os.path.join(Config.MODEL_DIR, 'inference_model.h5')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_logs, exist_ok=True)
        
        self._setup_gpu_and_mixed_precision()
        self.htr_instance = None # Keep reference to extract inference model later
        self.model = self._initialize_model()

    def _setup_gpu_and_mixed_precision(self):
        """Detects GPU and strictly enables mixed precision for memory and speed efficiency if available."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU Detected: {len(gpus)} GPU(s) available.")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Enable Mixed Precision (FP16)
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision (mixed_float16) enabled for faster training.")
            except Exception as e:
                logger.warning(f"Could not fully configure GPU/Mixed Precision. Proceeding with defaults. Error: {e}")
        else:
            logger.info("No GPU detected. Training will run on CPU with standard FP32 precision.")

    def _initialize_model(self):
        """Initializes the HTR model, compiles it, and restores weights if resuming."""
        logger.info("Initializing HTR Training Model...")
        self.htr_instance = HTRModel()
        training_model = self.htr_instance.get_training_model()
        
        # CTC Loss computation is embedded directly inside the custom CTCLayer during forward pass.
        # Keras requires a loss function to compile if targets are passed during fit().
        # We pass a dummy loss (lambda y_true, y_pred: y_pred) because the real loss is added via `self.add_loss` in the layer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        training_model.compile(
            optimizer=optimizer,
            loss={'ctc_loss': lambda y_true, y_pred: y_pred}
        )
        
        # Handle Resuming Training seamlessly
        if self.resume_training and os.path.exists(self.latest_model_path):
            logger.info(f"Resuming training from checkpoint: {self.latest_model_path}")
            try:
                training_model.load_weights(self.latest_model_path)
                logger.info("Weights restored successfully.")
            except Exception as e:
                logger.error(f"Critical error restoring weights: {e}")
                logger.info("Falling back to training from scratch.")
        
        return training_model

    def _get_callbacks(self):
        """Configures and returns an industry-standard list of Keras callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=8, 
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.best_model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True, # Use weights_only=True for Custom CTC layers
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.latest_model_path,
                monitor='val_loss',
                save_best_only=False,
                save_weights_only=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=3, 
                min_lr=1e-6,
                verbose=1
            ),
            CSVLogger(
                filename=self.csv_log_path,
                append=self.resume_training
            )
        ]
        
        try:
            import tensorboard
            callbacks.insert(0, TensorBoard(
                log_dir=self.tensorboard_logs, 
                histogram_freq=1,
                update_freq='epoch'
            ))
            logger.info("TensorBoard callback initialized.")
        except ImportError:
            logger.warning("TensorBoard is not installed. Skipping TensorBoard callback.")
            
        return callbacks
        
    def save_inference_model(self):
        """Extracts and saves the compiled inference model."""
        logger.info("Extracting Inference Model from Training Model...")
        # Get inference model and save it completely
        inference_model = self.htr_instance.get_inference_model()
        inference_model.save(self.inference_model_path)
        logger.info(f"Inference Model successfully saved to {self.inference_model_path}")

    def _plot_training_graphs(self, history):
        """Generates and saves visual training graphs upon completion."""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss (CTC)')
            
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss (CTC)')
            
            plt.title('HTR Model Convergence Over Epochs')
            plt.ylabel('CTC Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            graph_path = os.path.join(Config.BASE_DIR, 'logs', 'loss_convergence_graph.png')
            plt.savefig(graph_path)
            logger.info(f"Training convergence graph saved to {graph_path}")
        except Exception as e:
            logger.error(f"Failed to plot training graphs (Matplotlib Error): {e}")

    def train(self, train_dataset, val_dataset, epochs=Config.EPOCHS):
        """
        Executes the training loop with complete exception handling.
        
        :param train_dataset: tf.data.Dataset or Sequence yielding (inputs, targets)
        :param val_dataset: tf.data.Dataset or Sequence yielding (inputs, targets)
        :param epochs: Number of epochs to train for. Defaults to Config.EPOCHS.
        :return: True if training completes (or is manually interrupted safely), False on critical error.
        """
        logger.info(f"Commencing training pipeline for {epochs} epochs...")
        callbacks = self._get_callbacks()
        
        try:
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks
            )
            logger.info("Training cycle completed successfully.")
            self._plot_training_graphs(history)
            return True
            
        except KeyboardInterrupt:
            logger.info("Training interrupted manually by user (KeyboardInterrupt). Latest epoch checkpoint saved.")
            return True
        except Exception as e:
            logger.error(f"A critical error occurred during the training loop: {e}")
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    logger.info("Verifying Training Pipeline structure...")
    try:
        trainer = ModelTrainer(resume_training=False)
        logger.info("Training pipeline initialized and verified successfully.")
        logger.info("Ready for Datasets to execute `trainer.train(train_data, val_data)`.")
    except Exception as e:
        logger.error(f"Failed to initialize training pipeline: {e}")
        logger.error(traceback.format_exc())
