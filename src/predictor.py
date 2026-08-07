import os
import sys
import logging
import numpy as np

# Add project root to sys.path to allow importing from config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.model import HTRModel
from src.preprocessor import ImagePreprocessor

# Ensure log directory exists
os.makedirs(os.path.join(Config.BASE_DIR, 'logs'), exist_ok=True)

# Configure logging
log_file = os.path.join(Config.BASE_DIR, 'logs', 'prediction.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HTRPredictor:
    """
    Production-ready Inference Pipeline for Handwritten Text Recognition.
    Handles image loading, preprocessing, inference, and CTC decoding securely.
    """
    def __init__(self, model_path=None, char_map=None):
        """
        Initializes the inference pipeline.
        
        :param model_path: Path to the best trained weights. 
                           Defaults to 'models/best_htr_model.h5'.
        :param char_map: Dictionary mapping integer indices to characters.
        """
        self.model_path = model_path or os.path.join(Config.MODEL_DIR, 'best_htr_model.h5')
        self.preprocessor = ImagePreprocessor()
        
        # Initialize default character map if none provided
        if char_map is None:
            # Assuming standard alphanumeric + punctuation dataset for IAM
            chars = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            self.char_map = {i: c for i, c in enumerate(chars)}
            # Overwrite Config.VOCAB_SIZE dynamically to avoid shape mismatches
            Config.VOCAB_SIZE = len(chars)
        else:
            self.char_map = char_map
            
        self.inference_model = self._load_model()
        
    def _load_model(self):
        """Loads the HTR inference model and its pre-trained weights."""
        logger.info("Initializing HTR Inference Architecture...")
        
        # Instantiate the full model and extract the inference portion
        htr = HTRModel(vocab_size=Config.VOCAB_SIZE)
        training_model = htr.get_training_model()
        inference_model = htr.get_inference_model()
        
        if not os.path.exists(self.model_path):
            logger.warning(f"Trained model weights not found at {self.model_path}.")
            logger.warning("Predictor will return random predictions initialized by He-normal.")
            return inference_model
            
        logger.info(f"Loading trained weights from {self.model_path}...")
        try:
            # Load weights into the full training model (which shares references with inference_model)
            # We use by_name=True and skip_mismatch=True to be robust against CTC layer quirks
            training_model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise
            
        return inference_model
        
    def predict_image(self, image_path):
        """
        Runs the complete prediction pipeline on a single image file.
        
        :param image_path: Path to the image file (JPG, PNG, JPEG).
        :return: Tuple of (predicted_text, confidence_score) or (None, 0.0) on failure.
        """
        logger.info(f"Processing inference request for: {image_path}")
        
        # Validate File Type and Existence
        valid_extensions = ('.jpg', '.jpeg', '.png')
        if not str(image_path).lower().endswith(valid_extensions):
            logger.error(f"Unsupported file format. Please provide one of {valid_extensions}")
            return None, 0.0
            
        if not os.path.exists(image_path):
            logger.error(f"Image not found at path: {image_path}")
            return None, 0.0
            
        # 1. Preprocess using existing modular preprocessor
        processed_img, success = self.preprocessor.preprocess_image(image_path)
        
        if not success or processed_img is None:
            logger.error(f"Failed to preprocess image: {image_path}. Image may be corrupted.")
            return None, 0.0
            
        # 2. Format Data for Neural Network (Batch, H, W, Channels)
        # Preprocessor returned 0-255 uint8 to allow saving. We must normalize to 0.0-1.0 for the network.
        normalized_img = processed_img / 255.0
        
        # Add channel dimension if missing, and then batch dimension
        img_batch = np.expand_dims(normalized_img, axis=-1)
        img_batch = np.expand_dims(img_batch, axis=0)
        
        # 3. Predict Softmax Distributions
        try:
            preds = self.inference_model.predict(img_batch, verbose=0)
            
            # 4. Decode CTC Predictions using modular decoder
            results, confidences = HTRModel.decode_predictions(preds, self.char_map)
            
            predicted_text = results[0]
            confidence_score = confidences[0]
            
            logger.info(f"Prediction: '{predicted_text}' | Confidence: {confidence_score:.4f}")
            return predicted_text, confidence_score
            
        except Exception as e:
            logger.error(f"Inference failed during network execution: {e}")
            return None, 0.0

if __name__ == "__main__":
    # Test script initialization
    logger.info("Starting Predictor Standalone Test...")
    try:
        predictor = HTRPredictor()
        logger.info("Predictor module ready for production inference API integration.")
        # Example Usage: 
        # text, conf = predictor.predict_image("path/to/handwriting.jpg")
    except Exception as e:
        logger.error(f"Failed to initialize Predictor: {e}")
