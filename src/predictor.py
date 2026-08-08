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

import functools
import tensorflow as tf

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
        self.model_path = model_path or os.path.join(Config.MODEL_DIR, 'best_htr_model.weights.h5')
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
            
        self.inference_model_path = os.path.join(Config.MODEL_DIR, 'inference_model.h5')
        self.inference_model = self._load_model()
        self._predict_fn = self._build_predict_fn()
        
    def _load_model(self):
        """Loads the HTR inference model."""
        logger.info("Initializing HTR Inference Architecture...")
        
        # 1. Try loading the optimized standalone inference model
        if os.path.exists(self.inference_model_path):
            logger.info(f"Loading standalone inference model from {self.inference_model_path}...")
            try:
                inference_model = tf.keras.models.load_model(self.inference_model_path, compile=False)
                logger.info("Inference model loaded successfully.")
                return inference_model
            except Exception as e:
                logger.warning(f"Failed to load standalone model: {e}. Falling back to weight loading.")
        
        # 2. Fallback: Instantiate the full model and load weights
        htr = HTRModel(vocab_size=Config.VOCAB_SIZE)
        training_model = htr.get_training_model()
        inference_model = htr.get_inference_model()
        
        if not os.path.exists(self.model_path):
            logger.warning(f"Trained model weights not found at {self.model_path}.")
            logger.warning("Predictor will return random predictions initialized by He-normal.")
            return inference_model
            
        logger.info(f"Loading trained weights from {self.model_path}...")
        try:
            training_model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise
            
        return inference_model
        
    def _build_predict_fn(self):
        """Builds an optimized prediction function using XLA if enabled."""
        if getattr(Config, 'ENABLE_XLA', False):
            logger.info("Enabling XLA Compilation for faster inference...")
            return tf.function(self.inference_model, jit_compile=True)
        return self.inference_model

    def predict_image(self, image_path):
        """
        Runs the complete prediction pipeline on a single image file.
        Wraps the internal logic to allow for optional caching.
        """
        if getattr(Config, 'ENABLE_CACHING', False):
            return self._predict_image_cached(image_path)
        return self._predict_image_uncached(image_path)

    @functools.lru_cache(maxsize=128)
    def _predict_image_cached(self, image_path):
        """Cached version of the prediction pipeline (safe for repeated evaluations)."""
        logger.debug(f"Cache miss for {image_path}. Running inference.")
        return self._predict_image_uncached(image_path)

    def _predict_image_uncached(self, image_path):
        """
        Core inference logic on a single image file.
        Attempts advanced document segmentation. Falls back to single image processing.
        
        :param image_path: Path to the image file (JPG, PNG, JPEG).
        :return: Tuple of (predicted_text, confidence_score, timings, pipeline_images) 
                 or (None, 0.0, None, None) on failure.
        """
        import time
        logger.info(f"Processing inference request for: {image_path}")
        
        timings = {}
        pipeline_images = None
        
        valid_extensions = ('.jpg', '.jpeg', '.png')
        if not str(image_path).lower().endswith(valid_extensions):
            logger.error(f"Unsupported file format. Please provide one of {valid_extensions}")
            return None, 0.0, None, None
            
        if not os.path.exists(image_path):
            logger.error(f"Image not found at path: {image_path}")
            return None, 0.0, None, None
            
        # 1. Attempt Advanced Document Segmentation
        t0 = time.perf_counter()
        is_doc, segmented_lines, b64_orig, b64_prep, b64_lines = self.preprocessor.segment_document(image_path)
        
        if is_doc and segmented_lines and len(segmented_lines) > 0:
            t1 = time.perf_counter()
            timings['preprocessing_ms'] = round((t1 - t0) * 1000, 2)
            
            pipeline_images = {
                "original": b64_orig,
                "preprocessed": b64_prep,
                "lines": b64_lines
            }
            
            merged_text = []
            total_conf = 0.0
            inf_time = 0.0
            dec_time = 0.0
            
            for line_tensor in segmented_lines:
                img_batch = np.expand_dims(line_tensor, axis=-1)
                img_batch = np.expand_dims(img_batch, axis=0)
                input_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
                
                try:
                    t_inf_start = time.perf_counter()
                    preds = self._predict_fn(input_tensor)
                    t_inf_end = time.perf_counter()
                    inf_time += (t_inf_end - t_inf_start)
                    
                    t_dec_start = time.perf_counter()
                    results, confidences = HTRModel.decode_predictions(preds, self.char_map)
                    t_dec_end = time.perf_counter()
                    dec_time += (t_dec_end - t_dec_start)
                    
                    merged_text.append(results[0])
                    total_conf += confidences[0]
                except Exception as e:
                    logger.error(f"Inference failed on segmented line: {e}")
                    
            timings['inference_ms'] = round(inf_time * 1000, 2)
            timings['decoding_ms'] = round(dec_time * 1000, 2)
            
            final_text = " ".join(merged_text)
            avg_conf = total_conf / len(segmented_lines) if len(segmented_lines) > 0 else 0.0
            
            logger.info(f"Multi-line Prediction: '{final_text}' | Confidence: {avg_conf:.4f} | Latency: {timings}")
            return final_text, avg_conf, timings, pipeline_images
            
        else:
            # 2. Fallback to Single Image Preprocessing
            logger.info("Segmentation failed or not applicable. Falling back to single image pipeline.")
            processed_img, success = self.preprocessor.preprocess_image(image_path)
            t1 = time.perf_counter()
            timings['preprocessing_ms'] = round((t1 - t0) * 1000, 2)
            
            if not success or processed_img is None:
                logger.error(f"Failed to preprocess image: {image_path}. Image may be corrupted.")
                return None, 0.0, None, None
                
            img_batch = np.expand_dims(processed_img, axis=-1)
            img_batch = np.expand_dims(img_batch, axis=0)
            input_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
            
            try:
                t2 = time.perf_counter()
                preds = self._predict_fn(input_tensor)
                t3 = time.perf_counter()
                timings['inference_ms'] = round((t3 - t2) * 1000, 2)
                
                t4 = time.perf_counter()
                results, confidences = HTRModel.decode_predictions(preds, self.char_map)
                t5 = time.perf_counter()
                timings['decoding_ms'] = round((t5 - t4) * 1000, 2)
                
                predicted_text = results[0]
                confidence_score = confidences[0]
                
                logger.info(f"Fallback Prediction: '{predicted_text}' | Confidence: {confidence_score:.4f} | Latency: {timings}")
                return predicted_text, confidence_score, timings, None
                
            except Exception as e:
                logger.error(f"Inference failed during fallback network execution: {e}")
                return None, 0.0, None, None

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
