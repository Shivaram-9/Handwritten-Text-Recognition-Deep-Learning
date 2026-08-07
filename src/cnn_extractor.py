import sys
import os
import logging
import tensorflow as tf
from tensorflow.keras import layers, Model

# Add project root to sys.path to allow importing from config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Ensure directories exist
os.makedirs(os.path.join(Config.DATA_DIR, 'processed'), exist_ok=True)

# Configure logging
log_file = os.path.join(Config.DATA_DIR, 'processed', 'model_build.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CNNFeatureExtractor:
    """
    A Convolutional Neural Network (CNN) Feature Extractor for Handwritten Text Recognition.
    Built using the TensorFlow/Keras Functional API.
    
    This module is responsible for taking a 2D image array and extracting a 1D sequence 
    of dense feature vectors that will subsequently be passed to a Recurrent Neural Network (RNN).
    """
    def __init__(self, input_shape=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1)):
        """
        Initializes the CNN Feature Extractor.
        
        :param input_shape: Tuple defining (height, width, channels).
                            Default is (32, 128, 1) as defined in config.py.
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Constructs the CNN architecture.
        Returns a tf.keras.Model that outputs extracted feature sequences.
        """
        logger.info(f"Building CNN Feature Extractor with input shape: {self.input_shape}")
        
        inputs = layers.Input(shape=self.input_shape, name="image_input")
        
        # Block 1
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name="conv_1")(inputs)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)  # H: 16, W: 64
        
        # Block 2
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name="conv_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)  # H: 8, W: 32
        
        # Block 3
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name="conv_3")(x)
        x = layers.BatchNormalization(name="bn_3")(x)
        x = layers.MaxPooling2D(pool_size=(2, 1), name="pool_3")(x)  # H: 4, W: 32
        
        # Block 4
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name="conv_4")(x)
        x = layers.BatchNormalization(name="bn_4")(x)
        x = layers.MaxPooling2D(pool_size=(2, 1), name="pool_4")(x)  # H: 2, W: 32
        
        # Block 5
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name="conv_5")(x)
        x = layers.BatchNormalization(name="bn_5")(x)
        x = layers.MaxPooling2D(pool_size=(2, 1), name="pool_5")(x)  # H: 1, W: 32
        
        # Regularization
        x = layers.Dropout(0.2, name="dropout")(x)
        
        # The output shape here is (Batch, 1, 32, 256)
        # We need to squeeze the height dimension out to make it a sequence of 32 time steps, 
        # each with 256 features, forming a shape of (Batch, 32, 256)
        
        # Using lambda layer to squeeze the axis (cleaner than reshape when dynamic sizing is needed)
        x = layers.Lambda(lambda t: tf.squeeze(t, axis=1), name="squeeze_height")(x)
        
        # Final dense feature mapping to condense features before RNN processing
        features = layers.Dense(64, activation='relu', name="dense_features")(x)
        
        model = Model(inputs=inputs, outputs=features, name="CNN_Feature_Extractor")
        
        logger.info("CNN Feature Extractor built successfully.")
        
        # Log model summary
        # Passing logger.info directly into print_fn captures the summary in our logs
        model.summary(print_fn=logger.info)
        
        return model
        
    def get_model(self):
        """
        Returns the constructed Keras model.
        """
        return self.model

if __name__ == "__main__":
    # Test initialization and logging
    cnn_extractor = CNNFeatureExtractor()
    model = cnn_extractor.get_model()
