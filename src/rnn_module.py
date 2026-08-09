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

# Configure logging to append to the same model build log
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

class RNNSequenceLearner:
    """
    A Recurrent Neural Network (RNN) module utilizing Bidirectional LSTMs.
    Built using the TensorFlow/Keras Functional API.
    
    This module takes the sequence of features extracted by the CNN and 
    learns the contextual dependencies between them. It is designed to be completely 
    modular and compatible with the preceding CNN output.
    """
    def __init__(self, input_shape=(32, 64), lstm_units=256):
        """
        Initializes the RNN module.
        
        :param input_shape: Tuple defining (time_steps, features). 
                            By default, the CNN extractor outputs sequences of length 32 
                            with 64 dense features per timestep.
        :param lstm_units: Number of hidden units in each LSTM layer.
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Constructs the RNN architecture.
        Returns a tf.keras.Model that outputs context-aware sequences.
        """
        logger.info(f"Building Bidirectional LSTM module with input shape: {self.input_shape}")
        
        inputs = layers.Input(shape=self.input_shape, name="rnn_input")
        
        # First Bidirectional LSTM Layer
        # return_sequences=True is critical for CTC Loss to map each time step to characters
        lstm_1 = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.3),
            name="bilstm_1"
        )(inputs)
        
        # Second Bidirectional LSTM Layer
        # Stacking a second layer allows the model to learn more complex sequence semantics
        lstm_2 = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.3),
            name="bilstm_2"
        )(lstm_1)
        
        # Output shape here will be (Batch, Time_Steps, lstm_units * 2) 
        # (multiplying by 2 because it's Bidirectional)
        
        model = Model(inputs=inputs, outputs=lstm_2, name="Bidirectional_LSTM_Learner")
        
        logger.info("Bidirectional LSTM module built successfully.")
        
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
    rnn_module = RNNSequenceLearner()
    model = rnn_module.get_model()
