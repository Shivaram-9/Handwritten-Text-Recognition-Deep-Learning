import os
import sys
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.cnn_extractor import CNNFeatureExtractor
from src.rnn_module import RNNSequenceLearner

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


class CTCLayer(layers.Layer):
    """
    Custom Keras layer to compute the CTC Loss.
    During training, it calculates and adds the loss to the model.
    During inference, it simply passes the predictions through.
    """
    def __init__(self, name="ctc_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Determine the batch length
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        
        # Sequence lengths
        # y_pred shape: (batch_size, time_steps, num_classes)
        # y_true shape: (batch_size, max_text_length)
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        
        # Expand lengths to match the batch dimension
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        # Compute the CTC loss
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        
        # Return predictions for inference
        return y_pred


class HTRModel:
    """
    Complete Handwritten Text Recognition (HTR) Model integrating CNN, BiLSTM, and CTC.
    """
    def __init__(self, vocab_size=Config.VOCAB_SIZE, max_text_length=Config.MAX_TEXT_LENGTH):
        self.vocab_size = vocab_size
        self.max_text_length = max_text_length
        self.image_shape = (Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1)
        
        # Initialize modules
        self.cnn_module = CNNFeatureExtractor(input_shape=self.image_shape).get_model()
        # CNN output shape will be (Batch, Time_Steps, Features)
        cnn_out_shape = self.cnn_module.output_shape[1:] 
        self.rnn_module = RNNSequenceLearner(input_shape=cnn_out_shape).get_model()
        
        # Build the full combined model
        self.training_model, self.inference_model = self._build_model()
        
    def _build_model(self):
        """
        Integrates CNN, BiLSTM and CTC Loss into a cohesive model.
        Returns both a training model (with CTC loss) and an inference model (predictions only).
        """
        logger.info("Integrating CNN, BiLSTM, and CTC Layer into Final HTR Model.")
        
        # Define Inputs
        image_input = layers.Input(shape=self.image_shape, name="image_input")
        labels = layers.Input(shape=(self.max_text_length,), name="label_input", dtype="float32")
        
        # Forward pass through CNN
        cnn_features = self.cnn_module(image_input)
        
        # Forward pass through BiLSTM
        rnn_features = self.rnn_module(cnn_features)
        
        # Output classification layer (maps to vocabulary size + 1 for CTC blank)
        # Activation is softmax to get probability distribution over characters
        preds = layers.Dense(self.vocab_size + 1, activation="softmax", name="dense_output")(rnn_features)
        
        # Add CTC layer for calculating loss during training
        ctc_output = CTCLayer(name="ctc_loss")(labels, preds)
        
        # Construct the Training Model
        training_model = Model(
            inputs=[image_input, labels], 
            outputs=ctc_output, 
            name="HTR_Training_Model"
        )
        
        # Construct the Inference Model (used for decoding during testing/production)
        inference_model = Model(
            inputs=image_input, 
            outputs=preds, 
            name="HTR_Inference_Model"
        )
        
        logger.info("Successfully constructed HTR Models.")
        
        logger.info("=== HTR Training Model Summary ===")
        training_model.summary(print_fn=logger.info)
        
        return training_model, inference_model

    def get_training_model(self):
        """Returns the model compiled with CTC Loss for training."""
        return self.training_model
        
    def get_inference_model(self):
        """Returns the model for inference."""
        return self.inference_model
        
    @staticmethod
    def decode_predictions(predictions, char_map):
        """
        CTC Decoder to convert raw model predictions back into text strings.
        Uses greedy search for simplicity.
        
        :param predictions: Raw softmax predictions from the inference model.
        :param char_map: Dictionary mapping integer indexes back to characters.
        :return: Tuple of (results_list, confidences_list)
        """
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
        
        # Use Keras backend CTC decode
        decoded, log_probs = K.ctc_decode(predictions, input_length=input_len, greedy=True)
        
        # Convert index sequence back to string
        results = []
        confidences = []
        
        for i, seq in enumerate(decoded[0]):
            seq_numpy = seq.numpy()
            res = ""
            for x in seq_numpy:
                # -1 means the CTC blank token
                if x != -1 and x in char_map:
                    res += char_map[x]
            results.append(res)
            
            # Confidence is derived from negative log probabilities
            # Taking exponential converts negative log prob back to a probability [0, 1]
            conf = np.exp(-log_probs[i][0].numpy())
            confidences.append(float(conf))
            
        return results, confidences

if __name__ == "__main__":
    logger.info("Initializing HTR Integration Test...")
    htr = HTRModel()
    train_model = htr.get_training_model()
    inference_model = htr.get_inference_model()
    logger.info("Integration Test Passed.")
