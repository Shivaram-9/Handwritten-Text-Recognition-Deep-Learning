import os
import sys
import json
import logging
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Highly optimized tf.data.Dataset pipeline for Handwritten Text Recognition.
    Handles loading images, encoding labels, batching, and prefetching.
    """
    def __init__(self, batch_size=Config.BATCH_SIZE, max_text_length=Config.MAX_TEXT_LENGTH):
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.processed_dir = os.path.join(Config.DATA_DIR, 'processed', 'IAM', 'words')
        
        # Vocabulary
        self.chars = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.vocab_size = len(self.chars)
        
        # Create a StringLookup layer for encoding labels
        # Mask token is '', OOV token is '[UNK]'. We use standard indices.
        self.char_to_num = tf.keras.layers.StringLookup(
            vocabulary=list(self.chars), mask_token=None, oov_token="[UNK]"
        )

    def _encode_sample(self, img_path, label):
        """
        Loads the image and encodes the label.
        """
        # Load and process image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32) # Normalizes to [0, 1]
        
        # The image from preprocessor is saved as H, W. We need to ensure it's H, W, 1
        # config.py has IMAGE_HEIGHT=32, IMAGE_WIDTH=128
        img = tf.image.resize(img, [Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH])
        
        # Split label into characters
        chars = tf.strings.unicode_split(label, input_encoding="UTF-8")
        
        # Map characters to numeric representation
        encoded_label = self.char_to_num(chars)
        
        # Pad label to max_text_length with -1 (CTC blank/padding)
        pad_size = self.max_text_length - tf.shape(encoded_label)[0]
        # tf.pad expects paddings as a tensor of shape [n, 2]
        encoded_label = tf.pad(encoded_label, paddings=[[0, pad_size]], constant_values=-1)
        
        # The Keras model expects a dictionary mapping to input layer names, and a dummy target since 
        # the loss is calculated inside the custom CTCLayer
        inputs = {
            "image_input": img,
            "label_input": tf.cast(encoded_label, tf.float32)
        }
        
        # We yield the same encoded_label as a dummy target, though Keras won't use it for loss
        return inputs, tf.cast(encoded_label, tf.float32)

    def load_split(self, split_json_path, is_training=False):
        """
        Creates a tf.data.Dataset from a split JSON file.
        """
        if not os.path.exists(split_json_path):
            logger.error(f"Split file not found: {split_json_path}")
            return None
            
        with open(split_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        img_paths = []
        labels = []
        
        for item in data:
            img_full_path = os.path.join(self.processed_dir, item['image'])
            # We must verify physical existence because we might be running a dry-run with a subset
            if os.path.exists(img_full_path):
                img_paths.append(img_full_path)
                labels.append(item['label'])
            
        if not img_paths:
            logger.warning(f"No valid images found for {split_json_path}")
            return None
            
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Use AUTOTUNE to dynamically parallelize the mapping
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = dataset.map(self._encode_sample, num_parallel_calls=AUTOTUNE)
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        
        return dataset

    def export_vocabulary(self, output_path):
        """
        Exports the character map to JSON for inference.
        """
        vocab = self.char_to_num.get_vocabulary()
        # Create a char_map exactly matching StringLookup output (including [UNK] at index 0)
        char_map = {i: c for i, c in enumerate(vocab)}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(char_map, f, indent=4)
        logger.info(f"Vocabulary exported to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DatasetLoader()
    train_split_path = os.path.join(Config.DATA_DIR, 'processed', 'splits', 'train_split.json')
    ds = loader.load_split(train_split_path)
    if ds:
        print("Dataset pipeline verified.")
