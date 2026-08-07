import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Add project root to sys.path to allow importing from config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Ensure the processed directory exists for the log file
os.makedirs(os.path.join(Config.DATA_DIR, 'processed'), exist_ok=True)

# Configure Logging for Preprocessing
log_file = os.path.join(Config.DATA_DIR, 'processed', 'preprocessing.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, target_size=(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)):
        """
        Initialize the ImagePreprocessor.
        :param target_size: Tuple of (width, height) for the model input.
        """
        self.target_size = target_size
        self.raw_dir = os.path.join(Config.DATA_DIR, 'raw', 'IAM', 'words')
        self.processed_dir = os.path.join(Config.DATA_DIR, 'processed', 'IAM', 'words')
        
        # Ensure processed IAM words directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def preprocess_image(self, image_path):
        """
        Applies the complete preprocessing pipeline to a single image.
        Returns the processed image for saving and boolean indicating success.
        """
        try:
            # 1. Load Image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Image could not be loaded or is corrupted.")

            # 2. Convert to Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 3. Remove Noise using Gaussian Blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 4. Adaptive Thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # 5. Resize to Model Input Size (with aspect ratio preservation through padding)
            h, w = thresh.shape
            target_w, target_h = self.target_size
            
            aspect_ratio = w / h
            target_aspect_ratio = target_w / target_h
            
            if aspect_ratio > target_aspect_ratio:
                # Image is wider than target aspect ratio
                new_w = target_w
                new_h = int(new_w / aspect_ratio)
            else:
                # Image is taller than target aspect ratio
                new_h = target_h
                new_w = int(new_h * aspect_ratio)
                
            # Prevent zero size errors during resizing
            if new_h == 0 or new_w == 0:
                raise ValueError("Invalid dimension after aspect ratio calculation.")
                
            resized = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad the rest with zeros (black, since we used THRESH_BINARY_INV)
            final_img = np.zeros((target_h, target_w), dtype=np.uint8)
            
            # Center the image
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            final_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized

            # 6. Normalize pixel values
            # Pixel values are strictly mapped between 0.0 and 1.0 (Required for Deep Learning)
            # We return the uint8 image for saving to disk, but during DataLoader execution,
            # this variable will be used for network input.
            normalized_img = final_img / 255.0 
            
            return final_img, True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None, False

    def process_dataset(self):
        """
        Iterates over the raw dataset, processes each image, and saves it.
        """
        logger.info(f"Starting Preprocessing Pipeline.")
        logger.info(f"Source: {self.raw_dir}")
        logger.info(f"Destination: {self.processed_dir}")
        logger.info(f"Target Size: {self.target_size}")
        
        if not os.path.exists(self.raw_dir):
            logger.error("Raw words directory does not exist. Please integrate IAM dataset first.")
            return

        # Gather all image files
        image_files = []
        for root, dirs, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
                    
        total_files = len(image_files)
        logger.info(f"Total image files found: {total_files}")
        
        if total_files == 0:
            logger.warning("No images found to process. Pipeline ended.")
            return

        success_count = 0
        corrupt_count = 0
        
        # 8. Skip corrupted files and Display Progress Bar
        for img_path in tqdm(image_files, desc="Preprocessing Images"):
            processed_img, success = self.preprocess_image(img_path)
            
            if success:
                # 9. Store Processed Images Separately
                # Maintain the same directory tree architecture as the raw dataset
                rel_path = os.path.relpath(img_path, self.raw_dir)
                dest_path = os.path.join(self.processed_dir, rel_path)
                
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                cv2.imwrite(dest_path, processed_img)
                success_count += 1
            else:
                corrupt_count += 1

        # 10. Generate Preprocessing Log
        logger.info("=================================")
        logger.info("    PREPROCESSING COMPLETED      ")
        logger.info("=================================")
        logger.info(f"Total Processed Successfully: {success_count}")
        logger.info(f"Total Skipped/Corrupted: {corrupt_count}")
        logger.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    preprocessor.process_dataset()
