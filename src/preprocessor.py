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
        
    def _pad_and_resize(self, thresh):
        """Helper to resize an image to target size with padding."""
        h, w = thresh.shape
        target_w, target_h = self.target_size
        
        aspect_ratio = w / h
        target_aspect_ratio = target_w / target_h
        
        if aspect_ratio > target_aspect_ratio:
            new_w = target_w
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = target_h
            new_w = int(new_h * aspect_ratio)
            
        if new_h == 0 or new_w == 0:
            raise ValueError("Invalid dimension after aspect ratio calculation.")
            
        resized = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        final_img = np.zeros((target_h, target_w), dtype=np.uint8)
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        final_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return final_img

    def segment_document(self, image_path):
        """
        Advanced Document Segmentation Pipeline.
        Returns: (success, segmented_lines, b64_original, b64_preprocessed, b64_lines)
        """
        import base64
        def img_to_b64(img):
            _, buffer = cv2.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')

        try:
            # 1. Load Original
            orig_bgr = cv2.imread(image_path)
            if orig_bgr is None: return False, None, None, None, None
            b64_orig = img_to_b64(orig_bgr)

            # 2. Convert to Grayscale
            gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
            
            # 3. Background Normalization (Scale-invariant polarity)
            if np.median(gray) < 127:
                gray = 255 - gray
                
            h, w = gray.shape
            
            # 4. Dynamic Bounding Box Thresholding
            block = max(11, (h // 50) | 1)
            if block % 2 == 0: block += 1
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 2)
            
            b64_prep = img_to_b64(255 - thresh) # Display the normalized background version to frontend

            # 5. Dynamic Component Grouping
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, w//500), max(2, h//500)))
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: return False, None, None, None, None
            
            char_boxes = []
            img_width = gray.shape[1]
            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                # Ignore noise and full-page borders dynamically
                if bw > max(2, w//1000) and bh > max(5, h//500) and bw < img_width * 0.9:
                    char_boxes.append((x, y, bw, bh))
                    
            if not char_boxes: return False, None, None, None, None
            
            # Sort top-to-bottom
            char_boxes.sort(key=lambda b: b[1])
            
            # Line clustering with dynamic vertical overlap
            lines = []
            current_line = [char_boxes[0]]
            line_top = char_boxes[0][1]
            line_bottom = char_boxes[0][1] + char_boxes[0][3]
            
            for box in char_boxes[1:]:
                by, bh = box[1], box[3]
                box_bottom = by + bh
                
                overlap_top = max(line_top, by)
                overlap_bottom = min(line_bottom, box_bottom)
                overlap = max(0, overlap_bottom - overlap_top)
                
                if overlap > min(bh, line_bottom - line_top) * 0.2:
                    current_line.append(box)
                    line_top = min(line_top, by)
                    line_bottom = max(line_bottom, box_bottom)
                else:
                    lines.append(current_line)
                    current_line = [box]
                    line_top = by
                    line_bottom = box_bottom
            if current_line:
                lines.append(current_line)
                
            # Sort lines top-to-bottom by vertical center
            lines.sort(key=lambda l: np.mean([b[1] + b[3]/2 for b in l]))
              
            segmented_lines = []
            b64_lines = []
            
            for line in lines:
                if not line: continue
                line.sort(key=lambda b: b[0])
                
                widths = [b[2] for b in line]
                median_w = np.median(widths) if widths else 15
                gap_threshold = median_w * 0.75
                
                word_boxes = []
                current_word = list(line[0])
                for i in range(1, len(line)):
                    curr = line[i]
                    gap = curr[0] - (current_word[0] + current_word[2])
                    if gap < gap_threshold:
                        new_x = min(current_word[0], curr[0])
                        new_y = min(current_word[1], curr[1])
                        new_w = max(current_word[0] + current_word[2], curr[0] + curr[2]) - new_x
                        new_h = max(current_word[1] + current_word[3], curr[1] + curr[3]) - new_y
                        current_word = [new_x, new_y, new_w, new_h]
                    else:
                        word_boxes.append(tuple(current_word))
                        current_word = list(curr)
                word_boxes.append(tuple(current_word))
                
                for (x, y, bw, bh) in word_boxes:
                    pad = max(2, int(bh * 0.1))
                    y1 = max(0, y - pad)
                    y2 = min(gray.shape[0], y + bh + pad)
                    x1 = max(0, x - pad)
                    x2 = min(gray.shape[1], x + bw + pad)
                    
                    word_crop_gray = gray[y1:y2, x1:x2]
                    
                    if word_crop_gray.shape[0] == 0 or word_crop_gray.shape[1] == 0:
                        continue
                        
                    h_c, w_c = word_crop_gray.shape
                        
                    target_w, target_h = self.target_size
                    
                    new_h = target_h
                    new_w = int(w_c * (target_h / h_c))
                    
                    if new_w > target_w:
                        new_w = target_w
                        new_h = int(h_c * (target_w / w_c))
                        if new_h == 0: new_h = 1
                        
                    if new_h == 0 or new_w == 0:
                        continue
                        
                    # DOWNSCALE FIRST to preserve stroke thickness!
                    resized_gray = cv2.resize(word_crop_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Otsu's binarization on the downscaled crop (Scale-invariant)
                    word_blurred = cv2.GaussianBlur(resized_gray, (3, 3), 0)
                    _, word_thresh = cv2.threshold(word_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    final_tensor = np.zeros((target_h, target_w), dtype=np.uint8)
                    start_x = (target_w - new_w) // 2
                    start_y = (target_h - new_h) // 2
                    final_tensor[start_y:start_y+new_h, start_x:start_x+new_w] = word_thresh
                    
                    segmented_lines.append(final_tensor)
                    b64_lines.append(img_to_b64(final_tensor))

            if len(segmented_lines) == 0:
                return False, None, None, None, None
                
            return True, segmented_lines, b64_orig, b64_prep, b64_lines

        except Exception as e:
            logger.error(f"Advanced segmentation failed: {e}")
            return False, None, None, None, None

    def preprocess_image(self, image_path):
        """
        Fallback Single-Image Preprocessing.
        Applies basic pipeline to a single image.
        """
        try:
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError("Image could not be loaded.")
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            final_img = self._pad_and_resize(thresh)
            return final_img, True
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None, False

    def _process_single_image(self, img_path):
        """Helper to process a single image for multiprocessing."""
        rel_path = os.path.relpath(img_path, self.raw_dir)
        dest_path = os.path.join(self.processed_dir, rel_path)
        
        if os.path.exists(dest_path):
            return True, True # (Success, Skipped)
            
        processed_img, success = self.preprocess_image(img_path)
        
        if success:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            cv2.imwrite(dest_path, processed_img)
            return True, False
        else:
            return False, False

    def process_dataset(self, sample_size=None):
        """
        Iterates over the raw dataset, processes each image, and saves it.
        Uses multiprocessing for speed and skips already processed files.
        """
        import concurrent.futures
        
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
                    
        if sample_size and sample_size > 0:
            image_files = image_files[:sample_size]
            
        total_files = len(image_files)
        logger.info(f"Total image files found: {total_files}")
        
        if total_files == 0:
            logger.warning("No images found to process. Pipeline ended.")
            return

        success_count = 0
        corrupt_count = 0
        skipped_count = 0
        
        max_workers = min(32, os.cpu_count() + 4)
        logger.info(f"Using {max_workers} worker processes.")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map returns results in order, which is fine, but imap_unordered or as_completed is better for tqdm
            futures = [executor.submit(self._process_single_image, path) for path in image_files]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Preprocessing Images"):
                success, skipped = future.result()
                if skipped:
                    skipped_count += 1
                elif success:
                    success_count += 1
                else:
                    corrupt_count += 1

        # 10. Generate Preprocessing Log
        logger.info("=================================")
        logger.info("    PREPROCESSING COMPLETED      ")
        logger.info("=================================")
        logger.info(f"Total Processed Successfully: {success_count}")
        logger.info(f"Total Skipped (Already Processed): {skipped_count}")
        logger.info(f"Total Corrupted/Failed: {corrupt_count}")
        logger.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    # Process 20 images first if passed as argument
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        preprocessor.process_dataset(sample_size=20)
    else:
        preprocessor.process_dataset()
