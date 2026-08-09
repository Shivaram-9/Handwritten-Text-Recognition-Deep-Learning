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

            # 2. Convert to Grayscale & Contrast (CLAHE)
            gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # 3. Document Edge Detection & Perspective Correction (Fast Heuristic)
            # Resize for faster edge detection
            ratio = enhanced.shape[0] / 500.0
            small = cv2.resize(enhanced, (int(enhanced.shape[1]/ratio), 500))
            blurred = cv2.GaussianBlur(small, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            
            cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
                doc_cnt = None
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:
                        doc_cnt = approx
                        break
                
                # If a large document-like rectangle is found, warp it (Deskew/Perspective)
                if doc_cnt is not None and cv2.contourArea(doc_cnt) > 20000:
                    doc_cnt = doc_cnt.reshape(4, 2) * ratio
                    rect = np.zeros((4, 2), dtype="float32")
                    s = doc_cnt.sum(axis=1)
                    rect[0] = doc_cnt[np.argmin(s)]
                    rect[2] = doc_cnt[np.argmax(s)]
                    diff = np.diff(doc_cnt, axis=1)
                    rect[1] = doc_cnt[np.argmin(diff)]
                    rect[3] = doc_cnt[np.argmax(diff)]
                    
                    (tl, tr, br, bl) = rect
                    widthA = np.linalg.norm(br - bl)
                    widthB = np.linalg.norm(tr - tl)
                    maxWidth = max(int(widthA), int(widthB))
                    heightA = np.linalg.norm(tr - br)
                    heightB = np.linalg.norm(tl - bl)
                    maxHeight = max(int(heightA), int(heightB))
                    
                    dst = np.array([
                        [0, 0], [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
                    ], dtype="float32")
                    
                    M = cv2.getPerspectiveTransform(rect, dst)
                    enhanced = cv2.warpPerspective(enhanced, M, (maxWidth, maxHeight))

            # 4. Adaptive Thresholding (Matched to training pipeline)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            cleaned = thresh
            b64_prep = img_to_b64(cleaned)

            # 5. Line Segmentation via Horizontal Projection Profile
            # Dilate horizontally to connect characters in a line
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
            dilated = cv2.dilate(cleaned, dilate_kernel, iterations=1)
            
            line_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours top-to-bottom
            if not line_cnts: return False, None, None, None, None
            bounding_boxes = [cv2.boundingRect(c) for c in line_cnts]
            (line_cnts, bounding_boxes) = zip(*sorted(zip(line_cnts, bounding_boxes), key=lambda b: b[1][1]))
            
            segmented_lines = []
            b64_lines = []
            
            for (x, y, w, h) in bounding_boxes:
                # 6. Filter non-handwritten regions (too small, too tall, likely borders)
                aspect_ratio = w / float(h)
                if w > 20 and h > 15 and aspect_ratio > 1.2:
                    # Pad slightly
                    pad = 5
                    y1 = max(0, y - pad)
                    y2 = min(cleaned.shape[0], y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(cleaned.shape[1], x + w + pad)
                    
                    line_crop = cleaned[y1:y2, x1:x2]
                    
                    # Convert to required neural network format
                    final_tensor = self._pad_and_resize(line_crop)
                    segmented_lines.append(final_tensor)
                    
                    # Store visual for frontend
                    b64_lines.append(img_to_b64(line_crop))

            # Fallback if no valid lines found
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
