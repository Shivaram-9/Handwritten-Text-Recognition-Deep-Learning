import os
import sys
import logging
import glob

# Add project root to sys.path to allow importing from config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class IAMDatasetIntegrator:
    def __init__(self):
        self.dataset_path = os.path.join(Config.DATA_DIR, 'raw', 'IAM')
        self.words_dir = os.path.join(self.dataset_path, 'words')
        self.lines_dir = os.path.join(self.dataset_path, 'lines')
        self.ascii_dir = os.path.join(self.dataset_path, 'ascii')
        self.report_path = os.path.join(self.dataset_path, 'validation_report.txt')

    def check_directories(self):
        """Check if the base directories exist."""
        directories = [self.words_dir, self.lines_dir, self.ascii_dir]
        all_exist = True
        for directory in directories:
            if not os.path.exists(directory):
                logger.error(f"Directory missing: {directory}")
                all_exist = False
        return all_exist

    def analyze_dataset(self):
        """Analyze the dataset to count images, labels, and missing files."""
        logger.info("Starting dataset analysis...")
        
        if not self.check_directories():
            logger.error("Dataset directories are incomplete. Analysis aborted.")
            self._write_empty_report()
            return
            
        words_txt = os.path.join(self.ascii_dir, 'words.txt')
        
        total_labels = 0
        total_images_found = 0
        missing_images = 0
        
        # Analyze words.txt
        if os.path.exists(words_txt):
            logger.info(f"Found label file: {words_txt}")
            with open(words_txt, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 9:
                    total_labels += 1
                    file_id = parts[0]
                    # The image path format is typically: words/part1/part1-part2/part1-part2-part3.png
                    file_parts = file_id.split('-')
                    if len(file_parts) >= 3:
                        folder1 = file_parts[0]
                        folder2 = f"{file_parts[0]}-{file_parts[1]}"
                        img_path = os.path.join(self.words_dir, folder1, folder2, f"{file_id}.png")
                        
                        if os.path.exists(img_path):
                            total_images_found += 1
                        else:
                            missing_images += 1
        else:
            logger.warning(f"Label file not found: {words_txt}")
            # Fallback: Just count the images in the directory
            logger.info("Counting raw images inside words/ directory...")
            for root, _, files in os.walk(self.words_dir):
                for file in files:
                    if file.endswith('.png'):
                        total_images_found += 1
                        
        self._generate_report(total_labels, total_images_found, missing_images)

    def _generate_report(self, total_labels, total_images_found, missing_images):
        """Generate a validation report and save it to the dataset directory."""
        report = (
            "========================================\n"
            "      IAM DATASET VALIDATION REPORT     \n"
            "========================================\n"
            f"Dataset Path : {self.dataset_path}\n"
            f"Total Labels : {total_labels}\n"
            f"Images Found : {total_images_found}\n"
            f"Missing Files: {missing_images}\n"
            "========================================\n"
        )
        
        logger.info(f"\n{report}")
        
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"Validation report saved to: {self.report_path}")

    def _write_empty_report(self):
        report = (
            "========================================\n"
            "      IAM DATASET VALIDATION REPORT     \n"
            "========================================\n"
            "STATUS: FAILED\n"
            "REASON: Dataset folders are missing. Please download the dataset.\n"
            "========================================\n"
        )
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(report)

if __name__ == "__main__":
    integrator = IAMDatasetIntegrator()
    integrator.analyze_dataset()
