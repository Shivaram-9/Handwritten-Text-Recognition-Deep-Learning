import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("DatasetVerifier")

def verify_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    words_txt = os.path.join(base_dir, 'data', 'raw', 'IAM', 'ascii', 'words.txt')
    words_dir = os.path.join(base_dir, 'data', 'raw', 'IAM', 'words')
    
    logger.info("--- Starting IAM Dataset Verification ---")
    
    # 1. Directory Checks
    missing_dirs = False
    if not os.path.exists(words_dir):
        logger.error(f"Missing directory: {words_dir}")
        missing_dirs = True
    else:
        logger.info(f"OK: Directory found - {words_dir}")
        
    if not os.path.exists(os.path.dirname(words_txt)):
        logger.error(f"Missing directory: {os.path.dirname(words_txt)}")
        missing_dirs = True
        
    if missing_dirs:
        logger.error("VERIFICATION FAILED: Required directories are missing. Please follow docs/IAM_SETUP_GUIDE.md")
        return False
        
    # 2. File Checks
    if not os.path.exists(words_txt):
        logger.error(f"Missing file: {words_txt}")
        logger.error("VERIFICATION FAILED: words.txt is missing. Please follow docs/IAM_SETUP_GUIDE.md")
        return False
    else:
        logger.info(f"OK: File found - {words_txt}")

    # 3. Parse words.txt and count labels
    logger.info("Parsing words.txt...")
    labels_map = {}
    with open(words_txt, 'r', encoding='utf-8') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 9:
                file_id = parts[0]
                label = " ".join(parts[8:])
                
                file_parts = file_id.split('-')
                if len(file_parts) >= 3:
                    folder1 = file_parts[0]
                    folder2 = f"{file_parts[0]}-{file_parts[1]}"
                    rel_path = os.path.join(folder1, folder2, f"{file_id}.png")
                    labels_map[rel_path] = label

    total_labels = len(labels_map)
    logger.info(f"Total labels parsed from words.txt: {total_labels}")
    
    if total_labels == 0:
        logger.error("VERIFICATION FAILED: words.txt is empty or improperly formatted.")
        return False

    # 4. Count Physical Images
    logger.info("Counting physical images in data/raw/IAM/words...")
    physical_images = glob.glob(os.path.join(words_dir, '**', '*.png'), recursive=True)
    total_images = len(physical_images)
    logger.info(f"Total physical images found: {total_images}")

    # 5. Check for missing images
    logger.info("Cross-referencing labels with physical images...")
    missing_count = 0
    missing_examples = []
    
    for rel_path in labels_map.keys():
        full_path = os.path.join(words_dir, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(full_path)
                
    if missing_count > 0:
        logger.error(f"VERIFICATION FAILED: {missing_count} images specified in words.txt are missing from the filesystem.")
        logger.error("Examples of missing files:")
        for ex in missing_examples:
            logger.error(f" - {ex}")
        return False

    if total_images == 0:
        logger.error("VERIFICATION FAILED: No images found. Did you extract words.tgz correctly?")
        return False
        
    logger.info("=========================================")
    logger.info("VERIFICATION SUCCESSFUL")
    logger.info("=========================================")
    logger.info(f"Labels mapped: {total_labels}")
    logger.info(f"Images verified: {total_images - missing_count}")
    logger.info("The IAM dataset is ready for processing.")
    return True

if __name__ == "__main__":
    verify_dataset()
