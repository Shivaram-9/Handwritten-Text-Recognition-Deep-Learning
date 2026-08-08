import os
import sys
import glob
import csv
import shutil
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("DatasetImporter")

def parse_labels(source_dir):
    possible_files = ['words.txt', 'words_new.txt', 'labels.csv', 'labels.txt', 'metadata.csv']
    label_file = None
    
    for f in possible_files:
        path = os.path.join(source_dir, f)
        if os.path.exists(path):
            label_file = path
            break
            
    if not label_file:
        for f in possible_files:
            matches = glob.glob(os.path.join(source_dir, '**', f), recursive=True)
            if matches:
                label_file = matches[0]
                break

    if not label_file:
        logger.error("Could not find any standard label files (words.txt, words_new.txt, labels.csv, etc.)")
        return None

    logger.info(f"Found label file: {label_file}")
    
    labels_map = {}
    ext = os.path.splitext(label_file)[1].lower()
    
    with open(label_file, 'r', encoding='utf-8') as f:
        if ext == '.csv':
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    fid = row[0].replace('.png', '').replace('.jpg', '').strip()
                    label = row[1].strip()
                    labels_map[fid] = label
        else:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 9 and parts[1] in ['ok', 'err']:
                    fid = parts[0]
                    label = " ".join(parts[8:])
                    labels_map[fid] = label
                elif len(parts) >= 2:
                    fid = parts[0].replace('.png', '').replace('.jpg', '')
                    label = " ".join(parts[1:])
                    labels_map[fid] = label

    return label_file, labels_map

def find_images_directory(source_dir):
    """
    Looks for a directory named 'words' that contains 'a01', 'a02', etc.
    """
    matches = glob.glob(os.path.join(source_dir, '**', 'words'), recursive=True)
    for match in matches:
        if os.path.isdir(match):
            subdirs = os.listdir(match)
            if any(d.startswith('a0') for d in subdirs):
                return match
    return None

def import_dataset(source_dir):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_raw_dir = os.path.join(base_dir, 'data', 'raw', 'IAM')
    target_words_dir = os.path.join(target_raw_dir, 'words')
    target_ascii_dir = os.path.join(target_raw_dir, 'ascii')
    target_words_txt = os.path.join(target_ascii_dir, 'words.txt')
    
    logger.info("=== Starting Dataset Import & Conversion ===")
    
    label_info = parse_labels(source_dir)
    if not label_info:
        return False
    label_file_path, labels_map = label_info
        
    nested_words_dir = find_images_directory(source_dir)
    
    os.makedirs(target_raw_dir, exist_ok=True)
    os.makedirs(target_ascii_dir, exist_ok=True)
    
    if nested_words_dir:
        logger.info(f"Found pre-nested IAM words directory: {nested_words_dir}")
        logger.info("Fast-moving directory to data/raw/IAM/words...")
        if os.path.exists(target_words_dir):
            if not os.listdir(target_words_dir) or (len(os.listdir(target_words_dir))==1 and os.listdir(target_words_dir)[0]=='.gitkeep'):
                shutil.rmtree(target_words_dir)
            else:
                logger.error(f"Target directory {target_words_dir} already exists and is not empty. Aborting.")
                return False
        shutil.move(nested_words_dir, target_words_dir)
        
        logger.info(f"Copying label file to {target_words_txt}...")
        shutil.copy2(label_file_path, target_words_txt)
        
        logger.info("Fast Import complete.")
    else:
        # Fallback to slow file-by-file copy if flat structure
        logger.info("Nested words directory not found. Falling back to deep search and slow copy...")
        # ... (implementation omitted for brevity in fast path)
        logger.error("Flat directory parsing is disabled. Kaggle IAM should be nested.")
        return False
    
    # Generate Report
    report = (
        "# Dataset Import Summary\n\n"
        f"- Fast imported nested structure.\n"
        f"- Total labels found: {len(labels_map)}\n"
        f"- Source label file used: {label_file_path}\n"
    )
    with open(os.path.join(base_dir, 'docs', 'IMPORT_REPORT.md'), 'w') as f:
        f.write(report)
        
    logger.info("Saved docs/IMPORT_REPORT.md")
    
    # Chain execute
    logger.info("Invoking verify_dataset.py...")
    os.system(f"{sys.executable} verify_dataset.py")
    

    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Kaggle IAM Dataset")
    parser.add_argument('--source', type=str, required=True, help="Path to extracted Kaggle dataset")
    args = parser.parse_args()
    
    import_dataset(args.source)
