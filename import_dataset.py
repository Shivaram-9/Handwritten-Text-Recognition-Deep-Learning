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
    """
    Scans the source directory for common label files and extracts file_id -> label mappings.
    """
    possible_files = ['words.txt', 'labels.csv', 'labels.txt', 'metadata.csv']
    label_file = None
    
    for f in possible_files:
        path = os.path.join(source_dir, f)
        if os.path.exists(path):
            label_file = path
            break
            
    # Also check recursive
    if not label_file:
        for f in possible_files:
            matches = glob.glob(os.path.join(source_dir, '**', f), recursive=True)
            if matches:
                label_file = matches[0]
                break

    if not label_file:
        logger.error("Could not find any standard label files (words.txt, labels.csv, etc.)")
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
                    # Assume column 0 is filename/id, column 1 is label
                    fid = row[0].replace('.png', '').replace('.jpg', '').strip()
                    label = row[1].strip()
                    labels_map[fid] = label
        else:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # If it's a standard IAM words.txt
                parts = line.split()
                if len(parts) >= 9 and parts[1] in ['ok', 'err']:
                    fid = parts[0]
                    label = " ".join(parts[8:])
                    labels_map[fid] = label
                elif len(parts) >= 2:
                    # Fallback generic space-separated format
                    fid = parts[0].replace('.png', '').replace('.jpg', '')
                    label = " ".join(parts[1:])
                    labels_map[fid] = label

    return labels_map

def find_images(source_dir):
    """
    Finds all image files in the source directory.
    Returns dict of file_id -> absolute_path
    """
    img_paths = glob.glob(os.path.join(source_dir, '**', '*.*'), recursive=True)
    images_map = {}
    for p in img_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            fid = os.path.basename(p).replace(ext, '')
            images_map[fid] = p
    return images_map

def is_already_formatted(source_dir):
    """
    Checks if the source directory is already a properly formatted IAM dataset.
    """
    words_txt = os.path.join(source_dir, 'ascii', 'words.txt')
    words_dir = os.path.join(source_dir, 'words')
    
    if os.path.exists(words_txt) and os.path.exists(words_dir):
        # Quick check inside words_dir for nested structure
        subdirs = [d for d in os.listdir(words_dir) if os.path.isdir(os.path.join(words_dir, d))]
        if subdirs:
            return True
    return False

def import_dataset(source_dir):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_raw_dir = os.path.join(base_dir, 'data', 'raw', 'IAM')
    
    logger.info("=== Starting Dataset Import & Conversion ===")
    
    if is_already_formatted(source_dir):
        logger.info("Source directory already matches the official IAM structure.")
        logger.info(f"Please manually move or symlink {source_dir} to {target_raw_dir}")
        return True

    labels_map = parse_labels(source_dir)
    if not labels_map:
        return False
        
    images_map = find_images(source_dir)
    if not images_map:
        logger.error("No images found in source directory.")
        return False
        
    logger.info(f"Found {len(labels_map)} labels and {len(images_map)} images.")
    
    # Cross reference
    valid_pairs = []
    missing_images = []
    
    for fid, label in labels_map.items():
        if fid in images_map:
            valid_pairs.append((fid, label, images_map[fid]))
        else:
            missing_images.append(fid)
            
    invalid_labels = len(labels_map) - len(valid_pairs)
    
    logger.info(f"Valid Image-Label Pairs: {len(valid_pairs)}")
    logger.info(f"Missing Images: {len(missing_images)}")
    logger.info(f"Unmatched/Invalid Labels: {invalid_labels}")
    
    if len(missing_images) > 0 and (len(missing_images) / len(labels_map)) > 0.5:
        logger.error("ABORTING: Massive mismatch detected (>50% missing). Ensure labels match file names.")
        return False
        
    if len(valid_pairs) == 0:
        logger.error("ABORTING: 0 valid image-label pairs found.")
        return False

    # Execute Conversion
    target_words_dir = os.path.join(target_raw_dir, 'words')
    target_ascii_dir = os.path.join(target_raw_dir, 'ascii')
    target_words_txt = os.path.join(target_ascii_dir, 'words.txt')
    
    os.makedirs(target_words_dir, exist_ok=True)
    os.makedirs(target_ascii_dir, exist_ok=True)
    
    logger.info(f"Copying images into deeply nested structure at {target_words_dir}...")
    
    with open(target_words_txt, 'w', encoding='utf-8') as fw:
        fw.write("# Formatted by import_dataset.py\n")
        
        for fid, label, img_path in valid_pairs:
            # Reconstruct dummy IAM hierarchy: a01/a01-000/a01-000-00.png
            parts = fid.split('-')
            if len(parts) >= 3:
                folder1 = parts[0]
                folder2 = f"{parts[0]}-{parts[1]}"
            else:
                # If the fid doesn't have hyphens, chunk it artificially or just use fallback dirs
                folder1 = "kaggle"
                folder2 = "kaggle-000"
                
            dest_dir = os.path.join(target_words_dir, folder1, folder2)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, f"{fid}.png")
            shutil.copy2(img_path, dest_path)
            
            # Write dummy IAM words.txt line
            # format: id ok 154 19 408 768 27 51 label
            fw.write(f"{fid} ok 154 19 408 768 27 51 {label}\n")

    logger.info("Conversion complete.")
    
    # Generate Report
    report = (
        "# Dataset Import Summary\n\n"
        f"- Total labels found: {len(labels_map)}\n"
        f"- Total images found: {len(images_map)}\n"
        f"- Successfully imported: {len(valid_pairs)}\n"
        f"- Missing images: {len(missing_images)}\n"
        f"- Unmatched labels: {invalid_labels}\n"
    )
    with open(os.path.join(base_dir, 'docs', 'IMPORT_REPORT.md'), 'w') as f:
        f.write(report)
        
    logger.info("Saved docs/IMPORT_REPORT.md")
    
    # Chain execute
    logger.info("Invoking verify_dataset.py...")
    os.system(f"{sys.executable} verify_dataset.py")
    
    logger.info("Invoking dataset_splitter.py...")
    os.system(f"{sys.executable} src/dataset_splitter.py")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Kaggle IAM Dataset")
    parser.add_argument('--source', type=str, required=True, help="Path to extracted Kaggle dataset")
    args = parser.parse_args()
    
    import_dataset(args.source)
