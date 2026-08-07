import os
import sys

# Add project root to sys.path to allow importing from config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def validate_dataset():
    """
    Validates the presence of the required IAM Handwriting Database files.
    """
    print("Validating Dataset Configuration...")
    
    dataset_path = os.path.join(Config.DATA_DIR, 'raw', 'IAM')
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset directory not found at: {dataset_path}")
        print("Please download the IAM Handwriting Database and place it in the 'data/raw/IAM' directory.")
        return False
        
    expected_folders = ['words', 'lines', 'ascii']
    all_present = True
    
    for folder in expected_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"[WARNING] Expected folder '{folder}' not found at {folder_path}")
            all_present = False
        else:
            print(f"[OK] Found folder: {folder}")
            
    if all_present:
        print("\n[SUCCESS] All required dataset components are present.")
        print("You can proceed to the preprocessing phase.")
    else:
        print("\n[INFO] Please ensure you have downloaded the required parts of the IAM dataset.")
        print("You can download it from: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
        
    return all_present

if __name__ == "__main__":
    validate_dataset()
