# IAM Dataset Setup Guide

This guide details exactly how to acquire the official IAM Handwriting Database required for production training of the Handwritten Text Recognition (HTR) system.

## 1. Request Access

The IAM dataset is hosted by the FKI (Research Group on Computer Vision and Artificial Intelligence).
1. Navigate to the official website: [IAM Handwriting Database Registration](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)
2. Follow the instructions to create a free account.
3. You will receive login credentials to access the download server.

## 2. Download the Required Files

Once authenticated, download the following specific archives:
1. **`words.tgz`**: Contains all the physically segmented word images.
2. **`words.txt`**: Contains the ground truth labels for all word images.

> [!IMPORTANT]  
> You **only** need the `words` subset. Do not download the `sentences` or `lines` datasets for this specific word-level CNN-BiLSTM-CTC model.

## 3. Directory Placement

Extract and place the files exactly as follows within your project root:

```text
Handwritten-Text-Recognition-Deep-Learning/
├── data/
│   ├── raw/
│   │   ├── IAM/
│   │   │   ├── ascii/
│   │   │   │   └── words.txt       <-- (Place words.txt here)
│   │   │   ├── words/              <-- (Extract words.tgz here)
│   │   │   │   ├── a01/
│   │   │   │   ├── a02/
│   │   │   │   └── ... (all subdirectories)
```

## 4. Verification

After placing the files, run the automated validation script to ensure no files were corrupted during extraction.

```bash
python verify_dataset.py
```

If the script passes all checks, the dataset is ready for production training!
