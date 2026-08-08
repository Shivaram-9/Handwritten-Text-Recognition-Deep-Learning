# Dataset Compatibility Report

## Overview
The architecture is designed to train on the official IAM Handwriting Database structure. However, because alternative mirrors (like the Kaggle dataset) often flatten directory structures and modify metadata labels, an automated compatibility layer was required.

This document details how the project seamlessly ingests alternative formats without breaking backwards compatibility with the core ML pipeline (`verify_dataset.py`, `dataset_splitter.py`, `trainer.py`).

## Architectural Decision
**No Core Pipeline Scripts Were Modified.** 
Instead of modifying the deep learning pipeline to accept variable data structures (which introduces instability), an abstraction layer `import_dataset.py` was introduced. This script translates the Kaggle dataset into the strict IAM format.

## Compatibility Rules Enforced by `import_dataset.py`

### 1. Label Normalization
- **Auto-Detection**: The script searches the source directory for `words.txt`, `labels.csv`, `labels.txt`, or `metadata.csv`.
- **Parsing**: It dynamically parses CSV headers or space-separated TXT rows, isolating the `<file_id>` and `<label>`.
- **IAM `words.txt` Generation**: A pristine `data/raw/IAM/ascii/words.txt` is compiled. Because `dataset_splitter.py` expects 9 columns where the 1st is the ID and the 9th is the text, `import_dataset.py` fills the intermediary columns with standard IAM dummy values (`ok 154 19 408 768 27 51`).

### 2. Directory Reconstruction
- The official IAM splits words into deep nested directories: `words/a01/a01-000/a01-000-00.png`.
- Kaggle versions usually flatten this into a single `words/` folder.
- **Translation**: `import_dataset.py` recursively finds all images, extracts their file ID, reconstructing the nested IAM directories based on the hyphenated ID structure (e.g. `a01-000-00` -> `a01/a01-000/`), and physically copies the images into `data/raw/IAM/words/`.

### 3. Failsafe Execution & Automatic Splitting
- The script computes missing images vs mismatched labels. If >50% mismatch occurs, the pipeline aborts to protect the training process from garbage input.
- **Automation Chain**: Once conversion succeeds, the script programmatically chains `verify_dataset.py` followed by `src/dataset_splitter.py`. This ensures `train_split.json`, `val_split.json` and `test_split.json` are instantly ready for the training engine.

### Result
By building this structural adapter, the dataset loaders and CTC mappings remain fully protected and 100% compliant with the original design specifications.
