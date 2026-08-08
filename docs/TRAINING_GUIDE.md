# NeuralText: Model Retraining Guide

Because a critical bug was identified in the CTC Loss function logic (detailed in `docs/MODEL_DEBUG_REPORT.md`), the currently saved weights in the `models/` directory are corrupted and invalid. 

**You must retrain the model from scratch.** Follow these steps to generate accurate model weights using the newly corrected codebase.

## Prerequisites

1. Ensure the raw IAM dataset is placed in `data/raw/IAM`.
2. Ensure you have run the dataset preprocessor and split scripts to generate the `processed` training JSON files:
   ```bash
   python src/preprocessor.py
   python src/dataset_splitter.py
   ```

## Retraining the Model

Because the weights are faulty, we want to completely overwrite them rather than resuming training on bad weights.

1. **Delete Existing Bad Checkpoints**
   Delete the existing weights to prevent the `ModelTrainer` from accidentally restoring bad weights.
   ```bash
   rm -f models/best_htr_model.h5
   rm -f models/checkpoints/latest_model.h5
   ```
   *(If on Windows PowerShell, use `Remove-Item` instead of `rm`)*

2. **Execute the Training Pipeline**
   We have not supplied a top-level `train.py` wrapper, but you can build a simple one or call the `ModelTrainer` inside a script.
   
   If you have a `train.py` at the project root, run it:
   ```bash
   python train.py
   ```
   
   *(Note: The trainer uses `tf.keras.mixed_precision.Policy('mixed_float16')`. Ensure you have an NVIDIA GPU for the fastest 3-4x speedup, otherwise it will safely fall back to CPU).*

## Post-Training Verification

1. **Run the Automated Evaluator**
   Once training completes and saves the new `best_htr_model.h5`, run the `evaluator.py` module to calculate the Word Error Rate (WER) and Character Error Rate (CER) on the unseen test split:
   ```bash
   python src/evaluator.py
   ```
2. **Review Metrics**
   Check the terminal output or the generated `evaluation_results/evaluation_report.json`. The new Character Error Rate should drastically drop from ~100% to acceptable margins, and the confidence metric will no longer be `0%`.

3. **Deploy**
   Since the architecture dimensions remained unchanged, you simply need to push the new `.h5` file to your server (or GitHub if using LFS), and the web application will automatically begin recognizing handwritten text accurately.
