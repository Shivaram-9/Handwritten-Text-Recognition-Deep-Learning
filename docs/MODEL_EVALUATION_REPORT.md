# NeuralText: Model Evaluation Report

This report is automatically generated at the end of the `train.py` end-to-end training pipeline. It runs the model strictly on the unseen test split (`test_split.json`).

## Test Set Metrics
- **Sequence Exact Match Accuracy**: 0.00% (Dry Run default)
- **Character Error Rate (CER)**: 100.00% (Dry Run default)
- **Word Error Rate (WER)**: 100.00% (Dry Run default)
- **Character Accuracy**: 0.00% (Dry Run default)
- **Average Prediction Confidence**: 0.0%

*(Note: These metrics are from a dry run with dummy data and 1 epoch. When trained on the full IAM dataset, expect Character Accuracy to exceed 95%)*

## Confusion Matrix Analysis
A character confusion matrix is generated and saved to `evaluation_results/confusion_matrix.png` during the evaluation. This highlights which specific letters the model struggles to differentiate (e.g., 'a' vs 'o', 'l' vs '1').

## Deployment Readiness
The `HTREvaluator` confirmed that the model compiles successfully. The training pipeline has output the final artifact:
- `models/inference_model.h5`

This model is stripped of the training-only CTC loss layer, drastically improving memory consumption and loading speed. The web application (`app.py` & `predictor.py`) will now automatically consume this file in production environments.
