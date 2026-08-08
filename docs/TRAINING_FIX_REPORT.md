# NeuralText: Training Pipeline Fix Report

This document outlines the bug fixes applied to stabilize the end-to-end training pipeline.

## Bug 1: Keras Input Dictionary Mismatch

### Issue
The `HTR_Training_Model` is defined using the Keras Functional API with multiple inputs:
1. `image_input` (The visual feature tensor)
2. `label_input` (The true labels used directly by the custom `CTCLayer` for loss calculation)

However, the `DatasetLoader` was originally mapping the pipeline as:
```python
return img, encoded_label
```
This caused a fatal `ValueError` because `model.fit()` interpreted this as a single input (`img`) and a target (`encoded_label`), leaving `label_input` completely unassigned.

### Resolution
I refactored the `DatasetLoader._encode_sample()` method to yield the inputs as a named dictionary matching the exact Keras layer names, along with a dummy target (since the `CTCLayer` calculates the true loss internally).
```python
inputs = {
    "image_input": img,
    "label_input": tf.cast(encoded_label, tf.float32)
}
return inputs, tf.cast(encoded_label, tf.float32)
```
This correctly feeds the tensors to the correct input layers in the computational graph.

## Bug 2: Predictor Uninitialized Reference

### Issue
During the evaluation phase, `train.py` invoked `HTREvaluator`, which internally bootstrapped the `HTRPredictor`.
However, the `HTRPredictor.__init__()` contained a sequential execution bug:
```python
self.inference_model = self._load_model()
self._predict_fn = self._build_predict_fn()
self.inference_model_path = os.path.join(Config.MODEL_DIR, 'inference_model.h5')
```
`_load_model()` attempts to check if `self.inference_model_path` exists, but the variable wasn't defined until *after* the function executed, leading to an `AttributeError`.

### Resolution
The variable initialization was moved above the `_load_model()` call to guarantee it is in scope before being accessed.

## Verification
A complete dry-run of the pipeline was executed via `train.py` with `EPOCHS=1`.
1. The Keras computational graph compiled without input mismatches.
2. `Epoch 1/1` executed successfully, and validation loss was computed.
3. `best_htr_model.weights.h5` was securely saved by `ModelCheckpoint`.
4. `inference_model.h5` was explicitly exported.
5. `evaluator.py` successfully consumed the inference model and generated the evaluation metrics.

The pipeline is now completely stable and ready for a full production retraining cycle on the complete IAM dataset.
