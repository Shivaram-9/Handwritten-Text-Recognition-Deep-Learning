# Preprocessing Validation Report

## Preprocessing Job Summary
- Total Processed: 55556
- Total Skipped (Resumed): 59762
- Total Failed/Corrupted: 2
- Elapsed Time: ~4 minutes 16 seconds (using multiprocessing)

## Raw Output Validation (Image Level)
- **PNG Dtype uint8**: PASS
- **Shape is 128x32**: PASS
- **Meaningful Pixel Range**: PASS (Min: 0, Max: 255)

## TensorFlow Dataset Loader Validation
- **TensorFlow Tensor Shape**: (32, 128, 1)
- **TensorFlow Tensor Dtype**: <dtype: 'float32'>
- **TensorFlow Pixel Min**: 0.0000
- **TensorFlow Pixel Max**: 1.0000
- **TensorFlow Label Encoded**: [62. 67. 73. 68. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.
 -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
- **Input Tensor Range Validation**: PASS (Network is receiving real image features)