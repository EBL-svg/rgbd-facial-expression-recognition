
# Real-Time Facial Expression Recognition Using RGB-D Vision

**Master's Capstone Project | Computer Vision & Deep Learning**

Academic project developed as part of the Master's programme in Signal and Image Processing at Aix-Marseille Université, France.

## Overview

This project investigates facial expression recognition using both RGB images and depth information.

The objective is to explore multimodal deep learning architectures and their potential for real-time facial expression recognition.

The project includes:
- RGB-D data preprocessing and augmentation
- TinyViT-based multimodal feature fusion
- EfficientNetV2-S with CBAM attention
- Model training and subject-based fine-tuning
- A real-time inference prototype using a ZED stereo camera and YOLO face detection

## Technical Stack

- Python
- PyTorch and torchvision
- timm
- OpenCV
- NumPy and scikit-learn
- Ultralytics YOLO
- Stereolabs ZED SDK

## Model Architectures

### 1. TinyViT with Gated RGB-D Fusion

Two TinyViT feature extractors process RGB and depth inputs separately.

A gating mechanism uses depth features to modulate RGB features. A Transformer encoder then combines the feature tokens for facial expression classification.

Inputs:
- RGB: `(B, 3, 224, 224)`
- Depth: `(B, 1, 224, 224)`

Output:
- Seven classification logits per sample.

### 2. EfficientNetV2-S with CBAM

A second architecture concatenates RGB and depth information into a four-channel input.

A convolutional projection maps the input to three channels before feature extraction using EfficientNetV2-S.

A CBAM attention block and classification head produce seven output logits.

Input:
- RGB-D: `(B, 4, 224, 224)`

Output:
- Seven classification logits per sample.

## Data Preprocessing

The TinyViT training pipeline uses:

- Aspect-ratio-preserving resize and padding
- Synchronized RGB-depth cropping
- Synchronized horizontal flipping
- Bilinear interpolation for RGB
- Nearest-neighbor interpolation for depth
- ImageNet normalization for RGB

The real-time EfficientNetV2 pipeline uses a separate preprocessing implementation, including depth conversion from metric values to an 8-bit representation.

These preprocessing pipelines should not be considered interchangeable.

## Project Structure

```text
src/
  models/
  data/
  training/
  inference/
  utils/
tests/
configs/
results/
docs/
README.md
requirements.txt
```

## Installation

Recommended environment: Python 3.11.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The ZED SDK must be installed separately for camera-based inference.

## Tests

The repository includes smoke tests using synthetic RGB-D inputs.

Run:

```bash
python -m tests.test_model_smoke
python -m tests.test_preprocessing
```

The tests check model output shapes, finite values, data transformations, and basic preprocessing reproducibility.

They do not measure classification accuracy.

## Experiments

The academic work explored pretraining using AffectNet and subject-based fine-tuning using FED-LFI.

The original experimental report includes results from these experiments.

Full reproduction requires access to the corresponding datasets, checkpoints and evaluation configurations.

## Real-Time Prototype

The original prototype integrates:

1. RGB acquisition using a ZED stereo camera
2. Depth estimation from the stereo camera
3. Face detection using YOLO
4. RGB-D preparation
5. Facial expression inference
6. Temporal smoothing of predictions

The real-time prototype requires compatible hardware and additional software dependencies.

## Limitations

- The public repository does not include proprietary or restricted datasets.
- Model checkpoints are not distributed unless redistribution is authorized.
- Synthetic smoke tests do not establish predictive accuracy.
- The two model architectures use different input preparation pipelines.
- Real-time operation requires separate validation on compatible hardware.

## Academic Context

Developed as a collaborative Master's capstone project.

Individual contributions, academic acknowledgements and relevant publication permissions will be documented before public release.

## License

No redistribution or reuse license is granted by default.

An open-source license may be added after confirming ownership and permissions for all included code and resources.
