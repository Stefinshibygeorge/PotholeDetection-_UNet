# Pothole Detection using U-Net

This repository provides a deep learning pipeline for detecting potholes in road images via semantic segmentation using a U-Net architecture. The project is implemented in Jupyter Notebook, designed to run on Google Colab, and uses TensorFlow/Keras.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Setup & Requirements](#setup--requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training & Validation](#training--validation)
- [Visualization](#visualization)
- [Results](#results)
- [References](#references)

---

## Overview

The goal of this project is to accurately detect and segment potholes in road images using deep learning. The approach leverages the U-Net model, which is effective for pixel-wise image segmentation tasks.

---

## Dataset Structure

The dataset should be organized as follows:

```
Pothole_dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── sample_video.mp4
```
- `images/`: Input road images.
- `labels/`: Polygon label files (YOLO format) for segmentation masks.

---

## Setup & Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab (recommended for easy access to Google Drive)

Install dependencies with:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

---

## Data Preparation

- Mount Google Drive in Colab to access the dataset.
- Resize and normalize images.
- Generate binary segmentation masks from polygon label files.
- Save preprocessed images and masks as NumPy arrays for efficient loading.

---

## Model Architecture

This project uses a custom [U-Net](https://arxiv.org/abs/1505.04597) for image segmentation:
- **Encoder**: Stacked convolutional blocks with downsampling.
- **Bottleneck**: Deepest layer capturing global features.
- **Decoder**: Upsampling blocks with skip connections for precise localization.
- **Output**: Pixelwise mask via sigmoid activation.

---

## Training & Validation

- Data is split into training and validation sets.
- The model is compiled and trained using binary cross-entropy loss and appropriate metrics (e.g., IoU, Dice).
- Training progress and validation performance are monitored.

---

## Visualization

- Built-in functions to visualize random pairs of images and masks.
- Qualitative inspection of model predictions alongside ground truth.

---

## Results

The U-Net model demonstrates robust performance in segmenting potholes. Below are sample qualitative results:

| Input Image | Ground Truth Mask | Predicted Mask |
|-------------|------------------|---------------|
| ![Input](results/input1.jpg) | ![GT](results/gt1.jpg) | ![Pred](results/pred1.jpg) |
| ![Input](results/input2.jpg) | ![GT](results/gt2.jpg) | ![Pred](results/pred2.jpg) |
| ![Input](results/input3.jpg) | ![GT](results/gt3.jpg) | ![Pred](results/pred3.jpg) |

> **Note:** Replace the image paths above with your actual result images, e.g., from `/results/` or `/images/` in your repository.

**Sample Metrics:**  
- *Add metrics such as Dice score, IoU, accuracy here after evaluation.*

---

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al.)](https://arxiv.org/abs/1505.04597)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)

---

**Presentation:**  
For a summary of the project, see [`Minor Project Ppt.pptx`](Minor%20Project%20Ppt.pptx) in the repository.

---

*Please adjust dataset paths and add result images as appropriate for your setup.*