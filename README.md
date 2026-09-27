# 🚦 Embedded AI - Lab 1: Traffic Sign Recognition using HOG + Classical Machine Learning

## Overview

This project implements a Traffic Sign Recognition (TSR) system based on traditional computer vision techniques. The proposed pipeline uses **Histogram of Oriented Gradients (HOG)** feature extraction combined with classical **Machine Learning** classifiers including:

* k-Nearest Neighbors (kNN)
* Logistic Regression
* Linear Support Vector Machine (Linear SVM)
* Random Forest

Experiments were conducted on Vietnamese Traffic Sign datasets (VNTS) and follow a leakage-free evaluation protocol using group-based dataset splitting.

---

# Project Structure

```text
├── data/
│   ├── vnts_raw/             # Original VNTS dataset (images + annotation txt files)
│   └── vnts_crops/           # Cropped traffic sign ROIs
│
├── src/
│   ├── prepare_data.py       # Dataset exploration and ROI cropping
│   ├── extract_features.py   # HOG feature extraction + feature normalization
│   └── train_eval.py         # Model training, tuning and evaluation
│
├── models/
│   └── best_model.joblib     # Best trained model
│
├── main.py                   # End-to-end training pipeline
├── benchmark.py              # Latency and FPS benchmark
│
├── hog_config.json           # HOG configuration
├── label_mapping.json        # Class ID → traffic sign label mapping
└── README.md
```

---

# Environment Setup

## Requirements

* Python 3.10+
* Recommended: Virtual Environment (`venv`)

---

## Step 1: Create Virtual Environment

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## Step 2: Install Dependencies

```bash
pip install --upgrade pip

pip install \
numpy \
pandas \
opencv-python \
scikit-image \
scikit-learn \
matplotlib \
seaborn \
joblib
```

---

# Running the Project

## 1. Run Full Training Pipeline

The main pipeline performs:

1. Dataset exploration
2. ROI extraction and filtering
3. HOG feature extraction
4. Group-based train/validation/test split
5. Model training
6. Hyperparameter tuning
7. Evaluation and report generation

```bash
python main.py
```

---

## 2. Run Performance Benchmark

Measure:

* Feature extraction latency
* Model inference latency
* Estimated FPS

```bash
python benchmark.py
```

---

# Methodology

## HOG Feature Extraction

Traffic sign images are resized before feature extraction.

### HOG Configuration

| Parameter           | Value   |
| ------------------- | ------- |
| Target Size         | 48 × 48 |
| Orientations        | 9       |
| Pixels per Cell     | (8, 8)  |
| Cells per Block     | (2, 2)  |
| Block Normalization | L2-Hys  |

---

## Machine Learning Models

The following classifiers are evaluated:

* kNN
* Logistic Regression
* Linear SVM
* Random Forest

All extracted HOG features are standardized using:

```python
StandardScaler()
```

before model training.

---

# Dataset Statistics

## VNTS Dataset

### ROI Cropping Summary

| Metric                | Value  |
| --------------------- | ------ |
| Total Crops Generated | 15,254 |
| Removed (<16 px)      | 1,414  |
| Final Samples         | 13,840 |

---

## Dataset Split

| Dataset    | Samples | Ratio |
| ---------- | ------- | ----- |
| Train      | 10,692  | ~70%  |
| Validation | 2,324   | ~15%  |
| Test       | 2,238   | ~15%  |

---

# Data Leakage Prevention

## Problem

Multiple traffic signs can be cropped from the same original image (`image_id`).

If a random split is used:

* Similar crops may appear in both training and testing sets.
* The model may memorize contextual information such as:

  * illumination
  * camera angle
  * road background

This leads to overly optimistic performance estimates.

---

## Solution

Use:

```python
GroupShuffleSplit
```

with:

```python
group = image_id
```

All crops originating from the same image are assigned exclusively to one subset.

---

## Leakage Verification

| Check              | Result |
| ------------------ | ------ |
| Train ∩ Validation | 0      |
| Train ∩ Test       | 0      |
| Validation ∩ Test  | 0      |

No overlapping source images exist between subsets.

---

# Experimental Results

## Validation Performance

| Model               | Accuracy   | Macro-F1   |
| ------------------- | ---------- | ---------- |
| kNN                 | **94.66%** | **0.9457** |
| Logistic Regression | 93.46%     | 0.9169     |
| Linear SVM          | 92.08%     | 0.8979     |
| Random Forest       | 90.88%     | 0.8971     |

---

## Computational Benchmark

| Model               | Training Time (s) | Inference Time (ms/sample) |
| ------------------- | ----------------- | -------------------------- |
| kNN                 | 0.01              | 0.215                      |
| Logistic Regression | 1.95              | **0.008**                  |
| Linear SVM          | 4367.35           | 0.024                      |
| Random Forest       | 46.12             | 0.034                      |

---

## Best Model

### k-Nearest Neighbors (kNN)

Selected as the final model because it achieved the best validation performance:

* Accuracy: **94.66%**
* Macro-F1: **0.9457**

while requiring almost no training time.

---

# Misclassification Analysis

Analysis was performed using the generated file:

```text
20_misclassified_samples.png
```

Three major error sources were identified.

---

## 1. Class Imbalance

Minority classes such as:

* Class 0
* Class 19
* Class 37

are frequently misclassified into dominant classes:

* Class 1
* Class 2
* Class 12
* Class 38

due to insufficient training examples.

---

## 2. HOG Limitation on Similar Geometries

HOG captures:

* edges
* gradients
* contour structures

but ignores color information.

As a result, signs sharing similar shapes can produce highly similar feature vectors.

Example:

* Speed Limit 30 km/h
* Speed Limit 50 km/h

Both signs have nearly identical circular boundaries and red borders.

---

## 3. Low-Quality ROI Images

Prediction errors also occur when:

* signs are partially occluded
* images are blurred
* contrast is poor

which distort the extracted HOG representation.

---

# Generated Artifacts

After execution, the following outputs are automatically generated.

| File                           | Description                            |
| ------------------------------ | -------------------------------------- |
| `vnts_class_distribution.png`  | Class distribution visualization       |
| `30_random_crops.png`          | Randomly sampled cropped traffic signs |
| `20_misclassified_samples.png` | Misclassified samples visualization    |
| `models/best_model.joblib`     | Best trained model                     |
| `hog_config.json`              | HOG parameter configuration            |
| `label_mapping.json`           | Traffic sign label mapping             |

---

# Key Findings

* Traditional HOG features remain highly effective for traffic sign recognition.
* Proper dataset splitting is critical to avoid data leakage.
* kNN achieved the best overall performance with:

  * Accuracy ≈ 94.66%
  * Macro-F1 ≈ 0.9457
* Most classification errors arise from:

  * class imbalance
  * visually similar traffic signs
  * low-quality image crops

---

# Future Improvements

Potential directions for future work:

1. Integrate color descriptors alongside HOG features.
2. Apply class balancing techniques.
3. Evaluate advanced descriptors such as:

   * LBP
   * SIFT
   * ORB
4. Compare against lightweight deep learning models:

   * MobileNetV3
   * EfficientNet-Lite
   * TinyViT
5. Deploy the final pipeline on embedded AI platforms for real-time inference.

```

README này đã ở mức phù hợp để đưa trực tiếp lên GitHub cho một đồ án Embedded AI/Machine Learning, với cấu trúc thường thấy trong báo cáo nghiên cứu hoặc portfolio kỹ thuật.
```
