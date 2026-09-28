# Chest X-Ray Image Classifier

A Streamlit application for classifying chest X-ray images into four categories with a fine-tuned DenseNet121 model.

## Overview

This project applies transfer learning to chest X-ray image classification using the COVID-19 Radiography Dataset. Given an uploaded image, the application predicts one of four classes—COVID, Normal, Viral Pneumonia, or Lung_Opacity—and shows the model's output scores for all classes.

This is an experimental machine-learning project for educational and research purposes. It is not a medical diagnostic tool, and its predictions should not be used to make medical decisions.

## Features

- Upload PNG, JPG, or JPEG chest X-ray images.
- Classify images into COVID, Normal, Viral Pneumonia, or Lung_Opacity.
- Display the predicted class and its model confidence.
- Show the probability distribution across all four classes as a table and bar chart.
- Run inference on CUDA when available, otherwise on CPU.
- Preview the uploaded image in the app.

## Model Architecture

The classifier uses DenseNet121 with ImageNet-pretrained weights as its starting backbone. Training uses two stages: first, the backbone is frozen while the classifier is trained; then the final DenseNet block is unfrozen for fine-tuning on chest X-ray images.

The classification head replaces DenseNet121's original classifier with Dropout (0.4) followed by a linear layer mapping the backbone features to four class scores. Inputs are resized to 224 × 224 and normalized with ImageNet mean and standard deviation.

```text
Chest X-ray image
        │
        ▼
Resize to 224 × 224
        │
        ▼
ImageNet normalization
        │
        ▼
DenseNet121 backbone
(ImageNet-pretrained; final block fine-tuned)
        │
        ▼
Dropout (0.4)
        │
        ▼
Linear layer (DenseNet features → 4 classes)
        │
        ▼
Class scores → Softmax probabilities
```

## Dataset

Training used the COVID-19 Radiography Dataset, containing 21,165 images across four classes:

- COVID
- Normal
- Viral Pneumonia
- Lung_Opacity

The data was split using stratified sampling into 80% training, 10% validation, and 10% testing. The dataset is imbalanced. Class weights were calculated from the training split and used with weighted CrossEntropyLoss to account for this imbalance.

No dataset download link is included because none is provided in this repository.

## Training Methodology

Training was performed in two stages:

1. **Feature extraction (10 epochs):** Freeze the pretrained DenseNet121 backbone and train the replacement classifier.
2. **Fine-tuning (10 epochs):** Unfreeze the final DenseNet block and continue training with a low learning rate.

Feature extraction lets the new classifier learn the target classes from the pretrained visual features. Fine-tuning the final block then allows higher-level features to adapt to chest X-ray images while retaining the earlier pretrained representations. Weighted CrossEntropyLoss was used in training to address class imbalance.

## Results

### Test set

The reported best validation accuracy was 88.00%. The following metrics are reported for the 2,117-image test set:

| Metric | Score |
|---|---:|
| Accuracy | 89.84% |
| Macro Precision | 89.54% |
| Macro Recall | 91.72% |
| Macro F1-score | 90.55% |

### Test-set results by class

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| COVID | 85.31% | 91.44% | 88.27% | 362 |
| Normal | 91.38% | 90.48% | 90.93% | 1,019 |
| Viral Pneumonia | 91.72% | 99.25% | 95.34% | 134 |
| Lung_Opacity | 89.74% | 85.71% | 87.68% | 602 |

## Confusion Matrix Analysis

A confusion matrix is not included in the repository or in the reported results, so the specific class-to-class confusions cannot be established from the available information. The per-class metrics show that Viral Pneumonia has the highest reported recall (99.25%), while Lung_Opacity has the lowest (85.71%). These recall values indicate how often examples of each class were correctly identified, but do not reveal which other class received the incorrect predictions.

## Streamlit Application

The application loads `model/best_densenet121_finetuned.pth` into the DenseNet121 architecture and selects CUDA when available, otherwise CPU. For an uploaded image, it:

1. Accepts a PNG, JPG, or JPEG file and converts it to RGB.
2. Resizes it to 224 × 224, converts it to a tensor, and applies ImageNet normalization.
3. Runs DenseNet121 inference without gradient calculation.
4. Applies softmax to obtain a score distribution across the four classes.
5. Displays the class with the highest score and its score as model confidence.
6. Shows all class scores in a table and a bar chart.

The displayed scores are the model's output distribution; they should not be interpreted as clinical probabilities.

### Run locally

From the project directory, install the listed dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model checkpoint must be present at `model/best_densenet121_finetuned.pth` for the app to load successfully.

## Project Structure

```text
Covid-XRay_App/
├── app.py
├── codex.txt
├── README.md
├── requirements.txt
├── .gitignore
└── model/
    └── best_densenet121_finetuned.pth
```
