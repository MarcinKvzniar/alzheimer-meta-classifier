# Alzheimer-Meta-Classifier

Repository for the **Artificial Intelligence 2 Laboratory** final project, focused on classifying Alzheimer’s disease severity from brain MRI scans.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Data](#data)  
   - [HuggingFace Dataset](#huggingface-dataset---falahalzheimer_mri)  
   - [Local Kaggle Dataset](#local-kaggle-dataset---augmented-alzheimer-mri-dataset)  
3. [Classifiers](#classifiers)  
4. [Usage Recommendations](#usage-recommendations)  

---

## Project Overview

This project classifies Alzheimer’s disease severity into **four classes** based on brain MRI scans:

| Label | Class Name          |
|-------|-------------------|
| 0     | Mild_Demented      |
| 1     | Moderate_Demented  |
| 2     | Non_Demented       |
| 3     | Very_Mild_Demented |

The pipeline supports multiple classifiers and can work with either a small HuggingFace dataset or a larger local Kaggle dataset.

---

## Data

The project supports **two ways** to load Alzheimer's MRI data:

1. **HuggingFace Dataset** via `data_loader.py`  
   [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI)

2. **Local Kaggle Dataset** via `local_data_loader.py`  
   [Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)  
   > Files should be saved in a `datasets` folder, organized into separate folders per class.

---

## Classifiers

Several machine learning classifiers are implemented for evaluation:

- Random Forest  
- Support Vector Machine (SVM)  
- Gradient Boosting  

These allow both baseline performance evaluation and comparison with deep learning approaches.

---

## HuggingFace Dataset — `Falah/Alzheimer_MRI`

**Source:** [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI)  
**Loader used:** `data_loader.py`

**Description:**  
A publicly available dataset of preprocessed brain MRI slices labeled for Alzheimer’s disease severity. Suitable for baseline experiments and rapid prototyping.

**Key Characteristics:**
- Total images: 6,400 (5,120 train / 1,280 test)  
- Image resolution: 128 × 128 px  
- Classes (4): Mild_Demented (0), Moderate_Demented (1), Non_Demented (2), Very_Mild_Demented (3)  
- Format: Image + label per example; train/test split provided by dataset  
- License: Apache-2.0  

**Advantages:**
- Small and easy to download → fast experiments and debugging  
- Easy HuggingFace integration (`datasets.load_dataset(...)`)  
- Covers four Alzheimer’s classes  

**Limitations:**
- Low resolution may hinder learning of subtle anatomical features  
- Small dataset → risk of overfitting or limited generalization  
- No patient metadata → potential risk of data leakage if the same subject appears across splits  

---

## Local Kaggle Dataset — "Augmented Alzheimer MRI Dataset"

**Source:** [Kaggle](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)  
**Loader used:** `local_data_loader.py`
**Description:**  
A larger augmented dataset of 2D MRI slices, created from an original small dataset and expanded via image augmentation. Useful for full-scale deep learning training.

**Key Characteristics:**
- Total images: ~33,984  
- Class-wise counts (approx.):  
  - Mild Demented: ~8,960  
  - Moderate Demented: ~6,464  
  - Very Mild Demented: ~8,960  
  - Non Demented: ~9,600  
- Format: JPEG/PNG slices in class-specific folders (compatible with PyTorch `ImageFolder`)  
- Augmentation: rotations, flips, scaling, zooming for class balance and variability  

**Advantages:**
- Large dataset supports deep CNNs → better generalization  
- Balanced class distribution mitigates class imbalance issues  
- Directory-based structure → straightforward to load  
- Augmentation increases robustness  

**Limitations:**
- 2D slices only → loss of full 3D context  
- Compressed formats (JPEG/PNG) → potential loss of fidelity compared to raw medical formats  
- No patient metadata → cannot guarantee unique subjects or avoid data leakage  
- Augmented images may reduce anatomical diversity → limited real-world generalization  

---

## Usage Recommendations

- **HuggingFace dataset** → Quick experiments, debugging, baseline model evaluation, limited computational resources.  
- **Kaggle augmented dataset** → Full-scale deep learning training, leveraging larger size and class balance.  
- Always document dataset limitations (metadata absence, 2D slices, potential leakage).  
- Clearly state dataset used and whether slices are 2D or augmented when reporting results, ensuring reproducibility.  
