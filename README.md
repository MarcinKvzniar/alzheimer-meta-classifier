# Alzheimer-Meta-Classifier
Repository containing final project for Artificial Intelligence 2 Laboratory. 

## Data
The project supports two ways to load Alzheimer's MRI data:

1. **HuggingFace Dataset** (via `data_loader.py`) - Downloads data from HuggingFace Hub (https://huggingface.co/datasets/Falah/Alzheimer_MRI)
2. **Local Dataset** (via `local_data_loader.py`) - Loads from local downloaded from Kaggle (https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset). Files hould be saved to 'datasets' folder in separate folders per class

Classifies Alzheimer severity based on the brain MRI scans into 4 classes:
    - '0': Mild_Demented
    - '1': Moderate_Demented
    - '2': Non_Demented
    - '3': Very_Mild_Demented

## Classifiers
Encompasses several classifiers to evaluate final results, including
    - Random Forest
    - Support Vector Machine (SVM)
    - Gradient Boosting