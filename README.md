# CS506_final project

# Alzheimer's Disease MRI Classification Project




# Project Goals

### Description of the Project

Our goal of this project is to implement deep learning-based system for automated classification of Alzheimer's disease stages using brain MRI scans. The system can categorize scans into four stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented.

### Data Collection

1. Primary Dataset: Kaggle Alzheimer's MRI Dataset (https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset/data)
   - Contains labeled MRI scans of brain
   - Multiple classes representing different stages of Alzheimer's
   - 6400 images across all classes
   

2. Supplementary Datasets:
   - ADNI (Alzheimer's Disease Neuroimaging Initiative) database
   - OASIS (Open Access Series of Imaging Studies)

### Data Cleaning
1. Direct download from Kaggle platform
2. Application for access to ADNI and OASIS databases
3. Data preprocessing pipeline:
   - Image standardization
   - Noise reduction
   - Normalization
   - Data augmentation through rotation, scaling, and intensity adjustments

### Modeling Approach
1. Primary Model:
   - Deep Learning using Convolutional Neural Networks (CNN) Tensorflow/Pytorch 
   
2. Model Optimization:
   - Hyperparameter tuning

### Feature Extraction
- possible feautres
        1. Hippocampal Region:
         - Track the progressive atrophy visible in the highlighted areas
         - Measure volumetric changes across stages
         - Analyze shape deformation patterns
        2. Ventricle Size:
         - Monitor the enlargement of ventricles   
         - Measure rate of expansion between stages
         - Compare bilateral differences
        3. Temporal Lobe:
         - Analyze gray matter density changes
         - Track cortical thickness reduction
         - Measure regional volume loss

### Data Visualization ( HMMM?? )
1. Pre-training Visualizations:
   - Sample MRI visualizations across different classes
   - Distribution plots of pixel intensities
   - Heat maps showing class distribution
   - Box plots of image statistics across classes

2. Model Analysis Visualizations:
   - Confusion matrix with interactive hover details
   - ROC curves and AUC scores for each class
   - Grad-CAM visualizations showing model focus areas
   - Interactive learning curves (training vs. validation)
   - t-SNE plots of learned feature representations
   - Feature maps from different CNN layers

### Test Plan
  Data Splitting Strategy:
   - Training set: 80% of total data
   - Testing set: 20% of total data
   - Within training set, 10% used for validation
   - Cross-validation
