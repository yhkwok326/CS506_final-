# Alzheimer's Disease MRI Classification Project

### Summary 

This project aims to implement deep learning-based system for automated classification of Alzheimer's disease stages using brain MRI scans. The goal of this system is to categorize scans into four stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented; ultimately aiming to aide with more precise and efficent diagnose of Alzheimers Disease.

### Data Collection

We began the pipeline by integrating two distinct MRI datasets: a [folder based dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/data) containing MRI images in standard image formats (JPEG, PNG) organized by impairment level folders, and a [parquet based dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset/data) containing binary image data stored in Apache Parquet format. Both datasets contain brain MRI scans used for dementia classification with four severity levels: Non-Demented (Label 2), Very Mild Demented (Label 3), Mild Demented (Label 0), and Moderate Demented (Label 1). In total, 6400 photos were extracted across all four classes. 

![brain samples](https://github.com/yhkwok326/CS506_final/blob/main/brain_samples.png?raw=true)

### Data Cleaning
1. Direct download from Kaggle platform
2. ADNI and OASIS databases (Maybe, we are trying to get access )
3. Data preprocessing pipeline:
   - Image standardization
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

### Data Visualization ( TBD )
1. Pre-training Visualizations:
   - Sample MRI visualizations across different classes

2. Model Analysis Visualizations:
   - Confusion matrix
   - ROC curves and AUC scores for each class
   - Interactive learning curves (training vs. validation)
   - Feature maps from different CNN layers

### Test Plan
  Data Splitting Strategy:
   - Training set: 70% of total data
   - Testing set: 20% of total data
   - 10% of the data used for validation

3. Supplementary Datasets:
   - ADNI (Alzheimer's Disease Neuroimaging Initiative) database
   - OASIS (Open Access Series of Imaging Studies)
