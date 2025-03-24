# Alzheimer's Disease MRI Classification Project

## Summary 

This project aims to implement deep learning-based system for automated classification of Alzheimer's disease stages using brain MRI scans. The goal of this system is to categorize scans into four stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented; ultimately aiming to aide with more precise and efficent diagnose of Alzheimers Disease.

## Data Collection

We began the pipeline by integrating two distinct MRI datasets: a [folder based dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/data) containing MRI images in standard image formats (JPEG, PNG) organized by impairment level folders, and a [parquet based dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset/data) containing binary image data stored in Apache Parquet format. Both datasets contain brain MRI scans used for dementia classification with four severity levels: Non-Demented (Label 2), Very Mild Demented (Label 3), Mild Demented (Label 0), and Moderate Demented (Label 1). In total, 6400 photos were extracted across all four classes. 

![brain samples](images/brain_samples.png)

## Data Preprocessing

For data standardization, we applied several procedures to the raw images. These included dimension normalization where all images were resized to 128×128 pixels, conversion to grayscale (single-channel) to ensure consistent formatting, and intensity scaling where pixel values were normalized to the range [0-1] by dividing by 255 to ensure compatibility with neural networks.

To balance the dataset, we employed stratified data splitting using sklearn's **train_test_split function** to maintain class distribution, resulting in a three-way partition: 70% for training, 10% for validation, and 20% for testing. The directories were organized with structured folders following a train/val/test organization, each containing class subfolders. This approach ensured that the proportion of diagnostic classes was preserved in each split, supporting proper model training and evaluation.

![class_distribution](images/class_distribution.png)

## Feature Extraction

We began our feature extraction process by implementing percentile-based normalization. This technique allows us to capture meaningful intensity variations in brain tissue because it effectively removes extreme outliers/noise such as very bright and very dark regions. The pixel values were then rescaled to the range [0-1], focusing on relevant tissue intensities and reducing the impact of outlier pixel values commonly found in MRI images.

For structure enhancement, we applied Contrast Limited Adaptive Histogram Equalization (CLAHE) with an 8×8 tile grid and a clip limit of 2.0. This approach enhances local contrast, improving the visualization of brain structures. The chosen tile size helps find a balance between local and global contrast enhancement, this is essential because smaller tiles would introduce excessive local detail and noise, while larger tiles would not sufficiently enhance local structures. Moreover, the clip limit of 2.0 prevents over-amplification of noise while still enhancing meaningful contrast.

Lastly we focused on the lateral ventricles of the MRI images as ventricle enlargement is a key biomarker of dementia. First, we inverted the normalized images (1 - normalized_image), making dark ventricles appear bright. For segmentation, we generated ventricle masks by thresholding at 50% of the mean intensity, followed by morphological operations (opening and closing) to refine the binary masks. This threshold was selected to maximize ventricle pixel inclusion while minimizing non-ventricle regions. Opening was applied to remove small noise and weakly connected areas, whereas closing filled small gaps and connected adjacent regions. In general, these steps helped maintain the integrity of ventricle structures while reducing artifacts, with the structuring element size chosen to distinguish between noise and true ventricle features.

![ventricle_extraction](images/feature_extraction.png)

## Modeling Approach
1. Primary Model:
   - Deep Learning using Convolutional Neural Networks (CNN) Tensorflow/Pytorch 
   
2. Model Optimization:
   - Hyperparameter tuning


## Results/Data Visualization
1. Pre-training Visualizations:
   - Sample MRI visualizations across different classes

2. Model Analysis Visualizations:
   - Confusion matrix
   - ROC curves and AUC scores for each class
   - Interactive learning curves (training vs. validation)
   - Feature maps from different CNN layers


3. Supplementary Datasets:
   - ADNI (Alzheimer's Disease Neuroimaging Initiative) database
   - OASIS (Open Access Series of Imaging Studies)
