# Alzheimer's Disease MRI Classification

## Summary 

This project aims to implement deep learning-based system for automated classification of Alzheimer's disease stages using brain MRI scans. The goal of this system is to categorize scans into four stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented; ultimately aiming to aide with more precise and efficent diagnose of Alzheimers Disease.

## Data Collection

We began the pipeline by integrating two distinct MRI datasets: a [folder based dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/data) containing MRI images in standard image formats (JPEG, PNG) organized by impairment level folders, and a [parquet based dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset/data) containing binary image data stored in Apache Parquet format. Both datasets contain brain MRI scans used for dementia classification with four severity levels: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. 

![brain samples](images/brain_samples.png)

## Data Preprocessing

For data standardization, we applied several procedures to the raw images. These included dimension normalization where all images were resized to 128×128 pixels, conversion to grayscale (single-channel) to ensure consistent formatting, and intensity scaling where pixel values were normalized to the range [0-1] by dividing by 255 to ensure compatibility with neural networks.

To balance the dataset, we employed stratified data splitting using sklearn's **train_test_split function** to maintain class distribution, resulting in a three-way partition: 70% for training, 10% for validation, and 20% for testing. The directories were organized with structured folders following a train/val/test organization, each containing class subfolders. This approach ensured that the proportion of diagnostic classes was preserved in each split, supporting proper model training and evaluation.

![class_distribution](images/class_distribution.png)

## Feature Extraction

We first applied percentile-based normalization to adjust intensity values while removing extreme outliers like overly bright or dark regions. This helped highlight meaningful brain tissue variations. To ensure consistency and to redue the impact of noise commonly found in MRI scans, pixel values were then rescaled to [0-1]. To enhance structural details, we used Contrast Limited Adaptive Histogram Equalization (CLAHE) with an 8×8 tile grid and a clip limit of 2.0. This helped improve local contrast making brain structures more distinguishable. In addition, the parameters were chosen to balance detail enhancement without over-amplifying noise.

Since ventricle enlargement is a key biomarker for dementia, we focused on segmenting the lateral ventricles. First, we inverted the images so that dark ventricles appeared bright. Then, we created ventricle masks by thresholding at 50% of the mean intensity, followed by morphological operations (opening and closing) to refine the segmentation. These steps helped isolate ventricles while reducing noise and preserving structural integrity.

![ventricle_extraction](images/feature_extraction.png)

## Modeling Approach

This project uses a custom Convolutional Neural Network (CNN) to classify dementia based on ventricle features in 128×128 grayscale MRI scans. The model consists of three convolutional blocks that progressively extract important image features, followed by fully connected layers for classification. Each convolutional block increases the number of filters (64 → 128 → 256) and applies convolution, ReLU activation, batch normalization, and max pooling to reduce spatial dimensions while retaining important features. After feature extraction, the model flattens the output and processes it through two fully connected layers (512 and 128 units) with dropout to prevent overfitting.

For training, we used the AdamW optimizer with weight decay (1e-4) to improve generalization. The model was trained for 10 epochs with a batch size of 32 images, using CrossEntropyLoss for multi-class classification. A learning rate scheduler (ReduceLROnPlateau) adjusted the learning rate when performance plateaued, and early stopping with model checkpointing ensured the best model was saved based on validation accuracy.

## Results

![validation_results](images/accuracy_results.png)

Our training phase consisted of 10 epochs, with significant improvements in accuracy throughout the process. We started with a training accuracy of 56.03% and validation accuracy of 66.41% in epoch 1, and progressively improved to 98.76% training accuracy by epoch 10. The model reached its peak performance in epoch 9 with an accuracy of 99.22%, at which point we saved the best model checkpoint.

Intrestingly, our per-class validation accuracies show high accuracy for the Moderate Demented class, which is close to 100% for each epoch. Yet, this class has the least number of sample, so we may need to oversample the Moderate Demented class to ensure the reliability of the model's accuracy. In addition, the accuracy of our model seeems extremely high and might have to deal with the simplicity of our current dataset. As such, we are working on obtaining more images from supplementary datasets from ADNI (Alzheimer's Disease Neuroimaging Initiative) databas and OASIS (Open Access Series of Imaging Studies) to increase the overall range our model can cover in MRI brain scans.
