# Alzheimer's Disease MRI Classification

youtube link for the Midterm report: https://youtu.be/vV8o-egBJ6o
<br>
youtube link for Final Report: https://youtu.be/9rCDjcYDpLk

## Summary 

This project is to implement a deep learning-based model for automated classification of Alzheimer's disease stages using brain MRI scans. The goal of this model is to categorize scans into four stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented; ultimately aiming to aide with more precise and efficent diagnose of Alzheimers Disease.

## Data Collection

We began the pipeline by integrating two distinct MRI datasets: a [folder based dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/data) containing MRI images in standard image formats (JPEG, PNG) organized by impairment level folders, and a [parquet based dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset/data) containing binary image data stored in Apache Parquet format. Both datasets contain brain MRI scans used for dementia classification with four severity levels: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. 

![brain samples](images/brain_samples.png)

## Data Preprocessing

For data standardization, we applied several procedures to the raw images. These included dimension normalization where all images were resized to 128×128 pixels, conversion to grayscale (single-channel) to ensure consistent formatting, and intensity scaling where pixel values were normalized to the range [0-1] by dividing by 255 to ensure compatibility with neural networks.

To balance the dataset, we employed stratified data splitting using sklearn's train_test_split function to maintain class distribution, resulting in a three-way partition: **70% for training, 10% for validation, and 20% for testing**. The directories were organized with structured folders following a train/val/test organization, each containing class subfolders. This approach ensured that the proportion of diagnostic classes was preserved in each split, supporting proper model training and evaluation.

![class_distribution](images/class_distribution.png)

## Feature Extraction

To ensure consistency and to reduce the impact of noise commonly found in MRI scans, pixel values were rescaled to [0-1]. To enhance structural details, we used Contrast Limited Adaptive Histogram Equalization (CLAHE) with an 8×8 tile grid and a clip limit of 2.0. This helped improve local contrast making brain structures more distinguishable. The parameters were chosen to balance detail enhancement without over-amplifying noise.

Since ventricle enlargement is a key biomarker for dementia, we focused on segmenting the lateral ventricles. First, we inverted the images so that dark ventricles appeared bright. Then, we created ventricle masks using Otsu thresholding, an adaptive method that automatically determines the optimal threshold value, followed by applying a 50% factor to this threshold. We then applied morphological operations (opening and closing) to refine the segmentation. These steps helped isolate ventricles while reducing noise and preserving structural integrity.

![ventricle_extraction](images/feature_extraction.png)

## Modeling Approach

This project uses a custom CNN to classify dementia from 128×128 grayscale MRI scans by analyzing ventricle features. The model has two convolutional blocks that extract important details—each block applies convolution, ReLU activation, batch normalization, and max pooling to reduce image size while keeping key features. The first block uses 64 filters, and the second increases to 128 to capture more complex patterns. After feature extraction, the model flattens the data and processes it through two dense layers (512 and 128 units) with dropout (30% and 20%) to prevent overfitting. Despite its compact design, the model performs well, achieving 95.2% accuracy on the test set while remaining efficient.

For training, we used the AdamW optimizer with weight decay (1e-4) to improve generalization. The model was trained for 10 epochs with a batch size of 32 images, using CrossEntropyLoss for multi-class classification. A learning rate scheduler (ReduceLROnPlateau) adjusted the learning rate when performance plateaued, and early stopping with model checkpointing ensured the best model was saved based on validation accuracy.

## Training Results

![validation_results](images/n_trainingcurve.png)


The model was trained for 10 epochs on 12,543 images, with 1,792 images used for validation. Starting at 48% training accuracy in the first epoch, it improved steadily, reaching 96% accuracy by the final epoch. Validation accuracy followed a similar trend, starting at 58% and ending at 91%. Training remained stable throughout, with no signs of overfitting. These results show the model learned effectively while maintaining a good performance.

## Testing results 
![Confusion_matrix](images/confusion_matrix.png)

Our trained VentricleCNN model achieves an impressive overall accuracy of 91.41% in evaluation, with detailed metrics showing particularly strong performance for Moderate_Demented cases (high precision and recall) and Mild_Demented cases (95% precision, 94.5% recall). However, the model faces challenges in differentiating between Non_Demented and Very_Mild_Demented cases, with 66 misclassifications, suggesting that while it excels at identifying later-stage dementia, it struggles with the subtle anatomical changes characteristic of early dementia progression.

## GRADCAM visualization
![visualization_results](images/HeatMap.png)

To better understand the model's decision-making patterns, we employ a specialized Grad-CAM visualization technique tailored for ventricle masks. Using a hook mechanism on the second convolutional layer, we capture both feature maps during the forward pass and gradient flow during backpropagation. This enables us to generate attention heatmaps by computing a weighted combination of activation maps based on gradient importance, revealing which brain regions most strongly influence the model's predictions. In these visualizations, red and orange areas highlight regions with the highest predictive influence, while blue areas indicate minimal contribution. Interestingly, even when enhancing the ventricle region, we observe that the model relies on broader brain regions for classification, suggesting it incorporates contextual features beyond just ventricular structures.

## Usage 

Here, we developed a simple brain MRI classifier that tests a deep learning model on new brain images. Our tool processes MRI scans to highlight ventricle features and predicts whether a scan indicates Non-Demented, Very Mild, Mild, or Moderate dementia.

To evaluate the model's performance on a new dataset (consisting of unseen brain scan images, confirmed to be distinct from our training set using perceptual hashing), first download the ZIP file containing brain MRI images from this [Kaggle dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset).



After downloading the files, navigate to the directory containing the **mriclassifier.mk** script and run the following command in your conda environment:

```
make -f mriclassifier.mk
```
Running this command will install the required dependencies, retrieve the model, normalize and extract features from the new dataset, and make a prediction for a sample image.

---
Alternatively, if you prefer to run the scripts manually without using the Makefile, follow these steps after downloading the zip files from the Kaggle website:

First, run the following command to retrieve the model, normalize, and extract features from the new dataset:

```
pip install -r requirements.txt
python mri_dementia_classification.py --zip_file1 "Alzheimer MRI Disease Classification Dataset-2.zip" --zip_file2 "Combined Dataset.zip" --output_dir "Combined_MRI_Dataset" --epochs 10
```

Next, execute the following command to test if the model correctly predicts the class of an image:

```
simple_mri_classifier.py --image 14.png --model best_model.pt
```

To test other images from the dataset, simply replace 14.png with the filename of any image you would like to classify!
