def extract_zip_file(zip_path, extract_to=None):
    """Extract a zip file to a temporary directory or specified path"""
    if extract_to is None:
        extract_dir = tempfile.mkdtemp()
    else:
        extract_dir = extract_to
        os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    return extract_dir

def detect_dataset_structure(base_dir):
    """Auto-detect the structure of the extracted dataset"""
    folder_train_path = None
    folder_test_path = None
    train_parquet_path = None
    test_parquet_path = None
    
    # Look for folder structure
    for root, dirs, files in os.walk(base_dir):
        # Look for folder dataset structure
        if "train" in dirs:
            potential_train = os.path.join(root, "train")
            if os.path.isdir(potential_train):
                folder_train_path = potential_train
                print(f"Found folder train path: {folder_train_path}")
        
        if "test" in dirs:
            potential_test = os.path.join(root, "test")
            if os.path.isdir(potential_test):
                folder_test_path = potential_test
                print(f"Found folder test path: {folder_test_path}")
        
        # Look for parquet files
        for file in files:
            if file.lower().endswith('.parquet'):
                if 'train' in file.lower():
                    train_parquet_path = os.path.join(root, file)
                    print(f"Found train parquet path: {train_parquet_path}")
                if 'test' in file.lower():
                    test_parquet_path = os.path.join(root, file)
                    print(f"Found test parquet path: {test_parquet_path}")
    
    return folder_train_path, folder_test_path, train_parquet_path, test_parquet_path#!/usr/bin/env python3
"""
MRI Dementia Classification Pipeline

This script combines data processing, feature extraction, and model training
for MRI-based dementia classification into a single executable.
It supports input from zip files and outputs training accuracy and the best model.
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pyarrow.parquet as pq
import json
import time
import zipfile
import tempfile
import shutil

# Constants
IMG_SIZE = 128
N_CLASSES = 4
CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
BATCH_SIZE = 32

# Class mapping
classes = {
    0: 'Mild_Demented',
    1: 'Moderate_Demented', 
    2: 'Non_Demented',
    3: 'Very_Mild_Demented'
}

# Reverse mapping for feature extraction
class_labels = {
    'Mild_Demented': 0,
    'Moderate_Demented': 1,
    'Non_Demented': 2,
    'Very_Mild_Demented': 3
}

# PART 1: DATA PROCESSING FUNCTIONS
def load_folder_dataset(base_dir, target_size=(IMG_SIZE, IMG_SIZE)):
    """Load and preprocess images from a folder-based dataset"""
    dataset = []
    
    # Map folder names to label system
    folder_to_label = {
        "No Impairment": 2,           # Non Demented
        "Very Mild Impairment": 3,    # Very Mild Demented
        "Mild Impairment": 0,         # Mild Demented
        "Moderate Impairment": 1      # Moderate Demented
    }
    
    for folder_name, label in folder_to_label.items():
        folder_path = os.path.join(base_dir, folder_name)
        
        if os.path.exists(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, img_name)
                    try:
                        # Load and convert image to grayscale
                        img = Image.open(img_path).convert('L')
                        
                        # Resize to standard size
                        img = img.resize(target_size)
                        
                        # Convert to numpy array and normalize
                        img_arr = np.array(img) / 255.0
                        
                        # Add to dataset
                        dataset.append({
                            'label': label,
                            'image': img_arr,
                            'source': 'folder_dataset',
                            'original_path': img_path
                        })
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        else:
            print(f"Folder not found: {folder_path}")
    
    return dataset

def bytes_to_image(img_bytes):
    """Convert bytes to image for parquet files"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

def load_parquet_data(file_path):
    """Load and preprocess images from parquet files"""
    dataset = []
    
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        for i, row in df.iterrows():
            img_bytes = row['image']['bytes']
            img = bytes_to_image(img_bytes)
            
            # Resize image to standard size
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalize pixel values to [0,1]
            img_normalized = img_resized / 255.0
            
            dataset.append({
                'image': img_normalized,
                'label': row['label'],
                'source': 'parquet_dataset'
            })
            
    except Exception as e:
        print(f"Error loading parquet file {file_path}: {e}")
    
    return dataset

def resampling_data(train_data, test_data):
    """Resample data to ensure balanced class distribution"""
    all_data = train_data + test_data
    all_data_df = pd.DataFrame(all_data)  # converting to pd dataframe for stratification
    
    # Split data into train, test, and validation sets with stratification
    train, test = train_test_split(all_data_df, test_size=0.2, random_state=42, stratify=all_data_df['label'])
    train, val = train_test_split(train, test_size=0.125, random_state=42, stratify=train['label'])
    
    # Convert back to dictionaries
    resampled_train = train.to_dict('records')
    resampled_test = test.to_dict('records')
    resampled_val = val.to_dict('records')
    
    return resampled_train, resampled_test, resampled_val

def save_images(data, output_dir, output_subdir):
    """Save processed images to specified directory"""
    # Count images saved per class
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i, item in enumerate(data):
        label = item['label']
        img_array = item['image']
        
        # Convert back to 0-255 range for saving
        img_array_255 = (img_array * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(img_array_255)
        
        # Generate filename
        class_name = classes[label]
        class_counts[label] += 1
        
        # Generate unique filename based on source and class count
        source_prefix = 'folder' if item.get('source') == 'folder_dataset' else 'parquet'
        filename = f"{source_prefix}_{class_name}_{class_counts[label]:04d}.png"
        
        # Save path
        save_path = os.path.join(output_dir, output_subdir, class_name, filename)
        
        # Save image
        img.save(save_path)
    
    return class_counts

def process_data(folder_train_path, folder_test_path, train_parquet_path, test_parquet_path, output_dir):
    """Process and combine datasets from both folder and parquet sources"""
    print("\n=== PART 1: DATA PROCESSING ===")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    
    # Create class directories
    for class_name in classes.values():
        os.makedirs(os.path.join(output_dir, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val", class_name), exist_ok=True)
    
    # Load folder-based datasets
    folder_train_data = []
    folder_test_data = []
    if folder_train_path and os.path.exists(folder_train_path):
        print("Loading Folder Dataset (Train)...")
        folder_train_data = load_folder_dataset(folder_train_path)
        print(f"Loaded {len(folder_train_data)} images from folder train")
    
    if folder_test_path and os.path.exists(folder_test_path):
        print("Loading Folder Dataset (Test)...")
        folder_test_data = load_folder_dataset(folder_test_path)
        print(f"Loaded {len(folder_test_data)} images from folder test")
    
    # Load parquet datasets
    parquet_train_data = []
    parquet_test_data = []
    if train_parquet_path and os.path.exists(train_parquet_path):
        print("Loading Parquet Dataset (Train)...")
        parquet_train_data = load_parquet_data(train_parquet_path)
        print(f"Loaded {len(parquet_train_data)} images from parquet train")
    
    if test_parquet_path and os.path.exists(test_parquet_path):
        print("Loading Parquet Dataset (Test)...")
        parquet_test_data = load_parquet_data(test_parquet_path)
        print(f"Loaded {len(parquet_test_data)} images from parquet test")
    
    # Combine datasets
    combined_train_data = folder_train_data + parquet_train_data
    combined_test_data = folder_test_data + parquet_test_data
    
    if not combined_train_data and not combined_test_data:
        print("No data found. Please check input paths.")
        return None
    
    print(f"Combined training data: {len(combined_train_data)} images")
    print(f"Combined test data: {len(combined_test_data)} images")
    
    # Resample data for balanced distribution
    resampled_train_data, resampled_test_data, resampled_val_data = resampling_data(
        combined_train_data, combined_test_data
    )
    
    print(f"Resampled training data: {len(resampled_train_data)} images")
    print(f"Resampled test data: {len(resampled_test_data)} images")
    print(f"Resampled validation data: {len(resampled_val_data)} images")
    
    # Save images
    print("Saving processed images...")
    train_counts = save_images(resampled_train_data, output_dir, "train")
    test_counts = save_images(resampled_test_data, output_dir, "test")
    val_counts = save_images(resampled_val_data, output_dir, "val")
    
    # Display distribution
    print("\nTraining data distribution:")
    for label, count in train_counts.items():
        print(f"  {classes[label]}: {count} images")
    
    print("\nTest data distribution:")
    for label, count in test_counts.items():
        print(f"  {classes[label]}: {count} images")
    
    print("\nValidation data distribution:")
    for label, count in val_counts.items():
        print(f"  {classes[label]}: {count} images")
    
    print("\nData processing complete!")
    return output_dir

# PART 2: FEATURE EXTRACTION FUNCTIONS
def normalize_mri_for_ventricles(item):
    """Apply normalization techniques to extract ventricle features"""
    image = item['image']
    
    # 1. Simple normalization - just use the original normalized image
    item['image_normalized'] = image
    
    # 2. Ventricle enhancement
    # Create version optimized for dark ventricle regions
    img_uint8 = (item['image_normalized'] * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_uint8)
    item['image_enhanced'] = enhanced / 255.0
    
    # Create inverted version to highlight ventricles
    inverted = 1 - item['image_normalized']
    item['image_ventricle_focus'] = inverted
    
    # 3. Ventricle segmentation using Otsu thresholding
    otsu_thresh, _ = cv2.threshold(
        img_uint8, 
        0, 
        255, 
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    _, ventricle_mask = cv2.threshold(
        img_uint8, 
        int(otsu_thresh * 0.5),   
        255, 
        cv2.THRESH_BINARY_INV
    )
    
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    ventricle_mask = cv2.morphologyEx(ventricle_mask, cv2.MORPH_OPEN, kernel)
    ventricle_mask = cv2.morphologyEx(ventricle_mask, cv2.MORPH_CLOSE, kernel)
    
    item['ventricle_mask'] = ventricle_mask / 255.0
    
    return item

def load_and_normalize_dataset(base_dir):
    """Load images from processed dataset and apply feature extraction"""
    normalized_data = []
    
    # Process train, test, and val folders
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(base_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        # Process each class folder
        for class_name, label in class_labels.items():
            class_dir = os.path.join(split_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} does not exist, skipping...")
                continue
                
            # Process each image in the class folder
            print(f"Processing {split}/{class_name}...")
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    
                    try:
                        # Load image
                        img = Image.open(img_path).convert('L')
                        img_array = np.array(img) / 255.0  # Normalize to [0,1]
                        
                        # Process and store
                        item = {
                            'original_path': img_path,
                            'label': label,
                            'class_name': class_name,
                            'split': split,
                            'image': img_array,
                            'dataset': 'folder_dataset' if 'folder' in img_file else 'parquet_dataset'
                        }
                        
                        # Apply normalizations
                        normalized_item = normalize_mri_for_ventricles(item)
                        normalized_data.append(normalized_item)
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    
    # Split back into train, test, and val
    train_data = [item for item in normalized_data if item['split'] == 'train']
    test_data = [item for item in normalized_data if item['split'] == 'test']
    val_data = [item for item in normalized_data if item['split'] == 'val']
    
    return train_data, test_data, val_data

def save_normalized_dataset(base_dir, train_data, test_data, val_data):
    """Save normalized images to output directory"""
    # Create output directories
    output_dir = os.path.join(base_dir, 'normalized')
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    
    for split, data in [('train', train_data), ('test', test_data), ('val', val_data)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create class directories
        class_names = set(item['class_name'] for item in data)
        for class_name in class_names:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Save normalized images
        for item in tqdm(data, desc=f"Saving {split} images"):
            # Generate filename
            original_filename = os.path.basename(item['original_path'])
            base_name = os.path.splitext(original_filename)[0]
            
            # Define paths for different normalizations
            class_dir = os.path.join(split_dir, item['class_name'])
            
            # Save normalized image
            norm_img = (item['image_normalized'] * 255).astype(np.uint8)
            norm_path = os.path.join(class_dir, f"{base_name}_norm.png")
            Image.fromarray(norm_img).save(norm_path)
            
            # Save enhanced image
            enhanced_img = (item['image_enhanced'] * 255).astype(np.uint8)
            enhanced_path = os.path.join(class_dir, f"{base_name}_enhanced.png")
            Image.fromarray(enhanced_img).save(enhanced_path)
            
            # Save ventricle focused image
            ventricle_img = (item['image_ventricle_focus'] * 255).astype(np.uint8)
            ventricle_path = os.path.join(class_dir, f"{base_name}_ventricle.png")
            Image.fromarray(ventricle_img).save(ventricle_path)
            
            # Save ventricle mask
            mask_img = (item['ventricle_mask'] * 255).astype(np.uint8)
            mask_path = os.path.join(class_dir, f"{base_name}_mask.png")
            Image.fromarray(mask_img).save(mask_path)
            
            saved_count += 1
    
    print(f"Total of {saved_count} images processed and saved with normalizations")

def extract_features(base_dir):
    """Extract features from processed images"""
    print("\n=== PART 2: FEATURE EXTRACTION ===")
    
    print("Loading and normalizing dataset...")
    train_data, test_data, val_data = load_and_normalize_dataset(base_dir)
    
    if not train_data and not test_data and not val_data:
        print("No data found for feature extraction. Please check the processed data.")
        return False
    
    print(f"Processed {len(train_data)} training images, {len(test_data)} test images, and {len(val_data)} validation images")
    
    # Class distribution summary
    for split_name, split_data in [("Training", train_data), ("Test", test_data), ("Validation", val_data)]:
        print(f"\n{split_name} set class distribution:")
        class_counts = {}
        for item in split_data:
            class_name = item['class_name']
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images ({count/len(split_data)*100:.1f}%)")
    
    # Save normalized dataset
    print("\nSaving normalized dataset...")
    save_normalized_dataset(base_dir, train_data, test_data, val_data)
    
    print("\nFeature extraction complete!")
    return True

# PART 3: MODEL TRAINING FUNCTIONS
class VentricleDataset(Dataset):
    """PyTorch dataset for ventricle mask images"""
    def __init__(self, base_dir, split='train'):
        self.data_dir = os.path.join(base_dir, "normalized", split)
        self.images = []
        self.labels = []
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith('mask.png'):
                    self.images.append(os.path.join(class_dir, img_file))
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images for {split} split")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        return TF.to_tensor(img), torch.tensor(self.labels[idx], dtype=torch.long)

class VentricleCNN(nn.Module):
    """CNN model for ventricle-based dementia classification"""
    def __init__(self):
        super().__init__()
        # Two convolutional blocks
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        
        # Flattened size after 2 pooling layers (4x reduction)
        self.fc_input_size = 128 * (IMG_SIZE//4) * (IMG_SIZE//4)
        
        # Classifier
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.out = nn.Linear(128, N_CLASSES)
        
    def forward(self, x):
        x = self.batchnorm1(self.pool1(F.relu(self.conv1(x))))
        x = self.batchnorm2(self.pool2(F.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        return self.out(x)

def evaluate(model, loader, criterion):
    """Evaluate model performance"""
    model.eval()
    correct, total, loss = 0, 0, 0
    class_correct = [0] * N_CLASSES
    class_total = [0] * N_CLASSES
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for label, pred in zip(labels, predicted):
                class_correct[label] += (pred == label).item()
                class_total[label] += 1
    
    accuracy = correct / total
    avg_loss = loss / len(loader)
    class_acc = {CLASS_NAMES[i]: class_correct[i]/class_total[i] 
                for i in range(N_CLASSES) if class_total[i] > 0}
    
    return accuracy, avg_loss, class_acc

def train_model(model, train_loader, val_loader, output_dir, epochs=10, lr=0.001):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        val_acc, val_loss, class_acc = evaluate(model, val_loader, criterion)
        train_acc = train_correct / train_total
        
        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        
        scheduler.step(val_loss)
    
    # Save training history for future reference
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f)
    
    return model, history, best_acc

def train_and_evaluate(base_dir, epochs=10):
    """Train model and evaluate performance using only training and validation data"""
    print("\n=== PART 3: MODEL TRAINING ===")
    
    # Check if normalized data exists
    norm_dir = os.path.join(base_dir, "normalized")
    if not os.path.exists(norm_dir):
        print("Normalized data not found. Run feature extraction first.")
        return None
    
    # Create output directory
    output_dir = os.path.join(base_dir, "model_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    train_set = VentricleDataset(base_dir, 'train')
    val_set = VentricleDataset(base_dir, 'val')
    
    if len(train_set) == 0 or len(val_set) == 0:
        print("Insufficient data for training. Check normalized dataset.")
        return None
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    print("Initializing model...")
    model = VentricleCNN()
    
    print(f"Training model for {epochs} epochs...")
    model, history, best_val_acc = train_model(model, train_loader, val_loader, output_dir, epochs=epochs)
    
    # Final validation evaluation
    print("\nFinal evaluation on validation set...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    val_acc, val_loss, class_acc = evaluate(model, val_loader, nn.CrossEntropyLoss())
    
    # Print only the essential output: validation accuracy and model path
    print(f"\n=== RESULTS ===")
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model.pt')}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    return val_acc
    
    return results

def main():
    """Main function to run the entire pipeline"""
    parser = argparse.ArgumentParser(description="MRI Dementia Classification Pipeline")
    
    # Data paths - support for zip files and direct paths
    parser.add_argument("--zip_file1", type=str, default="", help="First zip file containing dataset")
    parser.add_argument("--zip_file2", type=str, default="", help="Second zip file containing dataset")
    parser.add_argument("--folder_train", type=str, default="", help="Path to folder-based training dataset")
    parser.add_argument("--folder_test", type=str, default="", help="Path to folder-based test dataset")
    parser.add_argument("--parquet_train", type=str, default="", help="Path to parquet training dataset")
    parser.add_argument("--parquet_test", type=str, default="", help="Path to parquet test dataset")
    
    # Output and model parameters
    parser.add_argument("--output_dir", type=str, default="Combined_MRI_Dataset", help="Output directory for processed data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Extract zip files if provided
    temp_dirs = []
    
    # Process first zip file if provided
    if args.zip_file1 and os.path.exists(args.zip_file1):
        temp_dir1 = extract_zip_file(args.zip_file1)
        temp_dirs.append(temp_dir1)
        
        # Auto-detect dataset structure in first zip
        ft1, fs1, pt1, ps1 = detect_dataset_structure(temp_dir1)
        
        # Use these paths if the command line args aren't provided
        if not args.folder_train and ft1:
            args.folder_train = ft1
        if not args.folder_test and fs1:
            args.folder_test = fs1
        if not args.parquet_train and pt1:
            args.parquet_train = pt1
        if not args.parquet_test and ps1:
            args.parquet_test = ps1
    
    # Process second zip file if provided
    if args.zip_file2 and os.path.exists(args.zip_file2):
        temp_dir2 = extract_zip_file(args.zip_file2)
        temp_dirs.append(temp_dir2)
        
        # Auto-detect dataset structure in second zip
        ft2, fs2, pt2, ps2 = detect_dataset_structure(temp_dir2)
        
        # Only use these paths if the corresponding paths weren't found in the first zip
        # and weren't provided as command line args
        if not args.folder_train and ft2:
            args.folder_train = ft2
        if not args.folder_test and fs2:
            args.folder_test = fs2
        if not args.parquet_train and pt2:
            args.parquet_train = pt2
        if not args.parquet_test and ps2:
            args.parquet_test = ps2
    
    # Step 1: Data Processing
    processed_dir = process_data(
        args.folder_train, 
        args.folder_test, 
        args.parquet_train, 
        args.parquet_test, 
        args.output_dir
    )
    
    if not processed_dir:
        print("Data processing failed. Exiting.")
        # Clean up temp dirs
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return
    
    # Step 2: Feature Extraction
    features_extracted = extract_features(processed_dir)
    
    if not features_extracted:
        print("Feature extraction failed. Exiting.")
        # Clean up temp dirs
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return
    
    # Step 3: Model Training
    val_acc = train_and_evaluate(processed_dir, args.epochs)
    
    # Clean up temp dirs
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time/60:.2f} minutes")
    
    # Final output - just the essential information
    if val_acc:
        print(f"\n=== FINAL OUTPUT ===")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Model Path: {os.path.join(processed_dir, 'model_output', 'best_model.pt')}")

if __name__ == "__main__":
    main()