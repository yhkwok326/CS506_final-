import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import argparse
import cv2

# Constants
N_CLASSES = 4
CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
IMG_SIZE = (128, 128)

class VentricleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Two convolutional blocks
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        
        # Calculate flattened size after 2 pooling layers (4x reduction)
        self.fc_input_size = 128 * (IMG_SIZE[0]//4) * (IMG_SIZE[1]//4)
        
        # Classifier - fixed the input size based on calculated dimensions
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.out = nn.Linear(128, N_CLASSES)
    
    def forward(self, x):
        x = self.batchnorm1(self.pool1(F.relu(self.conv1(x))))
        x = self.batchnorm2(self.pool2(F.relu(self.conv2(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        return self.out(x)

def load_model(model_path):
    """Load a trained model from disk"""
    model = VentricleCNN()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found.")
        exit(1)
    
    return model

def normalize_mri_for_ventricles(image):
    """Apply the same preprocessing steps from the training pipeline"""
    # Ensure image is normalized to [0,1]
    if image.max() > 1.0:
        image = image / 255.0
        
    # 1. Simple normalization
    normalized = image
    
    # 2. Convert to uint8 for preprocessing
    img_uint8 = (image * 255).astype(np.uint8)
    
    # 3. Ventricle segmentation using Otsu thresholding
    otsu_thresh, _ = cv2.threshold(
        img_uint8, 
        0, 
        255, 
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    
    _, ventricle_mask = cv2.threshold(
        img_uint8, 
        int(otsu_thresh * 0.5),  # 50% of the Otsu threshold
        255, 
        cv2.THRESH_BINARY_INV
    )
    
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    ventricle_mask = cv2.morphologyEx(ventricle_mask, cv2.MORPH_OPEN, kernel)
    ventricle_mask = cv2.morphologyEx(ventricle_mask, cv2.MORPH_CLOSE, kernel)
    
    return ventricle_mask / 255.0

def preprocess_image(image_path):
    """Preprocess an image for model prediction using same pipeline as training"""
    try:
        # Load and convert to grayscale
        img = Image.open(image_path).convert('L')
        # Resize to model's expected input size
        img = img.resize((IMG_SIZE[0], IMG_SIZE[1]))
        
        # Convert to numpy array
        img_array = np.array(img)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
            
        # Apply ventricle mask preprocessing
        ventricle_mask = normalize_mri_for_ventricles(img_array)
        
        # Convert to tensor and add batch dimension - ENSURE FLOAT32 TYPE
        tensor = TF.to_tensor(ventricle_mask).unsqueeze(0).float()  # Explicitly convert to float32
            
        return tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_class(model, image_tensor, device='cpu'):
    """Predict class from preprocessed image tensor"""
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Ensure consistent data type
        image_tensor = image_tensor.to(device).float()  # Explicitly convert to float32
        outputs = model(image_tensor)
        
        # Get probabilities
        probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get predicted class
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence = probabilities[predicted_idx].item()
        
    return predicted_class, confidence

def interactive_prediction():
    """Run an interactive prediction session"""
    parser = argparse.ArgumentParser(description='Simple MRI Classification Tool')
    parser.add_argument('--image', type=str, help='Path to the MRI image to classify')
    parser.add_argument('--model', type=str, default='best_model.pt',
                      help='Path to the trained model (default: best_model.pt)')
    args = parser.parse_args()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model)
    # Ensure model is in float32 precision
    model = model.float()
    model.to(device)
    model.eval()
    
    if args.image:
        # Process a single image from command line argument
        process_single_image(args.image, model, device)
    else:
        # Interactive mode
        while True:
            image_path = input("\nEnter the path to an MRI image (or 'q' to quit): ")
            if image_path.lower() == 'q':
                break
            
            process_single_image(image_path, model, device)

def process_single_image(image_path, model, device):
    """Process a single image and print the predicted class"""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    
    if image_tensor is not None:
        # Make prediction
        predicted_class, confidence = predict_class(model, image_tensor, device)
        
        # Print just the class name and confidence
        print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.4f})")
        
        return predicted_class
    else:
        print("Failed to process the image.")
        return None

if __name__ == "__main__":
    interactive_prediction()