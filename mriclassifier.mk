# Makefile for MRI Dementia Classification Project

# ===== Configuration Variables =====
# Script files
TRAIN_SCRIPT := mri_dementia_classification.py
PREDICT_SCRIPT := simple_mri_classifier.py
REQ := requirements.txt

# Dataset files 
ZIP_FILE1 := "Alzheimer MRI Disease Classification Dataset-2.zip"
ZIP_FILE2 := "Combined Dataset.zip"
OUTPUT_DIR := Combined_MRI_Dataset

# Model files
MODEL := best_model.pt
IMAGE := 14.png

# ===== Targets =====
.PHONY: all run deps train predict clean

# Default target - runs full pipeline
all: run

# Complete workflow
run: deps train predict

# Install Python dependencies
deps:
	@echo "Installing dependencies..."
	pip install -r $(REQ)

# Train the model
train:
	@echo "Training model..."
	python $(TRAIN_SCRIPT) \
		--zip_file1 $(ZIP_FILE1) \
		--zip_file2 $(ZIP_FILE2) \
		--output_dir $(OUTPUT_DIR) \
		--epochs 10

# Make predictions
predict:
	@echo "Running predictions..."
	python $(PREDICT_SCRIPT) \
		--image $(IMAGE) \
		--model $(MODEL)

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf \
		__pycache__ \
		*.pyc \
		$(OUTPUT_DIR) \
		$(MODEL)