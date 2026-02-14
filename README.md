# Digit Recognition CNN Model

A deep learning project for recognizing handwritten digits (0-9) using Convolutional Neural Networks (CNN) with TensorFlow/Keras, optimized for deployment with TensorFlow Lite.

## Overview

This project implements a custom CNN model trained to recognize handwritten digits from images. The model is trained on a custom dataset, includes data augmentation techniques, and is exported in TensorFlow Lite format for efficient deployment on mobile and edge devices.

## Features

- Custom CNN architecture with batch normalization
- Data augmentation for improved model generalization
- Model quantization for reduced size and faster inference
- TensorFlow Lite export for mobile deployment
- Visual prediction interface with confidence scores

## Requirements

```
tensorflow
numpy
matplotlib
pillow
scikit-learn
ai-edge-litert
```

## Project Structure

The notebook follows these main steps:

### 1. Environment Setup and Data Loading

**Libraries imported:**
- TensorFlow/Keras for deep learning
- NumPy and PIL for image processing
- Matplotlib for visualization
- Scikit-learn for data splitting

**Key configurations:**
- GPU acceleration enabled (T4)
- Image input size: 20×32 pixels
- Number of classes: 10 (digits 0-9)

### 2. Data Preprocessing

**Steps:**
1. Extract dataset from zip file (`data.zip`)
2. Resize all images to standard dimensions (20×32)
3. Convert images to numpy arrays (float32)
4. Apply one-hot encoding to labels
5. Shuffle the dataset

**Label mapping:**
- Digits 0-9 are mapped to their respective indices
- Special handling for 'N' label → mapped to class 10

### 3. Train-Test Split

- Configurable training percentage (default: 0.0 for using all data)
- When `Training_Percentage > 0`, data is split into training and validation sets
- Uses scikit-learn's `train_test_split`

### 4. Model Architecture

**CNN Architecture:**
```
Input Layer: (32, 20, 3)
  ↓
Batch Normalization
  ↓
Conv2D (32 filters, 3×3) + ReLU → MaxPool2D (2×2)
  ↓
Conv2D (32 filters, 3×3) + ReLU → MaxPool2D (2×2)
  ↓
Conv2D (32 filters, 3×3) + ReLU → MaxPool2D (2×2)
  ↓
Flatten
  ↓
Dense (256 units) + ReLU
  ↓
Dropout (0.5)
  ↓
Dense (10 units) + Softmax
```

**Model compilation:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

### 5. Data Augmentation

**ImageDataGenerator parameters:**
- Width shift: ±1 pixel
- Height shift: ±1 pixel
- Brightness: 0.8-1.2 range
- Zoom: ±30%
- Rotation: ±5 degrees

**Training configuration:**
- Batch size: 4
- Epochs: 100
- Validation data (if split enabled)

### 6. Model Training

The model is trained using the augmented data generator with the specified hyperparameters.

### 7. Model Export

**Two export formats:**

1. **SavedModel format** (`test/`)
   - Full TensorFlow format
   - Used for conversion to TFLite

2. **TensorFlow Lite models:**
   - `4TrainedModel.tflite` - Quantized model
     - Optimizations: DEFAULT
     - Representative dataset for quantization
     - Smaller file size, faster inference
   
   - `5TrainedModel.tflite` - Standard TFLite model
     - No quantization (optional)
     - Better accuracy potential

### 8. Model Inference

**Prediction pipeline:**
1. Load TFLite interpreter
2. Preprocess input image:
   - Resize to 20×32
   - Convert to RGB
   - Normalize pixel values
   - Add batch dimension
3. Run inference
4. Get predicted class and confidence scores
5. Visualize results with matplotlib

**Visualization includes:**
- Original image display
- Predicted digit
- Confidence score
- Full probability distribution

## Usage

### Training the Model

1. Prepare your dataset in the required format
2. Update the `input_folder` path to your dataset
3. Run all cells sequentially
4. The model will be saved in TFLite format

### Making Predictions

```python
# Load and predict on a single image
predict_and_plot("/path/to/your/image.jpg")
```

The function will display:
- The input image
- Predicted digit
- Confidence percentage
- Probability distribution across all classes

## Model Performance

The model uses:
- **Batch Normalization** for stable training
- **Dropout (0.5)** to prevent overfitting
- **Data Augmentation** to improve generalization
- **Multiple Conv layers** for feature extraction

## Optimization

The quantized TFLite model (`4TrainedModel.tflite`) offers:
- Reduced model size (typically 4x smaller)
- Faster inference on mobile/edge devices
- Minimal accuracy loss
- INT8 quantization with representative dataset

## Notes

- The project is designed to run on Google Colab with GPU support
- Dataset is stored in Google Drive for persistence
- Model can be easily deployed to mobile apps using TensorFlow Lite
- Adjust `epochs` and `batch_size` based on your dataset size

## Future Improvements

- Implement early stopping and model checkpointing
- Add learning rate scheduling
- Experiment with different architectures (ResNet, MobileNet)
- Add confusion matrix and detailed performance metrics
- Implement real-time digit recognition from camera feed

## Author

Yassine OUJAMA