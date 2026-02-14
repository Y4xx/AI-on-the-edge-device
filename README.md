# Digit Recognition CNN Model

A deep learning project for recognizing handwritten digits (0-9) using Convolutional Neural Networks (CNN) with TensorFlow/Keras, optimized for deployment with TensorFlow Lite.

## Overview

This project implements a custom CNN model trained to recognize handwritten digits from images. The model is trained on a custom dataset, includes data augmentation techniques, and is exported in TensorFlow Lite format for efficient deployment on mobile and edge devices.

**New in v1.0:** Complete refactoring with modular architecture, CLI tools, comprehensive documentation, and extensive improvements!

## Features

### Core Features
- Custom CNN architecture with batch normalization and improved dropout strategy
- Data augmentation for improved model generalization
- Model quantization for reduced size and faster inference
- TensorFlow Lite export for mobile deployment
- Visual prediction interface with confidence scores

### New Features (v1.0)
- **Modular Architecture**: Clean separation of concerns with organized modules
- **Configuration Management**: YAML-based configuration for easy experimentation
- **CLI Tools**: Command-line interfaces for training, inference, and analysis
- **Advanced Callbacks**: Early stopping, model checkpointing, learning rate scheduling
- **Comprehensive Evaluation**: Confusion matrix, precision/recall/F1 metrics, visualization
- **Batch Prediction**: Process multiple images efficiently
- **Data Visualization**: Explore dataset statistics and distribution
- **Model Comparison**: Compare different models by size, speed, and accuracy
- **Unit Tests**: Automated testing for core functionality
- **Production Ready**: Proper package structure with setup.py

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Y4xx/AI-on-the-edge-device.git
cd AI-on-the-edge-device

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Training

```bash
# Using the training script
python train.py --config config.yaml --epochs 50 --batch-size 8 --output-dir output

# Or using the CLI tool (if installed with setup.py)
digit-train --config config.yaml --epochs 50
```

### Inference

```bash
# Single image prediction with visualization
python predict.py path/to/image.jpg --model 5TrainedModel.tflite --visualize

# Batch prediction
python predict.py image1.jpg image2.jpg image3.jpg --batch --output results.json

# Using the CLI tool
digit-predict path/to/image.jpg --visualize
```

### Data Visualization

```bash
python visualize_data.py /path/to/dataset --show-plots
```

### Model Comparison

```bash
python compare_models.py 4TrainedModel.tflite 5TrainedModel.tflite --data-folder /path/to/test/data
```

## Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=10.0.0
scikit-learn>=1.3.0
tensorflow>=2.15.0
pyyaml>=6.0
seaborn>=0.12.0
```

## Project Structure

```
AI-on-the-edge-device/
├── src/                          # Source code package
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/                   # Model architecture and training
│   │   ├── __init__.py
│   │   ├── model_builder.py
│   │   └── trainer.py
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── config_loader.py      # Configuration management
│       ├── evaluation.py         # Model evaluation and metrics
│       ├── inference.py          # TFLite inference
│       ├── logger.py             # Logging setup
│       └── model_export.py       # Model export utilities
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_config_loader.py
│   ├── test_data_loader.py
│   └── test_model_builder.py
├── train.py                      # Main training script
├── predict.py                    # Inference CLI tool
├── visualize_data.py             # Data visualization tool
├── compare_models.py             # Model comparison utility
├── run_tests.py                  # Test runner
├── config.yaml                   # Configuration file
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
├── train.ipynb                   # Original Jupyter notebook
├── README.md                     # This file
├── USAGE.md                      # Detailed usage guide
├── .gitignore                    # Git ignore rules
└── data.zip                      # Dataset archive
```

## Documentation

- **[USAGE.md](USAGE.md)**: Comprehensive usage guide with examples
- **README.md**: This file - project overview and quick start

## Configuration

All training parameters can be configured via `config.yaml`:

```yaml
data:
  image_width: 20
  image_height: 32
  num_classes: 10

training:
  batch_size: 4
  epochs: 100
  validation_split: 0.2

model:
  conv_filters: [32, 32, 32]
  dense_units: 256
  dropout_rate: 0.5

optimizer:
  name: "adam"
  learning_rate: 0.001

callbacks:
  early_stopping:
    enabled: true
    patience: 10
  model_checkpoint:
    enabled: true
    save_best_only: true
```

See [USAGE.md](USAGE.md) for complete configuration options.

## Testing

Run unit tests:

```bash
python run_tests.py
```

Or run individual test files:

```bash
python -m unittest tests.test_data_loader
python -m unittest tests.test_model_builder
python -m unittest tests.test_config_loader
```

## Model Architecture

The CNN model architecture includes:

- **Input Layer**: 32×20×3 (height, width, channels)
- **Batch Normalization**: For stable training
- **3 Convolutional Blocks**: Each with:
  - Conv2D layer (32 filters, 3×3 kernel)
  - Batch Normalization
  - ReLU activation
  - MaxPooling2D (2×2)
  - Dropout (configurable)
- **Flatten Layer**
- **Dense Layer**: 256 units with ReLU
- **Dropout**: 0.5 (configurable)
- **Output Layer**: 10 units with Softmax

**Key Improvements:**
- Added dropout after convolutional layers (not just dense)
- Batch normalization after each conv layer for stability
- Configurable optimizer (Adam, Adadelta, SGD, RMSprop)
- Learning rate scheduling support

## Training Features

### Data Augmentation
- Width/height shift
- Brightness adjustment
- Zoom
- Rotation
- Configurable via YAML

### Callbacks
- **Early Stopping**: Prevent overfitting
- **Model Checkpointing**: Save best model
- **Reduce Learning Rate on Plateau**: Adaptive learning rate

### Evaluation Metrics
- Accuracy and Loss curves
- Confusion Matrix
- Per-class Precision, Recall, F1-Score
- Sample prediction visualization

## Model Export

The training pipeline exports models in multiple formats:

1. **SavedModel** (`test/`): Full TensorFlow format
2. **Quantized TFLite** (`4TrainedModel.tflite`): 
   - INT8 quantization
   - ~4x size reduction
   - Faster inference on mobile/edge devices
3. **Standard TFLite** (`5TrainedModel.tflite`):
   - No quantization
   - Better accuracy potential

## CLI Tools

### 1. Training Tool (`train.py`)
```bash
python train.py --config config.yaml --epochs 50 --batch-size 8
```

Options:
- `--config`: Configuration file path
- `--data-folder`: Override data folder
- `--epochs`: Number of epochs
- `--batch-size`: Batch size
- `--output-dir`: Output directory

### 2. Inference Tool (`predict.py`)
```bash
python predict.py image.jpg --model 5TrainedModel.tflite --visualize
```

Options:
- `--model`: TFLite model path
- `--visualize`: Show prediction visualization
- `--batch`: Batch prediction mode
- `--output`: Save results to JSON

### 3. Data Visualization (`visualize_data.py`)
```bash
python visualize_data.py /path/to/data --show-plots
```

Generates:
- Class distribution plot
- Sample images
- Image statistics (pixel values, brightness)

### 4. Model Comparison (`compare_models.py`)
```bash
python compare_models.py model1.tflite model2.tflite --data-folder /path/to/test/data
```

Compares:
- Model file sizes
- Inference speed
- Accuracy metrics
- Efficiency scores

## API Usage

### Training

```python
from src.data.data_loader import load_images_from_folder, split_data
from src.models.model_builder import create_digit_recognition_model, compile_model
from src.models.trainer import train_model, create_data_generator, create_callbacks
from src.utils.evaluation import evaluate_model
from src.utils.model_export import export_models

# Load data
x_data, y_data = load_images_from_folder("/path/to/data")
X_train, X_val, y_train, y_val = split_data(x_data, y_data, test_size=0.2)

# Create and compile model
model = create_digit_recognition_model()
model = compile_model(model, optimizer_name='adam', learning_rate=0.001)

# Train
datagen = create_data_generator(config['augmentation'])
callbacks = create_callbacks(config['callbacks'])
history = train_model(model, X_train, y_train, X_val, y_val, 
                     batch_size=4, epochs=100, datagen=datagen, callbacks=callbacks)

# Evaluate and export
metrics = evaluate_model(model, X_val, y_val)
export_models(model, X_train)
```

### Inference

```python
from src.utils.inference import TFLiteInference

# Load model
inference = TFLiteInference("5TrainedModel.tflite")

# Predict single image
predicted_class, confidence, probabilities = inference.predict_from_image("test.jpg")
print(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")

# Visualize prediction
inference.predict_and_visualize("test.jpg", show_plot=True)

# Batch prediction
results = inference.batch_predict(["img1.jpg", "img2.jpg", "img3.jpg"])
```

## Improvements Over Original

### Code Quality
✅ Modular architecture with separation of concerns  
✅ Type hints for better code documentation  
✅ Comprehensive error handling and logging  
✅ Unit tests for core functionality  
✅ Proper package structure with setup.py  

### Features
✅ Configuration management via YAML  
✅ CLI tools for common tasks  
✅ Batch prediction capability  
✅ Model comparison utility  
✅ Data visualization tools  
✅ Comprehensive evaluation metrics  

### Model Training
✅ Improved dropout strategy (conv + dense layers)  
✅ Early stopping to prevent overfitting  
✅ Model checkpointing to save best model  
✅ Learning rate scheduling  
✅ Configurable optimizer  
✅ Training history visualization  

### Deployment
✅ Better TFLite export with quantization  
✅ Inference API for easy integration  
✅ Batch prediction for efficiency  
✅ Model size and speed comparison  

## Best Practices

1. **Always use validation data** to monitor overfitting
2. **Enable early stopping** for efficient training
3. **Use quantized models** for deployment on edge devices
4. **Monitor training curves** to detect issues
5. **Test on sample images** before batch processing
6. **Use configuration files** for reproducibility

## Troubleshooting

See [USAGE.md](USAGE.md) for detailed troubleshooting guide covering:
- Out of memory errors
- Model convergence issues
- Overfitting problems
- TFLite model size optimization

## Future Enhancements

- [ ] Add more advanced architectures (ResNet, MobileNet)
- [ ] Implement data pipeline optimization
- [ ] Add distributed training support
- [ ] Create web interface for inference
- [ ] Add model pruning and optimization
- [ ] Implement real-time camera inference
- [ ] Add more data augmentation techniques
- [ ] Create Docker container for deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Yassine OUJAMA