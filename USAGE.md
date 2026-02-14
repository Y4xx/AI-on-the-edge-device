# Advanced Usage Guide

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)

## Installation

### Using pip
```bash
pip install -r requirements.txt
```

### Using setup.py
```bash
pip install -e .
```

This will install the package in editable mode and create command-line tools:
- `digit-train`: Training script
- `digit-predict`: Inference script

## Quick Start

### 1. Training a Model

**Using the Python script:**
```bash
python train.py --config config.yaml --epochs 50 --batch-size 8
```

**Using the installed CLI tool:**
```bash
digit-train --config config.yaml --epochs 50 --output-dir my_output
```

**Command-line options:**
- `--config`: Path to configuration file (default: config.yaml)
- `--data-folder`: Override data folder path
- `--epochs`: Override number of epochs
- `--batch-size`: Override batch size
- `--output-dir`: Directory for output files (default: output)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### 2. Running Inference

**Single image prediction with visualization:**
```bash
python predict.py path/to/image.jpg --model 5TrainedModel.tflite --visualize
```

**Batch prediction on multiple images:**
```bash
python predict.py image1.jpg image2.jpg image3.jpg --batch --output results.json
```

**Using the installed CLI tool:**
```bash
digit-predict path/to/image.jpg --model 5TrainedModel.tflite --visualize
```

## Training

### Configuration File

The `config.yaml` file allows you to customize all aspects of training:

```yaml
# Data Configuration
data:
  output_folder: "/path/to/your/dataset"
  image_width: 20
  image_height: 32
  num_classes: 10

# Training Configuration
training:
  batch_size: 4
  epochs: 100
  training_percentage: 0.0  # 0 = use all data
  validation_split: 0.2
  shuffle: true
  random_seed: 42

# Model Architecture
model:
  input_shape: [32, 20, 3]
  conv_filters: [32, 32, 32]
  dense_units: 256
  dropout_rate: 0.5

# Optimizer Configuration
optimizer:
  name: "adam"
  learning_rate: 0.001

# Data Augmentation
augmentation:
  enabled: true
  width_shift_range: 1
  height_shift_range: 1
  brightness_range: [0.8, 1.2]
  zoom_range: 0.3
  rotation_range: 5

# Callbacks
callbacks:
  early_stopping:
    enabled: true
    patience: 10
  
  model_checkpoint:
    enabled: true
    monitor: "val_accuracy"
    save_best_only: true
```

### Training Outputs

After training, the following files will be generated:

- `output/best_model.h5`: Best model checkpoint
- `output/training_history.png`: Loss and accuracy plots
- `output/confusion_matrix.png`: Confusion matrix visualization
- `output/prediction_samples.png`: Sample predictions
- `output/metrics.json`: Detailed evaluation metrics
- `test/`: SavedModel format
- `4TrainedModel.tflite`: Quantized TFLite model
- `5TrainedModel.tflite`: Standard TFLite model

## Inference

### Using Python API

```python
from src.utils.inference import TFLiteInference

# Load model
inference = TFLiteInference("5TrainedModel.tflite")

# Predict single image
predicted_class, confidence, probabilities = inference.predict_from_image("test.jpg")
print(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")

# Predict with visualization
inference.predict_and_visualize("test.jpg", show_plot=True)

# Batch prediction
results = inference.batch_predict(["img1.jpg", "img2.jpg", "img3.jpg"])
for result in results:
    print(f"{result['image_path']}: Class {result['predicted_class']}")
```

## Advanced Features

### 1. Custom Model Architecture

Modify the model configuration in `config.yaml`:

```yaml
model:
  conv_filters: [64, 128, 256]  # Deeper network
  dense_units: 512
  dropout_rate: 0.6
```

### 2. Learning Rate Scheduling

Enable in `config.yaml`:

```yaml
callbacks:
  reduce_lr:
    enabled: true
    monitor: "val_loss"
    factor: 0.5
    patience: 5
    min_lr: 0.00001
```

### 3. Data Augmentation

Customize augmentation in `config.yaml`:

```yaml
augmentation:
  enabled: true
  width_shift_range: 2
  height_shift_range: 2
  brightness_range: [0.7, 1.3]
  zoom_range: 0.4
  rotation_range: 10
  horizontal_flip: false
  vertical_flip: false
```

### 4. Model Quantization

Quantization is enabled by default. To disable:

```yaml
export:
  quantize: false
```

### 5. Custom Training Script

```python
from src.utils.config_loader import load_config
from src.data.data_loader import load_images_from_folder, split_data
from src.models.model_builder import create_digit_recognition_model, compile_model
from src.models.trainer import create_data_generator, create_callbacks, train_model
from src.utils.evaluation import evaluate_model
from src.utils.model_export import export_models

# Load config
config = load_config("config.yaml")

# Load data
x_data, y_data = load_images_from_folder("/path/to/data")

# Split data
X_train, X_val, y_train, y_val = split_data(x_data, y_data, test_size=0.2)

# Create model
model = create_digit_recognition_model(
    input_shape=(32, 20, 3),
    num_classes=10,
    conv_filters=[32, 32, 32],
    dense_units=256,
    dropout_rate=0.5
)

# Compile
model = compile_model(model, optimizer_name='adam', learning_rate=0.001)

# Create training components
datagen = create_data_generator(config['augmentation'])
callbacks = create_callbacks(config['callbacks'])

# Train
history = train_model(
    model, X_train, y_train, X_val, y_val,
    batch_size=4, epochs=100,
    datagen=datagen, callbacks=callbacks
)

# Evaluate
metrics = evaluate_model(model, X_val, y_val)

# Export
export_models(model, X_train)
```

## API Reference

### Data Module (`src.data.data_loader`)

- `load_images_from_folder(folder_path, image_width, image_height, num_classes)`: Load and preprocess images
- `split_data(x_data, y_data, test_size, random_state)`: Split data into train/test
- `preprocess_single_image(image_path, target_width, target_height)`: Preprocess single image

### Model Module (`src.models.model_builder`)

- `create_digit_recognition_model(...)`: Create CNN model
- `compile_model(model, optimizer_name, learning_rate)`: Compile model

### Training Module (`src.models.trainer`)

- `create_data_generator(augmentation_config)`: Create ImageDataGenerator
- `create_callbacks(callbacks_config, model_checkpoint_path)`: Create training callbacks
- `train_model(model, X_train, y_train, ...)`: Train model

### Evaluation Module (`src.utils.evaluation`)

- `plot_training_history(history, save_path)`: Plot training curves
- `plot_confusion_matrix(y_true, y_pred, class_names, save_path)`: Plot confusion matrix
- `evaluate_model(model, X_test, y_test)`: Evaluate and return metrics
- `plot_prediction_samples(model, X_test, y_test, num_samples)`: Plot sample predictions

### Export Module (`src.utils.model_export`)

- `export_models(model, X_train, ...)`: Export in multiple formats
- `export_tflite_model(saved_model_path, output_path, quantize)`: Export TFLite model

### Inference Module (`src.utils.inference`)

- `TFLiteInference(model_path)`: TFLite inference wrapper class
  - `predict_from_image(image_path)`: Predict single image
  - `predict_and_visualize(image_path)`: Predict with visualization
  - `batch_predict(image_paths)`: Predict multiple images

## Tips and Best Practices

1. **Start with small epochs** to test your setup, then increase for final training
2. **Use validation data** to monitor overfitting (set `training_percentage > 0`)
3. **Enable early stopping** to prevent overfitting
4. **Use learning rate scheduling** for better convergence
5. **Monitor training curves** to detect issues
6. **Use quantized models** for deployment on edge devices
7. **Test on sample images** before batch processing

## Troubleshooting

### Issue: Out of Memory
- Reduce `batch_size` in config
- Reduce `conv_filters` or `dense_units`
- Use gradient accumulation

### Issue: Model not converging
- Increase `epochs`
- Adjust `learning_rate`
- Enable `reduce_lr` callback
- Check data quality

### Issue: Overfitting
- Increase `dropout_rate`
- Enable data augmentation
- Reduce model complexity
- Get more training data

### Issue: TFLite model too large
- Enable quantization
- Reduce model complexity
- Use pruning techniques
