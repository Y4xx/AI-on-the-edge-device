# Quick Reference Guide

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Training

```bash
# Quick start
python train.py --config config.yaml

# Custom settings
python train.py --epochs 50 --batch-size 8 --output-dir my_output

# With specific data folder
python train.py --data-folder /path/to/data --epochs 100
```

## Inference

```bash
# Single image with visualization
python predict.py image.jpg --visualize

# Batch prediction
python predict.py img1.jpg img2.jpg img3.jpg --batch

# Save results to JSON
python predict.py image.jpg --output results.json

# Use specific model
python predict.py image.jpg --model 4TrainedModel.tflite
```

## Data Visualization

```bash
# Explore dataset
python visualize_data.py /path/to/data --show-plots

# Save to specific directory
python visualize_data.py /path/to/data --output-dir viz_output
```

## Model Comparison

```bash
# Compare two models
python compare_models.py model1.tflite model2.tflite --data-folder /path/to/test/data

# Compare with plots
python compare_models.py model1.tflite model2.tflite --data-folder /path/to/test/data --show-plot
```

## Testing

```bash
# Run all tests
python run_tests.py

# Run specific test
python -m unittest tests.test_data_loader
```

## Python API

### Training
```python
from src.data.data_loader import load_images_from_folder, split_data
from src.models.model_builder import create_digit_recognition_model, compile_model
from src.models.trainer import train_model, create_data_generator, create_callbacks
from src.utils.model_export import export_models

# Load data
x_data, y_data = load_images_from_folder("/path/to/data")
X_train, X_val, y_train, y_val = split_data(x_data, y_data, test_size=0.2)

# Create model
model = create_digit_recognition_model()
model = compile_model(model, optimizer_name='adam', learning_rate=0.001)

# Train
history = train_model(model, X_train, y_train, X_val, y_val, epochs=100)

# Export
export_models(model, X_train)
```

### Inference
```python
from src.utils.inference import TFLiteInference

# Initialize
inference = TFLiteInference("5TrainedModel.tflite")

# Predict
predicted_class, confidence, probabilities = inference.predict_from_image("test.jpg")

# Visualize
inference.predict_and_visualize("test.jpg", show_plot=True)

# Batch predict
results = inference.batch_predict(["img1.jpg", "img2.jpg"])
```

## Configuration

Edit `config.yaml` to change:
- Data parameters (image size, classes)
- Training settings (batch size, epochs)
- Model architecture (filters, units, dropout)
- Optimizer settings (type, learning rate)
- Data augmentation parameters
- Callbacks (early stopping, checkpointing)

## Common Tasks

### Change Model Architecture
```yaml
# In config.yaml
model:
  conv_filters: [64, 128, 256]  # Deeper network
  dense_units: 512
  dropout_rate: 0.6
```

### Enable Early Stopping
```yaml
# In config.yaml
callbacks:
  early_stopping:
    enabled: true
    patience: 10
```

### Adjust Learning Rate
```yaml
# In config.yaml
optimizer:
  name: "adam"
  learning_rate: 0.0001  # Lower learning rate

callbacks:
  reduce_lr:
    enabled: true
    factor: 0.5
    patience: 5
```

### Disable Data Augmentation
```yaml
# In config.yaml
augmentation:
  enabled: false
```

## Output Files

After training:
- `output/best_model.h5` - Best model checkpoint
- `output/training_history.png` - Training curves
- `output/confusion_matrix.png` - Confusion matrix
- `output/prediction_samples.png` - Sample predictions
- `output/metrics.json` - Evaluation metrics
- `test/` - SavedModel format
- `4TrainedModel.tflite` - Quantized model
- `5TrainedModel.tflite` - Standard model

## Troubleshooting

### Out of Memory
```yaml
# Reduce batch size
training:
  batch_size: 2

# Or reduce model complexity
model:
  conv_filters: [16, 32]
  dense_units: 128
```

### Model Not Converging
```yaml
# Try different learning rate
optimizer:
  learning_rate: 0.01

# Or more epochs
training:
  epochs: 200
```

### Overfitting
```yaml
# Increase dropout
model:
  dropout_rate: 0.7

# Enable augmentation
augmentation:
  enabled: true
```

## Tips

1. Start with small epochs (10-20) to test
2. Use validation data to monitor overfitting
3. Enable early stopping for efficiency
4. Use quantized models for deployment
5. Monitor training curves regularly
6. Test on sample images first

## Links

- **Full Documentation**: [USAGE.md](USAGE.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Repository**: https://github.com/Y4xx/AI-on-the-edge-device
