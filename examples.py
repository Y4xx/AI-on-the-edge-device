#!/usr/bin/env python
"""
Example script demonstrating the improved API usage.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("Digit Recognition CNN Model - Example Usage")
print("=" * 70)

print("\n1. Configuration Management")
print("-" * 70)
print("""
from src.utils.config_loader import load_config

config = load_config('config.yaml')
print(f"Batch size: {config['training']['batch_size']}")
print(f"Epochs: {config['training']['epochs']}")
print(f"Model filters: {config['model']['conv_filters']}")
""")

print("\n2. Data Loading with Error Handling")
print("-" * 70)
print("""
from src.data.data_loader import load_images_from_folder

x_data, y_data = load_images_from_folder(
    "/path/to/dataset",
    image_width=20,
    image_height=32,
    num_classes=10
)

print(f"Loaded {len(x_data)} images")
print(f"Data shape: {x_data.shape}")
print(f"Labels shape: {y_data.shape}")
""")

print("\n3. Model Building with Custom Architecture")
print("-" * 70)
print("""
from src.models.model_builder import create_digit_recognition_model, compile_model

# Create model with custom parameters
model = create_digit_recognition_model(
    input_shape=(32, 20, 3),
    num_classes=10,
    conv_filters=[64, 128, 256],  # Deeper network
    dense_units=512,
    dropout_rate=0.6,
    use_batch_normalization=True
)

# Compile with Adam optimizer
model = compile_model(
    model,
    optimizer_name='adam',
    learning_rate=0.001
)

model.summary()
""")

print("\n4. Training with Advanced Callbacks")
print("-" * 70)
print("""
from src.models.trainer import train_model, create_data_generator, create_callbacks

# Create data augmentation
datagen = create_data_generator({
    'enabled': True,
    'width_shift_range': 1,
    'height_shift_range': 1,
    'rotation_range': 5
})

# Create callbacks (early stopping, checkpointing, lr scheduling)
callbacks = create_callbacks({
    'early_stopping': {'enabled': True, 'patience': 10},
    'model_checkpoint': {'enabled': True, 'save_best_only': True},
    'reduce_lr': {'enabled': True, 'patience': 5}
})

# Train model
history = train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    batch_size=8,
    epochs=100,
    datagen=datagen,
    callbacks=callbacks
)
""")

print("\n5. Comprehensive Model Evaluation")
print("-" * 70)
print("""
from src.utils.evaluation import (
    plot_training_history,
    plot_confusion_matrix,
    evaluate_model,
    plot_prediction_samples
)

# Plot training curves
plot_training_history(history, save_path='training_history.png')

# Evaluate and get detailed metrics
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")

# Generate confusion matrix
y_pred = model.predict(X_test)
plot_confusion_matrix(y_test, y_pred, save_path='confusion_matrix.png')

# Visualize predictions
plot_prediction_samples(model, X_test, y_test, save_path='samples.png')
""")

print("\n6. Model Export in Multiple Formats")
print("-" * 70)
print("""
from src.utils.model_export import export_models

# Export to SavedModel, quantized TFLite, and standard TFLite
export_info = export_models(
    model,
    X_train,
    saved_model_path='test',
    quantized_path='quantized_model.tflite',
    standard_path='standard_model.tflite',
    quantize=True,
    representative_dataset_size=100
)

print(f"Quantized size: {export_info['quantized_model']['size']} bytes")
print(f"Standard size: {export_info['standard_model']['size']} bytes")
""")

print("\n7. Advanced Inference with Batch Prediction")
print("-" * 70)
print("""
from src.utils.inference import TFLiteInference

# Initialize inference engine
inference = TFLiteInference('quantized_model.tflite')

# Single image prediction
predicted_class, confidence, probabilities = inference.predict_from_image('test.jpg')
print(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")

# Visualize prediction
inference.predict_and_visualize('test.jpg', show_plot=True)

# Batch prediction for multiple images
results = inference.batch_predict([
    'image1.jpg',
    'image2.jpg', 
    'image3.jpg'
])

for result in results:
    print(f"{result['image_path']}: Class {result['predicted_class']} "
          f"({result['confidence']:.2%})")
""")

print("\n8. Complete Training Pipeline")
print("-" * 70)
print("""
from src.utils.config_loader import load_config
from src.data.data_loader import load_images_from_folder, split_data
from src.models.model_builder import create_digit_recognition_model, compile_model
from src.models.trainer import create_data_generator, create_callbacks, train_model
from src.utils.evaluation import evaluate_model, plot_training_history
from src.utils.model_export import export_models

# Load configuration
config = load_config('config.yaml')

# Load and split data
x_data, y_data = load_images_from_folder(config['data']['output_folder'])
X_train, X_val, y_train, y_val = split_data(x_data, y_data, test_size=0.2)

# Create and compile model
model = create_digit_recognition_model(**config['model'])
model = compile_model(model, **config['optimizer'])

# Train with augmentation and callbacks
datagen = create_data_generator(config['augmentation'])
callbacks = create_callbacks(config['callbacks'])
history = train_model(model, X_train, y_train, X_val, y_val,
                     datagen=datagen, callbacks=callbacks,
                     **config['training'])

# Evaluate
metrics = evaluate_model(model, X_val, y_val)
plot_training_history(history)

# Export
export_models(model, X_train, **config['export'])

print("Training complete! Models exported successfully.")
""")

print("\n" + "=" * 70)
print("Key Improvements Over Original:")
print("=" * 70)
print("""
✅ Modular architecture with reusable components
✅ Configuration-driven development
✅ Type hints and comprehensive error handling
✅ Advanced callbacks (early stopping, checkpointing, LR scheduling)
✅ Detailed evaluation metrics and visualizations
✅ Batch prediction support
✅ Multiple export formats with quantization
✅ CLI tools for common tasks
✅ Unit tests for reliability
✅ Extensive documentation
""")

print("\n" + "=" * 70)
print("Quick Commands:")
print("=" * 70)
print("""
# Training
python train.py --config config.yaml --epochs 50

# Inference
python predict.py image.jpg --model model.tflite --visualize

# Data visualization
python visualize_data.py /path/to/data --show-plots

# Model comparison
python compare_models.py model1.tflite model2.tflite --data-folder /path/to/data

# Testing
python run_tests.py
""")

print("\n" + "=" * 70)
print("For more examples, see USAGE.md and QUICKSTART.md")
print("=" * 70)
