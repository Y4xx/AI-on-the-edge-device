"""
Model export utilities for TensorFlow Lite.
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


def export_saved_model(
    model: tf.keras.Model,
    export_path: str = "test"
) -> None:
    """
    Export model in SavedModel format.
    
    Args:
        model: Keras model to export
        export_path: Path to save the model
    """
    model.export(export_path)
    logger.info(f"✔ Model exported to '{export_path}'")


def create_representative_dataset(
    X_train: np.ndarray,
    num_samples: int = 100
) -> Callable:
    """
    Create a representative dataset function for quantization.
    
    Args:
        X_train: Training data
        num_samples: Number of samples to use
        
    Returns:
        Representative dataset function
    """
    def representative_dataset():
        for i in range(min(num_samples, len(X_train))):
            yield [X_train[i:i+1]]
    
    return representative_dataset


def export_tflite_model(
    saved_model_path: str,
    output_path: str,
    quantize: bool = False,
    representative_dataset: Optional[Callable] = None
) -> int:
    """
    Export model to TensorFlow Lite format.
    
    Args:
        saved_model_path: Path to SavedModel
        output_path: Output path for TFLite model
        quantize: Whether to apply quantization
        representative_dataset: Function for representative dataset (required for quantization)
        
    Returns:
        Size of the exported model in bytes
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    if quantize:
        if representative_dataset is None:
            logger.warning("Quantization requested but no representative dataset provided")
        else:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            # Optional: disable per-channel quantization for dense layers
            converter._experimental_disable_per_channel_quantization_for_dense_layers = True
            logger.info("Applying quantization optimizations")
    
    tflite_model = converter.convert()
    
    # Save model
    model_path = Path(output_path)
    model_path.write_bytes(tflite_model)
    
    model_size = model_path.stat().st_size
    logger.info(f"✅ TFLite model saved: {output_path}")
    logger.info(f"📦 Size: {model_size:,} bytes ({model_size / 1024:.2f} KB)")
    
    return model_size


def export_models(
    model: tf.keras.Model,
    X_train: np.ndarray,
    saved_model_path: str = "test",
    quantized_path: str = "4TrainedModel.tflite",
    standard_path: str = "5TrainedModel.tflite",
    quantize: bool = True,
    representative_dataset_size: int = 100
) -> dict:
    """
    Export model in multiple formats.
    
    Args:
        model: Keras model to export
        X_train: Training data for representative dataset
        saved_model_path: Path for SavedModel format
        quantized_path: Path for quantized TFLite model
        standard_path: Path for standard TFLite model
        quantize: Whether to create quantized model
        representative_dataset_size: Number of samples for representative dataset
        
    Returns:
        Dictionary with export information
    """
    export_info = {}
    
    # Export SavedModel
    export_saved_model(model, saved_model_path)
    export_info['saved_model_path'] = saved_model_path
    
    # Create representative dataset for quantization
    rep_dataset = create_representative_dataset(X_train, representative_dataset_size)
    
    # Export quantized TFLite model
    if quantize:
        quantized_size = export_tflite_model(
            saved_model_path,
            quantized_path,
            quantize=True,
            representative_dataset=rep_dataset
        )
        export_info['quantized_model'] = {
            'path': quantized_path,
            'size': quantized_size
        }
    
    # Export standard TFLite model
    standard_size = export_tflite_model(
        saved_model_path,
        standard_path,
        quantize=False
    )
    export_info['standard_model'] = {
        'path': standard_path,
        'size': standard_size
    }
    
    # Log size comparison
    if quantize:
        reduction = (1 - quantized_size / standard_size) * 100
        logger.info(f"Size reduction: {reduction:.1f}% (quantized vs standard)")
    
    return export_info
