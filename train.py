#!/usr/bin/env python
"""
Main training script for digit recognition model.
"""
import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.config_loader import load_config
from src.data.data_loader import load_images_from_folder, split_data
from src.models.model_builder import create_digit_recognition_model, compile_model
from src.models.trainer import create_data_generator, create_callbacks, train_model
from src.utils.evaluation import (
    plot_training_history,
    plot_confusion_matrix,
    evaluate_model,
    plot_prediction_samples
)
from src.utils.model_export import export_models
import logging

logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train digit recognition model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-folder',
        type=str,
        help='Override data folder path from config'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs from config'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size from config'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory for output files'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting digit recognition model training")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']
    optimizer_config = config['optimizer']
    augmentation_config = config['augmentation']
    callbacks_config = config['callbacks']
    export_config = config['export']
    
    if args.data_folder:
        data_config['output_folder'] = args.data_folder
    if args.epochs:
        training_config['epochs'] = args.epochs
    if args.batch_size:
        training_config['batch_size'] = args.batch_size
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    x_data, y_data = load_images_from_folder(
        data_config['output_folder'],
        data_config['image_width'],
        data_config['image_height'],
        data_config['num_classes']
    )
    
    if len(x_data) == 0:
        logger.error("No data loaded. Exiting.")
        return
    
    # Split data
    logger.info("Splitting data...")
    test_size = training_config.get('validation_split', 0.2) \
                if training_config['training_percentage'] > 0 else 0
    
    X_train, X_val, y_train, y_val = split_data(
        x_data,
        y_data,
        test_size=test_size,
        random_state=training_config.get('random_seed', 42),
        shuffle_data=training_config.get('shuffle', True)
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_digit_recognition_model(
        input_shape=tuple(model_config['input_shape']),
        num_classes=data_config['num_classes'],
        conv_filters=model_config['conv_filters'],
        conv_kernel_size=model_config['conv_kernel_size'],
        pool_size=model_config['pool_size'],
        dense_units=model_config['dense_units'],
        dropout_rate=model_config['dropout_rate'],
        use_batch_normalization=model_config['use_batch_normalization']
    )
    
    # Compile model
    logger.info("Compiling model...")
    model = compile_model(
        model,
        optimizer_name=optimizer_config['name'],
        learning_rate=optimizer_config['learning_rate']
    )
    
    model.summary()
    
    # Create data generator
    datagen = create_data_generator(augmentation_config)
    
    # Create callbacks
    checkpoint_path = output_dir / 'best_model.h5'
    callbacks = create_callbacks(callbacks_config, str(checkpoint_path))
    
    # Train model
    logger.info("Training model...")
    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=training_config['batch_size'],
        epochs=training_config['epochs'],
        datagen=datagen if augmentation_config['enabled'] else None,
        callbacks=callbacks
    )
    
    # Plot training history
    logger.info("Plotting training history...")
    plot_training_history(
        history,
        save_path=str(output_dir / 'training_history.png'),
        show_plot=False
    )
    
    # Evaluate model if validation data exists
    if X_val is not None:
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_val, y_val)
        
        # Plot confusion matrix
        y_pred = model.predict(X_val)
        plot_confusion_matrix(
            y_val,
            y_pred,
            save_path=str(output_dir / 'confusion_matrix.png'),
            show_plot=False
        )
        
        # Plot prediction samples
        plot_prediction_samples(
            model,
            X_val,
            y_val,
            num_samples=10,
            save_path=str(output_dir / 'prediction_samples.png'),
            show_plot=False
        )
        
        # Save metrics
        import json
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {output_dir / 'metrics.json'}")
    
    # Export models
    logger.info("Exporting models...")
    export_info = export_models(
        model,
        X_train,
        saved_model_path=export_config['saved_model_path'],
        quantized_path=export_config['quantized_model_path'],
        standard_path=export_config['standard_tflite_path'],
        quantize=export_config['quantize'],
        representative_dataset_size=export_config['representative_dataset_size']
    )
    
    # Save export info
    import json
    with open(output_dir / 'export_info.json', 'w') as f:
        json.dump(export_info, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"All outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
