#!/usr/bin/env python
"""
Inference CLI tool for digit recognition.
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logging
from src.utils.inference import TFLiteInference
import logging
import json

logger = logging.getLogger(__name__)


def main():
    """Main inference CLI."""
    parser = argparse.ArgumentParser(description='Run inference on digit images')
    parser.add_argument(
        'image_path',
        type=str,
        nargs='+',
        help='Path(s) to image file(s)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='5TrainedModel.tflite',
        help='Path to TFLite model'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization of predictions'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple images in batch mode'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize pixel values (0-1)'
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
    
    # Check model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    inference = TFLiteInference(args.model)
    
    results = []
    
    if args.batch:
        # Batch prediction
        logger.info(f"Running batch prediction on {len(args.image_path)} images")
        results = inference.batch_predict(args.image_path, args.normalize)
        
        # Print results
        for result in results:
            if 'error' in result:
                print(f"{result['image_path']}: ERROR - {result['error']}")
            else:
                print(f"{result['image_path']}: "
                      f"Class {result['predicted_class']} "
                      f"(confidence: {result['confidence']:.2%})")
    else:
        # Single or multiple image prediction with visualization
        for image_path in args.image_path:
            if not Path(image_path).exists():
                logger.error(f"Image not found: {image_path}")
                continue
            
            logger.info(f"Processing {image_path}")
            
            if args.visualize:
                predicted_class, confidence = inference.predict_and_visualize(
                    image_path,
                    show_plot=True
                )
            else:
                predicted_class, confidence, probabilities = inference.predict_from_image(
                    image_path,
                    args.normalize
                )
            
            result = {
                'image_path': image_path,
                'predicted_class': int(predicted_class),
                'confidence': float(confidence)
            }
            results.append(result)
            
            print(f"\nImage: {image_path}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print("-" * 50)
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
