#!/usr/bin/env python
"""
Model comparison utility for comparing different models.
"""
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logging
from src.utils.inference import TFLiteInference
from src.data.data_loader import load_images_from_folder
import logging

logger = logging.getLogger(__name__)


def compare_model_sizes(model_paths: List[str]) -> Dict[str, int]:
    """
    Compare file sizes of different models.
    
    Args:
        model_paths: List of model file paths
        
    Returns:
        Dictionary of model sizes
    """
    sizes = {}
    for path in model_paths:
        model_file = Path(path)
        if model_file.exists():
            size = model_file.stat().st_size
            sizes[path] = size
            logger.info(f"{model_file.name}: {size:,} bytes ({size/1024:.2f} KB)")
        else:
            logger.warning(f"Model not found: {path}")
    
    return sizes


def benchmark_inference_speed(
    model_path: str,
    test_images: np.ndarray,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark inference speed of a model.
    
    Args:
        model_path: Path to TFLite model
        test_images: Test images
        num_runs: Number of inference runs for averaging
        
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    inference = TFLiteInference(model_path)
    
    # Warmup
    for i in range(10):
        _ = inference.predict(test_images[i:i+1])
    
    # Benchmark
    times = []
    for i in range(num_runs):
        idx = i % len(test_images)
        start = time.time()
        _ = inference.predict(test_images[idx:idx+1])
        end = time.time()
        times.append(end - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }


def compare_model_accuracy(
    model_paths: List[str],
    x_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compare accuracy of different models.
    
    Args:
        model_paths: List of model file paths
        x_test: Test images
        y_test: Test labels (one-hot encoded)
        
    Returns:
        Dictionary with accuracy metrics for each model
    """
    results = {}
    
    for model_path in model_paths:
        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        logger.info(f"Evaluating {Path(model_path).name}...")
        inference = TFLiteInference(model_path)
        
        predictions = []
        for i in range(len(x_test)):
            output = inference.predict(x_test[i:i+1])
            predictions.append(output)
        
        predictions = np.array(predictions)
        
        # Calculate accuracy
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        
        # Calculate top-3 accuracy
        top3_preds = np.argsort(predictions, axis=1)[:, -3:]
        top3_accuracy = np.mean([true_classes[i] in top3_preds[i] for i in range(len(true_classes))])
        
        results[model_path] = {
            'accuracy': float(accuracy),
            'top3_accuracy': float(top3_accuracy)
        }
        
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Top-3 Accuracy: {top3_accuracy:.4f}")
    
    return results


def plot_comparison(
    model_names: List[str],
    sizes: Dict[str, int],
    speeds: Dict[str, Dict[str, float]],
    accuracies: Dict[str, Dict[str, float]],
    save_path: str = None,
    show_plot: bool = True
):
    """
    Plot comprehensive model comparison.
    
    Args:
        model_names: List of model names
        sizes: Model sizes
        speeds: Speed benchmarks
        accuracies: Accuracy results
        save_path: Path to save plot (optional)
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Model Size
    sizes_kb = [sizes.get(name, 0) / 1024 for name in model_names]
    axes[0, 0].bar(range(len(model_names)), sizes_kb, color='skyblue', edgecolor='navy')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels([Path(n).stem for n in model_names], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Size (KB)')
    axes[0, 0].set_title('Model Size Comparison')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    for i, size in enumerate(sizes_kb):
        axes[0, 0].text(i, size, f'{size:.1f}', ha='center', va='bottom')
    
    # Plot 2: Inference Speed
    speeds_mean = [speeds.get(name, {}).get('mean_ms', 0) for name in model_names]
    axes[0, 1].bar(range(len(model_names)), speeds_mean, color='lightgreen', edgecolor='darkgreen')
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels([Path(n).stem for n in model_names], rotation=45, ha='right')
    axes[0, 1].set_ylabel('Time (ms)')
    axes[0, 1].set_title('Inference Speed (Lower is Better)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for i, speed in enumerate(speeds_mean):
        axes[0, 1].text(i, speed, f'{speed:.2f}', ha='center', va='bottom')
    
    # Plot 3: Accuracy
    acc_values = [accuracies.get(name, {}).get('accuracy', 0) * 100 for name in model_names]
    axes[1, 0].bar(range(len(model_names)), acc_values, color='salmon', edgecolor='darkred')
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels([Path(n).stem for n in model_names], rotation=45, ha='right')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Model Accuracy')
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for i, acc in enumerate(acc_values):
        axes[1, 0].text(i, acc, f'{acc:.2f}%', ha='center', va='bottom')
    
    # Plot 4: Efficiency Score (accuracy / (size * speed))
    efficiency = []
    for name in model_names:
        acc = accuracies.get(name, {}).get('accuracy', 0)
        size = sizes.get(name, 1) / 1024  # KB
        speed = speeds.get(name, {}).get('mean_ms', 1)
        eff = (acc * 100) / (size * speed) if size > 0 and speed > 0 else 0
        efficiency.append(eff)
    
    axes[1, 1].bar(range(len(model_names)), efficiency, color='gold', edgecolor='orange')
    axes[1, 1].set_xticks(range(len(model_names)))
    axes[1, 1].set_xticklabels([Path(n).stem for n in model_names], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Efficiency Score')
    axes[1, 1].set_title('Model Efficiency (Higher is Better)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for i, eff in enumerate(efficiency):
        axes[1, 1].text(i, eff, f'{eff:.2f}', ha='center', va='bottom')
    
    plt.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main model comparison script."""
    parser = argparse.ArgumentParser(description='Compare different models')
    parser.add_argument(
        'models',
        type=str,
        nargs='+',
        help='Paths to TFLite models to compare'
    )
    parser.add_argument(
        '--data-folder',
        type=str,
        help='Path to test data folder (required for accuracy/speed comparison)'
    )
    parser.add_argument(
        '--benchmark-runs',
        type=int,
        default=100,
        help='Number of runs for speed benchmark'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model_comparison.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--plot',
        type=str,
        default='model_comparison.png',
        help='Output plot file'
    )
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display plots interactively'
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
    logger.info("Starting model comparison")
    
    # Compare sizes
    logger.info("\n" + "="*60)
    logger.info("Model Size Comparison")
    logger.info("="*60)
    sizes = compare_model_sizes(args.models)
    
    results = {
        'sizes': sizes,
        'speeds': {},
        'accuracies': {}
    }
    
    # Load test data if provided
    if args.data_folder:
        logger.info(f"\nLoading test data from {args.data_folder}")
        x_test, y_test = load_images_from_folder(args.data_folder, 20, 32, 10)
        
        if len(x_test) > 0:
            # Benchmark speed
            logger.info("\n" + "="*60)
            logger.info("Inference Speed Benchmark")
            logger.info("="*60)
            for model_path in args.models:
                if Path(model_path).exists():
                    logger.info(f"\nBenchmarking {Path(model_path).name}...")
                    speed_stats = benchmark_inference_speed(
                        model_path,
                        x_test,
                        args.benchmark_runs
                    )
                    results['speeds'][model_path] = speed_stats
                    logger.info(f"  Mean: {speed_stats['mean_ms']:.2f} ms")
                    logger.info(f"  Std: {speed_stats['std_ms']:.2f} ms")
            
            # Compare accuracy
            logger.info("\n" + "="*60)
            logger.info("Model Accuracy Comparison")
            logger.info("="*60)
            accuracies = compare_model_accuracy(args.models, x_test, y_test)
            results['accuracies'] = accuracies
            
            # Generate comparison plot
            plot_comparison(
                args.models,
                sizes,
                results['speeds'],
                accuracies,
                save_path=args.plot,
                show_plot=args.show_plot
            )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")
    
    logger.info("\n" + "="*60)
    logger.info("Comparison Complete")
    logger.info("="*60)


if __name__ == '__main__':
    main()
