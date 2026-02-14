#!/usr/bin/env python
"""
Data visualization and exploration script.
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logging
from src.data.data_loader import load_images_from_folder
import logging

logger = logging.getLogger(__name__)


def plot_class_distribution(y_data, save_path=None, show_plot=True):
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        y_data: One-hot encoded labels
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    # Convert one-hot to class indices
    class_indices = np.argmax(y_data, axis=1)
    class_counts = Counter(class_indices)
    
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(classes)
    plt.grid(True, alpha=0.3, axis='y')
    
    total_samples = sum(counts)
    plt.text(0.02, 0.98, f'Total Samples: {total_samples}',
             transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class distribution plot to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_sample_images(x_data, y_data, num_samples=20, save_path=None, show_plot=True):
    """
    Plot sample images from the dataset.
    
    Args:
        x_data: Image data
        y_data: One-hot encoded labels
        num_samples: Number of samples to display
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    indices = np.random.choice(len(x_data), min(num_samples, len(x_data)), replace=False)
    
    rows = (num_samples + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = x_data[idx].astype(np.uint8)
        label = np.argmax(y_data[idx])
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Class: {label}', fontsize=11)
    
    # Hide extra subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sample images to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_image_statistics(x_data, save_path=None, show_plot=True):
    """
    Plot statistics about image properties.
    
    Args:
        x_data: Image data
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Mean pixel values per channel
    mean_values = np.mean(x_data, axis=(0, 1, 2))
    axes[0, 0].bar(['Red', 'Green', 'Blue'], mean_values, color=['red', 'green', 'blue'], alpha=0.7)
    axes[0, 0].set_ylabel('Mean Pixel Value')
    axes[0, 0].set_title('Mean Pixel Values per Channel')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Pixel value distribution (all channels combined)
    axes[0, 1].hist(x_data.flatten(), bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Overall Pixel Value Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Standard deviation per channel
    std_values = np.std(x_data, axis=(0, 1, 2))
    axes[1, 0].bar(['Red', 'Green', 'Blue'], std_values, color=['red', 'green', 'blue'], alpha=0.7)
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].set_title('Pixel Value Standard Deviation per Channel')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Image brightness distribution (mean across channels and pixels)
    brightness = np.mean(x_data, axis=(1, 2, 3))
    axes[1, 1].hist(brightness, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Mean Brightness')
    axes[1, 1].set_ylabel('Number of Images')
    axes[1, 1].set_title('Image Brightness Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Dataset Image Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved image statistics to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main visualization script."""
    parser = argparse.ArgumentParser(description='Visualize and explore dataset')
    parser.add_argument(
        'data_folder',
        type=str,
        help='Path to dataset folder'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Directory for output visualizations'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of sample images to display'
    )
    parser.add_argument(
        '--image-width',
        type=int,
        default=20,
        help='Image width'
    )
    parser.add_argument(
        '--image-height',
        type=int,
        default=32,
        help='Image height'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=10,
        help='Number of classes'
    )
    parser.add_argument(
        '--show-plots',
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
    logger.info("Starting data visualization")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_folder}")
    x_data, y_data = load_images_from_folder(
        args.data_folder,
        args.image_width,
        args.image_height,
        args.num_classes
    )
    
    if len(x_data) == 0:
        logger.error("No data loaded. Exiting.")
        return
    
    logger.info(f"Loaded {len(x_data)} images")
    
    # Generate visualizations
    logger.info("Generating class distribution plot...")
    plot_class_distribution(
        y_data,
        save_path=str(output_dir / 'class_distribution.png'),
        show_plot=args.show_plots
    )
    
    logger.info("Generating sample images plot...")
    plot_sample_images(
        x_data,
        y_data,
        num_samples=args.num_samples,
        save_path=str(output_dir / 'sample_images.png'),
        show_plot=args.show_plots
    )
    
    logger.info("Generating image statistics plots...")
    plot_image_statistics(
        x_data,
        save_path=str(output_dir / 'image_statistics.png'),
        show_plot=args.show_plots
    )
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("Dataset Summary")
    logger.info("="*60)
    logger.info(f"Total Images: {len(x_data)}")
    logger.info(f"Image Shape: {x_data[0].shape}")
    logger.info(f"Data Type: {x_data.dtype}")
    
    class_indices = np.argmax(y_data, axis=1)
    class_counts = Counter(class_indices)
    logger.info(f"\nClass Distribution:")
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / len(x_data)) * 100
        logger.info(f"  Class {cls}: {count} images ({percentage:.2f}%)")
    
    logger.info(f"\nPixel Value Statistics:")
    logger.info(f"  Min: {x_data.min():.2f}")
    logger.info(f"  Max: {x_data.max():.2f}")
    logger.info(f"  Mean: {x_data.mean():.2f}")
    logger.info(f"  Std: {x_data.std():.2f}")
    
    logger.info(f"\nAll visualizations saved to {output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
