"""
Model evaluation and visualization utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_recall_fscore_support
)
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Any,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> np.ndarray:
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true: True labels (one-hot encoded or integers)
        y_pred: Predicted labels (one-hot encoded or integers)
        class_names: List of class names
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        Confusion matrix as numpy array
    """
    # Convert one-hot to integers if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return cm


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Evaluate model and return comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        class_names: List of class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_classes, 
        y_pred_classes, 
        average='weighted'
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(y_true_classes, y_pred_classes, average=None)
    
    # Overall accuracy
    accuracy = np.mean(y_pred_classes == y_true_classes)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class_metrics': {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1_score': per_class_f1.tolist(),
            'support': per_class_support.tolist()
        }
    }
    
    # Print classification report
    if class_names is None:
        class_names = [str(i) for i in range(len(per_class_precision))]
    
    logger.info("\n" + "="*50)
    logger.info("Model Evaluation Metrics")
    logger.info("="*50)
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted Precision: {precision:.4f}")
    logger.info(f"Weighted Recall: {recall:.4f}")
    logger.info(f"Weighted F1-Score: {f1:.4f}")
    logger.info("\nPer-Class Metrics:")
    logger.info("-"*50)
    
    for i, class_name in enumerate(class_names):
        logger.info(
            f"Class {class_name}: "
            f"Precision={per_class_precision[i]:.4f}, "
            f"Recall={per_class_recall[i]:.4f}, "
            f"F1={per_class_f1[i]:.4f}, "
            f"Support={per_class_support[i]}"
        )
    
    return metrics


def plot_prediction_samples(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_samples: int = 10,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot sample predictions from the test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        num_samples: Number of samples to display
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    predictions = model.predict(X_test[:num_samples])
    
    rows = (num_samples + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3))
    axes = axes.flatten()
    
    for i in range(num_samples):
        pred_class = np.argmax(predictions[i])
        true_class = np.argmax(y_test[i])
        confidence = predictions[i][pred_class]
        
        # Display image
        axes[i].imshow(X_test[i].astype(np.uint8))
        axes[i].axis('off')
        
        # Color code: green for correct, red for incorrect
        color = 'green' if pred_class == true_class else 'red'
        axes[i].set_title(
            f'True: {true_class}\nPred: {pred_class} ({confidence:.2%})',
            color=color,
            fontsize=10
        )
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction samples to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
