"""
Inference utilities for TensorFlow Lite models.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TFLiteInference:
    """
    TensorFlow Lite model inference wrapper.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize TFLite interpreter.
        
        Args:
            model_path: Path to TFLite model file
        """
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.info(f"Loaded TFLite model from {model_path}")
        logger.info(f"Input shape: {self.input_details[0]['shape']}")
        logger.info(f"Output shape: {self.output_details[0]['shape']}")
    
    def preprocess_image(
        self,
        image_path: str,
        normalize: bool = False
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to image file
            normalize: Whether to normalize pixel values
            
        Returns:
            Tuple of (preprocessed_array, original_image)
        """
        # Get expected input shape
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Load and resize image
        img = Image.open(image_path).convert("RGB").resize((width, height))
        img_array = np.array(img, dtype=np.float32)
        
        if normalize:
            img_array /= 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Preprocessed input array
            
        Returns:
            Output predictions
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        return output_data
    
    def predict_from_image(
        self,
        image_path: str,
        normalize: bool = False
    ) -> Tuple[int, float, np.ndarray]:
        """
        Predict from image file.
        
        Args:
            image_path: Path to image file
            normalize: Whether to normalize pixel values
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        input_data, _ = self.preprocess_image(image_path, normalize)
        output_data = self.predict(input_data)
        
        predicted_class = int(np.argmax(output_data))
        confidence = float(output_data[predicted_class])
        
        return predicted_class, confidence, output_data
    
    def predict_and_visualize(
        self,
        image_path: str,
        class_names: Optional[list] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Tuple[int, float]:
        """
        Predict and visualize results.
        
        Args:
            image_path: Path to image file
            class_names: List of class names
            save_path: Path to save visualization (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        input_data, original_img = self.preprocess_image(image_path)
        output_data = self.predict(input_data)
        
        predicted_class = np.argmax(output_data)
        confidence = output_data[predicted_class]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Display image
        ax1.imshow(original_img)
        ax1.axis("off")
        ax1.set_title(
            f"Predicted: {predicted_class} ({confidence:.2%})",
            fontsize=14,
            fontweight='bold'
        )
        
        # Display bar chart
        if class_names is None:
            class_names = [str(i) for i in range(len(output_data))]
        
        colors = ['green' if i == predicted_class else 'skyblue' 
                 for i in range(len(output_data))]
        
        ax2.bar(range(len(output_data)), output_data, color=colors)
        ax2.set_xticks(range(len(output_data)))
        ax2.set_xticklabels(class_names)
        ax2.set_ylim(0, 1)
        ax2.set_title("Prediction Probabilities", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Confidence", fontsize=12)
        ax2.set_xlabel("Class", fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return int(predicted_class), float(confidence)
    
    def batch_predict(
        self,
        image_paths: list,
        normalize: bool = False
    ) -> list:
        """
        Predict on multiple images.
        
        Args:
            image_paths: List of image paths
            normalize: Whether to normalize pixel values
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                predicted_class, confidence, probabilities = self.predict_from_image(
                    image_path, normalize
                )
                results.append({
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities.tolist()
                })
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results


def create_inference_function(model_path: str):
    """
    Create a simple inference function.
    
    Args:
        model_path: Path to TFLite model
        
    Returns:
        Inference function
    """
    inference = TFLiteInference(model_path)
    
    def predict_and_plot(image_path: str):
        """Predict and visualize a single image."""
        return inference.predict_and_visualize(image_path)
    
    return predict_and_plot
