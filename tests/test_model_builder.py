"""
Unit tests for model builder.
"""
import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_builder import (
    create_digit_recognition_model,
    compile_model,
    get_model_summary
)


class TestModelBuilder(unittest.TestCase):
    """Test cases for model building functions."""
    
    def test_create_model_default(self):
        """Test creating model with default parameters."""
        model = create_digit_recognition_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers) > 0, True)
        self.assertEqual(model.input_shape, (None, 32, 20, 3))
        self.assertEqual(model.output_shape, (None, 10))
    
    def test_create_model_custom(self):
        """Test creating model with custom parameters."""
        model = create_digit_recognition_model(
            input_shape=(64, 40, 3),
            num_classes=5,
            conv_filters=[16, 32],
            dense_units=128,
            dropout_rate=0.3
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 64, 40, 3))
        self.assertEqual(model.output_shape, (None, 5))
    
    def test_compile_model_adam(self):
        """Test compiling model with Adam optimizer."""
        model = create_digit_recognition_model()
        compiled_model = compile_model(model, optimizer_name='adam', learning_rate=0.001)
        
        self.assertIsNotNone(compiled_model.optimizer)
        self.assertEqual(compiled_model.optimizer.__class__.__name__, 'Adam')
    
    def test_compile_model_adadelta(self):
        """Test compiling model with Adadelta optimizer."""
        model = create_digit_recognition_model()
        compiled_model = compile_model(model, optimizer_name='adadelta', learning_rate=1.0)
        
        self.assertIsNotNone(compiled_model.optimizer)
        self.assertEqual(compiled_model.optimizer.__class__.__name__, 'Adadelta')
    
    def test_get_model_summary(self):
        """Test getting model summary."""
        model = create_digit_recognition_model()
        summary = get_model_summary(model)
        
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
        self.assertIn('Total params', summary)
    
    def test_model_parameters_count(self):
        """Test that model has trainable parameters."""
        model = create_digit_recognition_model()
        
        param_count = model.count_params()
        self.assertGreater(param_count, 0)


if __name__ == '__main__':
    unittest.main()
