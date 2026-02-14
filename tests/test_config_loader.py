"""
Unit tests for configuration loader.
"""
import unittest
import tempfile
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import (
    load_config,
    get_data_config,
    get_training_config,
    get_model_config
)


class TestConfigLoader(unittest.TestCase):
    """Test cases for configuration loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'data': {
                'image_width': 20,
                'image_height': 32,
                'num_classes': 10
            },
            'training': {
                'batch_size': 4,
                'epochs': 100
            },
            'model': {
                'conv_filters': [32, 32, 32],
                'dense_units': 256
            }
        }
        
        # Create temporary config file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.test_config, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        Path(self.temp_file.name).unlink()
    
    def test_load_config(self):
        """Test loading configuration from file."""
        config = load_config(self.temp_file.name)
        
        self.assertIsInstance(config, dict)
        self.assertIn('data', config)
        self.assertIn('training', config)
        self.assertIn('model', config)
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with self.assertRaises(FileNotFoundError):
            load_config('nonexistent.yaml')
    
    def test_get_data_config(self):
        """Test extracting data configuration."""
        config = load_config(self.temp_file.name)
        data_config = get_data_config(config)
        
        self.assertEqual(data_config['image_width'], 20)
        self.assertEqual(data_config['image_height'], 32)
        self.assertEqual(data_config['num_classes'], 10)
    
    def test_get_training_config(self):
        """Test extracting training configuration."""
        config = load_config(self.temp_file.name)
        training_config = get_training_config(config)
        
        self.assertEqual(training_config['batch_size'], 4)
        self.assertEqual(training_config['epochs'], 100)
    
    def test_get_model_config(self):
        """Test extracting model configuration."""
        config = load_config(self.temp_file.name)
        model_config = get_model_config(config)
        
        self.assertEqual(model_config['conv_filters'], [32, 32, 32])
        self.assertEqual(model_config['dense_units'], 256)


if __name__ == '__main__':
    unittest.main()
