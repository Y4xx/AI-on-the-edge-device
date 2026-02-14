"""
Unit tests for data loading and preprocessing.
"""
import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import (
    load_images_from_folder,
    resize_images_in_folder,
    split_data,
    preprocess_single_image
)


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_width = 20
        self.test_height = 32
        self.num_classes = 10
        
        # Create test images
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
            img.save(Path(self.temp_dir) / f"{i}_test.jpg")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_images_from_folder(self):
        """Test loading images from folder."""
        x_data, y_data = load_images_from_folder(
            self.temp_dir,
            self.test_width,
            self.test_height,
            self.num_classes
        )
        
        self.assertEqual(len(x_data), 5)
        self.assertEqual(len(y_data), 5)
        self.assertEqual(y_data.shape[1], self.num_classes)
    
    def test_resize_images_in_folder(self):
        """Test resizing images."""
        output_dir = Path(self.temp_dir) / "resized"
        count = resize_images_in_folder(
            self.temp_dir,
            str(output_dir),
            self.test_width,
            self.test_height
        )
        
        self.assertEqual(count, 5)
        self.assertTrue(output_dir.exists())
        
        # Check if images are properly resized
        resized_img = Image.open(output_dir / "0_test.jpg")
        self.assertEqual(resized_img.size, (self.test_width, self.test_height))
    
    def test_split_data(self):
        """Test data splitting."""
        x_data = np.random.rand(100, 32, 20, 3)
        y_data = np.random.rand(100, 10)
        
        X_train, X_test, y_train, y_test = split_data(
            x_data, y_data, test_size=0.2, shuffle_data=True
        )
        
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
    
    def test_split_data_no_test(self):
        """Test data splitting with no test set."""
        x_data = np.random.rand(100, 32, 20, 3)
        y_data = np.random.rand(100, 10)
        
        X_train, X_test, y_train, y_test = split_data(
            x_data, y_data, test_size=0, shuffle_data=True
        )
        
        self.assertEqual(len(X_train), 100)
        self.assertIsNone(X_test)
        self.assertEqual(len(y_train), 100)
        self.assertIsNone(y_test)
    
    def test_preprocess_single_image(self):
        """Test preprocessing a single image."""
        test_image_path = Path(self.temp_dir) / "0_test.jpg"
        
        img_array, original_img = preprocess_single_image(
            str(test_image_path),
            self.test_width,
            self.test_height,
            normalize=False
        )
        
        self.assertEqual(img_array.shape, (1, self.test_height, self.test_width, 3))
        self.assertIsInstance(original_img, Image.Image)
    
    def test_preprocess_single_image_normalize(self):
        """Test preprocessing with normalization."""
        test_image_path = Path(self.temp_dir) / "0_test.jpg"
        
        img_array, _ = preprocess_single_image(
            str(test_image_path),
            self.test_width,
            self.test_height,
            normalize=True
        )
        
        # Check if values are normalized
        self.assertTrue(np.all(img_array >= 0.0))
        self.assertTrue(np.all(img_array <= 1.0))


if __name__ == '__main__':
    unittest.main()
