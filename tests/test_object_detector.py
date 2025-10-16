"""
Test suite for the Street Scene Object Detection system.

This module contains unit tests for the main components of the system.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

# Import modules to test
from src.object_detector import StreetSceneDetector
from src.config import ConfigManager, ModelConfig, DataConfig, UIConfig, AppConfig


class TestStreetSceneDetector(unittest.TestCase):
    """Test cases for the StreetSceneDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = StreetSceneDetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.5
        )
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.model_path, "yolov8n.pt")
        self.assertEqual(self.detector.confidence_threshold, 0.5)
        self.assertIsNotNone(self.detector.model)
        self.assertIsInstance(self.detector.street_classes, dict)
    
    def test_street_classes(self):
        """Test street-relevant class definitions."""
        expected_classes = {0, 1, 2, 3, 5, 7, 9, 11, 13}
        self.assertEqual(set(self.detector.street_classes.keys()), expected_classes)
        
        # Check that important classes are included
        self.assertIn('person', self.detector.street_classes.values())
        self.assertIn('car', self.detector.street_classes.values())
        self.assertIn('truck', self.detector.street_classes.values())
    
    @patch('src.object_detector.YOLO')
    def test_model_loading_error(self, mock_yolo):
        """Test error handling during model loading."""
        mock_yolo.side_effect = Exception("Model loading failed")
        
        with self.assertRaises(Exception):
            StreetSceneDetector(model_path="invalid_model.pt")
    
    def test_extract_detection_data_empty(self):
        """Test extraction of detection data with no detections."""
        # Create a mock result with no detections
        mock_result = Mock()
        mock_result.boxes = None
        
        data = self.detector._extract_detection_data(mock_result)
        
        self.assertEqual(data['detections'], [])
        self.assertEqual(data['total_detections'], 0)
        self.assertEqual(data['street_relevant_detections'], 0)
    
    def test_extract_detection_data_with_detections(self):
        """Test extraction of detection data with detections."""
        # Create a mock result with detections
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
        mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([2])  # car
        mock_result.model.names = {2: 'car'}
        
        data = self.detector._extract_detection_data(mock_result)
        
        self.assertEqual(len(data['detections']), 1)
        self.assertEqual(data['total_detections'], 1)
        self.assertEqual(data['street_relevant_detections'], 1)
        
        detection = data['detections'][0]
        self.assertEqual(detection['class_name'], 'car')
        self.assertEqual(detection['confidence'], 0.8)
        self.assertTrue(detection['is_street_relevant'])
    
    def test_visualize_detections(self):
        """Test detection visualization."""
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create mock detection data
        detection_data = {
            'detections': [{
                'class_name': 'car',
                'confidence': 0.8,
                'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50},
                'is_street_relevant': True
            }]
        }
        
        annotated_image = self.detector.visualize_detections(test_image, detection_data)
        
        self.assertIsInstance(annotated_image, np.ndarray)
        self.assertEqual(annotated_image.shape, test_image.shape)
    
    def test_get_detection_summary(self):
        """Test detection summary generation."""
        detection_data = {
            'total_detections': 3,
            'street_relevant_detections': 2,
            'detections': [
                {'class_name': 'car', 'confidence': 0.8},
                {'class_name': 'person', 'confidence': 0.9},
                {'class_name': 'dog', 'confidence': 0.7}
            ]
        }
        
        summary = self.detector.get_detection_summary(detection_data)
        
        self.assertIn("Total objects detected: 3", summary)
        self.assertIn("Street-relevant objects: 2", summary)
        self.assertIn("car: 1", summary)
        self.assertIn("person: 1", summary)
        self.assertIn("dog: 1", summary)


class TestConfigManager(unittest.TestCase):
    """Test cases for the ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.config_manager = ConfigManager(self.config_path)
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = self.config_manager.get_config()
        
        self.assertIsInstance(config, AppConfig)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.ui, UIConfig)
    
    def test_config_update(self):
        """Test configuration updates."""
        self.config_manager.update_config(
            model_confidence_threshold=0.7,
            ui_show_labels=False
        )
        
        config = self.config_manager.get_config()
        self.assertEqual(config.model.confidence_threshold, 0.7)
        self.assertEqual(config.ui.show_labels, False)
    
    def test_config_save_and_load(self):
        """Test configuration save and load."""
        # Update configuration
        self.config_manager.update_config(
            model_confidence_threshold=0.8,
            data_input_dir="custom_input"
        )
        
        # Save configuration
        self.config_manager.save_config()
        
        # Create new config manager and load
        new_config_manager = ConfigManager(self.config_path)
        loaded_config = new_config_manager.get_config()
        
        self.assertEqual(loaded_config.model.confidence_threshold, 0.8)
        self.assertEqual(loaded_config.data.input_dir, "custom_input")


class TestModelConfig(unittest.TestCase):
    """Test cases for the ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        self.assertEqual(config.model_path, "yolov8n.pt")
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertIsNone(config.device)
        self.assertEqual(config.max_detections, 1000)
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            model_path="yolov8s.pt",
            confidence_threshold=0.7,
            device="cuda",
            max_detections=500
        )
        
        self.assertEqual(config.model_path, "yolov8s.pt")
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.max_detections, 500)


class TestDataConfig(unittest.TestCase):
    """Test cases for the DataConfig dataclass."""
    
    def test_default_values(self):
        """Test default data configuration values."""
        config = DataConfig()
        
        self.assertEqual(config.input_dir, "data/input")
        self.assertEqual(config.output_dir, "data/output")
        self.assertIn(".jpg", config.supported_formats)
        self.assertIn(".png", config.supported_formats)
    
    def test_post_init(self):
        """Test post-initialization behavior."""
        config = DataConfig()
        
        # Check that supported_formats is initialized
        self.assertIsNotNone(config.supported_formats)
        self.assertIsInstance(config.supported_formats, list)
        self.assertGreater(len(config.supported_formats), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main()
