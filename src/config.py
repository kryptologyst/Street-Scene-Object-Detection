"""
Configuration module for the Object Detection project.

This module handles all configuration settings including model parameters,
file paths, and application settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for the object detection model."""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    device: Optional[str] = None
    max_detections: int = 1000


@dataclass
class DataConfig:
    """Configuration for data handling."""
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    sample_image_url: str = "https://images.pexels.com/photos/167832/pexels-photo-167832.jpeg"
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']


@dataclass
class UIConfig:
    """Configuration for the user interface."""
    title: str = "Street Scene Object Detection"
    description: str = "Detect objects in street scenes using YOLOv8"
    max_file_size_mb: int = 10
    show_confidence: bool = True
    show_labels: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    data: DataConfig
    ui: UIConfig
    log_level: str = "INFO"
    debug: bool = False


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                return self._create_config_from_dict(config_data)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration.")
        
        return self._get_default_config()
    
    def _create_config_from_dict(self, config_data: Dict) -> AppConfig:
        """Create AppConfig from dictionary."""
        model_config = ModelConfig(**config_data.get('model', {}))
        data_config = DataConfig(**config_data.get('data', {}))
        ui_config = UIConfig(**config_data.get('ui', {}))
        
        return AppConfig(
            model=model_config,
            data=data_config,
            ui=ui_config,
            log_level=config_data.get('log_level', 'INFO'),
            debug=config_data.get('debug', False)
        )
    
    def _get_default_config(self) -> AppConfig:
        """Get default configuration."""
        return AppConfig(
            model=ModelConfig(),
            data=DataConfig(),
            ui=UIConfig()
        )
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = config_path or self.config_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        config_dict = {
            'model': {
                'model_path': self.config.model.model_path,
                'confidence_threshold': self.config.model.confidence_threshold,
                'device': self.config.model.device,
                'max_detections': self.config.model.max_detections
            },
            'data': {
                'input_dir': self.config.data.input_dir,
                'output_dir': self.config.data.output_dir,
                'sample_image_url': self.config.data.sample_image_url,
                'supported_formats': self.config.data.supported_formats
            },
            'ui': {
                'title': self.config.ui.title,
                'description': self.config.ui.description,
                'max_file_size_mb': self.config.ui.max_file_size_mb,
                'show_confidence': self.config.ui.show_confidence,
                'show_labels': self.config.ui.show_labels
            },
            'log_level': self.config.log_level,
            'debug': self.config.debug
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                # Handle nested attributes
                parts = key.split('.')
                if len(parts) == 2:
                    parent, child = parts
                    if hasattr(self.config, parent):
                        parent_obj = getattr(self.config, parent)
                        if hasattr(parent_obj, child):
                            setattr(parent_obj, child, value)


# Global configuration instance
config_manager = ConfigManager()
config = config_manager.get_config()
