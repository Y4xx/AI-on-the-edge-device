"""
Configuration loader for the digit recognition model.
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data configuration."""
    return config.get('data', {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration."""
    return config.get('training', {})


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration."""
    return config.get('model', {})


def get_optimizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract optimizer configuration."""
    return config.get('optimizer', {})


def get_augmentation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract augmentation configuration."""
    return config.get('augmentation', {})


def get_callbacks_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract callbacks configuration."""
    return config.get('callbacks', {})


def get_export_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract export configuration."""
    return config.get('export', {})


def get_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inference configuration."""
    return config.get('inference', {})
