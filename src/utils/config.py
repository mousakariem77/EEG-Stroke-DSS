"""
Configuration loader for the project.
Loads YAML config files and provides access to all hyperparameters.
"""
import yaml
import os
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, loads default config.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Default config path relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "default.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent
