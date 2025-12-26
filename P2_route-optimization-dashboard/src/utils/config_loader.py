"""
Configuration file loading utilities.

Supports YAML format for ship specifications, scenarios, and optimizer parameters.
"""

from typing import Any, Dict
import yaml
from pathlib import Path


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary with configuration data

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is invalid YAML
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config or {}


def save_yaml(data: Dict[str, Any], file_path: str):
    """Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
