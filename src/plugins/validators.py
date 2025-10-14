"""Configuration validators for plugins."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class PluginValidator:
    """Validator for plugin configurations."""
    
    @staticmethod
    def validate_path_exists(path: str, name: str = "Path") -> None:
        """Validate that a path exists."""
        if not path:
            raise ValueError(f"{name} cannot be empty")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")
    
    @staticmethod
    def validate_positive_int(value: int, name: str = "Value") -> None:
        """Validate that a value is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got: {value}")
    
    @staticmethod
    def validate_dict(value: Any, name: str = "Value") -> None:
        """Validate that a value is a dictionary."""
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a dictionary, got: {type(value)}")
    
    @staticmethod
    def validate_choice(value: Any, choices: List[Any], name: str = "Value") -> None:
        """Validate that a value is in a list of choices."""
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices}, got: {value}")


class ModelPluginValidator(PluginValidator):
    """Validator specifically for model plugins."""
    
    @staticmethod
    def validate_strategy(strategy: str) -> None:
        """Validate model strategy."""
        valid_strategies = ["adapter", "linear", "full"]
        PluginValidator.validate_choice(strategy, valid_strategies, "Strategy")
    
    @staticmethod
    def validate_adapter_config(config: Dict[str, Any]) -> None:
        """Validate adapter configuration."""
        PluginValidator.validate_dict(config, "Adapter config")
        
        if "last_k" in config:
            PluginValidator.validate_positive_int(config["last_k"], "adapter_last_k")
        
        if "bottleneck" in config:
            PluginValidator.validate_positive_int(config["bottleneck"], "adapter_bottleneck")


class DatasetPluginValidator(PluginValidator):
    """Validator specifically for dataset plugins."""
    
    @staticmethod
    def validate_root_dir(root: str) -> None:
        """Validate dataset root directory."""
        PluginValidator.validate_path_exists(root, "Dataset root directory")
        
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Dataset root must be a directory: {root}")
