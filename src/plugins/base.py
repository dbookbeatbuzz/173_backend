"""Base classes for plugin system."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ModelType(Enum):
    """Model architecture types."""
    VISION_TRANSFORMER = "vit"
    CONVOLUTIONAL_NEURAL_NETWORK = "cnn"
    LSTM = "lstm"
    BERT = "bert"
    RESNET = "resnet"


class InputType(Enum):
    """Input data types."""
    TEXT = "text"
    IMAGE = "image"


@dataclass
class ModelMetadata:
    """Model metadata - auto-populated via class attributes."""
    model_id: str                           # Unique string ID
    name: str                               # Display name
    model_type: ModelType                   # Model type
    input_type: InputType                   # Input type
    description: str                        # Description
    numeric_id: Optional[int] = None        # Numeric ID for frontend compatibility
    author: str = "Unknown"                 # Author
    version: str = "1.0.0"                  # Version
    tags: List[str] = field(default_factory=list)  # Tags


@dataclass
class DatasetMetadata:
    """Dataset metadata."""
    dataset_id: str                         # Unique ID
    name: str                               # Display name
    input_type: InputType                   # Input type
    num_classes: int                        # Number of classes
    description: str                        # Description
    default_root: str = None                # Default data path
    tags: List[str] = field(default_factory=list)  # Tags


class BaseModelPlugin(ABC):
    """Base class for model plugins - all models must inherit from this."""
    
    # ===== Required class attributes =====
    metadata: ModelMetadata = None          # Subclass must define
    dataset_plugin_id: str = None           # Associated dataset ID
    
    # ===== Optional configuration =====
    model_path: str = None                  # Model file path
    checkpoint_pattern: str = None          # Checkpoint pattern
    num_labels: Optional[int] = None        # Number of labels
    
    # Strategy related configuration
    strategy: str = "adapter"
    adapter_config: Dict[str, Any] = None
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin, can accept runtime config override."""
        self.config = config or {}
        if self.adapter_config is None:
            self.adapter_config = {}
        self.validate()
    
    @abstractmethod
    def build_model(self, num_labels: int, **kwargs) -> nn.Module:
        """Build model instance."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
        """Load checkpoint."""
        pass
    
    def get_model_path(self, client_id: Optional[int] = None) -> str:
        """Get model file path."""
        if self.checkpoint_pattern and client_id is not None:
            checkpoint = self.checkpoint_pattern.format(client_id=client_id)
            return os.path.join(self.model_path, checkpoint)
        return self.model_path
    
    def validate_model_exists(self, client_id: Optional[int] = None) -> bool:
        """Check if model file exists."""
        model_path = self.get_model_path(client_id)
        return model_path is not None and os.path.exists(model_path)
    
    def validate(self):
        """Validate configuration completeness."""
        if self.metadata is None:
            raise ValueError(f"{self.__class__.__name__} must define metadata attribute")
        if self.dataset_plugin_id is None:
            raise ValueError(f"{self.__class__.__name__} must define dataset_plugin_id attribute")
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Return preprocessing configuration."""
        return self.config.get("preprocessing", {})


class BaseDatasetPlugin(ABC):
    """Base class for dataset plugins."""
    
    # ===== Required class attributes =====
    metadata: DatasetMetadata = None
    
    def __init__(self, root: str = None, config: Optional[Dict[str, Any]] = None):
        """Initialize dataset plugin."""
        self.root = root or (self.metadata.default_root if self.metadata else None)
        self.config = config or {}
        self.validate()
        
        # Cache datasets
        self._train_set = None
        self._val_set = None
        self._test_set = None
        self._class_names = None
    
    @abstractmethod
    def build_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Build train/val/test datasets."""
        pass
    
    @abstractmethod
    def get_class_name(self, class_idx: int) -> str:
        """Get class name given its index."""
        pass
    
    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Get datasets (with caching)."""
        if self._train_set is None:
            self._train_set, self._val_set, self._test_set = self.build_datasets()
        return self._train_set, self._val_set, self._test_set
    
    def get_test_dataset(self) -> Tuple[Dataset, int]:
        """Get test dataset and number of classes."""
        if self._test_set is None:
            _, _, self._test_set = self.build_datasets()
        return self._test_set, self.get_num_classes()
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.metadata.num_classes
    
    def validate(self):
        """Validate configuration."""
        if self.metadata is None:
            raise ValueError(f"{self.__class__.__name__} must define metadata attribute")
        if not self.root:
            raise ValueError(f"{self.__class__.__name__} must provide root path")
