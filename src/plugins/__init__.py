"""Plugin system for models and datasets."""

from __future__ import annotations

import logging

from src.plugins.base import (
    BaseDatasetPlugin,
    BaseModelPlugin,
    DatasetMetadata,
    InputType,
    ModelMetadata,
    ModelType,
)
from src.plugins.registry import plugin_registry

logger = logging.getLogger(__name__)


def init_plugins():
    """Initialize plugin system - auto-discover all plugins."""
    try:
        plugin_registry.auto_discover()
        
        models = plugin_registry.list_models()
        datasets = plugin_registry.list_datasets()
        
        logger.info(f"Plugin system initialized successfully")
        logger.info(f"  Models: {[m.model_id for m in models]}")
        logger.info(f"  Datasets: {[d.dataset_id for d in datasets]}")
        
    except Exception as e:
        logger.error(f"Failed to initialize plugin system: {e}")
        raise


__all__ = [
    'plugin_registry',
    'init_plugins',
    'BaseModelPlugin',
    'BaseDatasetPlugin',
    'ModelMetadata',
    'DatasetMetadata',
    'ModelType',
    'InputType',
]
