"""Plugin auto-discovery and registry system."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Dict, List, Optional, Type

from src.plugins.base import BaseDatasetPlugin, BaseModelPlugin, DatasetMetadata, ModelMetadata

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Plugin auto-discovery and registry system."""
    
    def __init__(self):
        self._model_plugins: Dict[str, Type[BaseModelPlugin]] = {}
        self._dataset_plugins: Dict[str, Type[BaseDatasetPlugin]] = {}
        self._model_id_to_string: Dict[int, str] = {}  # numeric_id -> model_id mapping
        self._initialized = False
    
    def auto_discover(self):
        """Auto-discover all plugins."""
        if self._initialized:
            return
        
        logger.info("Starting plugin auto-discovery...")
        self._discover_plugins('src.plugins.models', BaseModelPlugin, self._model_plugins)
        self._discover_plugins('src.plugins.datasets', BaseDatasetPlugin, self._dataset_plugins)
        self._build_numeric_id_mapping()
        self._initialized = True
        
        logger.info(f"✓ Loaded {len(self._model_plugins)} model plugin(s)")
        logger.info(f"✓ Loaded {len(self._dataset_plugins)} dataset plugin(s)")
    
    def _build_numeric_id_mapping(self):
        """Build numeric ID to string ID mapping from plugin metadata."""
        for model_id, plugin_cls in self._model_plugins.items():
            if plugin_cls.metadata and plugin_cls.metadata.numeric_id is not None:
                numeric_id = plugin_cls.metadata.numeric_id
                self._model_id_to_string[numeric_id] = model_id
                logger.debug(f"Mapped numeric ID {numeric_id} -> {model_id}")
    
    def _discover_plugins(self, package_name: str, base_class: Type, registry: Dict):
        """Scan all plugin classes under specified package."""
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.warning(f"Could not import package {package_name}: {e}")
            return
        
        # Iterate through all modules in the package
        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__, 
            prefix=f"{package_name}."
        ):
            try:
                module = importlib.import_module(modname)
                
                # Find all classes that inherit from base class
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, base_class) and 
                        obj is not base_class and
                        not inspect.isabstract(obj)):
                        
                        # Use ID from metadata as key
                        if hasattr(obj, 'metadata') and obj.metadata:
                            if isinstance(obj.metadata, DatasetMetadata):
                                plugin_id = obj.metadata.dataset_id
                            else:
                                plugin_id = obj.metadata.model_id
                            
                            registry[plugin_id] = obj
                            logger.info(f"  ✓ Discovered plugin: {plugin_id} ({obj.__name__})")
            except Exception as e:
                logger.error(f"  ✗ Failed to load module {modname}: {e}")
    
    def resolve_model_id(self, model_id) -> Optional[str]:
        """Resolve model_id (int or str) to string model_id."""
        # If it's a string, return as-is if exists
        if isinstance(model_id, str):
            return model_id if model_id in self._model_plugins else None
        
        # If it's an int, look up in numeric mapping
        try:
            numeric_id = int(model_id)
            return self._model_id_to_string.get(numeric_id)
        except (ValueError, TypeError):
            return None
    
    def get_model_plugin(self, model_id) -> Optional[Type[BaseModelPlugin]]:
        """Get model plugin class. Accepts int or str model_id."""
        # Resolve to string ID first
        string_id = self.resolve_model_id(model_id)
        if not string_id:
            return None
        return self._model_plugins.get(string_id)
    
    def get_dataset_plugin(self, dataset_id: str) -> Optional[Type[BaseDatasetPlugin]]:
        """Get dataset plugin class."""
        return self._dataset_plugins.get(dataset_id)
    
    def list_models(self) -> List[ModelMetadata]:
        """List all available models."""
        return [cls.metadata for cls in self._model_plugins.values()]
    
    def list_datasets(self) -> List[DatasetMetadata]:
        """List all available datasets."""
        return [cls.metadata for cls in self._dataset_plugins.values()]
    
    def get_model_metadata(self, model_id) -> Optional[ModelMetadata]:
        """Get model metadata by ID. Accepts int or str model_id."""
        plugin_cls = self.get_model_plugin(model_id)
        return plugin_cls.metadata if plugin_cls else None
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by ID."""
        plugin_cls = self.get_dataset_plugin(dataset_id)
        return plugin_cls.metadata if plugin_cls else None
    
    def validate_model_exists(self, model_id, client_id: Optional[int] = None) -> bool:
        """Validate if model file exists. Accepts int or str model_id."""
        plugin_cls = self.get_model_plugin(model_id)
        if not plugin_cls:
            return False
        plugin = plugin_cls()
        return plugin.validate_model_exists(client_id)


# Global singleton
plugin_registry = PluginRegistry()
