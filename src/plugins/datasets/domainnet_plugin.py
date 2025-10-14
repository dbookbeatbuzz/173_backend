"""DomainNet dataset plugin."""

from __future__ import annotations

import logging
from typing import Tuple

from torch.utils.data import Dataset

from src.datasets.domainnet import build_domainnet_splits
from src.plugins.base import BaseDatasetPlugin, DatasetMetadata, InputType

logger = logging.getLogger(__name__)


class DomainNetPlugin(BaseDatasetPlugin):
    """DomainNet dataset plugin."""
    
    metadata = DatasetMetadata(
        dataset_id="domainnet",
        name="DomainNet",
        input_type=InputType.IMAGE,
        num_classes=170,
        description="Multi-domain image classification dataset with 6 domains",
        default_root="/root/domainnet",
        tags=["multi-domain", "classification", "vision"]
    )
    
    def build_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Build train/val/test datasets."""
        preprocessor_path = self.config.get(
            "preprocessor_path",
            "pretrained_models/clip-vit-base-patch16/preprocessor_config.json"
        )
        
        seed = self.config.get("seed", 42)
        
        logger.info(f"Building DomainNet datasets from {self.root}")
        logger.info(f"  Preprocessor: {preprocessor_path}")
        logger.info(f"  Seed: {seed}")
        
        train_set, val_set, test_set, num_labels = build_domainnet_splits(
            root=self.root,
            preprocessor_path=preprocessor_path,
            seed=seed,
        )
        
        logger.info(f"  Train: {len(train_set)} samples")
        logger.info(f"  Val: {len(val_set)} samples")
        logger.info(f"  Test: {len(test_set)} samples")
        logger.info(f"  Classes: {num_labels}")
        
        return train_set, val_set, test_set
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name given its index."""
        _, _, test_set = self.get_datasets()
        
        # Try to extract class names from the dataset
        try:
            # Navigate through the Subset and TransformWrapper layers
            if hasattr(test_set, 'dataset'):
                dataset = test_set.dataset
                
                # TransformWrapper
                if hasattr(dataset, 'dataset'):
                    domainnet_dataset = dataset.dataset
                    
                    # DomainnetDataset
                    if hasattr(domainnet_dataset, 'idx_to_class'):
                        classes = domainnet_dataset.idx_to_class
                        if 0 <= class_idx < len(classes):
                            return classes[class_idx]
            
            # Direct access attempt
            if hasattr(test_set, 'dataset') and hasattr(test_set.dataset, 'idx_to_class'):
                classes = test_set.dataset.idx_to_class
                if 0 <= class_idx < len(classes):
                    return classes[class_idx]
                    
        except (AttributeError, IndexError, TypeError) as e:
            logger.debug(f"Could not extract class name for index {class_idx}: {e}")
        
        return f"class_{class_idx}"
