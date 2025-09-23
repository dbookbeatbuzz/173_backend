"""
数据集管理模块
提供统一的数据集访问接口，支持多种数据集类型
"""

import os
from typing import Dict, List, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader

from data_domainnet import build_domainnet_splits


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    root: str
    input_type: str  # 'text' or 'image'
    num_classes: int
    class_names: Optional[List[str]] = None
    preprocessing_config: Optional[str] = None
    additional_config: Optional[Dict[str, Any]] = None


class BaseDatasetManager(ABC):
    """数据集管理器基类"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._train_set = None
        self._val_set = None
        self._test_set = None
        self._class_names = None
        self._num_classes = None
    
    @abstractmethod
    def _build_datasets(self) -> Tuple[Dataset, Dataset, Dataset, int]:
        """构建训练、验证、测试数据集，返回 (train_set, val_set, test_set, num_classes)"""
        pass
    
    @abstractmethod
    def get_class_name(self, class_idx: int) -> str:
        """根据类别索引获取类别名称"""
        pass
    
    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset, int]:
        """获取数据集"""
        if self._train_set is None:
            self._train_set, self._val_set, self._test_set, self._num_classes = self._build_datasets()
        return self._train_set, self._val_set, self._test_set, self._num_classes
    
    def get_test_dataset(self) -> Tuple[Dataset, int]:
        """只获取测试数据集（用于模型测试）"""
        if self._test_set is None:
            _, _, self._test_set, self._num_classes = self._build_datasets()
        return self._test_set, self._num_classes
    
    def get_sample(self, dataset_split: str, index: int) -> Tuple[Any, int]:
        """获取指定数据集分割的特定样本"""
        datasets = {'train': self._train_set, 'val': self._val_set, 'test': self._test_set}
        if dataset_split not in datasets:
            raise ValueError(f"Invalid dataset split: {dataset_split}")
        
        if datasets[dataset_split] is None:
            self.get_datasets()
        
        return datasets[dataset_split][index]


class DomainNetDatasetManager(BaseDatasetManager):
    """DomainNet数据集管理器"""
    
    def _build_datasets(self) -> Tuple[Dataset, Dataset, Dataset, int]:
        """构建DomainNet数据集"""
        preprocessor_path = self.config.preprocessing_config
        
        train_set, val_set, test_set, num_labels = build_domainnet_splits(
            root=self.config.root,
            preprocessor_path=preprocessor_path,
            seed=12345
        )
        
        return train_set, val_set, test_set, num_labels
    
    def get_class_name(self, class_idx: int) -> str:
        """获取DomainNet类别名称"""
        # 确保数据集已加载
        if self._test_set is None:
            self.get_datasets()
        
        # 尝试从数据集对象获取类别名称
        # 对于 Subset(TransformWrapper(DomainnetDataset)) 的结构
        try:
            # 尝试访问实际的 DomainnetDataset
            if hasattr(self._test_set, 'dataset'):
                # self._test_set 是 Subset
                transform_wrapper = self._test_set.dataset
                if hasattr(transform_wrapper, 'dataset'):
                    # transform_wrapper 是 TransformWrapper
                    domainnet_dataset = transform_wrapper.dataset
                    if hasattr(domainnet_dataset, 'idx_to_class'):
                        # domainnet_dataset 是 DomainnetDataset
                        if 0 <= class_idx < len(domainnet_dataset.idx_to_class):
                            return domainnet_dataset.idx_to_class[class_idx]
            
            # 回退方案：直接从 transform_wrapper 获取
            if hasattr(self._test_set, 'dataset') and hasattr(self._test_set.dataset, 'idx_to_class'):
                if 0 <= class_idx < len(self._test_set.dataset.idx_to_class):
                    return self._test_set.dataset.idx_to_class[class_idx]
        
        except (AttributeError, IndexError, TypeError):
            pass
        
        # 最后的回退方案
        return f"class_{class_idx}"


class TextDatasetManager(BaseDatasetManager):
    """文本数据集管理器（示例实现）"""
    
    def _build_datasets(self) -> Tuple[Dataset, Dataset, Dataset, int]:
        """构建文本数据集（示例实现）"""
        # 这里应该实现具体的文本数据集加载逻辑
        # 暂时返回空实现
        raise NotImplementedError("Text dataset manager not implemented yet")
    
    def get_class_name(self, class_idx: int) -> str:
        """获取文本分类类别名称"""
        # 根据具体数据集实现
        return f"class_{class_idx}"


class DatasetManagerFactory:
    """数据集管理器工厂"""
    
    _managers = {
        'domainnet': DomainNetDatasetManager,
        'text_classification': TextDatasetManager,
    }
    
    @classmethod
    def create_manager(cls, dataset_name: str, config: DatasetConfig) -> BaseDatasetManager:
        """创建数据集管理器"""
        if dataset_name not in cls._managers:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        manager_class = cls._managers[dataset_name]
        return manager_class(config)
    
    @classmethod
    def register_manager(cls, dataset_name: str, manager_class: type):
        """注册新的数据集管理器"""
        cls._managers[dataset_name] = manager_class
    
    @classmethod
    def list_supported_datasets(cls) -> List[str]:
        """列出支持的数据集类型"""
        return list(cls._managers.keys())