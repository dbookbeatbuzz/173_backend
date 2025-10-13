"""Dataset manager abstractions used by the service layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from src.datasets.domainnet import build_domainnet_splits


@dataclass(slots=True)
class DatasetConfig:
    """Definition for a dataset that can be loaded by the service layer."""

    name: str
    root: str
    input_type: str  # 'text' or 'image'
    num_classes: int
    class_names: Optional[List[str]] = None
    preprocessing_config: Optional[str] = None
    additional_config: Optional[Dict[str, Any]] = None


class BaseDatasetManager(ABC):
    """Common dataset manager contract."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self._train_set: Optional[Dataset] = None
        self._val_set: Optional[Dataset] = None
        self._test_set: Optional[Dataset] = None
        self._class_names: Optional[List[str]] = None
        self._num_classes: Optional[int] = None

    @abstractmethod
    def _build_datasets(self) -> Tuple[Dataset, Dataset, Dataset, int]:
        """Produce train/val/test datasets and the number of labels."""

    @abstractmethod
    def get_class_name(self, class_idx: int) -> str:
        """Resolve a human readable class name given its index."""

    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset, int]:
        if self._train_set is None:
            self._train_set, self._val_set, self._test_set, self._num_classes = self._build_datasets()
        return self._train_set, self._val_set, self._test_set, self._num_classes  # type: ignore[return-value]

    def get_test_dataset(self) -> Tuple[Dataset, int]:
        if self._test_set is None:
            _, _, self._test_set, self._num_classes = self._build_datasets()
        return self._test_set, int(self._num_classes or 0)

    def get_sample(self, dataset_split: str, index: int):  # pragma: no cover - helper not used in tests
        datasets = {"train": self._train_set, "val": self._val_set, "test": self._test_set}
        if dataset_split not in datasets:
            raise ValueError(f"Invalid dataset split: {dataset_split}")

        if datasets[dataset_split] is None:
            self.get_datasets()

        split_dataset = datasets[dataset_split]
        if split_dataset is None:
            raise RuntimeError(f"Dataset split '{dataset_split}' is not available")
        return split_dataset[index]


class DomainNetDatasetManager(BaseDatasetManager):
    def _build_datasets(self) -> Tuple[Dataset, Dataset, Dataset, int]:
        preprocessor_path = self.config.preprocessing_config
        train_set, val_set, test_set, num_labels = build_domainnet_splits(
            root=self.config.root,
            preprocessor_path=preprocessor_path,
            seed=12345,
        )
        return train_set, val_set, test_set, num_labels

    def get_class_name(self, class_idx: int) -> str:
        if self._test_set is None:
            self.get_datasets()

        try:
            if hasattr(self._test_set, "dataset"):
                transform_wrapper = self._test_set.dataset  # type: ignore[attr-defined]
                if hasattr(transform_wrapper, "dataset"):
                    domainnet_dataset = transform_wrapper.dataset
                    if hasattr(domainnet_dataset, "idx_to_class"):
                        classes = domainnet_dataset.idx_to_class
                        if 0 <= class_idx < len(classes):
                            return classes[class_idx]

            if hasattr(self._test_set, "dataset") and hasattr(self._test_set.dataset, "idx_to_class"):
                classes = self._test_set.dataset.idx_to_class  # type: ignore[attr-defined]
                if 0 <= class_idx < len(classes):
                    return classes[class_idx]

        except (AttributeError, IndexError, TypeError):
            pass

        return f"class_{class_idx}"


class TextDatasetManager(BaseDatasetManager):  # pragma: no cover - placeholder
    def _build_datasets(self) -> Tuple[Dataset, Dataset, Dataset, int]:
        raise NotImplementedError("Text dataset manager not implemented yet")

    def get_class_name(self, class_idx: int) -> str:
        return f"class_{class_idx}"


class DatasetManagerFactory:
    _managers = {
        "domainnet": DomainNetDatasetManager,
        "text_classification": TextDatasetManager,
    }

    @classmethod
    def create_manager(cls, dataset_name: str, config: DatasetConfig) -> BaseDatasetManager:
        if dataset_name not in cls._managers:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        manager_class = cls._managers[dataset_name]
        return manager_class(config)

    @classmethod
    def register_manager(cls, dataset_name: str, manager_class: type):
        cls._managers[dataset_name] = manager_class

    @classmethod
    def list_supported_datasets(cls) -> List[str]:
        return list(cls._managers.keys())
