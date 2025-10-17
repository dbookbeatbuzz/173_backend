"""Graph Multi-Domain Molecular dataset plugin."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset

from src.plugins.base import BaseDatasetPlugin, DatasetMetadata, InputType

logger = logging.getLogger(__name__)


class GraphDatasetWrapper(Dataset):
    """Wrapper for TUDataset that adds edge_attr补边逻辑."""

    def __init__(self, tu_dataset: TUDataset, transform=None):
        """
        Initialize wrapper.

        Args:
            tu_dataset: TUDataset instance
            transform: Optional runtime transform
        """
        self.dataset = tu_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # 补边: 如果没有edge_attr或为None,补充为全0
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            num_edges = data.edge_index.size(1)
            data.edge_attr = torch.zeros((num_edges, 1), dtype=torch.float)

        # 应用runtime transform
        if self.transform:
            data = self.transform(data)

        return data


class GraphMultiDomainMolPlugin(BaseDatasetPlugin):
    """Graph Multi-Domain Molecular dataset plugin.

    Combines multiple molecular graph datasets from TUDataset:
    - MUTAG
    - BZR
    - COX2
    - DHFR
    - PTC_MR
    - AIDS
    - NCI1
    """

    metadata = DatasetMetadata(
        dataset_id="graph_multi_domain_mol",
        name="Graph Multi-Domain Molecular",
        input_type=InputType.GRAPH,
        num_classes=2,  # Most molecular datasets are binary classification
        description="Multi-domain graph classification dataset with 7 molecular domains",
        default_root="./data/graph",
        tags=["multi-domain", "classification", "graph", "molecular"]
    )

    # 子域列表
    SUB_DOMAINS = ['MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1']

    def build_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Build train/val/test datasets.

        Note: 不划分客户端,所有数据合并为一个整体数据集.
        每个子域的数据会按8:1:1的比例划分为train/val/test.
        """
        from torch_geometric import transforms
        from torch.utils.data import ConcatDataset, Subset

        # 获取transform配置
        runtime_transform = self.config.get("transform", None)
        pre_transform = self.config.get("pre_transform", None)

        # 对IMDB类数据集添加constant transform
        transforms_funcs = {}
        if pre_transform:
            transforms_funcs['pre_transform'] = pre_transform

        logger.info(f"Building Graph Multi-Domain Molecular datasets from {self.root}")
        logger.info(f"  Sub-domains: {', '.join(self.SUB_DOMAINS)}")

        all_train_sets = []
        all_val_sets = []
        all_test_sets = []
        total_samples = 0

        # 加载每个子域并划分
        for domain_name in self.SUB_DOMAINS:
            logger.info(f"  Loading domain: {domain_name}")

            # 对IMDB数据集特殊处理
            if domain_name.startswith('IMDB') and 'pre_transform' not in transforms_funcs:
                transforms_funcs['pre_transform'] = transforms.Constant(value=1.0, cat=False)

            # 加载TUDataset
            try:
                tu_dataset = TUDataset(self.root, domain_name, **transforms_funcs)
                dataset_size = len(tu_dataset)
                total_samples += dataset_size

                # 划分为train/val/test (8:1:1)
                train_size = int(0.8 * dataset_size)
                val_size = int(0.1 * dataset_size)
                test_size = dataset_size - train_size - val_size

                # 创建索引
                indices = list(range(dataset_size))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:train_size + val_size]
                test_indices = indices[train_size + val_size:]

                # 创建wrapper并添加到列表
                wrapped_dataset = GraphDatasetWrapper(tu_dataset, transform=runtime_transform)

                train_subset = Subset(wrapped_dataset, train_indices)
                val_subset = Subset(wrapped_dataset, val_indices)
                test_subset = Subset(wrapped_dataset, test_indices)

                all_train_sets.append(train_subset)
                all_val_sets.append(val_subset)
                all_test_sets.append(test_subset)

                logger.info(f"    {domain_name}: {dataset_size} samples "
                           f"(train: {train_size}, val: {val_size}, test: {test_size})")

            except Exception as e:
                logger.error(f"    Failed to load {domain_name}: {e}")
                raise

        # 合并所有子域的数据集
        train_set = ConcatDataset(all_train_sets)
        val_set = ConcatDataset(all_val_sets)
        test_set = ConcatDataset(all_test_sets)

        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Train: {len(train_set)} samples")
        logger.info(f"  Val: {len(val_set)} samples")
        logger.info(f"  Test: {len(test_set)} samples")

        return train_set, val_set, test_set

    def get_class_name(self, class_idx: int) -> str:
        """Get class name given its index.

        For molecular datasets, typically:
        - 0: non-active/negative
        - 1: active/positive
        """
        class_names = {
            0: "non-active",
            1: "active"
        }
        return class_names.get(class_idx, f"class_{class_idx}")
