"""Evaluation helpers used by CLI tools and HTTP handlers."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

from src.plugins import plugin_registry

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_client(
    model_id: str = "domainnet_vit_fedsak",
    client_id: int = 1,
    split: str = "test",
    limit: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate a client model on a dataset split."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of 'train', 'val', 'test'")

    # Get model plugin
    model_plugin_cls = plugin_registry.get_model_plugin(model_id)
    if not model_plugin_cls:
        raise ValueError(f"Model plugin not found: {model_id}")
    
    model_plugin = model_plugin_cls()
    logger.info(f"Evaluating model: {model_plugin.metadata.name}")
    
    # Get dataset plugin
    dataset_plugin_cls = plugin_registry.get_dataset_plugin(model_plugin.dataset_plugin_id)
    if not dataset_plugin_cls:
        raise ValueError(f"Dataset plugin not found: {model_plugin.dataset_plugin_id}")
    
    dataset_plugin = dataset_plugin_cls()
    logger.info(f"Using dataset: {dataset_plugin.metadata.name}")
    
    # Get datasets
    train_set, val_set, test_set = dataset_plugin.get_datasets()
    datasets = {"train": train_set, "val": val_set, "test": test_set}
    dataset = datasets[split]
    
    if limit is not None and limit > 0:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Get number of labels
    num_labels = dataset_plugin.get_num_classes()
    
    # Load model
    model_path = model_plugin.get_model_path(client_id)
    state_dict = model_plugin.load_checkpoint(model_path, device="cpu")
    
    model = model_plugin.build_model(num_labels=num_labels)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading client %s: %s ...", client_id, missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading client %s: %s ...", client_id, unexpected[:5])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    accuracy = correct / max(total, 1)
    return {
        "model_id": model_id,
        "client_id": client_id,
        "split": split,
        "samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
