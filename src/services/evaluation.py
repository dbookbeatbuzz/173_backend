"""Evaluation helpers used by CLI tools and HTTP handlers."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.domainnet import build_domainnet_splits
from src.services.eval_model import build_model_for_eval

logger = logging.getLogger(__name__)


def load_client_checkpoint(save_root: str, client_id: int) -> Dict[str, torch.Tensor]:
    checkpoint_path = os.path.join(save_root, "client", f"client_model_{client_id}.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _infer_model_hyperparams_from_state(state: Dict[str, torch.Tensor]):
    strategy = "adapter" if any("_adapter" in key for key in state.keys()) else "linear"
    adapter_last_k = 3
    adapter_bottleneck = 64
    return strategy, adapter_last_k, adapter_bottleneck


@torch.no_grad()
def evaluate_client(
    client_id: int,
    split: str = "test",
    limit: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[str] = None,
    models_root: str = "exp_models/Domainnet_ViT_fedsak_lda",
    data_root: str = "/root/domainnet",
    preprocessor_json: Optional[str] = None,
) -> Dict[str, object]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of 'train', 'val', 'test'")

    train_set, val_set, test_set, num_labels = build_domainnet_splits(
        root=data_root,
        preprocessor_path=preprocessor_json or None,
        seed=12345,
    )
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

    state = load_client_checkpoint(models_root, client_id)
    strategy, adapter_last_k, adapter_bottleneck = _infer_model_hyperparams_from_state(state)

    model = build_model_for_eval(
        num_labels=num_labels,
        strategy=strategy,
        adapter_last_k=adapter_last_k,
        adapter_bottleneck=adapter_bottleneck,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
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
        "client_id": client_id,
        "split": split,
        "samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
