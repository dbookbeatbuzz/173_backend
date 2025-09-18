import json
import logging
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from eval_model import build_model_for_eval
from data_domainnet import build_domainnet_splits

logger = logging.getLogger(__name__)


def load_client_checkpoint(save_root: str, client_id: int) -> Dict[str, torch.Tensor]:
    ckpt_path = os.path.join(save_root, "client", f"client_model_{client_id}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    # fallback: assume it's a pure state_dict
    return obj


def _infer_model_hyperparams_from_state(state: Dict[str, torch.Tensor]):
    # Heuristic defaults aligning with training setup
    strategy = "adapter" if any("_adapter" in k for k in state.keys()) else "linear"
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
) -> Dict:
    assert split in {"train", "val", "test"}

    # Dataset
    train_set, val_set, test_set, num_labels = build_domainnet_splits(
        root=data_root, preprocessor_path=preprocessor_json or None, seed=12345
    )
    ds_map = {"train": train_set, "val": val_set, "test": test_set}
    dataset = ds_map[split]

    if limit is not None and limit > 0:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True)

    # Checkpoint
    state = load_client_checkpoint(models_root, client_id)
    strategy, adapter_last_k, adapter_bottleneck = _infer_model_hyperparams_from_state(state)

    # Model
    model = build_model_for_eval(
        num_labels=num_labels,
        strategy=strategy,
        adapter_last_k=adapter_last_k,
        adapter_bottleneck=adapter_bottleneck,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading client {client_id}: {missing[:5]} ...")
    if unexpected:
        logger.warning(f"Unexpected keys when loading client {client_id}: {unexpected[:5]} ...")

    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Eval loop
    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = correct / max(total, 1)
    result = {
        "client_id": client_id,
        "split": split,
        "samples": total,
        "correct": correct,
        "accuracy": acc,
    }
    return result
