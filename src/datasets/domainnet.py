"""DomainNet dataset loaders and transforms."""

import json
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class DomainnetDataset(Dataset):
    """DomainNet multi-domain dataset reader."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.domain_ids: List[int] = []
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: List[str] = []
        self.domain_names: List[str] = []

        domains = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        domains.sort()
        self.domain_names = domains

        domain_to_selected = {}
        selected_class_names = set()
        for domain_name in domains:
            domain_path = os.path.join(root_dir, domain_name)
            class_names = [
                cls for cls in os.listdir(domain_path)
                if os.path.isdir(os.path.join(domain_path, cls))
            ]
            class_names.sort()
            picked = class_names[:170]
            domain_to_selected[domain_name] = set(picked)
            selected_class_names.update(picked)

        self.idx_to_class = sorted(selected_class_names)
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.idx_to_class)}

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        for domain_id, domain_name in enumerate(domains):
            domain_path = os.path.join(root_dir, domain_name)
            for class_name in sorted(domain_to_selected.get(domain_name, [])):
                class_dir = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                class_idx = self.class_to_idx[class_name]
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(valid_ext):
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(class_idx)
                        self.domain_ids.append(domain_id)

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return len(self.image_paths)

    def __getitem__(self, idx: int):  # pragma: no cover - simple passthrough
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return img, label


class TransformWrapper(Dataset):
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):  # pragma: no cover - thin wrapper
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):  # pragma: no cover - thin wrapper
        return len(self.dataset)

    def __getattr__(self, name):
        if name in ("dataset", "transform"):
            return super().__getattribute__(name)
        return getattr(self.dataset, name)


def _load_preprocessor_config(path: str):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def _pil_resample_to_interpolation_mode(resample: int) -> InterpolationMode:
    mapping = {
        0: InterpolationMode.NEAREST,
        1: InterpolationMode.LANCZOS,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
    }
    return mapping.get(resample, InterpolationMode.BICUBIC)


def _build_clip_transforms(pre_cfg: dict):
    size = pre_cfg.get("size", 224)
    crop_size = pre_cfg.get("crop_size", size)
    do_resize = pre_cfg.get("do_resize", True)
    do_center_crop = pre_cfg.get("do_center_crop", True)
    image_mean = pre_cfg.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
    image_std = pre_cfg.get("image_std", [0.26862954, 0.26130258, 0.27577711])
    resample = _pil_resample_to_interpolation_mode(pre_cfg.get("resample", 3))

    steps = []
    if do_resize:
        steps.append(transforms.Resize(size, interpolation=resample))
    if do_center_crop:
        steps.append(transforms.CenterCrop(crop_size))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])
    transform = transforms.Compose(steps)
    return transform, transform


def build_domainnet_splits(
    root: str = "/root/domainnet",
    preprocessor_path: Optional[str] = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, int]:
    """Return train/val/test datasets and the number of labels."""

    if preprocessor_path is None:
        preprocessor_path = (
            os.environ.get("CLIP_PREPROCESSOR_JSON")
            or "pretrained_models/clip-vit-base-patch16/preprocessor_config.json"
        )
    pre_cfg = _load_preprocessor_config(preprocessor_path) or {}
    train_transform, eval_transform = _build_clip_transforms(pre_cfg)

    full_dataset = DomainnetDataset(root)
    num_labels = len(full_dataset.class_to_idx)

    indices = np.arange(len(full_dataset))
    labels = [full_dataset.labels[i] for i in indices]

    train_tmp_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_tmp_idx,
        test_size=0.25,
        stratify=[labels[i] for i in train_tmp_idx],
        random_state=seed,
    )

    def _apply(idx_list, transform):
        return Subset(TransformWrapper(full_dataset, transform=transform), idx_list)

    train_set = _apply(train_idx, train_transform)
    val_set = _apply(val_idx, eval_transform)
    test_set = _apply(test_idx, eval_transform)

    return train_set, val_set, test_set, num_labels
