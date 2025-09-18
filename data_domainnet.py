import json
import os
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class DomainnetDataset(Dataset):
    """DomainNet multi-domain dataset reader.

    Assumes directory structure:
    root_dir/
      <domain_A>/
        <class_1>/*.jpg
        <class_2>/*.jpg
        ...
      <domain_B>/
      <domain_C>/
        ...
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.domain_ids: List[int] = []
        self.class_to_idx = {}
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
        for d in domains:
            dpath = os.path.join(root_dir, d)
            class_names = [
                cls for cls in os.listdir(dpath)
                if os.path.isdir(os.path.join(dpath, cls))
            ]
            class_names.sort()
            picked = class_names[:170]
            domain_to_selected[d] = set(picked)
            selected_class_names.update(picked)

        self.idx_to_class = sorted(selected_class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.idx_to_class)}

        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        for dom_id, d in enumerate(domains):
            dpath = os.path.join(root_dir, d)
            for cls in sorted(domain_to_selected.get(d, [])):
                cdir = os.path.join(dpath, cls)
                if not os.path.isdir(cdir):
                    continue
                cls_idx = self.class_to_idx[cls]
                for f in os.listdir(cdir):
                    if f.lower().endswith(valid_ext):
                        self.image_paths.append(os.path.join(cdir, f))
                        self.labels.append(cls_idx)
                        self.domain_ids.append(dom_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        return img, label


class TransformWrapper(Dataset):
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name in ('dataset', 'transform'):
            return super().__getattribute__(name)
        return getattr(self.dataset, name)


def _load_preprocessor_config(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
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
    size = pre_cfg.get('size', 224)
    crop_size = pre_cfg.get('crop_size', size)
    do_resize = pre_cfg.get('do_resize', True)
    do_center_crop = pre_cfg.get('do_center_crop', True)
    image_mean = pre_cfg.get('image_mean', [0.48145466, 0.4578275, 0.40821073])
    image_std = pre_cfg.get('image_std', [0.26862954, 0.26130258, 0.27577711])
    resample = _pil_resample_to_interpolation_mode(pre_cfg.get('resample', 3))

    steps = []
    if do_resize:
        steps.append(transforms.Resize(size, interpolation=resample))
    if do_center_crop:
        steps.append(transforms.CenterCrop(crop_size))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])
    t = transforms.Compose(steps)
    return t, t


def build_domainnet_splits(
    root: str = "/root/domainnet",
    preprocessor_path: Optional[str] = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, int]:
    """
    Returns (train_set, val_set, test_set, num_labels)
    """
    if preprocessor_path is None:
        preprocessor_path = (
            os.environ.get("CLIP_PREPROCESSOR_JSON")
            or "pretrained_models/clip-vit-base-patch16/preprocessor_config.json"
        )
    pre_cfg = _load_preprocessor_config(preprocessor_path) or {}
    train_t, eval_t = _build_clip_transforms(pre_cfg)

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

    def _apply(idx_list, t):
        return TransformWrapper(Subset(full_dataset, idx_list), transform=t)

    train_set = _apply(train_idx, train_t)
    val_set = _apply(val_idx, eval_t)
    test_set = _apply(test_idx, eval_t)

    return train_set, val_set, test_set, num_labels
