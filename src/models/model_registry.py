"""Model registry and configuration definitions."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelType(Enum):
    VISION_TRANSFORMER = "vit"
    CONVOLUTIONAL_NEURAL_NETWORK = "cnn"
    LSTM = "lstm"
    BERT = "bert"


class InputType(Enum):
    TEXT = "text"
    IMAGE = "image"


@dataclass(slots=True)
class ModelConfig:
    model_id: str
    name: str
    model_type: ModelType
    input_type: InputType
    description: str

    model_path: str
    checkpoint_pattern: Optional[str] = None

    num_labels: Optional[int] = None
    strategy: str = "adapter"
    adapter_last_k: int = 3
    adapter_bottleneck: int = 64

    dataset_name: str = "domainnet"
    dataset_config: Optional[Dict[str, Any]] = None

    preprocessing_config: Optional[str] = None
    device_preference: str = "auto"

    def __post_init__(self) -> None:
        if self.dataset_config is None:
            self.dataset_config = {}


class ModelRegistry:
    def __init__(self, config_file: Optional[str] = None):
        self._models: Dict[str, ModelConfig] = {}
        self._config_file = config_file or "model_registry.json"
        self._load_default_models()
        self._load_from_file()

    def _load_default_models(self) -> None:
        domainnet_vit = ModelConfig(
            model_id="1",
            name="ViT-DomainNet-FedSAK",
            model_type=ModelType.VISION_TRANSFORMER,
            input_type=InputType.IMAGE,
            description="Vision Transformer trained on DomainNet",
            model_path="exp_models/Domainnet_ViT_fedsak_lda",
            checkpoint_pattern="client/client_model_{client_id}.pt",
            num_labels=None,
            strategy="adapter",
            adapter_last_k=3,
            adapter_bottleneck=64,
            dataset_name="domainnet",
            dataset_config={
                "root": "/root/domainnet",
                "preprocessor_path": "pretrained_models/clip-vit-base-patch16/preprocessor_config.json",
                "classes_per_domain": 170,
            },
        )
        self._models[domainnet_vit.model_id] = domainnet_vit

    def _load_from_file(self) -> None:
        if not os.path.exists(self._config_file):
            return
        try:
            with open(self._config_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: Failed to load model registry from {self._config_file}: {exc}")
            return

        for model_data in data.get("models", []):
            config = ModelConfig(
                model_type=ModelType(model_data["model_type"]),
                input_type=InputType(model_data["input_type"]),
                **{k: v for k, v in model_data.items() if k not in {"model_type", "input_type"}},
            )
            self._models[config.model_id] = config

    def save_to_file(self) -> None:
        payload = {
            "models": [
                {
                    **asdict(config),
                    "model_type": config.model_type.value,
                    "input_type": config.input_type.value,
                }
                for config in self._models.values()
            ]
        }
        try:
            with open(self._config_file, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: Failed to save model registry to {self._config_file}: {exc}")

    def register_model(self, config: ModelConfig) -> None:
        self._models[config.model_id] = config

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        return self._models.get(model_id)

    def list_models(self) -> List[ModelConfig]:
        return list(self._models.values())

    def list_models_by_input_type(self, input_type: InputType) -> List[ModelConfig]:
        return [config for config in self._models.values() if config.input_type == input_type]

    def get_model_path(self, model_id: str, client_id: Optional[int] = None) -> Optional[str]:
        config = self.get_model(model_id)
        if not config:
            return None

        if config.checkpoint_pattern and client_id is not None:
            checkpoint_name = config.checkpoint_pattern.format(client_id=client_id)
            return os.path.join(config.model_path, checkpoint_name)
        return config.model_path

    def validate_model_exists(self, model_id: str, client_id: Optional[int] = None) -> bool:
        model_path = self.get_model_path(model_id, client_id)
        return model_path is not None and os.path.exists(model_path)


model_registry = ModelRegistry()
