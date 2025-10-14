"""DomainNet ViT model plugin with complete model implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import CLIPModel

from src.plugins.base import BaseModelPlugin, InputType, ModelMetadata, ModelType

logger = logging.getLogger(__name__)


# ===== Model Architecture Implementation =====

def _adapter_residual_hook(module, _input, output):
    """Forward hook to inject adapter into transformer layer."""
    adapter_module = getattr(module, "_adapter", None)
    if adapter_module is None:
        return output

    if isinstance(output, tuple):
        hidden_states = output[0]
        hidden_states = adapter_module(hidden_states)
        return (hidden_states,) + output[1:]
    if hasattr(output, "hidden_states"):
        hidden_states = adapter_module(output.hidden_states)
        output.hidden_states = hidden_states
        return output
    hidden_states = adapter_module(output)
    return hidden_states


class Adapter(nn.Module):
    """Lightweight adapter module for parameter-efficient fine-tuning."""
    
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)

    def forward(self, inputs):
        """Apply adapter with residual connection."""
        return inputs + self.up(self.act(self.down(inputs)))


class CLIPVisionWithHead(nn.Module):
    """CLIP Vision model with classification head and optional adapters."""
    
    def __init__(
        self,
        num_labels: int,
        pretrained_dir: str = "pretrained_models/clip-vit-base-patch16",
        strategy: str = "adapter",
        adapter_last_k: int = 3,
        adapter_bottleneck: int = 64,
        local_files_only: bool = True,
    ):
        super().__init__()
        
        # Load pretrained CLIP model
        try:
            clip = CLIPModel.from_pretrained(
                pretrained_dir,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            message = (
                f"Failed to load CLIPModel from '{pretrained_dir}'. "
                "Ensure the pretrained files exist locally or set CLIP_PRETRAINED_DIR. "
                f"Original error: {exc}"
            )
            logger.error(message)
            raise RuntimeError(message) from exc

        self.vision = clip.vision_model
        hidden_dim = self.vision.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # Setup adapters if needed
        self.adapters: Optional[nn.ModuleDict] = None
        if strategy == "adapter":
            layers = self.vision.encoder.layers
            if not 1 <= adapter_last_k <= len(layers):
                raise ValueError(f"adapter_last_k must be within 1 and {len(layers)}")
            
            target_indices = range(len(layers) - adapter_last_k, len(layers))
            self.adapters = nn.ModuleDict()
            self._hooks = []
            
            for idx in target_indices:
                adapter_module = Adapter(hidden_dim, adapter_bottleneck)
                self.adapters[f"layer_{idx}"] = adapter_module
                setattr(layers[idx], "_adapter", adapter_module)
                hook = layers[idx].register_forward_hook(_adapter_residual_hook)
                self._hooks.append(hook)
                
            logger.debug(f"Initialized adapters in layers {list(target_indices)}")
        elif strategy == "linear":
            logger.debug("Using linear probe strategy (no adapters)")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @torch.no_grad()
    def forward(self, pixel_values):
        """Forward pass through vision model and classifier."""
        outputs = self.vision(pixel_values=pixel_values, output_hidden_states=False)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits


# ===== Plugin Implementation =====


class DomainNetViTPlugin(BaseModelPlugin):
    """DomainNet ViT model plugin - trained with FedSAK."""
    
    # ===== Metadata definition =====
    metadata = ModelMetadata(
        model_id="domainnet_vit_fedsak",
        name="ViT-DomainNet-FedSAK",
        model_type=ModelType.VISION_TRANSFORMER,
        input_type=InputType.IMAGE,
        description="Vision Transformer trained on DomainNet with FedSAK (Federated Learning with Adapter)",
        numeric_id=1,
        author="Federated Learning Team",
        version="1.0.0",
        tags=["federated", "vision", "adapter", "vit"]
    )
    
    # ===== Configuration definition =====
    dataset_plugin_id = "domainnet"
    model_path = "exp_models/Domainnet_ViT_fedsak_lda"
    checkpoint_pattern = "client/client_model_{client_id}.pt"
    
    strategy = "adapter"
    adapter_config = {
        "last_k": 3,
        "bottleneck": 64
    }
    
    def build_model(self, num_labels: int, **kwargs) -> nn.Module:
        """Build model instance."""
        strategy = kwargs.get("strategy", self.strategy)
        adapter_last_k = kwargs.get("adapter_last_k", self.adapter_config.get("last_k", 3))
        adapter_bottleneck = kwargs.get("adapter_bottleneck", self.adapter_config.get("bottleneck", 64))
        pretrained_dir = kwargs.get("pretrained_dir", "pretrained_models/clip-vit-base-patch16")
        local_files_only = kwargs.get("local_files_only", True)
        
        logger.info(f"Building {self.metadata.name} model:")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Adapter last_k: {adapter_last_k}")
        logger.info(f"  Adapter bottleneck: {adapter_bottleneck}")
        logger.info(f"  Num labels: {num_labels}")
        logger.info(f"  Pretrained dir: {pretrained_dir}")
        
        model = CLIPVisionWithHead(
            num_labels=num_labels,
            pretrained_dir=pretrained_dir,
            strategy=strategy,
            adapter_last_k=adapter_last_k,
            adapter_bottleneck=adapter_bottleneck,
            local_files_only=local_files_only,
        )
        
        return model
    
    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
        """Load checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        state = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(state, dict) and "model" in state:
            logger.debug("Extracting model state from checkpoint dict")
            return state["model"]
        
        return state
    
    def infer_hyperparams_from_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Infer hyperparameters from state dict."""
        has_adapter = any("_adapter" in key for key in state_dict.keys())
        
        strategy = "adapter" if has_adapter else "linear"
        
        return {
            "strategy": strategy,
            "adapter_last_k": self.adapter_config.get("last_k", 3),
            "adapter_bottleneck": self.adapter_config.get("bottleneck", 64)
        }
