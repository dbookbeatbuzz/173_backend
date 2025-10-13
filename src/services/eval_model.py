"""Model construction helpers for evaluation time."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch import nn
from transformers import CLIPModel

logger = logging.getLogger(__name__)


def _adapter_residual_hook(module, _input, output):
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
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)

    def forward(self, inputs):
        return inputs + self.up(self.act(self.down(inputs)))


class EvalCLIPVisionWithHead(nn.Module):
    def __init__(
        self,
        num_labels: int,
        pretrained_dir: str = "pretrained_models/clip-vit-base-patch16",
        strategy: str = "adapter",
        adapter_last_k: int = 3,
        adapter_bottleneck: int = 64,
        unfreeze_ln: bool = True,  # kept for parity; not used in eval
        local_files_only: bool = True,
    ):
        super().__init__()
        try:
            clip = CLIPModel.from_pretrained(
                pretrained_dir,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # pragma: no cover - external dependency
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

        self.adapters: Optional[nn.ModuleDict] = None
        if strategy == "adapter":
            layers = self.vision.encoder.layers
            if not 1 <= adapter_last_k <= len(layers):
                raise ValueError("adapter_last_k must be within the encoder depth")
            target_indices = range(len(layers) - adapter_last_k, len(layers))
            self.adapters = nn.ModuleDict()
            self._hooks = []
            for idx in target_indices:
                adapter_module = Adapter(hidden_dim, adapter_bottleneck)
                self.adapters[f"layer_{idx}"] = adapter_module
                setattr(layers[idx], "_adapter", adapter_module)
                hook = layers[idx].register_forward_hook(_adapter_residual_hook)
                self._hooks.append(hook)
        elif strategy == "linear":
            pass
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @torch.no_grad()
    def forward(self, pixel_values):
        outputs = self.vision(pixel_values=pixel_values, output_hidden_states=False)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits


def build_model_for_eval(
    num_labels: int,
    pretrained_dir: Optional[str] = None,
    strategy: str = "adapter",
    adapter_last_k: int = 3,
    adapter_bottleneck: int = 64,
    local_files_only: bool = True,
):
    if pretrained_dir is None:
        pretrained_dir = (
            os.environ.get("CLIP_PRETRAINED_DIR")
            or "pretrained_models/clip-vit-base-patch16"
        )
    model = EvalCLIPVisionWithHead(
        num_labels=num_labels,
        pretrained_dir=pretrained_dir,
        strategy=strategy,
        adapter_last_k=adapter_last_k,
        adapter_bottleneck=adapter_bottleneck,
        local_files_only=local_files_only,
    )
    return model
