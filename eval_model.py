import logging
from typing import Optional

import torch
from torch import nn
from transformers import CLIPModel

logger = logging.getLogger(__name__)


def _adapter_residual_hook(module, input, output):
    """
    Inject module._adapter output as residual into hidden states.
    Works with different output structures from CLIPVisionTransformer.
    """
    adp = getattr(module, "_adapter", None)
    if adp is None:
        return output

    if isinstance(output, tuple):
        hs = output[0]
        hs = adp(hs)
        return (hs,) + output[1:]
    elif hasattr(output, "hidden_states"):
        hs = output.hidden_states
        hs = adp(hs)
        output.hidden_states = hs
        return output
    else:
        hs = output
        hs = adp(hs)
        return hs


class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class EvalCLIPVisionWithHead(nn.Module):
    """
    Minimal inference wrapper for CLIP ViT-B/16 vision backbone with classification head.
    This mirrors the training-time architecture to make checkpoint loading compatible,
    without importing federatedscope.
    """

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
        # Load CLIP vision from local directory by default
        try:
            clip = CLIPModel.from_pretrained(pretrained_dir,
                                             local_files_only=local_files_only)
        except Exception as e:
            msg = (
                f"Failed to load CLIPModel from '{pretrained_dir}'. "
                "Ensure the pretrained files exist locally or set CLIP_PRETRAINED_DIR. "
                f"Original error: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        self.vision = clip.vision_model
        hidden = self.vision.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

        # Setup adapters (structure only; weights will come from checkpoint)
        self.adapters: Optional[nn.ModuleDict] = None
        if strategy == "adapter":
            layers = self.vision.encoder.layers
            assert 1 <= adapter_last_k <= len(layers)
            target_idx = list(range(len(layers) - adapter_last_k, len(layers)))
            self.adapters = nn.ModuleDict()
            self._hooks = []
            for i in target_idx:
                adp = Adapter(hidden, adapter_bottleneck)
                self.adapters[f"layer_{i}"] = adp
                setattr(layers[i], "_adapter", adp)
                h = layers[i].register_forward_hook(_adapter_residual_hook)
                self._hooks.append(h)
        elif strategy == "linear":
            # no adapters
            pass
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @torch.no_grad()
    def forward(self, pixel_values):
        outputs = self.vision(pixel_values=pixel_values,
                              output_hidden_states=False)
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
        import os
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
