"""Application configuration handling.

Provides a single source of truth for runtime configuration so that
settings are not scattered across the codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List


@dataclass(slots=True)
class Settings:
    """Runtime configuration values loaded from environment variables."""

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}

    models_root: str = os.getenv("MODELS_ROOT", "exp_models/Domainnet_ViT_fedsak_lda")
    data_root: str = os.getenv("DATA_ROOT", "/root/domainnet")
    preprocessor_json: str = os.getenv(
        "CLIP_PREPROCESSOR_JSON",
        "pretrained_models/clip-vit-base-patch16/preprocessor_config.json",
    )

    cors_allow_origins: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

    @property
    def cors_origins(self) -> List[str] | str:
        """Return parsed CORS origins; keep wildcard as-is."""

        if self.cors_allow_origins.strip() == "*":
            return "*"
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton Settings instance."""

    return Settings()
