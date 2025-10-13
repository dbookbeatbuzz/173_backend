"""API package exposing Flask application factory."""

from .app import create_app

__all__ = ["create_app"]
