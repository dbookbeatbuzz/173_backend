"""Development server entrypoint.

Provides ``python -m src.cli.serve`` as a convenience for running the
Flask development server without relying on top-level modules.
"""
from __future__ import annotations

from src.api import create_app
from src.config import get_settings


def build_app():
    """Create and return a configured Flask application."""

    return create_app(settings=get_settings())


def main() -> None:
    settings = get_settings()
    app = create_app(settings=settings)
    app.run(host=settings.host, port=settings.port, debug=settings.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
