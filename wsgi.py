"""WSGI entrypoint for production servers."""

from src.api import create_app
from src.config import get_settings

settings = get_settings()
app = create_app(settings=settings)
