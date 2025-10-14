"""WSGI entrypoint for production servers."""

from src.api import create_app
from src.config import get_settings
from src.plugins import init_plugins

# Initialize plugin system at startup
init_plugins()

settings = get_settings()
app = create_app(settings=settings)
