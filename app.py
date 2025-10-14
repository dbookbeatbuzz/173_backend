"""Local development entrypoint for the Flask application."""

from src.api import create_app
from src.config import get_settings
from src.plugins import init_plugins

# Initialize plugin system at startup
init_plugins()


def main() -> None:
    settings = get_settings()
    app = create_app(settings=settings)
    app.run(host=settings.host, port=settings.port, debug=settings.debug)


app = create_app(settings=get_settings())


if __name__ == "__main__":
    main()
