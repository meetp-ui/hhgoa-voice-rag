from __future__ import annotations

from asgiref.wsgi import WsgiToAsgi
from flask import Flask
from flask_cors import CORS

from app.api.routes.health import health_bp
from app.api.routes.query import query_bp
from app.core.background_loop import BackgroundEventLoop
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> Flask:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings.flask_secret_key

    origins = [origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()]
    CORS(app, origins=origins)

    app.register_blueprint(health_bp)
    app.register_blueprint(query_bp)

    app.extensions["settings"] = settings
    app.extensions["pipeline_graph"] = None
    app.extensions["background_loop"] = BackgroundEventLoop()

    return app


app = create_app()
asgi_app = WsgiToAsgi(app)  # type: ignore[no-untyped-call]  # asgiref ships no type stubs
