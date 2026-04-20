import logging

from flask import Flask, jsonify

from .config import Config
from .model_service import ModelService
from .routes import api_bp


def create_app(config: Config | None = None) -> Flask:
    """Flask application factory."""
    app = Flask(__name__)
    app.config["APP_CONFIG"] = config or Config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Load model artifacts once during startup and keep service as app state.
    app.model_service = ModelService(app.config["APP_CONFIG"])
    app.register_blueprint(api_bp)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found"}), 404

    return app

