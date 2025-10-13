"""Flask application factory for the backend service."""

from __future__ import annotations

import os
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.config import Settings, get_settings
from src.api.blueprints import model_tests_bp
from src.models.model_registry import model_registry
from src.services.evaluation import evaluate_client


def create_app(settings: Optional[Settings] = None) -> Flask:
    settings = settings or get_settings()

    app = Flask(__name__)
    app.config["BACKEND_SETTINGS"] = settings

    cors_origins = settings.cors_origins
    CORS(
        app,
        supports_credentials=True,
        resources={
            r"/api/*": {
                "origins": cors_origins,
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "Last-Event-ID"],
            }
        },
    )

    app.register_blueprint(model_tests_bp, url_prefix="/api/model-tests")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/models")
    def list_models():
        models_payload = []
        for config in model_registry.list_models():
            model_exists = model_registry.validate_model_exists(config.model_id, client_id=1)
            models_payload.append({
                "id": config.model_id,
                "name": config.name,
                "type": config.model_type.value,
                "inputType": config.input_type.value,
                "description": config.description,
                "dataset": config.dataset_name,
                "available": model_exists,
            })
        return jsonify({"models": models_payload})

    @app.get("/api/models/<model_id>")
    def get_model_info(model_id: str):
        config = model_registry.get_model(model_id)
        if not config:
            return jsonify({"error": "Model not found"}), 404

        model_exists = model_registry.validate_model_exists(model_id, client_id=1)
        return jsonify({
            "id": config.model_id,
            "name": config.name,
            "type": config.model_type.value,
            "inputType": config.input_type.value,
            "description": config.description,
            "dataset": config.dataset_name,
            "strategy": config.strategy,
            "numLabels": config.num_labels,
            "available": model_exists,
            "modelPath": config.model_path,
        })

    @app.get("/clients")
    def list_clients():
        client_dir = settings.models_root
        clients = []
        client_path = f"{client_dir}/client"
        if os.path.isdir(client_path):
            for filename in os.listdir(client_path):
                if filename.startswith("client_model_") and filename.endswith(".pt"):
                    try:
                        client_id = int(filename[len("client_model_"):-3])
                        clients.append(client_id)
                    except Exception:
                        continue
        clients.sort()
        return jsonify({"models_root": settings.models_root, "clients": clients})

    @app.post("/evaluate")
    def evaluate_endpoint():
        body = request.get_json(force=True, silent=True) or {}
        try:
            client_id = int(body.get("client_id"))
        except Exception:
            return jsonify({"error": "client_id is required and must be int"}), 400

        split = body.get("split", "test")
        limit = body.get("limit")
        limit = int(limit) if (limit is not None) else None
        batch_size = int(body.get("batch_size", 64))
        num_workers = int(body.get("num_workers", 4))
        device = body.get("device")

        try:
            result = evaluate_client(
                client_id=client_id,
                split=split,
                limit=limit,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                models_root=settings.models_root,
                data_root=settings.data_root,
                preprocessor_json=settings.preprocessor_json,
            )
            return jsonify(result)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # pragma: no cover - defensive path
            return jsonify({"error": str(exc)}), 500

    return app
