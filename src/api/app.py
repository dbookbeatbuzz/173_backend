"""Flask application factory for the backend service."""

from __future__ import annotations

import os
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.config import Settings, get_settings
from src.api.blueprints import model_tests_bp
from src.plugins import plugin_registry
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
        for metadata in plugin_registry.list_models():
            model_exists = plugin_registry.validate_model_exists(metadata.model_id, client_id=1)
            
            # Get dataset name from plugin
            model_plugin_cls = plugin_registry.get_model_plugin(metadata.model_id)
            dataset_name = model_plugin_cls.dataset_plugin_id if model_plugin_cls else "unknown"
            
            models_payload.append({
                "id": metadata.numeric_id,
                "name": metadata.name,
                "type": metadata.model_type.value,
                "inputType": metadata.input_type.value,
                "description": metadata.description,
                "dataset": dataset_name,
                "available": model_exists,
            })
        return jsonify({"models": models_payload})

    @app.get("/api/models/<model_id>")
    def get_model_info(model_id: str):
        # Try to convert to int if it looks like a number
        try:
            model_id = int(model_id)
        except (ValueError, TypeError):
            pass  # Keep as string
        
        metadata = plugin_registry.get_model_metadata(model_id)
        if not metadata:
            return jsonify({"error": "Model not found"}), 404

        model_exists = plugin_registry.validate_model_exists(model_id, client_id=1)
        
        # Get additional info from plugin
        model_plugin_cls = plugin_registry.get_model_plugin(model_id)
        model_plugin = model_plugin_cls() if model_plugin_cls else None
        
        dataset_name = model_plugin.dataset_plugin_id if model_plugin else "unknown"
        strategy = model_plugin.strategy if model_plugin else "unknown"
        num_labels = model_plugin.num_labels if model_plugin else None
        model_path = model_plugin.model_path if model_plugin else None

        return jsonify({
            "id": metadata.numeric_id,
            "name": metadata.name,
            "type": metadata.model_type.value,
            "inputType": metadata.input_type.value,
            "description": metadata.description,
            "dataset": dataset_name,
            "strategy": strategy,
            "numLabels": num_labels,
            "available": model_exists,
            "modelPath": model_path,
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
        
        model_id = body.get("model_id", "domainnet_vit_fedsak")
        
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
                model_id=model_id,
                client_id=client_id,
                split=split,
                limit=limit,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
            )
            return jsonify(result)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # pragma: no cover - defensive path
            return jsonify({"error": str(exc)}), 500

    return app
