import json
import os
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from evaluator import evaluate_client
from blueprints.model_tests import bp as model_tests_bp
from models.model_registry import model_registry


def create_app():
    app = Flask(__name__)
    
    # 启用CORS支持
    CORS(app, supports_credentials=True, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Last-Event-ID"]
        }
    })
    
    # 注册模型测试API蓝图
    app.register_blueprint(model_tests_bp, url_prefix='/api/model-tests')

    MODELS_ROOT = os.environ.get(
        "MODELS_ROOT", "exp_models/Domainnet_ViT_fedsak_lda"
    )
    DATA_ROOT = os.environ.get("DATA_ROOT", "/root/domainnet")
    PREPROCESSOR_JSON = os.environ.get(
        "CLIP_PREPROCESSOR_JSON",
        "pretrained_models/clip-vit-base-patch16/preprocessor_config.json",
    )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/models")
    def list_models():
        """列出所有可用的模型"""
        models = []
        for config in model_registry.list_models():
            # 验证模型文件是否存在
            model_exists = model_registry.validate_model_exists(config.model_id, client_id=1)
            
            models.append({
                "id": config.model_id,
                "name": config.name,
                "type": config.model_type.value,
                "inputType": config.input_type.value,
                "description": config.description,
                "dataset": config.dataset_name,
                "available": model_exists
            })
        
        return jsonify({"models": models})

    @app.get("/api/models/<model_id>")
    def get_model_info(model_id: str):
        """获取特定模型的详细信息"""
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
            "modelPath": config.model_path
        })

    @app.get("/clients")
    def list_clients():
        client_dir = os.path.join(MODELS_ROOT, "client")
        clients = []
        if os.path.isdir(client_dir):
            for fn in os.listdir(client_dir):
                if fn.startswith("client_model_") and fn.endswith(".pt"):
                    try:
                        cid = int(fn[len("client_model_"):-3])
                        clients.append(cid)
                    except Exception:
                        pass
        clients.sort()
        return jsonify({"models_root": MODELS_ROOT, "clients": clients})

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
            res = evaluate_client(
                client_id=client_id,
                split=split,
                limit=limit,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                models_root=MODELS_ROOT,
                data_root=DATA_ROOT,
                preprocessor_json=PREPROCESSOR_JSON,
            )
            return jsonify(res)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    app = create_app()
    app.run(host=host, port=port, debug=True)
