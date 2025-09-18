import json
import os
from typing import Optional

from flask import Flask, jsonify, request

from evaluator import evaluate_client


def create_app():
    app = Flask(__name__)

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
    app.run(host=host, port=port, debug=False)
