"""Blueprint implementing the model testing HTTP API."""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Dict

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from src.config import Settings, get_settings
from src.models.job_store import create_job, get_job, jobs, lock
from src.plugins import plugin_registry
from src.services.model_test_runner import start_model_test_job

bp = Blueprint("model_tests", __name__)


def _settings() -> Settings:
    settings = current_app.config.get("BACKEND_SETTINGS")
    if settings is None:
        settings = get_settings()
        current_app.config["BACKEND_SETTINGS"] = settings
    return settings


@bp.route("/", methods=["POST"])
def start_test():
    data = request.get_json(force=True, silent=True) or {}

    model_id = data.get("modelId")
    if model_id is None:
        return jsonify({"error": "modelId is required"}), 400

    # Try to convert to int if it's numeric (frontend sends int)
    try:
        model_id = int(model_id)
    except (ValueError, TypeError):
        pass  # Keep as string if not numeric
    
    # Resolve model_id (handles both int and string)
    resolved_model_id = plugin_registry.resolve_model_id(model_id)
    if not resolved_model_id:
        return jsonify({"error": f"Model {model_id} not found"}), 404
    
    model_metadata = plugin_registry.get_model_metadata(resolved_model_id)
    if not model_metadata:
        return jsonify({"error": f"Model {model_id} not found"}), 404

    try:
        total = int(data.get("sampleCount", 50))
        if total < 1 or total > 2000:
            return jsonify({"error": "sampleCount must be between 1 and 2000"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "sampleCount must be a valid number"}), 400

    seed = data.get("randomSeed")
    if seed is not None:
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            return jsonify({"error": "randomSeed must be a valid number"}), 400

    input_type = data.get("inputType")
    if input_type is None:
        input_type = model_metadata.input_type.value
    elif input_type not in ("text", "image"):
        return jsonify({"error": 'inputType must be "text" or "image"'}), 400

    if input_type != model_metadata.input_type.value:
        return jsonify({
            "error": f'inputType "{input_type}" does not match model configuration "{model_metadata.input_type.value}"'
        }), 400

    client_id = data.get("clientId")
    if client_id is not None:
        try:
            client_id = int(client_id)
            if client_id < 1 or client_id > 30:
                return jsonify({"error": "clientId must be between 1 and 30"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "clientId must be a valid number"}), 400

    job_id = f"job_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

    try:
        # Use resolved_model_id (string) for internal processing
        job = create_job(job_id, resolved_model_id, total, input_type, seed, client_id)
        if start_model_test_job(job_id):
            return jsonify({"jobId": job_id, "total": total, "clientId": job.client_id})
        return jsonify({"error": "Failed to start job"}), 500
    except Exception as exc:  # pragma: no cover - defensive path
        return jsonify({"error": str(exc)}), 500


@bp.route("/<job_id>/stream", methods=["GET"])
def stream_events(job_id: str):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    def sse_format(event: str, payload: Dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        event_queue: list[tuple[str, Dict[str, Any]]] = []
        condition = threading.Condition()

        def event_callback(event: str, payload: Dict[str, Any]) -> None:
            with condition:
                event_queue.append((event, payload))
                condition.notify()

        with lock:
            job.subscribers.add(event_callback)

        try:
            with lock:
                elapsed_ms = int((time.time() - (job.started_at or job.created_at)) * 1000)
                event_queue.append(("progress", {
                    "processed": job.processed,
                    "total": job.total,
                    "elapsedMs": elapsed_ms,
                }))

            while True:
                with condition:
                    while not event_queue:
                        condition.wait(timeout=30)
                        if not event_queue:
                            yield sse_format("ping", {
                                "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            })
                            continue

                    event, payload = event_queue.pop(0)

                yield sse_format(event, payload)

                if event in ("summary", "error"):
                    break
        finally:
            with lock:
                job.subscribers.discard(event_callback)

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }

    return Response(stream_with_context(generate()), headers=headers)


@bp.route("/<job_id>/cancel", methods=["POST"])
def cancel_test(job_id: str):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    with lock:
        if job.status in ("success", "error", "cancelled"):
            return jsonify({"cancelled": True})

        job.cancelled = True

    return jsonify({"cancelled": True})


@bp.route("/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    with lock:
        # Get numeric_id for frontend compatibility
        metadata = plugin_registry.get_model_metadata(job.model_id)
        model_id_for_frontend = metadata.numeric_id if (metadata and metadata.numeric_id is not None) else job.model_id
        
        return jsonify({
            "jobId": job.job_id,
            "modelId": model_id_for_frontend,  # Return numeric ID for frontend
            "clientId": job.client_id,
            "status": job.status,
            "total": job.total,
            "processed": job.processed,
            "inputType": job.input_type,
            "randomSeed": job.random_seed,
            "startedAt": job.started_at,
            "finishedAt": job.finished_at,
            "error": job.error,
        })
