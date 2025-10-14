"""Long-running job execution for model evaluations."""

from __future__ import annotations

import base64
import io
import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

from PIL import Image
import torch

from src.models.job_store import jobs, lock
from src.plugins import plugin_registry

logger = logging.getLogger(__name__)

TRANSPORT_MODE = "data-url"


def bytes_to_data_url(mime: str, content: bytes) -> str:
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def pil_image_to_data_url(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    buffer = io.BytesIO()
    if format.upper() == "JPEG":
        image.save(buffer, format=format, quality=quality)
        mime = "image/jpeg"
    elif format.upper() == "PNG":
        image.save(buffer, format=format)
        mime = "image/png"
    else:
        image.save(buffer, format=format)
        mime = f"image/{format.lower()}"

    buffer.seek(0)
    return bytes_to_data_url(mime, buffer.getvalue())


def _emit(job, event: str, payload: Dict[str, Any]) -> None:
    subscribers = list(job.subscribers)
    for callback in subscribers:
        try:
            callback(event, payload)
        except Exception:
            job.subscribers.discard(callback)


def load_model_for_job(job):
    try:
        # Get model plugin
        model_plugin_cls = plugin_registry.get_model_plugin(job.model_id)
        if not model_plugin_cls:
            raise ValueError(f"Model plugin not found: {job.model_id}")
        
        model_plugin = model_plugin_cls()
        logger.info(f"Using model plugin: {model_plugin.metadata.name}")
        
        # Get dataset plugin
        dataset_plugin_cls = plugin_registry.get_dataset_plugin(model_plugin.dataset_plugin_id)
        if not dataset_plugin_cls:
            raise ValueError(f"Dataset plugin not found: {model_plugin.dataset_plugin_id}")
        
        # Configure dataset
        dataset_config = getattr(job, "dataset_config", {})
        if not dataset_config.get("root"):
            dataset_config["root"] = dataset_plugin_cls.metadata.default_root
        
        dataset_plugin = dataset_plugin_cls(config=dataset_config)
        logger.info(f"Using dataset plugin: {dataset_plugin.metadata.name}")
        
        # Get test dataset
        test_set, num_labels = dataset_plugin.get_test_dataset()
        
        # Get client ID
        client_id = getattr(job, "client_id", 1)
        
        # Get model path
        model_path = model_plugin.get_model_path(client_id)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = model_plugin.load_checkpoint(model_path, device="cpu")
        
        # Build model
        model = model_plugin.build_model(num_labels=num_labels)
        
        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading model: {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading model: {unexpected[:5]}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        return model, test_set, dataset_plugin, device, num_labels

    except Exception as exc:
        raise RuntimeError(f"Failed to load model {job.model_id}: {exc}") from exc


def run_job(job_id: str) -> None:
    with lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.status = "running"
        job.started_at = time.time()

    try:
        model, test_set, dataset_manager, device, num_labels = load_model_for_job(job)

        random_state = random.Random(job.random_seed)

        total_samples = len(test_set)
        sample_indices = random_state.sample(range(total_samples), min(job.total, total_samples))

        def create_test_case(case_idx: int, dataset_idx: int):
            try:
                image, true_label = test_set[dataset_idx]
                label_name = dataset_manager.get_class_name(true_label)

                start_time = time.time()
                with torch.no_grad():
                    image_batch = image.unsqueeze(0).to(device)
                    logits = model(image_batch)
                    probs = torch.softmax(logits, dim=1)
                    predicted_class = logits.argmax(dim=1).item()
                latency_ms = int((time.time() - start_time) * 1000)

                top_probs, top_indices = torch.topk(probs[0], min(5, num_labels))
                topk: List[Dict[str, Any]] = []
                for prob, idx in zip(top_probs, top_indices):
                    class_name = dataset_manager.get_class_name(idx.item())
                    topk.append({
                        "label": class_name,
                        "prob": float(prob),
                    })

                pred_label_name = dataset_manager.get_class_name(predicted_class)
                case_id = f"img_{case_idx:05d}"

                if TRANSPORT_MODE == "data-url":
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

                    denorm_image = image * std + mean
                    denorm_image = torch.clamp(denorm_image, 0, 1)

                    from torchvision.transforms.functional import to_pil_image

                    pil_img = to_pil_image(denorm_image)

                    if pil_img.size[0] > 224 or pil_img.size[1] > 224:
                        pil_img.thumbnail((224, 224), Image.Resampling.LANCZOS)

                    input_payload = {
                        "type": "image",
                        "dataUrl": pil_image_to_data_url(pil_img, "JPEG", 85),
                        "mime": "image/jpeg",
                        "width": pil_img.width,
                        "height": pil_img.height,
                    }
                else:
                    input_payload = {
                        "type": "image",
                        "url": f"https://picsum.photos/seed/{case_idx}/224/224",
                        "mime": "image/jpeg",
                        "width": 224,
                        "height": 224,
                    }

                return {
                    "caseId": case_id,
                    "input": input_payload,
                    "label": label_name,
                    "output": {
                        "predLabel": pred_label_name,
                        "topK": topk,
                    },
                    "correct": predicted_class == true_label,
                    "latencyMs": latency_ms,
                }

            except Exception as exc:  # pragma: no cover - defensive path
                return {
                    "caseId": f"img_{case_idx:05d}",
                    "input": {"type": "image", "dataUrl": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"},
                    "label": "unknown",
                    "output": {"predLabel": "error", "topK": []},
                    "correct": False,
                    "latencyMs": 0,
                    "error": str(exc),
                }

        for i, dataset_idx in enumerate(sample_indices, 1):
            with lock:
                if job.cancelled:
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    _emit(job, "error", {"message": "cancelled"})
                    return

            elapsed_ms = int((time.time() - job.started_at) * 1000)
            _emit(job, "progress", {
                "processed": i - 1,
                "total": job.total,
                "elapsedMs": elapsed_ms,
            })

            case_payload = create_test_case(i, dataset_idx)
            _emit(job, "case", case_payload)

            with lock:
                job.processed = i

            time.sleep(0.01)

        with lock:
            job.status = "success"
            job.finished_at = time.time()

        correct_count = 0
        _emit(job, "summary", {
            "processed": job.total,
            "total": job.total,
            "accuracy": correct_count / job.total if job.total > 0 else 0.0,
        })

    except Exception as exc:
        with lock:
            job.status = "error"
            job.finished_at = time.time()
            job.error = {"message": str(exc)}

        _emit(job, "error", {"message": str(exc)})


def start_model_test_job(job_id: str) -> bool:
    job = jobs.get(job_id)
    if not job:
        return False

    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    job.worker = thread
    thread.start()
    return True
