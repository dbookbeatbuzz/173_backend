"""In-memory job tracking used by model test runs."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional, Set


class Job:
    """Represents lifecycle information for a model test job."""

    def __init__(
        self,
        job_id: str,
        model_id: str,
        total: int,
        input_type: str,
        random_seed: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> None:
        self.job_id = job_id
        self.model_id = model_id
        self.total = total
        self.input_type = input_type
        self.random_seed = random_seed
        self.client_id = client_id or 1
        self.status = "pending"
        self.processed = 0
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.error: Optional[Dict[str, Any]] = None
        self.cancelled = False
        self.subscribers: Set[Callable[[str, Dict[str, Any]], None]] = set()
        self.worker: Optional[threading.Thread] = None


def create_job(
    job_id: str,
    model_id: str,
    total: int,
    input_type: str,
    random_seed: Optional[int] = None,
    client_id: Optional[int] = None,
) -> Job:
    job = Job(job_id, model_id, total, input_type, random_seed, client_id)
    with lock:
        jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with lock:
        return jobs.get(job_id)


def remove_job(job_id: str) -> bool:
    with lock:
        return jobs.pop(job_id, None) is not None


def list_jobs() -> Dict[str, Job]:  # pragma: no cover - diagnostic helper
    with lock:
        return jobs.copy()


jobs: Dict[str, Job] = {}
lock = threading.Lock()
