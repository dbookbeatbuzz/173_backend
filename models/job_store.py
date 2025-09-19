import time
import threading
from typing import Dict, Optional, Set, Callable, Any


class Job:
    """模型测试任务的数据模型"""
    
    def __init__(self, job_id: str, model_id: str, total: int, input_type: str, 
                 random_seed: Optional[int] = None):
        self.job_id = job_id
        self.model_id = model_id
        self.total = total
        self.input_type = input_type  # 'text' or 'image'
        self.random_seed = random_seed
        self.status = 'pending'  # pending|running|cancelled|success|error
        self.processed = 0
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.error: Optional[Dict[str, Any]] = None
        self.cancelled = False
        self.subscribers: Set[Callable[[str, Dict], None]] = set()
        self.worker = None  # threading.Thread reference


# 全局Job存储 (生产环境应使用Redis等持久化存储)
jobs: Dict[str, Job] = {}
lock = threading.Lock()


def create_job(job_id: str, model_id: str, total: int, input_type: str, 
               random_seed: Optional[int] = None) -> Job:
    """创建新的测试任务"""
    job = Job(job_id, model_id, total, input_type, random_seed)
    with lock:
        jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    """获取指定的测试任务"""
    with lock:
        return jobs.get(job_id)


def remove_job(job_id: str) -> bool:
    """移除指定的测试任务"""
    with lock:
        return jobs.pop(job_id, None) is not None


def list_jobs() -> Dict[str, Job]:
    """获取所有任务列表（调试用）"""
    with lock:
        return jobs.copy()