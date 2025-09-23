import json
import random
import time
import threading
import base64
import os
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
import io

from models.job_store import jobs, lock
from models.model_registry import model_registry
from models.dataset_manager import DatasetManagerFactory, DatasetConfig
from evaluator import load_client_checkpoint, _infer_model_hyperparams_from_state
from eval_model import build_model_for_eval


# 传输模式：'url' 或 'data-url'
TRANSPORT_MODE = 'data-url'

def bytes_to_data_url(mime: str, content: bytes) -> str:
    """将字节内容转换为data URL格式"""
    b64 = base64.b64encode(content).decode('ascii')
    return f"data:{mime};base64,{b64}"

def pil_image_to_data_url(image: Image.Image, format: str = 'JPEG', quality: int = 85) -> str:
    """将PIL图像转换为data URL"""
    buffer = io.BytesIO()
    if format.upper() == 'JPEG':
        image.save(buffer, format=format, quality=quality)
        mime = 'image/jpeg'
    elif format.upper() == 'PNG':
        image.save(buffer, format=format)
        mime = 'image/png'
    else:
        image.save(buffer, format=format)
        mime = f'image/{format.lower()}'
    
    buffer.seek(0)
    return bytes_to_data_url(mime, buffer.getvalue())

def _emit(job, event: str, payload: Dict[str, Any]):
    """向所有订阅者发送事件"""
    # 复制订阅者列表，避免遍历中修改
    subscribers = list(job.subscribers)
    for cb in subscribers:
        try:
            cb(event, payload)
        except Exception as e:
            # 移除失败的订阅者
            job.subscribers.discard(cb)

def load_model_for_job(job):
    """为指定任务加载模型"""
    try:
        # 获取模型配置
        model_config = model_registry.get_model(job.model_id)
        if not model_config:
            raise ValueError(f"Model {job.model_id} not found in registry")
        
        # 创建数据集管理器
        dataset_config = DatasetConfig(
            name=model_config.dataset_name,
            root=model_config.dataset_config.get('root'),
            input_type=model_config.input_type.value,
            num_classes=0,  # 将由数据集管理器动态确定
            preprocessing_config=model_config.dataset_config.get('preprocessor_path'),
            additional_config=model_config.dataset_config
        )
        
        dataset_manager = DatasetManagerFactory.create_manager(
            model_config.dataset_name, 
            dataset_config
        )
        
        # 获取测试数据集
        test_set, num_labels = dataset_manager.get_test_dataset()
        
        # 加载模型权重
        # 使用job中的客户端ID
        client_id = getattr(job, 'client_id', 1)  # 向后兼容
        
        model_path = model_registry.get_model_path(job.model_id, client_id)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 加载模型状态
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        
        # 推断模型超参数
        strategy, adapter_last_k, adapter_bottleneck = _infer_model_hyperparams_from_state(state)
        
        # 使用配置文件中的参数，如果可用的话
        strategy = model_config.strategy
        adapter_last_k = model_config.adapter_last_k
        adapter_bottleneck = model_config.adapter_bottleneck
        
        # 构建模型
        model = build_model_for_eval(
            num_labels=num_labels,
            strategy=strategy,
            adapter_last_k=adapter_last_k,
            adapter_bottleneck=adapter_bottleneck,
        )
        
        # 加载权重
        missing, unexpected = model.load_state_dict(state, strict=False)
        
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        return model, test_set, dataset_manager, device, num_labels
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {job.model_id}: {str(e)}")

def run_job(job_id: str):
    """运行模型测试任务的主函数"""
    with lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.status = 'running'
        job.started_at = time.time()

    try:
        # 加载模型和数据集
        model, test_set, dataset_manager, device, num_labels = load_model_for_job(job)
        
        # 设置随机种子
        rnd = random.Random(job.random_seed)
        
        # 从测试集中随机采样
        total_samples = len(test_set)
        sample_indices = rnd.sample(range(total_samples), min(job.total, total_samples))
        
        def create_test_case(case_idx: int, dataset_idx: int):
            """创建单个测试用例"""
            try:
                # 获取数据
                image, true_label = test_set[dataset_idx]
                
                # 使用数据集管理器获取标签名
                label_name = dataset_manager.get_class_name(true_label)
                
                # 模型推理
                start_time = time.time()
                with torch.no_grad():
                    image_batch = image.unsqueeze(0).to(device)
                    logits = model(image_batch)
                    probs = torch.softmax(logits, dim=1)
                    predicted_class = logits.argmax(dim=1).item()
                latency_ms = int((time.time() - start_time) * 1000)
                
                # 构建top-k结果
                top_probs, top_indices = torch.topk(probs[0], min(5, num_labels))
                topk = []
                for prob, idx in zip(top_probs, top_indices):
                    class_name = dataset_manager.get_class_name(idx.item())
                    topk.append({
                        'label': class_name,
                        'prob': float(prob)
                    })
                
                # 获取预测标签名
                pred_label_name = dataset_manager.get_class_name(predicted_class)
                
                # 构建输入数据
                case_id = f"img_{case_idx:05d}"
                
                if TRANSPORT_MODE == 'data-url':
                    # 将tensor转回PIL图像进行编码
                    # 反向标准化
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                    
                    # 反标准化
                    denorm_image = image * std + mean
                    denorm_image = torch.clamp(denorm_image, 0, 1)
                    
                    # 转换为PIL图像
                    from torchvision.transforms.functional import to_pil_image
                    pil_img = to_pil_image(denorm_image)
                    
                    # 缩放到合适大小用于传输
                    if pil_img.size[0] > 224 or pil_img.size[1] > 224:
                        pil_img.thumbnail((224, 224), Image.Resampling.LANCZOS)
                    
                    input_payload = {
                        'type': 'image',
                        'dataUrl': pil_image_to_data_url(pil_img, 'JPEG', 85),
                        'mime': 'image/jpeg',
                        'width': pil_img.width,
                        'height': pil_img.height
                    }
                else:
                    # URL模式（示例）
                    input_payload = {
                        'type': 'image',
                        'url': f'https://picsum.photos/seed/{case_idx}/224/224',
                        'mime': 'image/jpeg',
                        'width': 224,
                        'height': 224
                    }
                
                return {
                    'caseId': case_id,
                    'input': input_payload,
                    'label': label_name,
                    'output': {
                        'predLabel': pred_label_name,
                        'topK': topk
                    },
                    'correct': predicted_class == true_label,
                    'latencyMs': latency_ms,
                }
                
            except Exception as e:
                # 如果单个样本处理失败，返回错误样本
                return {
                    'caseId': f"img_{case_idx:05d}",
                    'input': {'type': 'image', 'dataUrl': 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'},
                    'label': 'unknown',
                    'output': {'predLabel': 'error', 'topK': []},
                    'correct': False,
                    'latencyMs': 0,
                    'error': str(e)
                }
        
        # 处理每个测试样本
        for i, dataset_idx in enumerate(sample_indices, 1):
            # 检查是否被取消
            with lock:
                if job.cancelled:
                    job.status = 'cancelled'
                    job.finished_at = time.time()
                    _emit(job, 'error', {'message': 'cancelled'})
                    return
            
            # 发送进度事件
            elapsed_ms = int((time.time() - job.started_at) * 1000)
            _emit(job, 'progress', {
                'processed': i - 1,
                'total': job.total,
                'elapsedMs': elapsed_ms
            })
            
            # 创建并发送测试用例
            case_payload = create_test_case(i, dataset_idx)
            _emit(job, 'case', case_payload)
            
            # 更新已处理数量
            with lock:
                job.processed = i
            
            # 模拟处理延迟（可选）
            time.sleep(0.01)
        
        # 完成任务
        with lock:
            job.status = 'success'
            job.finished_at = time.time()
            
        # 发送最终汇总
        correct_count = 0  # 这里可以统计正确数量，暂时简化
        _emit(job, 'summary', {
            'processed': job.total,
            'total': job.total,
            'accuracy': correct_count / job.total if job.total > 0 else 0.0
        })
        
    except Exception as e:
        # 任务执行出错
        with lock:
            job.status = 'error'
            job.finished_at = time.time()
            job.error = {'message': str(e)}
        
        _emit(job, 'error', {'message': str(e)})

def start_model_test_job(job_id: str):
    """启动模型测试任务"""
    job = jobs.get(job_id)
    if not job:
        return False
    
    # 创建并启动后台线程
    t = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    job.worker = t
    t.start()
    return True