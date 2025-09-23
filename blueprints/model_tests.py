from flask import Blueprint, request, Response, stream_with_context, jsonify
import json
import time
import threading
import uuid
from typing import Dict, Any

from models.job_store import jobs, Job, lock, create_job, get_job
from models.model_registry import model_registry
from services.model_test_runner import start_model_test_job

bp = Blueprint('model_tests', __name__)

@bp.route('/', methods=['POST'])
def start_test():
    """启动模型测试任务"""
    data = request.get_json(force=True, silent=True) or {}
    
    # 验证请求参数
    model_id = data.get('modelId')
    if model_id is None:
        return jsonify({'error': 'modelId is required'}), 400
    
    try:
        model_id = str(model_id)  # 支持字符串和数字
    except (ValueError, TypeError):
        return jsonify({'error': 'modelId must be string or number'}), 400
    
    # 验证模型是否存在
    model_config = model_registry.get_model(model_id)
    if not model_config:
        return jsonify({'error': f'Model {model_id} not found'}), 404
    
    try:
        total = int(data.get('sampleCount', 50))
        if total < 1 or total > 2000:
            return jsonify({'error': 'sampleCount must be between 1 and 2000'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'sampleCount must be a valid number'}), 400
    
    seed = data.get('randomSeed')
    if seed is not None:
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            return jsonify({'error': 'randomSeed must be a valid number'}), 400
    
    input_type = data.get('inputType')
    if input_type is None:
        # 从模型配置中推断输入类型
        input_type = model_config.input_type.value
    elif input_type not in ('text', 'image'):
        return jsonify({'error': 'inputType must be "text" or "image"'}), 400
    
    # 验证输入类型与模型配置是否匹配
    if input_type != model_config.input_type.value:
        return jsonify({
            'error': f'inputType "{input_type}" does not match model configuration "{model_config.input_type.value}"'
        }), 400
    
    # 生成唯一的任务ID
    job_id = f"job_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    
    # 创建任务
    try:
        job = create_job(job_id, model_id, total, input_type, seed)
        
        # 启动后台任务
        if start_model_test_job(job_id):
            return jsonify({'jobId': job_id, 'total': total})
        else:
            return jsonify({'error': 'Failed to start job'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/<job_id>/stream', methods=['GET'])
def stream_events(job_id: str):
    """订阅任务的流式事件（SSE）"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    def sse_format(event: str, payload: Dict[str, Any]) -> str:
        """格式化SSE消息"""
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        """SSE事件生成器"""
        # 创建事件队列和条件变量
        event_queue = []
        condition = threading.Condition()
        
        def event_callback(event: str, payload: Dict[str, Any]):
            """事件回调函数"""
            with condition:
                event_queue.append((event, payload))
                condition.notify()
        
        # 注册订阅者
        with lock:
            job.subscribers.add(event_callback)
        
        try:
            # 发送初始进度事件
            with lock:
                elapsed_ms = int((time.time() - (job.started_at or job.created_at)) * 1000)
                event_queue.append(('progress', {
                    'processed': job.processed,
                    'total': job.total,
                    'elapsedMs': elapsed_ms
                }))
            
            while True:
                with condition:
                    # 等待新事件或超时（用于心跳）
                    while not event_queue:
                        condition.wait(timeout=30)
                        if not event_queue:
                            # 发送心跳
                            yield sse_format('ping', {
                                't': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                            })
                            continue
                    
                    # 处理队列中的事件
                    event, payload = event_queue.pop(0)
                
                yield sse_format(event, payload)
                
                # 如果是终结事件，退出循环
                if event in ('summary', 'error'):
                    break
                    
        finally:
            # 取消订阅
            with lock:
                job.subscribers.discard(event_callback)

    # 设置SSE响应头
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive'
    }
    
    return Response(stream_with_context(generate()), headers=headers)

@bp.route('/<job_id>/cancel', methods=['POST'])
def cancel_test(job_id: str):
    """取消模型测试任务"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    with lock:
        if job.status in ('success', 'error', 'cancelled'):
            # 任务已结束，幂等返回
            return jsonify({'cancelled': True})
        
        # 标记为取消
        job.cancelled = True
    
    return jsonify({'cancelled': True})

@bp.route('/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """获取任务状态"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    with lock:
        return jsonify({
            'jobId': job.job_id,
            'modelId': job.model_id,
            'status': job.status,
            'total': job.total,
            'processed': job.processed,
            'inputType': job.input_type,
            'randomSeed': job.random_seed,
            'startedAt': job.started_at,
            'finishedAt': job.finished_at,
            'error': job.error
        })