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
from evaluator import load_client_checkpoint, _infer_model_hyperparams_from_state
from eval_model import build_model_for_eval
from data_domainnet import build_domainnet_splits


# DomainNet 类别标签 (根据实际数据集调整)
DOMAINNET_LABELS = [
    'aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 
    'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 
    'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 
    'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 
    'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 
    'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 
    'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 
    'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 
    'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 
    'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 
    'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 
    'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 
    'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 
    'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 
    'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 
    'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 
    'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 
    'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 
    'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 
    'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 
    'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 
    'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 
    'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 
    'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 
    'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 
    'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 
    'paintbrush', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 
    'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 
    'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 
    'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 
    'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 
    'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 
    'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 
    'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 
    'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 
    'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 
    'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 
    'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 
    'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 'table', 'teapot', 'teddy-bear', 
    'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 'The_Great_Wall_of_China', 
    'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 
    'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 
    'trumpet', 't-shirt', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 
    'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 
    'wristwatch', 'yoga', 'zebra', 'zigzag'
]

# 传输模式：'url' 或 'data-url'（推荐在实际数据集使用 'data-url'）
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
        # 获取环境配置
        models_root = os.environ.get("MODELS_ROOT", "exp_models/Domainnet_ViT_fedsak_lda")
        data_root = os.environ.get("DATA_ROOT", "/root/domainnet")
        preprocessor_json = os.environ.get(
            "CLIP_PREPROCESSOR_JSON", 
            "pretrained_models/clip-vit-base-patch16/preprocessor_config.json"
        )
        
        # 加载数据集信息
        _, _, test_set, num_labels = build_domainnet_splits(
            root=data_root, 
            preprocessor_path=preprocessor_json,
            seed=12345
        )
        
        # 加载客户端模型
        state = load_client_checkpoint(models_root, int(job.model_id))
        strategy, adapter_last_k, adapter_bottleneck = _infer_model_hyperparams_from_state(state)
        
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
        
        return model, test_set, device, num_labels
        
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
        model, test_set, device, num_labels = load_model_for_job(job)
        
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
                
                # 获取标签名
                if hasattr(test_set.dataset, 'idx_to_class'):
                    label_name = test_set.dataset.idx_to_class[true_label]
                else:
                    # 回退到数字标签
                    label_name = str(true_label)
                
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
                    if hasattr(test_set.dataset, 'idx_to_class'):
                        class_name = test_set.dataset.idx_to_class[idx.item()]
                    else:
                        class_name = str(idx.item())
                    topk.append({
                        'label': class_name,
                        'prob': float(prob)
                    })
                
                # 获取预测标签名
                if hasattr(test_set.dataset, 'idx_to_class'):
                    pred_label_name = test_set.dataset.idx_to_class[predicted_class]
                else:
                    pred_label_name = str(predicted_class)
                
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