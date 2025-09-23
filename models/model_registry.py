"""
模型注册和管理模块
提供模型配置的统一管理，支持多种模型类型和数据集的解耦
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class ModelType(Enum):
    """模型类型枚举"""
    VISION_TRANSFORMER = "vit"
    CONVOLUTIONAL_NEURAL_NETWORK = "cnn"
    LSTM = "lstm"
    BERT = "bert"


class InputType(Enum):
    """输入类型枚举"""
    TEXT = "text"
    IMAGE = "image"


@dataclass
class ModelConfig:
    """模型配置数据类"""
    model_id: str
    name: str
    model_type: ModelType
    input_type: InputType
    description: str
    
    # 模型文件路径配置
    model_path: str
    checkpoint_pattern: Optional[str] = None  # 支持多个checkpoint的模式，如 "client_model_{client_id}.pt"
    
    # 模型架构参数
    num_labels: Optional[int] = None
    strategy: str = "adapter"  # linear, adapter, full_finetune
    adapter_last_k: int = 3
    adapter_bottleneck: int = 64
    
    # 数据集配置
    dataset_name: str = "domainnet"
    dataset_config: Dict[str, Any] = None
    
    # 其他配置
    preprocessing_config: Optional[str] = None
    device_preference: str = "auto"  # auto, cpu, cuda
    
    def __post_init__(self):
        if self.dataset_config is None:
            self.dataset_config = {}


class ModelRegistry:
    """模型注册表，管理所有可用的模型配置"""
    
    def __init__(self, config_file: Optional[str] = None):
        self._models: Dict[str, ModelConfig] = {}
        self._config_file = config_file or "model_registry.json"
        self._load_default_models()
        self._load_from_file()
    
    def _load_default_models(self):
        """加载默认的模型配置"""
        # DomainNet ViT FedSAK模型
        domainnet_vit = ModelConfig(
            model_id="1",
            name="ViT-DomainNet-FedSAK",
            model_type=ModelType.VISION_TRANSFORMER,
            input_type=InputType.IMAGE,
            description="Vision Transformer trained on DomainNet",
            model_path="exp_models/Domainnet_ViT_fedsak_lda",
            checkpoint_pattern="client/client_model_{client_id}.pt",
            num_labels=None,  # 动态计算，取决于实际数据集中的类别数量
            strategy="adapter",
            adapter_last_k=3,
            adapter_bottleneck=64,
            dataset_name="domainnet",
            dataset_config={
                "root": "/root/domainnet",
                "preprocessor_path": "pretrained_models/clip-vit-base-patch16/preprocessor_config.json",
                "classes_per_domain": 170  # 每个域选择的类别数量
            }
        )
        self._models["1"] = domainnet_vit
        
        # 示例：添加更多模型配置
        # LSTM FedProx模型（示例）
        # lstm_fedprox = ModelConfig(
        #     model_id="2",
        #     name="LSTM-FedProx",
        #     model_type=ModelType.LSTM,
        #     input_type=InputType.TEXT,
        #     description="LSTM model trained with FedProx on text classification",
        #     model_path="exp_models/LSTM_fedprox",
        #     checkpoint_pattern="model_{client_id}.pt",
        #     num_labels=10,
        #     strategy="linear",
        #     dataset_name="text_classification",
        #     dataset_config={
        #         "root": "/root/text_data",
        #         "vocab_size": 10000
        #     }
        # )
        # self._models["2"] = lstm_fedprox
    
    def _load_from_file(self):
        """从配置文件加载模型注册信息"""
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_data in data.get('models', []):
                        config = ModelConfig(
                            model_type=ModelType(model_data['model_type']),
                            input_type=InputType(model_data['input_type']),
                            **{k: v for k, v in model_data.items() 
                               if k not in ['model_type', 'input_type']}
                        )
                        self._models[config.model_id] = config
            except Exception as e:
                print(f"Warning: Failed to load model registry from {self._config_file}: {e}")
    
    def save_to_file(self):
        """保存模型注册信息到文件"""
        try:
            data = {
                'models': [
                    {
                        **asdict(config),
                        'model_type': config.model_type.value,
                        'input_type': config.input_type.value
                    }
                    for config in self._models.values()
                ]
            }
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save model registry to {self._config_file}: {e}")
    
    def register_model(self, config: ModelConfig):
        """注册新的模型配置"""
        self._models[config.model_id] = config
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """获取指定模型的配置"""
        return self._models.get(model_id)
    
    def list_models(self) -> List[ModelConfig]:
        """列出所有可用的模型"""
        return list(self._models.values())
    
    def list_models_by_input_type(self, input_type: InputType) -> List[ModelConfig]:
        """按输入类型筛选模型"""
        return [config for config in self._models.values() 
                if config.input_type == input_type]
    
    def get_model_path(self, model_id: str, client_id: Optional[int] = None) -> Optional[str]:
        """获取模型文件的完整路径"""
        config = self.get_model(model_id)
        if not config:
            return None
        
        if config.checkpoint_pattern and client_id is not None:
            # 支持多客户端模式
            checkpoint_name = config.checkpoint_pattern.format(client_id=client_id)
            return os.path.join(config.model_path, checkpoint_name)
        else:
            # 单一模型文件
            return config.model_path
    
    def validate_model_exists(self, model_id: str, client_id: Optional[int] = None) -> bool:
        """验证模型文件是否存在"""
        model_path = self.get_model_path(model_id, client_id)
        return model_path is not None and os.path.exists(model_path)


# 全局模型注册表实例
model_registry = ModelRegistry()