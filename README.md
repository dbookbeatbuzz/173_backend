# 模型测试后端系统

基于Flask的模型评测服务，采用插件化架构设计，支持快速添加新模型和数据集。

## 项目特点

- **插件化架构**: 通过继承基类创建插件，自动发现和注册
- **类型安全**: 基于类的设计，编译时检查接口规范
- **零配置**: 无需维护配置文件，代码即配置
- **RESTful API**: 提供完整的HTTP接口和SSE流式响应
- **CLI工具**: 丰富的命令行工具用于开发和调试

## 目录结构

```
173_backend/
├── src/
│   ├── plugins/              # 插件系统核心
│   │   ├── base.py          # 基类定义
│   │   ├── registry.py      # 自动发现与注册
│   │   ├── validators.py    # 配置验证
│   │   ├── models/          # 模型插件
│   │   │   └── vit_domainnet.py
│   │   └── datasets/        # 数据集插件
│   │       └── domainnet_plugin.py
│   ├── api/                 # Flask应用和API端点
│   ├── cli/                 # 命令行工具
│   ├── services/            # 业务逻辑层
│   └── datasets/            # 数据集加载工具
├── exp_models/              # 模型文件目录
├── pretrained_models/       # 预训练模型
└── docs/                    # 文档

```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
# 开发模式
python app.py

# 生产模式
gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app
```

### 环境变量

```bash
# 可选配置
export MODELS_ROOT=exp_models/Domainnet_ViT_fedsak_lda
export DATA_ROOT=/root/domainnet
export CLIP_PRETRAINED_DIR=pretrained_models/clip-vit-base-patch16
export HOST=0.0.0.0
export PORT=8000
```

## API接口

### 模型管理

#### GET /api/models
列出所有可用模型

```bash
curl http://localhost:8000/api/models
```

响应示例:
```json
{
  "models": [
    {
      "id": "domainnet_vit_fedsak",
      "name": "ViT-DomainNet-FedSAK",
      "type": "vit",
      "inputType": "image",
      "description": "Vision Transformer trained on DomainNet with FedSAK",
      "dataset": "domainnet",
      "available": true
    }
  ]
}
```

#### GET /api/models/{model_id}
获取特定模型详情

```bash
curl http://localhost:8000/api/models/domainnet_vit_fedsak
```

### 模型测试

#### POST /api/model-tests/
启动模型测试任务

```bash
curl -X POST http://localhost:8000/api/model-tests/ \
  -H "Content-Type: application/json" \
  -d '{
    "modelId": "domainnet_vit_fedsak",
    "clientId": 1,
    "sampleCount": 50,
    "randomSeed": 42
  }'
```

响应: `{"jobId": "job_xxx", "total": 50}`

#### GET /api/model-tests/{jobId}/stream
订阅SSE事件流（实时接收测试结果）

```bash
curl -N http://localhost:8000/api/model-tests/{jobId}/stream
```

事件类型:
- `progress`: 进度更新
- `case`: 单个测试用例结果
- `summary`: 最终汇总
- `error`: 错误信息

#### POST /api/model-tests/{jobId}/cancel
取消测试任务

#### GET /api/model-tests/{jobId}
查询任务状态

### 其他接口

#### GET /health
健康检查

#### GET /clients
列出可用的客户端ID（联邦学习场景）

#### POST /evaluate
运行评估任务

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "domainnet_vit_fedsak",
    "client_id": 1,
    "split": "test",
    "batch_size": 64
  }'
```

## CLI工具

### 插件管理

```bash
# 列出所有插件
python -m src.cli.plugin_manager list

# 检查模型
python -m src.cli.plugin_manager check-model domainnet_vit_fedsak

# 检查数据集
python -m src.cli.plugin_manager check-dataset domainnet

# 测试模型（加载并推理）
python -m src.cli.plugin_manager test-model domainnet_vit_fedsak --client-id 1
```

### 评估

```bash
# 评估模型
python -m src.cli.evaluate --model-id domainnet_vit_fedsak --client-id 1 --split test

# 输出JSON格式
python -m src.cli.evaluate --model-id domainnet_vit_fedsak --client-id 5 --json
```

### 系统验证

```bash
# 运行所有验证测试
python -m src.cli.validate

# 运行单个测试
python -m src.cli.validate --test model
python -m src.cli.validate --test dataset
```

## 添加新模型

### 步骤1: 创建模型插件

在 `src/plugins/models/` 目录下创建新的Python文件，例如 `resnet_cifar10.py`:

```python
from src.plugins.base import BaseModelPlugin, ModelMetadata, ModelType, InputType
import torch
import torch.nn as nn

class ResNetCIFAR10Plugin(BaseModelPlugin):
    """CIFAR-10 ResNet模型插件"""
    
    # 元数据定义
    metadata = ModelMetadata(
        model_id="resnet18_cifar10",
        name="ResNet-18 CIFAR-10",
        model_type=ModelType.RESNET,
        input_type=InputType.IMAGE,
        description="ResNet-18 model trained on CIFAR-10",
        author="Your Team",
        version="1.0.0",
        tags=["resnet", "cifar10", "classification"]
    )
    
    # 配置
    dataset_plugin_id = "cifar10"
    model_path = "exp_models/cifar10_resnet18/model.pth"
    strategy = "full"
    
    def build_model(self, num_labels: int, **kwargs) -> nn.Module:
        """构建模型实例"""
        from torchvision import models
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        return model
    
    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu"):
        """加载模型权重"""
        return torch.load(checkpoint_path, map_location=device)
```

### 步骤2: 验证

```bash
# 检查新模型是否被发现
python -m src.cli.plugin_manager list

# 测试模型加载
python -m src.cli.plugin_manager test-model resnet18_cifar10
```

完成！模型已自动注册并可通过API使用。

## 添加新数据集

### 步骤1: 创建数据集插件

在 `src/plugins/datasets/` 目录下创建新的Python文件，例如 `cifar10_plugin.py`:

```python
from src.plugins.base import BaseDatasetPlugin, DatasetMetadata, InputType
from typing import Tuple
from torch.utils.data import Dataset

class CIFAR10Plugin(BaseDatasetPlugin):
    """CIFAR-10数据集插件"""
    
    metadata = DatasetMetadata(
        dataset_id="cifar10",
        name="CIFAR-10",
        input_type=InputType.IMAGE,
        num_classes=10,
        description="CIFAR-10 image classification dataset",
        default_root="./data/cifar10",
        tags=["classification", "10-class"]
    )
    
    def build_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """构建训练/验证/测试数据集"""
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_set = datasets.CIFAR10(
            self.root, train=True, download=True, transform=transform
        )
        test_set = datasets.CIFAR10(
            self.root, train=False, download=True, transform=transform
        )
        
        # 返回 (train, val, test)
        return train_set, test_set, test_set
    
    def get_class_name(self, class_idx: int) -> str:
        """获取类别名称"""
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        if 0 <= class_idx < len(classes):
            return classes[class_idx]
        return f"class_{class_idx}"
```

### 步骤2: 验证

```bash
# 检查新数据集
python -m src.cli.plugin_manager check-dataset cifar10
```

## 插件系统核心概念

### 模型插件基类

所有模型插件必须继承 `BaseModelPlugin` 并实现:

- `metadata`: 模型元数据（ModelMetadata类型）
- `dataset_plugin_id`: 关联的数据集ID
- `build_model(num_labels, **kwargs)`: 构建模型实例
- `load_checkpoint(checkpoint_path, device)`: 加载模型权重

可选属性:
- `model_path`: 模型文件路径
- `checkpoint_pattern`: 检查点文件名模式
- `strategy`: 训练策略（如"adapter", "linear", "full"）
- `adapter_config`: 适配器配置字典

### 数据集插件基类

所有数据集插件必须继承 `BaseDatasetPlugin` 并实现:

- `metadata`: 数据集元数据（DatasetMetadata类型）
- `build_datasets()`: 返回(train, val, test)三个数据集
- `get_class_name(class_idx)`: 根据索引返回类别名称

### 自动发现机制

插件系统在启动时自动扫描 `src/plugins/models/` 和 `src/plugins/datasets/` 目录：

1. 导入所有Python模块
2. 查找继承自基类的类
3. 提取元数据并注册到全局注册表
4. 使用`metadata.model_id`或`metadata.dataset_id`作为唯一标识

无需手动注册或修改配置文件。

## 在代码中使用插件

```python
from src.plugins import init_plugins, plugin_registry

# 初始化插件系统
init_plugins()

# 获取模型插件
model_plugin_cls = plugin_registry.get_model_plugin("domainnet_vit_fedsak")
model_plugin = model_plugin_cls()

# 获取数据集插件
dataset_plugin_cls = plugin_registry.get_dataset_plugin("domainnet")
dataset_plugin = dataset_plugin_cls()

# 加载数据
train_set, val_set, test_set = dataset_plugin.get_datasets()
num_classes = dataset_plugin.get_num_classes()

# 构建和加载模型
model = model_plugin.build_model(num_labels=num_classes)
checkpoint_path = model_plugin.get_model_path(client_id=1)
state_dict = model_plugin.load_checkpoint(checkpoint_path)
model.load_state_dict(state_dict, strict=False)
```

## 开发建议

### 代码规范

- 使用类型注解
- 添加详细的文档字符串
- 遵循PEP 8代码风格

### 插件开发

- 元数据信息尽量完整，方便管理和检索
- 在`build_model()`和`build_datasets()`中添加适当的错误处理
- 使用日志记录关键操作（推荐使用Python的logging模块）

### 测试

创建新插件后运行验证:

```bash
# 检查插件是否被发现
python -m src.cli.plugin_manager list

# 测试模型
python -m src.cli.plugin_manager test-model <model_id>

# 测试数据集
python -m src.cli.plugin_manager check-dataset <dataset_id>

# 完整验证
python -m src.cli.validate
```

## 故障排除

### 插件未被发现

- 确保文件在正确的目录（`src/plugins/models/` 或 `src/plugins/datasets/`）
- 检查类是否正确继承了基类
- 确认已定义`metadata`属性
- 重启应用重新扫描

### 模型加载失败

```bash
# 检查模型文件是否存在
python -m src.cli.plugin_manager check-model <model_id>

# 查看详细错误信息
python -m src.cli.plugin_manager test-model <model_id> --client-id 1
```

### 数据集加载失败

- 检查`metadata.default_root`路径是否正确
- 确认数据文件已准备就绪
- 使用`check-dataset`命令查看详细错误

## 技术栈

- **Web框架**: Flask
- **深度学习**: PyTorch, Transformers
- **数据处理**: NumPy, scikit-learn
- **图像处理**: Pillow, torchvision
- **CORS支持**: flask-cors
- **生产部署**: Gunicorn

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request。添加新的模型或数据集插件时，请确保:

1. 遵循现有代码风格
2. 添加必要的文档字符串
3. 通过所有验证测试
4. 更新相关文档

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
