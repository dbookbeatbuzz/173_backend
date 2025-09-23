#!/usr/bin/env python3
"""
模型注册管理脚本
用于管理和配置可用的模型
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_registry import model_registry, ModelConfig, ModelType, InputType


def list_models():
    """列出所有已注册的模型"""
    print("已注册的模型：")
    print("-" * 60)
    
    models = model_registry.list_models()
    if not models:
        print("没有找到任何模型")
        return
    
    for config in models:
        status = "✓" if model_registry.validate_model_exists(config.model_id, client_id=1) else "✗"
        print(f"{status} ID: {config.model_id}")
        print(f"  名称: {config.name}")
        print(f"  类型: {config.model_type.value}")
        print(f"  输入: {config.input_type.value}")
        print(f"  路径: {config.model_path}")
        print(f"  数据集: {config.dataset_name}")
        print()


def add_model():
    """交互式添加新模型"""
    print("添加新模型")
    print("-" * 30)
    
    # 收集模型信息
    model_id = input("模型ID: ").strip()
    if not model_id:
        print("模型ID不能为空")
        return
    
    if model_registry.get_model(model_id):
        print(f"模型ID {model_id} 已存在")
        return
    
    name = input("模型名称: ").strip()
    description = input("模型描述: ").strip()
    
    print("模型类型:")
    for i, model_type in enumerate(ModelType, 1):
        print(f"  {i}. {model_type.value}")
    
    try:
        type_choice = int(input("选择模型类型 (1-4): ")) - 1
        model_type = list(ModelType)[type_choice]
    except (ValueError, IndexError):
        print("无效的选择")
        return
    
    print("输入类型:")
    for i, input_type in enumerate(InputType, 1):
        print(f"  {i}. {input_type.value}")
    
    try:
        input_choice = int(input("选择输入类型 (1-2): ")) - 1
        input_type = list(InputType)[input_choice]
    except (ValueError, IndexError):
        print("无效的选择")
        return
    
    model_path = input("模型路径: ").strip()
    checkpoint_pattern = input("检查点文件模式 (可选，如 'client/client_model_{client_id}.pt'): ").strip()
    
    try:
        num_labels = int(input("类别数量: "))
    except ValueError:
        num_labels = None
    
    strategy = input("策略 (默认: adapter): ").strip() or "adapter"
    dataset_name = input("数据集名称 (默认: domainnet): ").strip() or "domainnet"
    
    # 创建配置
    config = ModelConfig(
        model_id=model_id,
        name=name,
        model_type=model_type,
        input_type=input_type,
        description=description,
        model_path=model_path,
        checkpoint_pattern=checkpoint_pattern if checkpoint_pattern else None,
        num_labels=num_labels,
        strategy=strategy,
        dataset_name=dataset_name
    )
    
    # 注册模型
    model_registry.register_model(config)
    model_registry.save_to_file()
    
    print(f"模型 {model_id} 已成功添加")


def remove_model():
    """删除模型"""
    model_id = input("要删除的模型ID: ").strip()
    if not model_id:
        return
    
    if not model_registry.get_model(model_id):
        print(f"模型 {model_id} 不存在")
        return
    
    confirm = input(f"确认删除模型 {model_id}? (y/N): ").strip().lower()
    if confirm == 'y':
        # 这里需要在model_registry中添加删除方法
        print("删除功能待实现")
    else:
        print("取消删除")


def check_models():
    """检查所有模型的文件状态"""
    print("检查模型文件状态：")
    print("-" * 40)
    
    models = model_registry.list_models()
    for config in models:
        exists = model_registry.validate_model_exists(config.model_id, client_id=1)
        status = "存在" if exists else "缺失"
        
        print(f"模型 {config.model_id} ({config.name}): {status}")
        if not exists:
            model_path = model_registry.get_model_path(config.model_id, client_id=1)
            print(f"  期望路径: {model_path}")


def test_dataset():
    """测试数据集加载"""
    print("测试 DomainNet 数据集加载...")
    try:
        from data_domainnet import build_domainnet_splits
        
        train_set, val_set, test_set, num_labels = build_domainnet_splits()
        
        print(f"✓ 数据集加载成功")
        print(f"  训练集: {len(train_set)} 样本")
        print(f"  验证集: {len(val_set)} 样本")
        print(f"  测试集: {len(test_set)} 样本")
        print(f"  类别数量: {num_labels}")
        
        # 获取类别名称示例
        if hasattr(test_set.dataset.dataset, 'idx_to_class'):
            class_names = test_set.dataset.dataset.idx_to_class
            print(f"  前10个类别: {class_names[:10]}")
            print(f"  实际选择的类别数: {len(class_names)}")
        
        # 测试数据集管理器
        print("\n测试数据集管理器...")
        from models.dataset_manager import DatasetManagerFactory, DatasetConfig
        
        dataset_config = DatasetConfig(
            name="domainnet",
            root="/root/domainnet",
            input_type="image",
            num_classes=0,
            preprocessing_config="pretrained_models/clip-vit-base-patch16/preprocessor_config.json"
        )
        
        dataset_manager = DatasetManagerFactory.create_manager("domainnet", dataset_config)
        test_set_mgr, num_classes_mgr = dataset_manager.get_test_dataset()
        
        print(f"✓ 数据集管理器工作正常")
        print(f"  管理器返回的类别数: {num_classes_mgr}")
        
        # 测试类别名称获取
        sample_indices = [0, 1, 5, 10, 20] if len(test_set_mgr) > 20 else [0]
        for idx in sample_indices:
            if idx < len(test_set_mgr):
                _, label = test_set_mgr[idx]
                class_name = dataset_manager.get_class_name(label)
                print(f"  样本 {idx}: 标签 {label} -> {class_name}")
        
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        print(f"详细错误: {traceback.format_exc()}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python manage_models.py <command>")
        print("命令:")
        print("  list     - 列出所有模型")
        print("  add      - 添加新模型")
        print("  remove   - 删除模型")
        print("  check    - 检查模型文件状态")
        print("  test     - 测试数据集加载")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_models()
    elif command == "add":
        add_model()
    elif command == "remove":
        remove_model()
    elif command == "check":
        check_models()
    elif command == "test":
        test_dataset()
    else:
        print(f"未知命令: {command}")


if __name__ == "__main__":
    main()