"""Utilities for managing registered models."""

from __future__ import annotations

import argparse
import sys

from src.models.model_registry import InputType, ModelConfig, ModelRegistry, ModelType, model_registry


def list_models(registry: ModelRegistry) -> None:
    models = registry.list_models()
    if not models:
        print("没有找到任何模型")
        return

    print("已注册的模型：")
    print("-" * 60)
    for config in models:
        status = "✓" if registry.validate_model_exists(config.model_id, client_id=1) else "✗"
        print(f"{status} ID: {config.model_id}")
        print(f"  名称: {config.name}")
        print(f"  类型: {config.model_type.value}")
        print(f"  输入: {config.input_type.value}")
        print(f"  路径: {config.model_path}")
        print(f"  数据集: {config.dataset_name}")
        print()


def add_model(registry: ModelRegistry) -> None:
    print("添加新模型")
    print("-" * 30)

    model_id = input("模型ID: ").strip()
    if not model_id:
        print("模型ID不能为空")
        return

    if registry.get_model(model_id):
        print(f"模型ID {model_id} 已存在")
        return

    name = input("模型名称: ").strip()
    description = input("模型描述: ").strip()

    print("模型类型:")
    for idx, model_type in enumerate(ModelType, 1):
        print(f"  {idx}. {model_type.value}")

    try:
        type_choice = int(input("选择模型类型 (1-4): ")) - 1
        model_type = list(ModelType)[type_choice]
    except (ValueError, IndexError):
        print("无效的选择")
        return

    print("输入类型:")
    for idx, input_type in enumerate(InputType, 1):
        print(f"  {idx}. {input_type.value}")

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

    config = ModelConfig(
        model_id=model_id,
        name=name,
        model_type=model_type,
        input_type=input_type,
        description=description,
        model_path=model_path,
        checkpoint_pattern=checkpoint_pattern or None,
        num_labels=num_labels,
        strategy=strategy,
        dataset_name=dataset_name,
    )

    registry.register_model(config)
    registry.save_to_file()

    print(f"模型 {model_id} 已成功添加")


def remove_model(registry: ModelRegistry) -> None:
    model_id = input("要删除的模型ID: ").strip()
    if not model_id:
        return

    if not registry.get_model(model_id):
        print(f"模型 {model_id} 不存在")
        return

    confirm = input(f"确认删除模型 {model_id}? (y/N): ").strip().lower()
    if confirm == "y":
        print("删除功能待实现")
    else:
        print("取消删除")


def check_models(registry: ModelRegistry) -> None:
    print("检查模型文件状态：")
    print("-" * 40)

    models = registry.list_models()
    for config in models:
        exists = registry.validate_model_exists(config.model_id, client_id=1)
        status = "存在" if exists else "缺失"
        print(f"模型 {config.model_id} ({config.name}): {status}")
        if not exists:
            model_path = registry.get_model_path(config.model_id, client_id=1)
            print(f"  期望路径: {model_path}")


def test_dataset() -> None:
    print("测试 DomainNet 数据集加载...")
    try:
        from src.datasets.domainnet import build_domainnet_splits
        from src.models.dataset_manager import DatasetConfig, DatasetManagerFactory

        train_set, val_set, test_set, num_labels = build_domainnet_splits()

        print(f"✓ 数据集加载成功")
        print(f"  训练集: {len(train_set)} 样本")
        print(f"  验证集: {len(val_set)} 样本")
        print(f"  测试集: {len(test_set)} 样本")
        print(f"  类别数量: {num_labels}")

        dataset_config = DatasetConfig(
            name="domainnet",
            root="/root/domainnet",
            input_type="image",
            num_classes=0,
            preprocessing_config="pretrained_models/clip-vit-base-patch16/preprocessor_config.json",
        )

        dataset_manager = DatasetManagerFactory.create_manager("domainnet", dataset_config)
        test_set_mgr, num_classes_mgr = dataset_manager.get_test_dataset()

        print(f"✓ 数据集管理器工作正常")
        print(f"  管理器返回的类别数: {num_classes_mgr}")

        sample_indices = [0, 1, 5, 10, 20] if len(test_set_mgr) > 20 else [0]
        for idx in sample_indices:
            if idx < len(test_set_mgr):
                _, label = test_set_mgr[idx]
                class_name = dataset_manager.get_class_name(label)
                print(f"  样本 {idx}: 标签 {label} -> {class_name}")

    except Exception as exc:  # pragma: no cover - diagnostic helper
        import traceback

        print(f"✗ 测试失败: {exc}")
        print(f"详细错误: {traceback.format_exc()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模型注册管理脚本")
    parser.add_argument("command", nargs="?", help="命令: list/add/remove/check/test")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    command = args.command
    registry = model_registry

    if command == "list":
        list_models(registry)
    elif command == "add":
        add_model(registry)
    elif command == "remove":
        remove_model(registry)
    elif command == "check":
        check_models(registry)
    elif command == "test":
        test_dataset()
    else:
        print(f"未知命令: {command}")
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
