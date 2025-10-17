"""Plugin management CLI tool."""

from __future__ import annotations

import argparse
import sys

from src.plugins import init_plugins, plugin_registry


def list_plugins() -> None:
    """List all available plugins."""
    init_plugins()
    
    print("\n" + "=" * 60)
    print("  Available Model Plugins")
    print("=" * 60)
    
    models = plugin_registry.list_models()
    if not models:
        print("  No model plugins found")
    else:
        for metadata in models:
            status = "✓" if plugin_registry.validate_model_exists(metadata.model_id, client_id=1) else "✗"
            print(f"\n{status} [{metadata.model_id}] {metadata.name}")
            print(f"    Type: {metadata.model_type.value}")
            print(f"    Input: {metadata.input_type.value}")
            print(f"    Description: {metadata.description}")
            print(f"    Version: {metadata.version}")
            if metadata.tags:
                print(f"    Tags: {', '.join(metadata.tags)}")
    
    print("\n" + "=" * 60)
    print("  Available Dataset Plugins")
    print("=" * 60)
    
    datasets = plugin_registry.list_datasets()
    if not datasets:
        print("  No dataset plugins found")
    else:
        for metadata in datasets:
            print(f"\n  [{metadata.dataset_id}] {metadata.name}")
            print(f"    Input Type: {metadata.input_type.value}")
            print(f"    Classes: {metadata.num_classes}")
            print(f"    Description: {metadata.description}")
            print(f"    Default Root: {metadata.default_root}")
            if metadata.tags:
                print(f"    Tags: {', '.join(metadata.tags)}")
    
    print("\n" + "=" * 60)


def check_model(model_id: str) -> None:
    """Check if a model's files exist."""
    init_plugins()
    
    print(f"\nChecking model: {model_id}")
    print("-" * 40)
    
    model_plugin_cls = plugin_registry.get_model_plugin(model_id)
    if not model_plugin_cls:
        print(f"✗ Model plugin '{model_id}' not found")
        return
    
    model_plugin = model_plugin_cls()
    print(f"✓ Plugin found: {model_plugin.metadata.name}")
    print(f"  Model path: {model_plugin.model_path}")
    
    if model_plugin.checkpoint_pattern:
        print(f"  Checkpoint pattern: {model_plugin.checkpoint_pattern}")
        print(f"\n  Checking client checkpoints:")
        
        for client_id in range(1, 6):
            exists = model_plugin.validate_model_exists(client_id)
            status = "✓" if exists else "✗"
            path = model_plugin.get_model_path(client_id)
            print(f"    {status} Client {client_id}: {path}")
    else:
        exists = model_plugin.validate_model_exists()
        status = "✓" if exists else "✗"
        print(f"  {status} Model file exists")


def check_dataset(dataset_id: str) -> None:
    """Check if a dataset can be loaded."""
    init_plugins()
    
    print(f"\nChecking dataset: {dataset_id}")
    print("-" * 40)
    
    dataset_plugin_cls = plugin_registry.get_dataset_plugin(dataset_id)
    if not dataset_plugin_cls:
        print(f"✗ Dataset plugin '{dataset_id}' not found")
        return
    
    try:
        dataset_plugin = dataset_plugin_cls()
        print(f"✓ Plugin found: {dataset_plugin.metadata.name}")
        print(f"  Root: {dataset_plugin.root}")
        
        print(f"\n  Loading datasets...")
        train_set, val_set, test_set = dataset_plugin.get_datasets()
        
        print(f"  ✓ Train set: {len(train_set)} samples")
        print(f"  ✓ Val set: {len(val_set)} samples")
        print(f"  ✓ Test set: {len(test_set)} samples")
        print(f"  ✓ Classes: {dataset_plugin.get_num_classes()}")
        
        # Test class name retrieval
        print(f"\n  Sample class names:")
        for idx in [0, 1, 5, 10]:
            class_name = dataset_plugin.get_class_name(idx)
            print(f"    Class {idx}: {class_name}")
        
        print(f"\n✓ Dataset '{dataset_id}' is working correctly")
        
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()


def test_model(model_id: str, client_id: int = 1) -> None:
    """Test if a model can be loaded and run inference."""
    init_plugins()
    
    print(f"\nTesting model: {model_id} (client {client_id})")
    print("-" * 40)
    
    try:
        import torch
        
        # Get model plugin
        model_plugin_cls = plugin_registry.get_model_plugin(model_id)
        if not model_plugin_cls:
            print(f"✗ Model plugin '{model_id}' not found")
            return
        
        model_plugin = model_plugin_cls()
        print(f"✓ Plugin: {model_plugin.metadata.name}")
        
        # Get dataset plugin
        dataset_plugin_cls = plugin_registry.get_dataset_plugin(model_plugin.dataset_plugin_id)
        if not dataset_plugin_cls:
            print(f"✗ Dataset plugin '{model_plugin.dataset_plugin_id}' not found")
            return
        
        dataset_plugin = dataset_plugin_cls()
        print(f"✓ Dataset: {dataset_plugin.metadata.name}")
        
        # Load dataset
        _, _, test_set = dataset_plugin.get_datasets()
        num_labels = dataset_plugin.get_num_classes()
        print(f"✓ Test set loaded: {len(test_set)} samples, {num_labels} classes")
        
        # Load model
        model_path = model_plugin.get_model_path(client_id)
        print(f"✓ Model path: {model_path}")
        
        state_dict = model_plugin.load_checkpoint(model_path)
        print(f"✓ Checkpoint loaded")
        
        model = model_plugin.build_model(num_labels=num_labels)
        print(f"✓ Model built")
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"✓ Model ready")
        
        # Test inference on a sample
        sample = test_set[0]
        print(f"\n  Testing inference on sample 0...")

        # Handle different data types (graph vs tensor)
        if hasattr(sample, 'y'):
            # Graph data (PyTorch Geometric Data object)
            label = sample.y.item() if sample.y.dim() == 1 else sample.y[0].item()
            print(f"  True label: {label} ({dataset_plugin.get_class_name(label)})")

            with torch.no_grad():
                output = model(sample)
                pred = output.argmax(dim=1).item()
                print(f"  Predicted: {pred} ({dataset_plugin.get_class_name(pred)})")
        else:
            # Tensor data (image datasets return (tensor, label) tuples)
            data, label = sample
            print(f"  True label: {label} ({dataset_plugin.get_class_name(label)})")

            with torch.no_grad():
                output = model(data.unsqueeze(0))
                pred = output.argmax(dim=1).item()
                print(f"  Predicted: {pred} ({dataset_plugin.get_class_name(pred)})")
        
        print(f"\n✓ Model '{model_id}' is working correctly")
        
    except Exception as e:
        print(f"\n✗ Failed to test model: {e}")
        import traceback
        traceback.print_exc()


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Plugin management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all plugins
  python -m src.cli.plugin_manager list
  
  # Check a model
  python -m src.cli.plugin_manager check-model domainnet_vit_fedsak
  
  # Check a dataset
  python -m src.cli.plugin_manager check-dataset domainnet
  
  # Test a model
  python -m src.cli.plugin_manager test-model domainnet_vit_fedsak --client-id 1
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    subparsers.add_parser("list", help="List all available plugins")
    
    # Check model command
    check_model_parser = subparsers.add_parser("check-model", help="Check if model files exist")
    check_model_parser.add_argument("model_id", help="Model ID to check")
    
    # Check dataset command
    check_dataset_parser = subparsers.add_parser("check-dataset", help="Check if dataset can be loaded")
    check_dataset_parser.add_argument("dataset_id", help="Dataset ID to check")
    
    # Test model command
    test_model_parser = subparsers.add_parser("test-model", help="Test if model can run inference")
    test_model_parser.add_argument("model_id", help="Model ID to test")
    test_model_parser.add_argument("--client-id", type=int, default=1, help="Client ID (default: 1)")
    
    return parser


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "list":
            list_plugins()
        elif args.command == "check-model":
            check_model(args.model_id)
        elif args.command == "check-dataset":
            check_dataset(args.dataset_id)
        elif args.command == "test-model":
            test_model(args.model_id, args.client_id)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
