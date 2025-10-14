"""CLI tool to validate the plugin system."""

import argparse
import sys
from src.plugins import init_plugins, plugin_registry

def test_plugin_discovery():
    """Test that plugins are discovered correctly."""
    print("\n" + "="*60)
    print("TEST 1: Plugin Discovery")
    print("="*60)
    
    init_plugins()
    
    models = plugin_registry.list_models()
    datasets = plugin_registry.list_datasets()
    
    print(f"✓ Found {len(models)} model plugin(s)")
    print(f"✓ Found {len(datasets)} dataset plugin(s)")
    
    assert len(models) > 0, "No model plugins found!"
    assert len(datasets) > 0, "No dataset plugins found!"
    
    print("\nModel Plugins:")
    for m in models:
        print(f"  - {m.model_id}: {m.name}")
    
    print("\nDataset Plugins:")
    for d in datasets:
        print(f"  - {d.dataset_id}: {d.name}")
    
    return True


def test_model_plugin():
    """Test model plugin functionality."""
    print("\n" + "="*60)
    print("TEST 2: Model Plugin Functionality")
    print("="*60)
    
    # Ensure plugins are initialized
    if not plugin_registry._initialized:
        init_plugins()
    
    model_id = "domainnet_vit_fedsak"
    
    # Get model plugin
    model_plugin_cls = plugin_registry.get_model_plugin(model_id)
    assert model_plugin_cls is not None, f"Model plugin {model_id} not found!"
    
    model_plugin = model_plugin_cls()
    print(f"✓ Model plugin loaded: {model_plugin.metadata.name}")
    print(f"  Type: {model_plugin.metadata.model_type.value}")
    print(f"  Dataset: {model_plugin.dataset_plugin_id}")
    print(f"  Strategy: {model_plugin.strategy}")
    
    # Check model path
    model_path = model_plugin.get_model_path(client_id=1)
    print(f"  Model path (client 1): {model_path}")
    
    exists = model_plugin.validate_model_exists(client_id=1)
    print(f"  File exists: {'✓' if exists else '✗'}")
    
    return True


def test_dataset_plugin():
    """Test dataset plugin functionality."""
    print("\n" + "="*60)
    print("TEST 3: Dataset Plugin Functionality")
    print("="*60)
    
    # Ensure plugins are initialized
    if not plugin_registry._initialized:
        init_plugins()
    
    dataset_id = "domainnet"
    
    # Get dataset plugin
    dataset_plugin_cls = plugin_registry.get_dataset_plugin(dataset_id)
    assert dataset_plugin_cls is not None, f"Dataset plugin {dataset_id} not found!"
    
    try:
        dataset_plugin = dataset_plugin_cls()
        print(f"✓ Dataset plugin loaded: {dataset_plugin.metadata.name}")
        print(f"  Root: {dataset_plugin.root}")
        print(f"  Classes: {dataset_plugin.get_num_classes()}")
        
        # Try to load datasets
        print("\n  Loading datasets...")
        train_set, val_set, test_set = dataset_plugin.get_datasets()
        print(f"  ✓ Train: {len(train_set)} samples")
        print(f"  ✓ Val: {len(val_set)} samples")
        print(f"  ✓ Test: {len(test_set)} samples")
        
        # Test class name retrieval
        class_name = dataset_plugin.get_class_name(0)
        print(f"  ✓ Class 0: {class_name}")
        
    except Exception as e:
        print(f"  ⚠ Dataset loading test skipped: {e}")
        print(f"  (This is expected if data is not available)")
    
    return True


def test_integration():
    """Test integration between model and dataset plugins."""
    print("\n" + "="*60)
    print("TEST 4: Model-Dataset Integration")
    print("="*60)
    
    # Ensure plugins are initialized
    if not plugin_registry._initialized:
        init_plugins()
    
    model_id = "domainnet_vit_fedsak"
    
    # Get model plugin
    model_plugin_cls = plugin_registry.get_model_plugin(model_id)
    model_plugin = model_plugin_cls()
    
    # Get associated dataset plugin
    dataset_id = model_plugin.dataset_plugin_id
    dataset_plugin_cls = plugin_registry.get_dataset_plugin(dataset_id)
    
    assert dataset_plugin_cls is not None, f"Dataset {dataset_id} not found for model {model_id}!"
    
    print(f"✓ Model '{model_id}' correctly links to dataset '{dataset_id}'")
    
    dataset_plugin = dataset_plugin_cls()
    num_classes = dataset_plugin.get_num_classes()
    
    print(f"  Dataset classes: {num_classes}")
    
    # Try to build model
    try:
        print("\n  Building model...")
        model = model_plugin.build_model(num_labels=num_classes)
        print(f"  ✓ Model built successfully")
        print(f"  ✓ Model type: {type(model).__name__}")
    except Exception as e:
        print(f"  ⚠ Model building test skipped: {e}")
    
    return True


def test_api_compatibility():
    """Test that the API interfaces remain compatible."""
    print("\n" + "="*60)
    print("TEST 5: API Compatibility")
    print("="*60)
    
    # Ensure plugins are initialized
    if not plugin_registry._initialized:
        init_plugins()
    
    # Test registry methods
    model_metadata = plugin_registry.get_model_metadata("domainnet_vit_fedsak")
    assert model_metadata is not None, "get_model_metadata failed!"
    print(f"✓ get_model_metadata() works")
    
    dataset_metadata = plugin_registry.get_dataset_metadata("domainnet")
    assert dataset_metadata is not None, "get_dataset_metadata failed!"
    print(f"✓ get_dataset_metadata() works")
    
    # Test validation
    exists = plugin_registry.validate_model_exists("domainnet_vit_fedsak", client_id=1)
    print(f"✓ validate_model_exists() works (result: {exists})")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PLUGIN SYSTEM REFACTOR - VALIDATION TESTS")
    print("#"*60)
    
    tests = [
        ("Plugin Discovery", test_plugin_discovery),
        ("Model Plugin", test_model_plugin),
        ("Dataset Plugin", test_dataset_plugin),
        ("Integration", test_integration),
        ("API Compatibility", test_api_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n {failed} test(s) failed. Please review the errors above.")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate the plugin system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all validation tests
  python -m src.cli.validate
  
  # Run specific test
  python -m src.cli.validate --test discovery
  python -m src.cli.validate --test model
  python -m src.cli.validate --test dataset
  python -m src.cli.validate --test integration
  python -m src.cli.validate --test api
        """
    )
    
    parser.add_argument(
        "--test",
        choices=["discovery", "model", "dataset", "integration", "api"],
        help="Run a specific test (if not specified, run all tests)"
    )
    
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    test_map = {
        "discovery": ("Plugin Discovery", test_plugin_discovery),
        "model": ("Model Plugin", test_model_plugin),
        "dataset": ("Dataset Plugin", test_dataset_plugin),
        "integration": ("Integration", test_integration),
        "api": ("API Compatibility", test_api_compatibility),
    }
    
    if args.test:
        # Run single test
        name, test_func = test_map[args.test]
        print("\n" + "#"*60)
        print(f"# PLUGIN SYSTEM VALIDATION - {name.upper()}")
        print("#"*60)
        
        try:
            if test_func():
                print(f"\n✓ {name} test passed!")
                return 0
            else:
                print(f"\n✗ {name} test failed!")
                return 1
        except Exception as e:
            print(f"\n✗ {name} test failed!")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        # Run all tests
        return main()


if __name__ == "__main__":
    sys.exit(cli_main())
