#!/usr/bin/env python3
"""
Example usage of the refactored CAPTCHA Recognizer system.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def example_basic_usage():
    """Demonstrate basic usage of the system."""
    print("🔧 Basic Usage Example")
    print("=" * 50)
    
    try:
        from captcha_recognizer import Config, CaptchaModel, create_model
        
        # Create configuration
        config = Config()
        print(f"✓ Created configuration with image size: {config.model.input_shape}")
        
        # Create model
        model = create_model(config.model)
        print(f"✓ Created model with {model.count_params()} parameters")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        return True
        
    except Exception as e:
        print(f"✗ Basic usage example failed: {e}")
        return False

def example_custom_configuration():
    """Demonstrate custom configuration."""
    print("\n🔧 Custom Configuration Example")
    print("=" * 50)
    
    try:
        from captcha_recognizer import Config, ModelConfig, TrainingConfig
        
        # Create custom configuration
        custom_config = Config(
            model=ModelConfig(
                image_height=64,
                image_width=256,
                chars_num=6
            ),
            training=TrainingConfig(
                batch_size=64,
                learning_rate=0.001,
                epochs=150
            )
        )
        
        print(f"✓ Custom configuration created:")
        print(f"  - Image size: {custom_config.model.input_shape}")
        print(f"  - Characters: {custom_config.model.chars_num}")
        print(f"  - Batch size: {custom_config.training.batch_size}")
        print(f"  - Learning rate: {custom_config.training.learning_rate}")
        print(f"  - Epochs: {custom_config.training.epochs}")
        
        return True
        
    except Exception as e:
        print(f"✗ Custom configuration example failed: {e}")
        return False

def example_data_loader():
    """Demonstrate data loader usage."""
    print("\n🔧 Data Loader Example")
    print("=" * 50)
    
    try:
        from captcha_recognizer import Config, CaptchaDataLoader
        
        config = Config()
        data_loader = CaptchaDataLoader(config.data, config.model)
        
        print("✓ Data loader created successfully")
        
        # Create synthetic dataset for demonstration
        synthetic_dataset = data_loader.create_synthetic_dataset(5)
        print(f"✓ Created synthetic dataset with 5 samples")
        
        # Show dataset structure
        for batch_idx, (images, labels) in enumerate(synthetic_dataset):
            print(f"  Batch {batch_idx + 1}:")
            print(f"    Images shape: {images.shape}")
            print(f"    Labels shape: {labels.shape}")
            if batch_idx >= 1:  # Just show first 2 batches
                break
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader example failed: {e}")
        return False

def example_environment_config():
    """Demonstrate environment variable configuration."""
    print("\n🔧 Environment Configuration Example")
    print("=" * 50)
    
    try:
        import os
        from captcha_recognizer import Config
        
        # Set environment variables
        os.environ['CAPTCHA_BATCH_SIZE'] = '32'
        os.environ['CAPTCHA_LEARNING_RATE'] = '0.0005'
        os.environ['CAPTCHA_EPOCHS'] = '200'
        
        # Create configuration from environment
        config = Config.from_env()
        
        print(f"✓ Configuration loaded from environment:")
        print(f"  - Batch size: {config.training.batch_size}")
        print(f"  - Learning rate: {config.training.learning_rate}")
        print(f"  - Epochs: {config.training.epochs}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment configuration example failed: {e}")
        return False

def main():
    """Run all examples."""
    print("🚀 CAPTCHA Recognizer - Example Usage\n")
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Configuration", example_custom_configuration),
        ("Data Loader", example_data_loader),
        ("Environment Configuration", example_environment_config),
    ]
    
    passed = 0
    total = len(examples)
    
    for example_name, example_func in examples:
        try:
            if example_func():
                passed += 1
            else:
                print(f"✗ {example_name} failed")
        except Exception as e:
            print(f"✗ {example_name} failed with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Example Results: {passed}/{total} examples completed successfully")
    
    if passed == total:
        print("🎉 All examples completed successfully!")
        print("\n💡 You can now use the system:")
        print("   - Run training: python -m trainer")
        print("   - Make predictions: python -m predictor --model_path ./models/model.h5")
        print("   - Test the system: python test_basic.py")
    else:
        print("❌ Some examples failed. Please check the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
