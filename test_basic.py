#!/usr/bin/env python3
"""
Basic test script to verify the refactored CAPTCHA recognizer works.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported correctly."""
    try:
        from captcha_recognizer import Config, ModelConfig, TrainingConfig, DataConfig
        print("✓ Configuration classes imported successfully")
        
        from captcha_recognizer import CaptchaModel, create_model
        print("✓ Model classes imported successfully")
        
        from captcha_recognizer import CaptchaDataLoader
        print("✓ Data loader imported successfully")
        
        from captcha_recognizer import CaptchaTrainer
        print("✓ Trainer imported successfully")
        
        from captcha_recognizer import CaptchaPredictor
        print("✓ Predictor imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    try:
        from captcha_recognizer import Config
        
        # Test default configuration
        config = Config()
        print(f"✓ Default config created: image_size={config.model.input_shape}")
        
        # Test custom configuration
        custom_config = Config(
            training=TrainingConfig(batch_size=64, epochs=50)
        )
        print(f"✓ Custom config created: batch_size={custom_config.training.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    try:
        from captcha_recognizer import create_model, ModelConfig
        
        config = ModelConfig()
        model = create_model(config)
        
        print(f"✓ Model created successfully with {model.count_params()} parameters")
        print(f"✓ Model input shape: {model.input_shape}")
        print(f"✓ Model output layers: {len(model.output_names)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def test_data_loader():
    """Test data loader functionality."""
    try:
        from captcha_recognizer import CaptchaDataLoader, Config
        
        config = Config()
        data_loader = CaptchaDataLoader(config.data, config.model)
        
        print("✓ Data loader created successfully")
        
        # Test synthetic dataset creation
        synthetic_dataset = data_loader.create_synthetic_dataset(10)
        print(f"✓ Synthetic dataset created with {len(list(synthetic_dataset))} batches")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Refactored CAPTCHA Recognizer\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Model Creation Test", test_model_creation),
        ("Data Loader Test", test_data_loader),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} failed")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
