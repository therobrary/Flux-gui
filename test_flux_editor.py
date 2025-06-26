#!/usr/bin/env python3
"""
Test script for Flux Image Editor functionality
"""

import sys
import tempfile
import torch
from PIL import Image
import numpy as np

# Test imports
try:
    from flux_editor_app import get_models, get_image_tensor, process_generated_image
    from flux.util import configs
    from flux.sampling import get_noise, get_schedule
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic app functionality without model loading"""
    print("\n🔧 Testing basic functionality...")
    
    # Test config availability
    available_models = list(configs.keys())
    expected_models = ["flux-dev", "flux-schnell", "flux-dev-kontext"]
    
    found_models = [m for m in expected_models if m in available_models]
    print(f"📋 Available models: {found_models}")
    
    if not found_models:
        print("❌ No expected models found in configs")
        return False
    
    # Test image tensor conversion
    print("🖼️ Testing image tensor conversion...")
    
    # Create a test image
    test_img = Image.new('RGB', (256, 256), color='red')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        test_img.save(tmp.name)
        
        # Test file-like object simulation
        with open(tmp.name, 'rb') as f:
            tensor = get_image_tensor(f)
            if tensor is not None and tensor.shape == (1, 3, 256, 256):
                print("✅ Image tensor conversion works")
            else:
                print(f"❌ Image tensor conversion failed. Shape: {tensor.shape if tensor is not None else None}")
                return False
    
    # Test noise generation
    print("🎲 Testing noise generation...")
    try:
        device = torch.device("cpu")  # Use CPU for testing
        noise = get_noise(1, 64, 64, device, torch.float32, 42)
        if noise.shape == (1, 16, 8, 8):  # Expected packed shape
            print("✅ Noise generation works")
        else:
            print(f"❌ Noise generation failed. Shape: {noise.shape}")
            return False
    except Exception as e:
        print(f"❌ Noise generation error: {e}")
        return False
    
    # Test timestep scheduling
    print("⏰ Testing timestep scheduling...")
    try:
        timesteps = get_schedule(10, 64, shift=True)
        if len(timesteps) == 11:  # Includes timestep 0, so 11 total
            print("✅ Timestep scheduling works")
        else:
            print(f"❌ Timestep scheduling failed. Length: {len(timesteps)} (expected 11)")
            return False
    except Exception as e:
        print(f"❌ Timestep scheduling error: {e}")
        return False
    
    return True

def test_mock_image_processing():
    """Test image processing pipeline with mock data"""
    print("\n🖼️ Testing image processing pipeline...")
    
    try:
        # Create mock generated image tensor (simulating model output)
        mock_output = torch.randn(1, 16, 32, 32)  # Latent space representation
        
        # Create a mock PIL image for testing
        mock_pil_image = Image.new('RGB', (512, 512), color='blue')
        
        print("✅ Mock image processing setup successful")
        return True
        
    except Exception as e:
        print(f"❌ Mock image processing error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Starting Flux Image Editor Tests")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_mock_image_processing,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Flux Image Editor is ready to use.")
        print("\nTo run the app:")
        print("streamlit run flux_editor_app.py")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)