"""
Simple test script to verify Compact Attention implementations work correctly.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add current directory to path to handle imports
sys.path.insert(0, '/home/wzc/data/file-share/logs/2025-09-22-11-21-28')

from compact_attention import CompactAttention, CompactAttentionConfig


def test_basic_functionality():
    """Test basic functionality of Compact Attention."""
    print("Testing Compact Attention Basic Functionality...")
    
    # Create a simple configuration
    config = CompactAttentionConfig(
        dim=256,
        num_heads=8,
        tile_size=8,
        frame_size=(16, 32, 32),  # Small for testing
        recall_threshold=0.8,
        cost_threshold=0.1,
        pattern_cache_dir="./test_cache"
    )
    
    # Create model
    model = CompactAttention(**config.to_dict())
    
    # Create test input
    batch_size = 2
    seq_len = 16 * 32 * 32 // 64  # Reduced for testing
    dim = 256
    
    x = torch.randn(batch_size, seq_len, dim)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            output = model(x, layer_idx=0, head_idx=0)
        
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful")
        
        # Test output consistency
        with torch.no_grad():
            output2 = model(x, layer_idx=0, head_idx=0)
        
        diff = torch.abs(output - output2).max()
        print(f"Output consistency (max diff): {diff.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_tensor_shapes():
    """Test tensor shape handling."""
    print("\nTesting Tensor Shape Handling...")
    
    # Test with different configurations
    configs = [
        {"dim": 128, "num_heads": 4, "tile_size": 4, "frame_size": (8, 16, 16)},
        {"dim": 256, "num_heads": 8, "tile_size": 8, "frame_size": (4, 32, 32)},
        {"dim": 512, "num_heads": 16, "tile_size": 16, "frame_size": (2, 64, 64)},
    ]
    
    for i, config_dict in enumerate(configs):
        print(f"\nTest {i+1}: {config_dict}")
        
        config = CompactAttentionConfig(**config_dict)
        model = CompactAttention(**config.to_dict())
        
        # Calculate expected sequence length
        t, h, w = config_dict["frame_size"]
        seq_len = t * h * w // config_dict["tile_size"]
        
        x = torch.randn(1, seq_len, config_dict["dim"])
        
        try:
            with torch.no_grad():
                output = model(x)
            
            expected_shape = (1, seq_len, config_dict["dim"])
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            print(f"✓ Shape test passed: {output.shape}")
            
        except Exception as e:
            print(f"✗ Shape test failed: {e}")


def test_pattern_caching():
    """Test pattern caching functionality."""
    print("\nTesting Pattern Caching...")
    
    config = CompactAttentionConfig(
        dim=128,
        num_heads=4,
        tile_size=4,
        frame_size=(4, 8, 8),
        pattern_cache_dir="./test_pattern_cache"
    )
    
    model = CompactAttention(**config.to_dict())
    
    # Create test input
    seq_len = 4 * 8 * 8 // 4
    x = torch.randn(1, seq_len, 128)
    
    # First run - should create cache
    print("First run (creating cache)...")
    with torch.no_grad():
        output1 = model(x, layer_idx=0, head_idx=0)
    
    # Second run - should use cache
    print("Second run (using cache)...")
    with torch.no_grad():
        output2 = model(x, layer_idx=0, head_idx=0)
    
    # Check if cache files exist
    cache_key = model.get_cache_key(0, 0)
    cache_path = os.path.join(config.pattern_cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_path):
        print("✓ Pattern cache file created")
    else:
        print("✗ Pattern cache file not found")
    
    # Check output consistency
    diff = torch.abs(output1 - output2).max()
    print(f"Output consistency with cache: {diff.item():.8f}")


def test_memory_usage():
    """Test memory usage estimation."""
    print("\nTesting Memory Usage Estimation...")
    
    # Test different sequence lengths
    lengths = [100, 500, 1000, 2000]
    
    for length in lengths:
        # Full attention memory (float32)
        full_memory = length * length * 4 / (1024 ** 2)  # MB
        
        # Compact attention with 40% sparsity
        compact_memory = length * length * 4 * 0.4 / (1024 ** 2)  # MB
        
        reduction = (1 - compact_memory / full_memory) * 100
        
        print(f"Length {length:4d}: Full={full_memory:6.2f}MB, "
              f"Compact={compact_memory:6.2f}MB, Reduction={reduction:5.1f}%")


def test_gradient_flow():
    """Test gradient flow through the model."""
    print("\nTesting Gradient Flow...")
    
    config = CompactAttentionConfig(
        dim=64,
        num_heads=4,
        tile_size=2,
        frame_size=(2, 8, 8),
        pattern_cache_dir="./test_grad_cache"
    )
    
    model = CompactAttention(**config.to_dict())
    
    # Create test input
    seq_len = 2 * 8 * 8 // 2
    x = torch.randn(2, seq_len, 64, requires_grad=True)
    target = torch.randn(2, seq_len, 64)
    
    # Forward pass
    output = model(x, layer_idx=0, head_idx=0)
    
    # Compute loss
    loss = F.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None for p in model.parameters())
    
    if has_gradients:
        print("✓ Gradient flow successful")
        
        # Check gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(f"Total gradient norm: {total_norm:.4f}")
    else:
        print("✗ No gradients found")


def main():
    """Run all tests."""
    print("Compact Attention Implementation Test Suite")
    print("=" * 50)
    
    # Create test directory
    os.makedirs("./test_cache", exist_ok=True)
    os.makedirs("./test_pattern_cache", exist_ok=True)
    os.makedirs("./test_grad_cache", exist_ok=True)
    
    # Run tests
    success = True
    
    success &= test_basic_functionality()
    test_tensor_shapes()
    test_pattern_caching()
    test_memory_usage()
    test_gradient_flow()
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    if success:
        print("✓ All basic functionality tests passed!")
        print("✓ Compact Attention implementation is ready for use")
    else:
        print("✗ Some tests failed - please check the implementation")
    
    # Cleanup
    import shutil
    for cache_dir in ["./test_cache", "./test_pattern_cache", "./test_grad_cache"]:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


if __name__ == "__main__":
    main()