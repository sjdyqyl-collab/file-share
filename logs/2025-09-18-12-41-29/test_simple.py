"""
Simple test script for XAttention implementation
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from baseline_attention import BaselineAttention
from xattention import XAttention


def test_basic_functionality():
    """Test basic functionality of both modules"""
    print("Testing XAttention Implementation...")
    
    # Test parameters
    B, L, D = 1, 64, 128
    num_heads = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Test config: B={B}, L={L}, D={D}, heads={num_heads}")
    
    try:
        # Create models
        baseline = BaselineAttention(D, num_heads, causal=True).to(device)
        xattention = XAttention(D, num_heads, block_size=8, stride=8, threshold=0.9, causal=True).to(device)
        
        print("✓ Models created successfully")
        
        # Create test input
        x = torch.randn(B, L, D, device=device)
        print(f"✓ Input tensor created: {x.shape}")
        
        # Test forward pass
        with torch.no_grad():
            baseline_out = baseline(x)
            xattention_out = xattention(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Baseline output: {baseline_out.shape}")
        print(f"  XAttention output: {xattention_out.shape}")
        
        # Test sparsity stats
        stats = xattention.get_sparsity_stats()
        print(f"✓ Sparsity stats: {stats}")
        
        # Test weight saving/loading
        state_dict = xattention.save_weights()
        xattention.load_weights(state_dict)
        print("✓ Weight save/load successful")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_configs():
    """Test different configurations"""
    print("\nTesting different configurations...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = [
        {"block_size": 4, "stride": 4, "threshold": 0.8},
        {"block_size": 8, "stride": 8, "threshold": 0.9},
        {"block_size": 16, "stride": 16, "threshold": 0.95},
    ]
    
    for config in configs:
        try:
            xattention = XAttention(
                dim=128,
                num_heads=8,
                **config
            ).to(device)
            
            x = torch.randn(1, 64, 128, device=device)
            
            with torch.no_grad():
                output = xattention(x)
                stats = xattention.get_sparsity_stats()
            
            print(f"✓ Config {config}: output={output.shape}, stats={stats}")
            
        except Exception as e:
            print(f"❌ Config {config}: {e}")


def test_performance():
    """Simple performance test"""
    print("\nRunning performance test...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, L, D = 1, 512, 256
    
    baseline = BaselineAttention(D, 8, causal=True).to(device)
    xattention = XAttention(D, 8, block_size=8, stride=8, threshold=0.9, causal=True).to(device)
    
    x = torch.randn(B, L, D, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = baseline(x)
            _ = xattention(x)
    
    # Time baseline
    start = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    end = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    
    if device == 'cuda':
        torch.cuda.synchronize()
        start.record()
        
        with torch.no_grad():
            for _ in range(20):
                _ = baseline(x)
        
        end.record()
        torch.cuda.synchronize()
        baseline_time = start.elapsed_time(end) / 20
        
        start.record()
        with torch.no_grad():
            for _ in range(20):
                _ = xattention(x)
        end.record()
        torch.cuda.synchronize()
        xattention_time = start.elapsed_time(end) / 20
        
        speedup = baseline_time / xattention_time
        
        print(f"Baseline: {baseline_time:.2f}ms")
        print(f"XAttention: {xattention_time:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
    else:
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = baseline(x)
        baseline_time = (time.time() - start) / 10
        
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = xattention(x)
        xattention_time = (time.time() - start) / 10
        
        speedup = baseline_time / xattention_time
        
        print(f"Baseline: {baseline_time*1000:.2f}ms")
        print(f"XAttention: {xattention_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    print("XAttention Simple Test Suite")
    print("=" * 40)
    
    success = test_basic_functionality()
    
    if success:
        test_different_configs()
        test_performance()
    else:
        print("Basic functionality failed, skipping additional tests")