"""
Demo script for XAttention implementation
Tests both baseline and XAttention methods
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_attention import BaselineAttention, BaselineAttentionFlash
from xattention import XAttention, XAttentionOptimized
from utils import validate_implementation, compare_attention_methods, save_model_checkpoint, load_model_checkpoint


def run_basic_tests():
    """Run basic functionality tests"""
    print("=" * 60)
    print("Running Basic Tests")
    print("=" * 60)
    
    # Test parameters
    B, L, D = 2, 128, 256
    num_heads = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Test parameters: B={B}, L={L}, D={D}, heads={num_heads}")
    
    # Create modules
    baseline = BaselineAttention(D, num_heads, causal=True).to(device)
    xattention = XAttention(D, num_heads, block_size=8, stride=8, threshold=0.9, causal=True).to(device)
    
    # Test forward pass
    x = torch.randn(B, L, D, device=device)
    
    print("\nTesting forward passes...")
    
    with torch.no_grad():
        baseline_out = baseline(x)
        xattention_out = xattention(x)
    
    print(f"✓ Baseline output shape: {baseline_out.shape}")
    print(f"✓ XAttention output shape: {xattention_out.shape}")
    
    # Test sparsity stats
    stats = xattention.get_sparsity_stats()
    print(f"✓ XAttention sparsity stats: {stats}")
    
    return True


def run_performance_comparison():
    """Run performance comparison between methods"""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different sequence lengths
    test_configs = [
        {"B": 1, "L": 1024, "D": 512, "heads": 8},
        {"B": 1, "L": 2048, "D": 512, "heads": 8},
        {"B": 1, "L": 4096, "D": 512, "heads": 8},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\nTesting L={config['L']}...")
        
        baseline = BaselineAttention(config['D'], config['heads']).to(device)
        xattention = XAttention(
            config['D'], 
            config['heads'], 
            block_size=8, 
            stride=8, 
            threshold=0.9
        ).to(device)
        
        # Compare methods
        comparison = compare_attention_methods(
            baseline, 
            xattention, 
            (config['B'], config['L'], config['D']), 
            device=device
        )
        
        results.append({
            'config': config,
            'comparison': comparison
        })
        
        print(f"  Speedup: {comparison['performance']['speedup']:.2f}x")
        print(f"  Density: {comparison['sparsity'].get('density', 0):.3f}")
        print(f"  MSE: {comparison['accuracy']['mse']:.6f}")
        print(f"  Cosine Similarity: {comparison['accuracy']['cosine_similarity']:.4f}")
    
    return results


def test_different_strides():
    """Test XAttention with different stride values"""
    print("\n" + "=" * 60)
    print("Testing Different Stride Values")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, L, D, heads = 1, 2048, 512, 8
    
    strides = [4, 8, 16, 32]
    
    for stride in strides:
        print(f"\nTesting stride={stride}...")
        
        xattention = XAttention(
            D, heads, 
            block_size=8, 
            stride=stride, 
            threshold=0.9
        ).to(device)
        
        x = torch.randn(B, L, D, device=device)
        
        with torch.no_grad():
            output = xattention(x)
            stats = xattention.get_sparsity_stats()
        
        print(f"  Output shape: {output.shape}")
        print(f"  Sparsity stats: {stats}")


def test_threshold_optimization():
    """Test dynamic threshold optimization"""
    print("\n" + "=" * 60)
    print("Testing Threshold Optimization")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, L, D, heads = 1, 1024, 512, 8
    
    # Create XAttention with dynamic threshold
    xattention = XAttention(
        D, heads, 
        block_size=8, 
        stride=8, 
        threshold=0.9,
        use_dynamic_threshold=True
    ).to(device)
    
    x = torch.randn(B, L, D, device=device)
    
    # Simulate performance scores for optimization
    performance_scores = torch.randn(heads)
    density_scores = torch.rand(heads)
    
    with torch.no_grad():
        _ = xattention(x)
        
        # Run threshold optimization
        xattention.optimize_thresholds(
            performance_scores, 
            density_scores, 
            max_adjustments=10
        )
    
    print(f"Optimized thresholds: {xattention.head_thresholds}")


def test_checkpointing():
    """Test model saving and loading"""
    print("\n" + "=" * 60)
    print("Testing Checkpointing")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create and save model
    xattention = XAttention(512, 8, block_size=8, stride=8).to(device)
    
    # Save checkpoint
    checkpoint_path = "xattention_test.pth"
    save_model_checkpoint(xattention, checkpoint_path)
    
    # Load checkpoint
    new_xattention = XAttention(512, 8, block_size=8, stride=8).to(device)
    checkpoint = load_model_checkpoint(new_xattention, checkpoint_path)
    
    # Verify loaded model
    x = torch.randn(1, 1024, 512, device=device)
    
    with torch.no_grad():
        original_output = xattention(x)
        loaded_output = new_xattention(x)
    
    # Check if outputs are the same
    diff = torch.abs(original_output - loaded_output).max().item()
    print(f"Max difference after loading: {diff:.8f}")
    
    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def run_memory_benchmark():
    """Test memory usage"""
    print("\n" + "=" * 60)
    print("Memory Usage Test")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping memory benchmark")
        return
    
    torch.cuda.empty_cache()
    
    B, L, D, heads = 1, 4096, 512, 8
    
    baseline = BaselineAttention(D, heads).to(device)
    xattention = XAttention(D, heads, block_size=8, stride=8, threshold=0.9).to(device)
    
    x = torch.randn(B, L, D, device=device)
    
    # Measure memory before
    torch.cuda.synchronize()
    memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Forward pass
    with torch.no_grad():
        baseline_out = baseline(x)
        torch.cuda.synchronize()
        memory_baseline = torch.cuda.memory_allocated() / 1024**2
        
        torch.cuda.empty_cache()
        xattention_out = xattention(x)
        torch.cuda.synchronize()
        memory_xattention = torch.cuda.memory_allocated() / 1024**2
    
    print(f"Memory usage - Baseline: {memory_baseline - memory_before:.1f} MB")
    print(f"Memory usage - XAttention: {memory_xattention - memory_before:.1f} MB")
    print(f"Memory reduction: {(memory_baseline - memory_xattention) / memory_baseline * 100:.1f}%")


def main():
    """Main demo function"""
    print("XAttention Implementation Demo")
    print("=" * 60)
    
    # Validate implementation
    if not validate_implementation():
        print("❌ Implementation validation failed!")
        return
    
    print("✅ Implementation validation passed!")
    
    # Run basic tests
    run_basic_tests()
    
    # Run performance comparison
    run_performance_comparison()
    
    # Test different configurations
    test_different_strides()
    test_threshold_optimization()
    test_checkpointing()
    run_memory_benchmark()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()