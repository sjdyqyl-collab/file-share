"""
Demo script to test XAttention implementations.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add the current directory to path for imports
sys.path.append('/home/wzc/data/file-share/logs/2025-09-19-10-22-22')

from xattention_original import XAttentionOriginal
from xattention_improved import XAttentionImproved


def benchmark_attention(model, x, num_iterations=10):
    """Benchmark attention model."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time, output


def test_original_xattention():
    """Test original XAttention implementation."""
    print("=== Testing Original XAttention ===")
    
    # Configuration
    batch_size = 2
    seq_len = 512
    dim = 256
    num_heads = 8
    
    # Create model
    model = XAttentionOriginal(
        dim=dim,
        num_heads=num_heads,
        block_size=8,
        stride=8,
        threshold=0.9,
        use_dynamic_threshold=False
    )
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test forward pass
    try:
        output = model(x, return_attention=True)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output['output'].shape}")
        print(f"  Attention shape: {output['attention_weights'].shape}")
        
        # Benchmark
        avg_time, _ = benchmark_attention(model, x)
        print(f"  Average inference time: {avg_time*1000:.2f}ms")
        
        # Check sparsity
        attn_weights = output['attention_weights']
        sparsity = (attn_weights.abs() < 1e-6).float().mean()
        print(f"  Attention sparsity: {sparsity.item()*100:.2f}%")
        
        stats = model.get_sparsity_stats()
        print(f"  Block size: {stats['block_size']}")
        print(f"  Stride: {stats['stride']}")
        print(f"  Threshold: {stats['threshold']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def test_improved_xattention():
    """Test improved XAttention implementation."""
    print("\n=== Testing Improved XAttention ===")
    
    # Configuration
    batch_size = 2
    seq_len = 512
    dim = 256
    num_heads = 8
    
    # Create model with all improvements enabled
    model = XAttentionImproved(
        dim=dim,
        num_heads=num_heads,
        default_block_size=8,
        strides=[4, 8, 16],
        default_threshold=0.9,
        use_adaptive_warmup=True,
        use_multi_scale=True,
        use_content_adaptive=True,
        use_gradient_optimization=True,
        use_dynamic_blocks=True
    )
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test forward pass
    try:
        # Test with different steps to see warmup behavior
        output_step_0 = model(x, return_attention=True, step=0)  # Warmup
        output_step_5 = model(x, return_attention=True, step=5)  # After warmup
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output_step_5['output'].shape}")
        print(f"  Attention shape: {output_step_5['attention_weights'].shape}")
        
        # Benchmark
        avg_time, _ = benchmark_attention(model, x)
        print(f"  Average inference time: {avg_time*1000:.2f}ms")
        
        # Check sparsity after warmup
        attn_weights = output_step_5['attention_weights']
        sparsity = (attn_weights.abs() < 1e-6).float().mean()
        print(f"  Attention sparsity: {sparsity.item()*100:.2f}%")
        
        stats = model.get_sparsity_stats()
        print(f"  Warmup steps: {stats['warmup_steps']}")
        print(f"  Thresholds: {[f'{t:.3f}' for t in stats['thresholds'][:4]]}")  # Show first 4
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def test_comparison():
    """Compare original and improved versions."""
    print("\n=== Performance Comparison ===")
    
    # Configuration
    batch_size = 2
    seq_len = 512
    dim = 256
    num_heads = 8
    
    # Create models
    original_model = XAttentionOriginal(
        dim=dim,
        num_heads=num_heads,
        block_size=8,
        stride=8,
        threshold=0.9
    )
    
    improved_model = XAttentionImproved(
        dim=dim,
        num_heads=num_heads,
        default_block_size=8,
        strides=[4, 8, 16],
        default_threshold=0.9,
        use_adaptive_warmup=False,  # Disable for fair comparison
        use_multi_scale=True,
        use_content_adaptive=False,
        use_gradient_optimization=False,
        use_dynamic_blocks=False
    )
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim)
    
    # Benchmark both
    original_time, original_out = benchmark_attention(original_model, x)
    improved_time, improved_out = benchmark_attention(improved_model, x)
    
    print(f"Original XAttention: {original_time*1000:.2f}ms")
    print(f"Improved XAttention: {improved_time*1000:.2f}ms")
    print(f"Speedup: {original_time/improved_time:.2f}x")
    
    # Check output similarity
    if 'output' in original_out and 'output' in improved_out:
        diff = torch.abs(original_out['output'] - improved_out['output']).mean()
        print(f"Output difference: {diff.item():.6f}")


def test_memory_usage():
    """Test memory usage."""
    print("\n=== Memory Usage Test ===")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        batch_size = 1
        seq_len = 1024
        dim = 512
        
        model = XAttentionImproved(
            dim=dim,
            num_heads=8,
            default_block_size=16,
            strides=[8, 16],
            use_adaptive_warmup=True,
            use_multi_scale=True,
            use_content_adaptive=True
        )
        
        x = torch.randn(batch_size, seq_len, dim).cuda()
        model = model.cuda()
        
        # Forward pass
        with torch.no_grad():
            output = model(x, step=10)  # After warmup
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        print(f"Sequence length: {seq_len}")
        print(f"Memory per token: {peak_memory*1024/seq_len:.2f} KB")
    else:
        print("CUDA not available, skipping memory test")


def main():
    """Main demo function."""
    print("XAttention Implementation Demo")
    print("=" * 40)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test original implementation
    success_original = test_original_xattention()
    
    # Test improved implementation
    success_improved = test_improved_xattention()
    
    # Run comparison if both succeed
    if success_original and success_improved:
        test_comparison()
        test_memory_usage()
    else:
        print("\nSkipping comparison due to test failures")
    
    print("\n" + "=" * 40)
    print("Demo completed!")


if __name__ == "__main__":
    main()