"""
Demo script for XAttention implementations.
Tests both the original XAttention and ABS-XAttention methods.
"""

import torch
import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xattention_base import XAttentionBase
from xattention_abs import ABSXAttention


def test_xattention_base():
    """Test the original XAttention implementation."""
    print("=== Testing XAttention Base ===")
    
    # Configuration
    batch_size = 2
    seq_len = 2048
    hidden_size = 512
    num_heads = 8
    
    # Initialize model
    model = XAttentionBase(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=16,
        stride=8,
        threshold=0.9,
        use_dynamic_threshold=True
    )
    
    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, hidden_size // num_heads)
    k = torch.randn(batch_size, num_heads, seq_len, hidden_size // num_heads)
    v = torch.randn(batch_size, num_heads, seq_len, hidden_size // num_heads)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output, info = model(q, k, v)
        elapsed_time = time.time() - start_time
    
    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Density: {info['density']:.4f}")
    print(f"Threshold: {info['threshold']:.4f}")
    print(f"Forward time: {elapsed_time:.4f}s")
    
    # Test saving/loading weights
    checkpoint_path = "/tmp/xattention_base_test.pt"
    model.save_weights(checkpoint_path)
    
    new_model = XAttentionBase(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=16,
        stride=8,
        threshold=0.9,
    )
    new_model.load_weights(checkpoint_path)
    
    print("✓ Base XAttention test passed")
    return True


def test_abs_xattention():
    """Test the ABS-XAttention implementation."""
    print("\n=== Testing ABS-XAttention ===")
    
    # Configuration
    batch_size = 2
    seq_len = 2048
    hidden_size = 512
    num_heads = 8
    
    # Initialize model
    model = ABSXAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        min_block_size=8,
        max_block_size=32,
        stride=8,
        threshold=0.9,
        use_hierarchical=True
    )
    
    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, hidden_size // num_heads)
    k = torch.randn(batch_size, num_heads, seq_len, hidden_size // num_heads)
    v = torch.randn(batch_size, num_heads, seq_len, hidden_size // num_heads)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output, info = model(q, k, v)
        elapsed_time = time.time() - start_time
    
    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Density: {info['density']:.4f}")
    print(f"Block sizes: {info['block_sizes']}")
    print(f"Individual densities: {[f'{d:.4f}' for d in info['densities']]}")
    print(f"Forward time: {elapsed_time:.4f}s")
    
    # Get efficiency metrics
    metrics = model.get_efficiency_metrics()
    print(f"Efficiency metrics: {metrics}")
    
    # Test saving/loading weights
    checkpoint_path = "/tmp/abs_xattention_test.pt"
    model.save_weights(checkpoint_path)
    
    new_model = ABSXAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        min_block_size=8,
        max_block_size=32,
        stride=8,
        threshold=0.9,
    )
    new_model.load_weights(checkpoint_path)
    
    print("✓ ABS-XAttention test passed")
    return True


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\n=== Testing Memory Efficiency ===")
    
    # Test with longer sequence
    batch_size = 1
    seq_len = 4096
    hidden_size = 512
    num_heads = 8
    
    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, hidden_size // num_heads)
    k = torch.randn(batch_size, num_heads, seq_len, hidden_size // num_heads)
    v = torch.randn(batch_size, num_heads, seq_len, hidden_size // num_heads)
    
    # Test base model
    base_model = XAttentionBase(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=16,
        stride=8,
        threshold=0.9,
    )
    
    # Test ABS model
    abs_model = ABSXAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        min_block_size=8,
        max_block_size=32,
        stride=8,
        threshold=0.9,
        use_hierarchical=True
    )
    
    base_model.eval()
    abs_model.eval()
    
    with torch.no_grad():
        # Base model
        _, base_info = base_model(q, k, v)
        
        # ABS model
        _, abs_info = abs_model(q, k, v)
    
    print(f"Base model density: {base_info['density']:.4f}")
    print(f"ABS model density: {abs_info['density']:.4f}")
    print(f"Density reduction: {((base_info['density'] - abs_info['density']) / base_info['density'] * 100):.2f}%")
    
    print("✓ Memory efficiency test completed")
    return True


def test_different_sequence_lengths():
    """Test with different sequence lengths."""
    print("\n=== Testing Different Sequence Lengths ===")
    
    hidden_size = 256
    num_heads = 4
    
    seq_lengths = [512, 1024, 2048, 4096]
    
    for seq_len in seq_lengths:
        print(f"\nTesting seq_len={seq_len}")
        
        # Create input tensors
        q = torch.randn(1, num_heads, seq_len, hidden_size // num_heads)
        k = torch.randn(1, num_heads, seq_len, hidden_size // num_heads)
        v = torch.randn(1, num_heads, seq_len, hidden_size // num_heads)
        
        # Test both models
        base_model = XAttentionBase(
            hidden_size=hidden_size,
            num_heads=num_heads,
            block_size=16,
            stride=8,
            threshold=0.9,
        )
        
        abs_model = ABSXAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            min_block_size=8,
            max_block_size=32,
            stride=8,
            threshold=0.9,
        )
        
        base_model.eval()
        abs_model.eval()
        
        with torch.no_grad():
            _, base_info = base_model(q, k, v)
            _, abs_info = abs_model(q, k, v)
        
        print(f"  Base density: {base_info['density']:.4f}")
        print(f"  ABS density: {abs_info['density']:.4f}")
        print(f"  ABS block sizes: {abs_info['block_sizes']}")
    
    print("✓ Different sequence lengths test completed")
    return True


if __name__ == "__main__":
    print("Starting XAttention demo...")
    
    try:
        # Run all tests
        test_xattention_base()
        test_abs_xattention()
        test_memory_efficiency()
        test_different_sequence_lengths()
        
        print("\n🎉 All tests passed successfully!")
        print("\nFiles generated:")
        print("- xattention_base.py: Original XAttention implementation")
        print("- xattention_abs.py: ABS-XAttention with adaptive block sizing")
        print("- demo.py: This demo script")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()