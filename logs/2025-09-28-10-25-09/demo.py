import torch
import torch.nn as nn
import sys
import os

# Add the current directory to path for imports
sys.path.append('/home/wzc/data/file-share/logs/2025-09-28-10-25-09')

from base_attention import BaseAttention
from xattention import XAttention

def test_attention_implementations():
    """Test both BaseAttention and XAttention implementations."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    hidden_size = 512
    num_heads = 8
    
    print("\n" + "="*50)
    print("Testing Attention Implementations")
    print("="*50)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_size).to(device)
    
    # Test BaseAttention
    print("\n1. Testing BaseAttention (Standard Multi-Head Attention)")
    base_attn = BaseAttention(hidden_size=hidden_size, num_heads=num_heads).to(device)
    
    with torch.no_grad():
        base_output, base_weights = base_attn(x, x, x, causal=True)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {base_output.shape}")
    print(f"   Attention weights shape: {base_weights.shape}")
    print(f"   ✓ BaseAttention test passed")
    
    # Test XAttention
    print("\n2. Testing XAttention (Sparse Attention with Antidiagonal Scoring)")
    xattn = XAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=32,
        stride=4,
        threshold=0.8
    ).to(device)
    
    with torch.no_grad():
        xattn_output, block_masks = xattn(x, x, x, causal=True)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {xattn_output.shape}")
    print(f"   Number of block masks: {len(block_masks)}")
    print(f"   Block mask shape: {block_masks[0].shape}")
    
    # Check sparsity statistics
    stats = xattn.get_sparsity_stats(block_masks)
    print(f"   Sparsity: {stats['sparsity']:.3f} ({stats['sparsity']*100:.1f}%)")
    print(f"   Density: {stats['density']:.3f} ({stats['density']*100:.1f}%)")
    print(f"   ✓ XAttention test passed")
    
    # Test with different sequence lengths
    print("\n3. Testing with different sequence lengths")
    test_lengths = [64, 256, 512]
    
    for seq_len in test_lengths:
        x_test = torch.randn(batch_size, seq_len, hidden_size).to(device)
        
        with torch.no_grad():
            xattn_output, block_masks = xattn(x_test, x_test, x_test, causal=True)
            stats = xattn.get_sparsity_stats(block_masks)
        
        print(f"   Seq_len={seq_len:3d}: Output={xattn_output.shape}, "
              f"Sparsity={stats['sparsity']:.3f}")
    
    # Test threshold adjustment
    print("\n4. Testing threshold adjustment")
    original_threshold = xattn.threshold
    
    for new_threshold in [0.5, 0.7, 0.9]:
        xattn.set_threshold(new_threshold)
        with torch.no_grad():
            _, block_masks = xattn(x, x, x, causal=True)
            stats = xattn.get_sparsity_stats(block_masks)
        
        print(f"   Threshold={new_threshold}: Sparsity={stats['sparsity']:.3f}, "
              f"Density={stats['density']:.3f}")
    
    # Restore original threshold
    xattn.set_threshold(original_threshold)
    
    # Test performance comparison
    print("\n5. Performance Comparison")
    import time
    
    # Warm up
    for _ in range(5):
        _ = base_attn(x, x, x, causal=True)
        _ = xattn(x, x, x, causal=True)
    
    # Time BaseAttention
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(10):
        _ = base_attn(x, x, x, causal=True)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    base_time = (time.time() - start_time) / 10
    
    # Time XAttention
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(10):
        _ = xattn(x, x, x, causal=True)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    xattn_time = (time.time() - start_time) / 10
    
    speedup = base_time / xattn_time
    print(f"   BaseAttention time: {base_time*1000:.2f}ms")
    print(f"   XAttention time: {xattn_time*1000:.2f}ms")
    print(f"   Speedup: {speedup:.2f}x")
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("="*50)

if __name__ == "__main__":
    test_attention_implementations()