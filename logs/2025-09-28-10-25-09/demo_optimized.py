import torch
import torch.nn as nn
import sys
import os
import time

# Add the current directory to path for imports
sys.path.append('/home/wzc/data/file-share/logs/2025-09-28-10-25-09')

from base_attention import BaseAttention
from xattention import XAttention
from xattention_optimized import XAttentionOptimized

def compare_implementations():
    """Compare original and optimized XAttention implementations."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 2
    seq_len = 256
    hidden_size = 512
    num_heads = 8
    
    print("\n" + "="*60)
    print("Comparing XAttention Implementations")
    print("="*60)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_size).to(device)
    
    # Initialize models
    print("\nInitializing models...")
    
    base_attn = BaseAttention(hidden_size=hidden_size, num_heads=num_heads).to(device)
    
    xattn_original = XAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=32,
        stride=4,
        threshold=0.8
    ).to(device)
    
    xattn_optimized = XAttentionOptimized(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=32,
        stride=4,
        threshold=0.8
    ).to(device)
    
    # Warm up
    print("Warming up...")
    for _ in range(3):
        _ = base_attn(x, x, x, causal=True)
        _ = xattn_original(x, x, x, causal=True)
        _ = xattn_optimized(x, x, x, causal=True)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # Test BaseAttention
    print("\n1. Testing BaseAttention (Standard)")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(10):
        base_output, base_weights = base_attn(x, x, x, causal=True)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    base_time = (time.time() - start_time) / 10
    
    print(f"   Average time: {base_time*1000:.2f}ms")
    print(f"   Output shape: {base_output.shape}")
    
    # Test Original XAttention
    print("\n2. Testing XAttention (Original Implementation)")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(10):
        xattn_output_orig, block_masks_orig = xattn_original(x, x, x, causal=True)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    xattn_orig_time = (time.time() - start_time) / 10
    
    print(f"   Average time: {xattn_orig_time*1000:.2f}ms")
    print(f"   Output shape: {xattn_output_orig.shape}")
    stats_orig = xattn_original.get_sparsity_stats(block_masks_orig)
    print(f"   Sparsity: {stats_orig['sparsity']:.3f} ({stats_orig['sparsity']*100:.1f}%)")
    
    # Test Optimized XAttention
    print("\n3. Testing XAttentionOptimized (Vectorized Implementation)")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(10):
        xattn_output_opt, block_masks_opt = xattn_optimized(x, x, x, causal=True)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    xattn_opt_time = (time.time() - start_time) / 10
    
    print(f"   Average time: {xattn_opt_time*1000:.2f}ms")
    print(f"   Output shape: {xattn_output_opt.shape}")
    stats_opt = xattn_optimized.get_sparsity_stats(block_masks_opt)
    print(f"   Sparsity: {stats_opt['sparsity']:.3f} ({stats_opt['sparsity']*100:.1f}%)")
    
    # Performance comparison
    print("\n4. Performance Comparison")
    print(f"   BaseAttention time: {base_time*1000:.2f}ms")
    print(f"   XAttention (original) time: {xattn_orig_time*1000:.2f}ms")
    print(f"   XAttention (optimized) time: {xattn_opt_time*1000:.2f}ms")
    
    speedup_orig = base_time / xattn_orig_time
    speedup_opt = base_time / xattn_opt_time
    optimization_gain = xattn_orig_time / xattn_opt_time
    
    print(f"   Speedup (original): {speedup_orig:.2f}x")
    print(f"   Speedup (optimized): {speedup_opt:.2f}x")
    print(f"   Optimization gain: {optimization_gain:.2f}x")
    
    # Test with different sequence lengths
    print("\n5. Scalability Test with Different Sequence Lengths")
    test_lengths = [64, 128, 256, 512]
    
    for seq_len in test_lengths:
        x_test = torch.randn(batch_size, seq_len, hidden_size).to(device)
        
        # Time BaseAttention
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(5):
            _ = base_attn(x_test, x_test, x_test, causal=True)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        base_time_test = (time.time() - start_time) / 5
        
        # Time Optimized XAttention
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(5):
            _, block_masks = xattn_optimized(x_test, x_test, x_test, causal=True)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        xattn_time_test = (time.time() - start_time) / 5
        
        speedup_test = base_time_test / xattn_time_test
        
        print(f"   Seq_len={seq_len:3d}: Base={base_time_test*1000:.1f}ms, "
              f"XAttn={xattn_time_test*1000:.1f}ms, Speedup={speedup_test:.2f}x")
    
    # Test accuracy comparison
    print("\n6. Accuracy Comparison (Output Similarity)")
    with torch.no_grad():
        # Compare outputs
        diff_orig_opt = torch.abs(xattn_output_orig - xattn_output_opt).mean()
        diff_base_opt = torch.abs(base_output - xattn_output_opt).mean()
        diff_base_orig = torch.abs(base_output - xattn_output_orig).mean()
        
        print(f"   |Base - Original|: {diff_base_orig:.6f}")
        print(f"   |Base - Optimized|: {diff_base_opt:.6f}")
        print(f"   |Original - Optimized|: {diff_orig_opt:.6f}")
        
        # Cosine similarity
        cos_sim_orig_opt = F.cosine_similarity(
            xattn_output_orig.view(-1), xattn_output_opt.view(-1), dim=0
        )
        cos_sim_base_opt = F.cosine_similarity(
            base_output.view(-1), xattn_output_opt.view(-1), dim=0
        )
        
        print(f"   Cosine similarity (Original vs Optimized): {cos_sim_orig_opt:.6f}")
        print(f"   Cosine similarity (Base vs Optimized): {cos_sim_base_opt:.6f}")
    
    print("\n" + "="*60)
    print("Comparison completed!")
    print("="*60)

if __name__ == "__main__":
    compare_implementations()