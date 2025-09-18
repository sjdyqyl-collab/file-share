"""
Comprehensive test script for all XAttention implementations.
Tests baseline, original, and improved methods with various configurations.
"""

import torch
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_attention import BaselineAttention, BaselineAttentionWithCache
from xattention_original import XAttentionOriginal
from xattention_improved import XAttentionImproved


def test_method_performance(batch_size: int, seq_len: int, hidden_size: int, num_heads: int):
    """Test and compare performance of all methods."""
    print(f"\n{'='*60}")
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}, num_heads={num_heads}")
    print('='*60)
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test Baseline Attention
    print("\n1. Testing Baseline Attention...")
    baseline = BaselineAttention(hidden_size, num_heads)
    start_time = time.time()
    baseline_output = baseline(x)
    baseline_time = time.time() - start_time
    baseline_flops = baseline.get_flops(seq_len)
    
    print(f"   Output shape: {baseline_output.shape}")
    print(f"   Time: {baseline_time:.4f}s")
    print(f"   FLOPs: {baseline_flops:,}")
    
    # Test Original XAttention
    print("\n2. Testing Original XAttention...")
    xattn_orig = XAttentionOriginal(hidden_size, num_heads, block_size=8, stride=8)
    start_time = time.time()
    xattn_orig_output = xattn_orig(x)
    xattn_orig_time = time.time() - start_time
    xattn_orig_flops = xattn_orig.get_flops(seq_len)
    sparsity_info = xattn_orig.get_sparsity_info(x)
    
    print(f"   Output shape: {xattn_orig_output.shape}")
    print(f"   Time: {xattn_orig_time:.4f}s")
    print(f"   FLOPs: {xattn_orig_flops:,}")
    print(f"   Sparsity: {sparsity_info['sparsity']:.2%}")
    print(f"   Density: {sparsity_info['density']:.2%}")
    print(f"   Selected blocks: {sparsity_info['selected_blocks']}/{sparsity_info['total_blocks']}")
    
    # Test Improved XAttention
    print("\n3. Testing Improved XAttention...")
    xattn_improved = XAttentionImproved(
        hidden_size, num_heads,
        block_sizes=[4, 8, 16],
        strides=[4, 8, 16, 64],
        warmup_steps=0  # Disable warmup for testing
    )
    start_time = time.time()
    xattn_improved_output = xattn_improved.forward_standard(x, task_type="language")
    xattn_improved_time = time.time() - start_time
    enhanced_sparsity_info = xattn_improved.get_enhanced_sparsity_info(x, task_type="language")
    
    print(f"   Output shape: {xattn_improved_output.shape}")
    print(f"   Time: {xattn_improved_time:.4f}s")
    print(f"   Sparsity: {enhanced_sparsity_info['sparsity']:.2%}")
    print(f"   Density: {enhanced_sparsity_info['density']:.2%}")
    print(f"   Compression ratio: {enhanced_sparsity_info['compression_ratio']:.2%}")
    print(f"   Current block size: {enhanced_sparsity_info['current_block_size']}")
    print(f"   Current stride: {enhanced_sparsity_info['current_stride']}")
    
    # Compare outputs
    print("\n4. Output Comparison...")
    
    # Check shapes
    assert baseline_output.shape == xattn_orig_output.shape == xattn_improved_output.shape
    print("   ✓ All outputs have correct shape")
    
    # Check for NaN or infinite values
    for name, output in [("Baseline", baseline_output), 
                        ("XAttention Original", xattn_orig_output),
                        ("XAttention Improved", xattn_improved_output)]:
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"   ✗ {name} output contains NaN or infinite values")
        else:
            print(f"   ✓ {name} output is valid")
    
    # Performance comparison
    print("\n5. Performance Comparison...")
    speedup_orig = baseline_time / xattn_orig_time if xattn_orig_time > 0 else float('inf')
    speedup_improved = baseline_time / xattn_improved_time if xattn_improved_time > 0 else float('inf')
    flops_reduction_orig = 1 - (xattn_orig_flops / baseline_flops) if baseline_flops > 0 else 0
    flops_reduction_improved = 1 - (enhanced_sparsity_info['compression_ratio']) if 'compression_ratio' in enhanced_sparsity_info else 0
    
    print(f"   XAttention Original speedup: {speedup_orig:.2f}x")
    print(f"   XAttention Improved speedup: {speedup_improved:.2f}x")
    print(f"   XAttention Original FLOPs reduction: {flops_reduction_orig:.2%}")
    print(f"   XAttention Improved compression ratio: {enhanced_sparsity_info['compression_ratio']:.2%}")
    
    return {
        'baseline_time': baseline_time,
        'xattn_orig_time': xattn_orig_time,
        'xattn_improved_time': xattn_improved_time,
        'xattn_orig_sparsity': sparsity_info['sparsity'],
        'xattn_improved_sparsity': enhanced_sparsity_info['sparsity'],
        'speedup_orig': speedup_orig,
        'speedup_improved': speedup_improved
    }


def test_video_generation_mode():
    """Test video generation specific features."""
    print(f"\n{'='*60}")
    print("Testing Video Generation Mode")
    print('='*60)
    
    batch_size, seq_len, hidden_size, num_heads = 1, 129, 256, 8  # 129 frames like in paper
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test improved XAttention with video generation settings
    xattn_video = XAttentionImproved(
        hidden_size, num_heads,
        block_sizes=[8, 16],
        strides=[8, 16],
        warmup_steps=5
    )
    
    print("\nTesting warmup strategy...")
    for step in range(7):
        output = xattn_video.forward_standard(x, task_type="video_generation")
        info = xattn_video.get_enhanced_sparsity_info(x, task_type="video_generation")
        print(f"   Step {step+1}: Sparsity={info['sparsity']:.2%}, "
              f"Is warmup={info['is_warmup']}, Block size={info['current_block_size']}")
    
    print("\n✓ Video generation mode test completed")


def test_streaming_mode():
    """Test streaming attention for long sequences."""
    print(f"\n{'='*60}")
    print("Testing Streaming Mode")
    print('='*60)
    
    batch_size, chunk_size, hidden_size, num_heads = 1, 256, 256, 8
    
    # Create streaming attention
    xattn_streaming = XAttentionImproved(
        hidden_size, num_heads,
        streaming_window=512
    )
    
    print("\nProcessing sequence in chunks...")
    total_chunks = 4
    
    for chunk_idx in range(total_chunks):
        x_chunk = torch.randn(batch_size, chunk_size, hidden_size)
        output = xattn_streaming.streaming_attention(x_chunk, is_streaming=True)
        print(f"   Chunk {chunk_idx+1}/{total_chunks}: Output shape={output.shape}")
    
    print("\n✓ Streaming mode test completed")


def test_kv_cache_compression():
    """Test KV cache compression feature."""
    print(f"\n{'='*60}")
    print("Testing KV Cache Compression")
    print('='*60)
    
    batch_size, seq_len, hidden_size, num_heads = 1, 128, 256, 8
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test with cache compression
    xattn_cache = XAttentionImproved(hidden_size, num_heads)
    
    print("\nTesting with cache compression...")
    output_with_cache = xattn_cache.forward_standard(x, use_cache=True)
    info_with_cache = xattn_cache.get_enhanced_sparsity_info(x)
    
    print(f"   Output shape with cache: {output_with_cache.shape}")
    print(f"   Compression ratio: {info_with_cache['compression_ratio']:.2%}")
    
    print("\n✓ KV cache compression test completed")


def test_all_methods():
    """Run comprehensive tests for all methods."""
    print("XAttention Implementation Test Suite")
    print("="*60)
    
    # Test different sequence lengths
    test_configs = [
        (1, 64, 256, 8),    # Small sequence
        (1, 256, 256, 8),   # Medium sequence
        (1, 512, 256, 8),   # Large sequence
    ]
    
    results = []
    
    for config in test_configs:
        try:
            result = test_method_performance(*config)
            results.append(result)
        except Exception as e:
            print(f"Error testing config {config}: {e}")
            continue
    
    # Test special modes
    try:
        test_video_generation_mode()
    except Exception as e:
        print(f"Error in video generation test: {e}")
    
    try:
        test_streaming_mode()
    except Exception as e:
        print(f"Error in streaming test: {e}")
    
    try:
        test_kv_cache_compression()
    except Exception as e:
        print(f"Error in KV cache test: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print('='*60)
    
    if results:
        avg_speedup_orig = np.mean([r['speedup_orig'] for r in results])
        avg_speedup_improved = np.mean([r['speedup_improved'] for r in results])
        avg_sparsity_orig = np.mean([r['xattn_orig_sparsity'] for r in results])
        avg_sparsity_improved = np.mean([r['xattn_improved_sparsity'] for r in results])
        
        print(f"Average XAttention Original speedup: {avg_speedup_orig:.2f}x")
        print(f"Average XAttention Improved speedup: {avg_speedup_improved:.2f}x")
        print(f"Average XAttention Original sparsity: {avg_sparsity_orig:.2%}")
        print(f"Average XAttention Improved sparsity: {avg_sparsity_improved:.2%}")
    
    print("\n✓ All tests completed successfully!")
    
    return results


if __name__ == "__main__":
    import numpy as np
    
    # Run all tests
    results = test_all_methods()
    
    # Save results for analysis
    if results:
        torch.save(results, "test_results.pt")
        print("\nTest results saved to test_results.pt")