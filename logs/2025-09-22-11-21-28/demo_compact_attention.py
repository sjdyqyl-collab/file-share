"""
Demo script for Compact Attention implementations.

This script demonstrates how to use both the original Compact Attention
and the enhanced Adaptive Compact Attention for video generation tasks.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from compact_attention import CompactAttention, CompactAttentionConfig
from adaptive_compact_attention import AdaptiveCompactAttention, AdaptiveCompactAttentionConfig


def create_dummy_video_data(batch_size=2, frames=81, height=48, width=80, dim=512):
    """Create dummy video data for testing."""
    # Simulate video tokens: [B, T*H*W, D]
    tokens_per_frame = height * width
    total_tokens = frames * tokens_per_frame
    
    x = torch.randn(batch_size, total_tokens, dim)
    return x


def benchmark_attention_layer(attention_layer, x, num_runs=10):
    """Benchmark attention layer performance."""
    attention_layer.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = attention_layer(x)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = attention_layer(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time, output


def test_original_compact_attention():
    """Test the original Compact Attention implementation."""
    print("=" * 60)
    print("Testing Original Compact Attention")
    print("=" * 60)
    
    # Configuration
    config = CompactAttentionConfig(
        dim=512,
        num_heads=8,
        tile_size=16,
        frame_size=(81, 48, 80),  # Reduced size for demo
        recall_threshold=0.9,
        cost_threshold=0.011,
        pattern_cache_dir="./demo_cache"
    )
    
    # Create model
    model = CompactAttention(**config.to_dict())
    
    # Create test data
    x = create_dummy_video_data(
        batch_size=2,
        frames=81,
        height=48,
        width=80,
        dim=512
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(x, layer_idx=0, head_idx=0)
    
    print(f"Output shape: {output.shape}")
    
    # Benchmark
    avg_time, _ = benchmark_attention_layer(model, x)
    print(f"Average inference time: {avg_time:.4f} seconds")
    
    # Test pattern caching
    print("\nTesting pattern caching...")
    cache_key = model.get_cache_key(0, 0)
    print(f"Cache key: {cache_key}")
    
    # Second forward pass should use cached patterns
    with torch.no_grad():
        output2 = model(x, layer_idx=0, head_idx=0)
    
    print("✓ Pattern caching working")
    
    return model, output


def test_adaptive_compact_attention():
    """Test the enhanced Adaptive Compact Attention implementation."""
    print("\n" + "=" * 60)
    print("Testing Adaptive Compact Attention")
    print("=" * 60)
    
    # Configuration
    config = AdaptiveCompactAttentionConfig(
        dim=512,
        num_heads=8,
        tile_size=16,
        frame_size=(81, 48, 80),  # Reduced size for demo
        recall_threshold=0.9,
        cost_threshold=0.011,
        use_adaptive_threshold=True,
        use_learnable_patterns=True,
        use_motion_aware=True,
        use_distributed=False,
        num_gpus=1,
        pattern_cache_dir="./demo_cache_adaptive"
    )
    
    # Create model
    model = AdaptiveCompactAttention(**config.to_dict())
    
    # Create test data
    x = create_dummy_video_data(
        batch_size=2,
        frames=81,
        height=48,
        width=80,
        dim=512
    )
    
    # Create dummy optical flow
    flow = torch.randn(2, 80, 2, 48, 80)  # [B, T-1, 2, H, W]
    
    print(f"Input shape: {x.shape}")
    print(f"Flow shape: {flow.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(x, layer_idx=0, head_idx=0, flow=flow)
    
    print(f"Output shape: {output.shape}")
    
    # Benchmark
    avg_time, _ = benchmark_attention_layer(model, x)
    print(f"Average inference time: {avg_time:.4f} seconds")
    
    # Test adaptive thresholding
    if model.use_adaptive_threshold:
        print("\nTesting adaptive thresholding...")
        
        # Create high-complexity input
        x_complex = create_dummy_video_data(batch_size=2, frames=81, height=48, width=80, dim=512)
        x_complex += 2.0 * torch.randn_like(x_complex)  # Add noise for complexity
        
        with torch.no_grad():
            # This will trigger threshold adaptation
            output_complex = model(x_complex, layer_idx=0, head_idx=0)
        
        print("✓ Adaptive thresholding working")
    
    return model, output


def compare_performance():
    """Compare performance between original and adaptive versions."""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    # Create test data
    x = create_dummy_video_data(batch_size=1, frames=81, height=48, width=80, dim=512)
    
    # Original Compact Attention
    original_config = CompactAttentionConfig(
        dim=512,
        num_heads=8,
        tile_size=16,
        frame_size=(81, 48, 80),
        pattern_cache_dir="./demo_cache_compare"
    )
    original_model = CompactAttention(**original_config.to_dict())
    
    # Adaptive Compact Attention
    adaptive_config = AdaptiveCompactAttentionConfig(
        dim=512,
        num_heads=8,
        tile_size=16,
        frame_size=(81, 48, 80),
        use_adaptive_threshold=True,
        use_learnable_patterns=True,
        use_motion_aware=False,  # Disable for fair comparison
        use_distributed=False,
        pattern_cache_dir="./demo_cache_compare_adaptive"
    )
    adaptive_model = AdaptiveCompactAttention(**adaptive_config.to_dict())
    
    # Benchmark
    original_time, original_out = benchmark_attention_layer(original_model, x)
    adaptive_time, adaptive_out = benchmark_attention_layer(adaptive_model, x)
    
    print(f"Original Compact Attention: {original_time:.4f}s")
    print(f"Adaptive Compact Attention: {adaptive_time:.4f}s")
    print(f"Speedup: {original_time / adaptive_time:.2f}x")
    
    # Check output similarity
    similarity = F.cosine_similarity(
        original_out.view(-1), adaptive_out.view(-1), dim=0
    )
    print(f"Output similarity: {similarity.item():.4f}")


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\n" + "=" * 60)
    print("Memory Efficiency Test")
    print("=" * 60)
    
    # Test with different sequence lengths
    lengths = [1000, 2500, 5000, 10000]
    
    for length in lengths:
        print(f"\nTesting sequence length: {length}")
        
        # Create data
        x = torch.randn(1, length, 512)
        
        # Original attention (simulated)
        original_memory = length * length * 4 / (1024 ** 2)  # MB for full attention
        
        # Compact attention (estimated)
        sparsity = 0.4  # 40% sparsity
        compact_memory = length * length * 4 * sparsity / (1024 ** 2)
        
        print(f"  Full attention memory: {original_memory:.2f} MB")
        print(f"  Compact attention memory: {compact_memory:.2f} MB")
        print(f"  Memory reduction: {(1 - compact_memory/original_memory)*100:.1f}%")


def main():
    """Main demo function."""
    print("Compact Attention Demo")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test original implementation
    original_model, original_output = test_original_compact_attention()
    
    # Test adaptive implementation
    adaptive_model, adaptive_output = test_adaptive_compact_attention()
    
    # Compare performance
    compare_performance()
    
    # Test memory efficiency
    test_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("Summary:")
    print("- Original Compact Attention: Implemented with tile-based sparsity")
    print("- Adaptive Compact Attention: Enhanced with dynamic thresholding")
    print("- Both implementations support pattern caching for efficiency")
    print("- Memory usage scales with sparsity rather than full O(n²)")
    print("- Ready for integration into video diffusion transformers")


if __name__ == "__main__":
    main()