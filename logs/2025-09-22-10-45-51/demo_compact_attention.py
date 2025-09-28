"""
Demo script for Compact Attention implementations.

Tests both the original CompactAttention and AdaptiveCompactAttention methods.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple
import sys
import os

# Add the current directory to Python path
sys.path.append('/home/wzc/data/file-share/logs/2025-09-22-10-45-51')

from compact_attention import CompactAttention, CompactAttentionConfig
from adaptive_compact_attention import AdaptiveCompactAttention, AdaptiveCompactAttentionConfig


def create_dummy_video_data(
    batch_size: int = 2,
    num_frames: int = 16,
    height: int = 64,
    width: int = 64,
    dim: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy video data for testing.
    
    Args:
        batch_size: Batch size
        num_frames: Number of frames
        height: Frame height
        width: Frame width
        dim: Feature dimension
        
    Returns:
        Tuple of (input_features, video_frames)
    """
    total_tokens = num_frames * height * width
    
    # Input features for attention
    input_features = torch.randn(batch_size, total_tokens, dim)
    
    # Video frames for content analysis
    video_frames = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    return input_features, video_frames


def test_compact_attention():
    """Test the original CompactAttention implementation."""
    print("=" * 60)
    print("Testing CompactAttention (Original Method)")
    print("=" * 60)
    
    # Configuration
    config = CompactAttentionConfig(model_type="hunyuan")
    
    # Create model
    model = CompactAttention(
        dim=config.dim,
        num_heads=config.num_heads,
        frame_size=config.frame_size,
        num_frames=config.num_frames,
        tile_size=config.tile_size,
        tau=config.tau,
        lambda_cost=0.04,
        device=config.device
    )
    
    # Create dummy data
    input_features, video_frames = create_dummy_video_data(
        batch_size=2,
        num_frames=config.num_frames,
        height=config.frame_size[0],
        width=config.frame_size[1],
        dim=config.dim
    )
    
    print(f"Model configuration:")
    print(f"  - Hidden dim: {config.dim}")
    print(f"  - Num heads: {config.num_heads}")
    print(f"  - Frame size: {config.frame_size}")
    print(f"  - Num frames: {config.num_frames}")
    print(f"  - Total tokens: {config.num_frames * config.frame_size[0] * config.frame_size[1]}")
    print(f"  - Device: {config.device}")
    
    print(f"\nInput shapes:")
    print(f"  - Input features: {input_features.shape}")
    print(f"  - Video frames: {video_frames.shape}")
    
    # Move to device
    device = torch.device(config.device)
    model = model.to(device)
    input_features = input_features.to(device)
    
    # Test offline auto-search
    print("\nRunning offline auto-search...")
    start_time = time.time()
    configs = model.offline_auto_search(input_features, num_samples=5, verbose=True)
    search_time = time.time() - start_time
    print(f"Auto-search completed in {search_time:.2f} seconds")
    
    # Print pattern configurations
    print("\nPattern configurations:")
    for pattern, config_data in configs.items():
        print(f"  {pattern}:")
        print(f"    Sparsity ratio: {config_data['sparsity_ratio']:.3f}")
        print(f"    Recall score: {config_data['recall_score']:.3f}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(input_features)
        forward_time = time.time() - start_time
        
    print(f"Forward pass completed in {forward_time:.4f} seconds")
    print(f"Output shape: {output.shape}")
    
    # Test with different timesteps
    print("\nTesting with timestep conditioning...")
    timesteps = torch.tensor([100, 500, 900], device=device)
    for t in timesteps:
        with torch.no_grad():
            output_t = model(input_features[:1], timestep=t.unsqueeze(0))
        print(f"  Timestep {t.item()}: output shape {output_t.shape}")
    
    return model, configs


def test_adaptive_compact_attention():
    """Test the AdaptiveCompactAttention implementation."""
    print("\n" + "=" * 60)
    print("Testing AdaptiveCompactAttention (Enhanced Method)")
    print("=" * 60)
    
    # Configuration
    config = AdaptiveCompactAttentionConfig(
        model_type="hunyuan",
        adaptive_tau=True,
        adaptive_lambda=True,
        content_adaptive=True,
        learned_patterns=True,
        multi_gpu=False
    )
    
    # Create model
    model = config.create_model()
    
    # Create dummy data
    input_features, video_frames = create_dummy_video_data(
        batch_size=2,
        num_frames=config.num_frames,
        height=config.frame_size[0],
        width=config.frame_size[1],
        dim=config.dim
    )
    
    print(f"Model configuration:")
    print(f"  - Hidden dim: {config.dim}")
    print(f"  - Num heads: {config.num_heads}")
    print(f"  - Frame size: {config.frame_size}")
    print(f"  - Num frames: {config.num_frames}")
    print(f"  - Adaptive tau: {config.adaptive_tau}")
    print(f"  - Adaptive lambda: {config.adaptive_lambda}")
    print(f"  - Content adaptive: {config.content_adaptive}")
    print(f"  - Learned patterns: {config.learned_patterns}")
    print(f"  - Multi-GPU: {config.multi_gpu}")
    
    print(f"\nInput shapes:")
    print(f"  - Input features: {input_features.shape}")
    print(f"  - Video frames: {video_frames.shape}")
    
    # Move to device
    device = torch.device(config.device)
    model = model.to(device)
    input_features = input_features.to(device)
    video_frames = video_frames.to(device)
    
    # Test content complexity analysis
    print("\nAnalyzing content complexity...")
    complexity_levels = model.analyze_content_complexity(video_frames)
    print(f"Content complexity levels: {complexity_levels}")
    
    # Test adaptive threshold scheduling
    print("\nTesting adaptive threshold scheduling...")
    test_timesteps = [100, 500, 900]
    for t in test_timesteps:
        tau, lambda_cost = model.get_adaptive_thresholds(t)
        print(f"  Timestep {t}: tau={tau:.3f}, lambda={lambda_cost:.3f}")
    
    # Test offline auto-search with content adaptation
    print("\nRunning enhanced offline auto-search...")
    start_time = time.time()
    configs = model.offline_auto_search(
        input_features,
        video_frames=video_frames,
        num_samples=5,
        verbose=True
    )
    search_time = time.time() - start_time
    print(f"Enhanced auto-search completed in {search_time:.2f} seconds")
    
    # Print enhanced pattern configurations
    print("\nEnhanced pattern configurations:")
    for pattern, config_data in configs.items():
        print(f"  {pattern}:")
        print(f"    Sparsity ratio: {config_data['sparsity_ratio']:.3f}")
        print(f"    Recall score: {config_data['recall_score']:.3f}")
    
    # Test forward pass with adaptive features
    print("\nTesting enhanced forward pass...")
    model.eval()
    with torch.no_grad():
        # Test with different timesteps
        for t in [100, 500, 900]:
            timestep = torch.tensor([t], device=device)
            start_time = time.time()
            output = model(
                input_features[:1],
                timestep=timestep,
                video_frames=video_frames[:1]
            )
            forward_time = time.time() - start_time
            print(f"  Timestep {t}: forward time {forward_time:.4f}s, output shape {output.shape}")
    
    # Test pattern complexity analysis
    print("\nPattern complexity analysis:")
    complexity = model.get_pattern_complexity()
    for key, value in complexity.items():
        print(f"  {key}: {value:.3f}")
    
    return model, configs


def compare_performance():
    """Compare performance between original and adaptive methods."""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    # Create test data
    config = CompactAttentionConfig(model_type="hunyuan")
    input_features, video_frames = create_dummy_video_data(
        batch_size=1,
        num_frames=config.num_frames,
        height=config.frame_size[0],
        width=config.frame_size[1],
        dim=config.dim
    )
    
    device = torch.device(config.device)
    input_features = input_features.to(device)
    video_frames = video_frames.to(device)
    
    # Test original method
    print("Testing original CompactAttention...")
    original_model = CompactAttention(
        dim=config.dim,
        num_heads=config.num_heads,
        frame_size=config.frame_size,
        num_frames=config.num_frames,
        tile_size=config.tile_size,
        device=config.device
    ).to(device)
    
    # Run auto-search for original
    original_model.offline_auto_search(input_features, num_samples=3, verbose=False)
    
    # Test adaptive method
    print("Testing AdaptiveCompactAttention...")
    adaptive_config = AdaptiveCompactAttentionConfig(
        model_type="hunyuan",
        adaptive_tau=True,
        adaptive_lambda=True,
        content_adaptive=True,
        learned_patterns=True
    )
    adaptive_model = adaptive_config.create_model().to(device)
    
    # Run enhanced auto-search
    adaptive_model.offline_auto_search(
        input_features,
        video_frames=video_frames,
        num_samples=3,
        verbose=False
    )
    
    # Benchmark forward passes
    num_runs = 10
    original_times = []
    adaptive_times = []
    
    print(f"\nBenchmarking {num_runs} forward passes...")
    
    with torch.no_grad():
        # Warm up
        _ = original_model(input_features)
        _ = adaptive_model(input_features, video_frames=video_frames)
        
        # Benchmark original
        for i in range(num_runs):
            start = time.time()
            _ = original_model(input_features)
            torch.cuda.synchronize()
            original_times.append(time.time() - start)
        
        # Benchmark adaptive
        for i in range(num_runs):
            start = time.time()
            _ = adaptive_model(input_features, video_frames=video_frames)
            torch.cuda.synchronize()
            adaptive_times.append(time.time() - start)
    
    # Calculate statistics
    original_mean = np.mean(original_times)
    adaptive_mean = np.mean(adaptive_times)
    
    print(f"\nPerformance Results:")
    print(f"  Original CompactAttention: {original_mean*1000:.2f}ms ± {np.std(original_times)*1000:.2f}ms")
    print(f"  AdaptiveCompactAttention: {adaptive_mean*1000:.2f}ms ± {np.std(adaptive_times)*1000:.2f}ms")
    print(f"  Overhead: {((adaptive_mean - original_mean) / original_mean * 100):.1f}%")
    
    # Compare sparsity
    original_sparsity = original_model.get_pattern_complexity()
    adaptive_sparsity = adaptive_model.get_pattern_complexity()
    
    print(f"\nSparsity Comparison:")
    for pattern in ['local_sparsity', 'cross_sparsity', 'global_sparsity']:
        if pattern in original_sparsity and pattern in adaptive_sparsity:
            orig_val = original_sparsity[pattern]
            adapt_val = adaptive_sparsity[pattern]
            print(f"  {pattern}: Original={orig_val:.3f}, Adaptive={adapt_val:.3f}")


def main():
    """Main demo function."""
    print("Compact Attention Demo")
    print("This demo tests both the original and enhanced Compact Attention implementations.")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("WARNING: CUDA not available, using CPU")
    
    try:
        # Test original method
        original_model, original_configs = test_compact_attention()
        
        # Test adaptive method
        adaptive_model, adaptive_configs = test_adaptive_compact_attention()
        
        # Performance comparison
        compare_performance()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
        # Save models for further testing
        print("\nSaving models...")
        torch.save(original_model.state_dict(), '/home/wzc/data/file-share/logs/2025-09-22-10-45-51/original_compact_attention.pth')
        torch.save(adaptive_model.state_dict(), '/home/wzc/data/file-share/logs/2025-09-22-10-45-51/adaptive_compact_attention.pth')
        print("Models saved successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()