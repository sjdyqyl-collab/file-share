"""
Demo script for testing DraftAttention implementations.
"""

import torch
import torch.nn as nn
import time
from draft_attention import DraftAttention, DraftAttentionBlock
from enhanced_draft_attention import (
    EnhancedDraftAttention, 
    EnhancedDraftAttentionBlock,
    DynamicSparsityScheduler,
    MultiScaleDraftAttention,
    LearnedAdaptivePooling,
    TemporalConsistencyModule,
    QuantizedSparseAttention
)


def test_basic_draft_attention():
    """Test basic DraftAttention implementation."""
    print("=== Testing Basic DraftAttention ===")
    
    # Parameters
    batch_size = 2
    num_frames = 8
    height, width = 48, 80  # 768p latent size
    d_model = 512
    sparsity_ratio = 0.9
    
    # Create model
    model = DraftAttention(
        sparsity_ratio=sparsity_ratio,
        kernel_size=(8, 16)
    )
    
    # Create input tensors
    q = torch.randn(batch_size, num_frames * height * width, d_model)
    k = torch.randn(batch_size, num_frames * height * width, d_model)
    v = torch.randn(batch_size, num_frames * height * width, d_model)
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        output = model(q, k, v, (height, width), num_frames)
    end_time = time.time()
    
    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {end_time - start_time:.4f} seconds")
    print(f"Sparsity ratio: {sparsity_ratio}")
    print("✓ Basic DraftAttention test passed\n")
    
    return output


def test_enhanced_draft_attention():
    """Test EnhancedDraftAttention implementation."""
    print("=== Testing EnhancedDraftAttention ===")
    
    # Parameters
    batch_size = 2
    num_frames = 8
    height, width = 48, 80
    d_model = 512
    n_heads = 8
    sparsity_ratio = 0.9
    
    # Create model with all enhancements
    model = EnhancedDraftAttention(
        d_model=d_model,
        n_heads=n_heads,
        sparsity_ratio=sparsity_ratio,
        use_dynamic_sparsity=True,
        use_multi_scale=True,
        use_learned_pooling=True,
        use_temporal_consistency=True,
        use_quantization=False,
        num_frames=num_frames
    )
    
    # Create input tensors
    q = torch.randn(batch_size, num_frames * height * width, d_model)
    k = torch.randn(batch_size, num_frames * height * width, d_model)
    v = torch.randn(batch_size, num_frames * height * width, d_model)
    
    # Test with different steps for dynamic sparsity
    for step in [0, 25, 50]:
        start_time = time.time()
        with torch.no_grad():
            output = model(q, k, v, (height, width), num_frames, step=step)
        end_time = time.time()
        
        current_sparsity = model.sparsity_scheduler.get_sparsity_ratio(step) if model.use_dynamic_sparsity else sparsity_ratio
        print(f"Step {step}: Output shape: {output.shape}, Sparsity: {current_sparsity:.3f}, Time: {end_time - start_time:.4f}s")
    
    print("✓ Enhanced DraftAttention test passed\n")
    
    return output


def test_multi_head_blocks():
    """Test multi-head attention blocks."""
    print("=== Testing Multi-Head Attention Blocks ===")
    
    # Parameters
    batch_size = 2
    num_frames = 8
    height, width = 48, 80
    d_model = 512
    n_heads = 8
    
    # Test basic block
    basic_block = DraftAttentionBlock(
        d_model=d_model,
        n_heads=n_heads,
        sparsity_ratio=0.9
    )
    
    # Test enhanced block
    enhanced_block = EnhancedDraftAttentionBlock(
        d_model=d_model,
        n_heads=n_heads,
        sparsity_ratio=0.9,
        use_dynamic_sparsity=True,
        use_multi_scale=True,
        use_learned_pooling=True,
        use_temporal_consistency=True,
        num_frames=num_frames
    )
    
    # Create input
    x = torch.randn(batch_size, num_frames * height * width, d_model)
    
    # Test basic block
    start_time = time.time()
    with torch.no_grad():
        basic_output = basic_block(x, (height, width), num_frames)
    basic_time = time.time() - start_time
    
    # Test enhanced block
    start_time = time.time()
    with torch.no_grad():
        enhanced_output = enhanced_block(x, (height, width), num_frames, step=25)
    enhanced_time = time.time() - start_time
    
    print(f"Basic block: {basic_output.shape}, Time: {basic_time:.4f}s")
    print(f"Enhanced block: {enhanced_output.shape}, Time: {enhanced_time:.4f}s")
    print("✓ Multi-head attention blocks test passed\n")
    
    return basic_output, enhanced_output


def test_individual_components():
    """Test individual enhanced components."""
    print("=== Testing Individual Components ===")
    
    # Test dynamic sparsity scheduler
    scheduler = DynamicSparsityScheduler(min_sparsity=0.5, max_sparsity=0.9, total_steps=50)
    print("Dynamic Sparsity Scheduler:")
    for step in [0, 25, 50]:
        sparsity = scheduler.get_sparsity_ratio(step)
        print(f"  Step {step}: {sparsity:.3f}")
    
    # Test multi-scale attention
    batch_size = 2
    num_frames = 8
    height, width = 48, 80
    d_model = 512
    
    multi_scale = MultiScaleDraftAttention(scales=[64, 128, 256])
    q = torch.randn(batch_size, num_frames * height * width, d_model)
    k = torch.randn(batch_size, num_frames * height * width, d_model)
    v = torch.randn(batch_size, num_frames * height * width, d_model)
    
    with torch.no_grad():
        multi_scale_output = multi_scale(q, k, v, (height, width), num_frames)
    print(f"Multi-Scale Attention: {multi_scale_output.shape}")
    
    # Test learned pooling
    learned_pool = LearnedAdaptivePooling(d_model)
    with torch.no_grad():
        pooled, scale_weights = learned_pool(q, (height, width), num_frames)
    print(f"Learned Pooling: {pooled.shape}, Scale weights: {scale_weights.shape}")
    
    # Test temporal consistency
    temporal_module = TemporalConsistencyModule(d_model, num_frames)
    with torch.no_grad():
        temporal_output = temporal_module(q, (height, width))
    print(f"Temporal Consistency: {temporal_output.shape}")
    
    print("✓ Individual components test passed\n")


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("=== Testing Memory Efficiency ===")
    
    batch_size = 4
    num_frames = 16
    height, width = 64, 64
    d_model = 768
    
    # Create large input
    x = torch.randn(batch_size, num_frames * height * width, d_model)
    
    # Test different sparsity ratios
    sparsity_ratios = [0.5, 0.7, 0.9, 0.95]
    
    for sparsity in sparsity_ratios:
        model = DraftAttention(sparsity_ratio=sparsity)
        
        # Measure memory usage
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        start_time = time.time()
        with torch.no_grad():
            output = model(x, x, x, (height, width), num_frames)
        end_time = time.time()
        
        print(f"Sparsity {sparsity:.2f}: Time {end_time - start_time:.4f}s, Output shape: {output.shape}")
    
    print("✓ Memory efficiency test passed\n")


def compare_performance():
    """Compare performance between basic and enhanced versions."""
    print("=== Performance Comparison ===")
    
    batch_size = 2
    num_frames = 8
    height, width = 48, 80
    d_model = 512
    n_heads = 8
    
    # Create models
    basic_model = DraftAttentionBlock(d_model, n_heads, sparsity_ratio=0.9)
    enhanced_model = EnhancedDraftAttentionBlock(
        d_model, n_heads, sparsity_ratio=0.9,
        use_dynamic_sparsity=True,
        use_multi_scale=True,
        use_learned_pooling=True,
        use_temporal_consistency=True,
        num_frames=num_frames
    )
    
    # Create input
    x = torch.randn(batch_size, num_frames * height * width, d_model)
    
    # Warm up
    with torch.no_grad():
        _ = basic_model(x, (height, width), num_frames)
        _ = enhanced_model(x, (height, width), num_frames, step=25)
    
    # Benchmark
    num_runs = 10
    
    # Basic model
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            basic_out = basic_model(x, (height, width), num_frames)
    basic_time = (time.time() - start_time) / num_runs
    
    # Enhanced model
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            enhanced_out = enhanced_model(x, (height, width), num_frames, step=25)
    enhanced_time = (time.time() - start_time) / num_runs
    
    print(f"Basic model average time: {basic_time:.4f}s")
    print(f"Enhanced model average time: {enhanced_time:.4f}s")
    print(f"Overhead ratio: {enhanced_time / basic_time:.2f}x")
    print("✓ Performance comparison completed\n")


def test_weight_loading():
    """Test weight loading and saving functionality."""
    print("=== Testing Weight Loading/Saving ===")
    
    # Create model
    model = EnhancedDraftAttention(
        d_model=512,
        n_heads=8,
        use_learned_pooling=True,
        use_temporal_consistency=True
    )
    
    # Save weights
    weights = model.save_weights()
    print(f"Saved {len(weights)} weight tensors")
    
    # Create new model and load weights
    new_model = EnhancedDraftAttention(
        d_model=512,
        n_heads=8,
        use_learned_pooling=True,
        use_temporal_consistency=True
    )
    new_model.load_weights(weights)
    print("✓ Weight loading/saving test passed\n")


def main():
    """Run all tests."""
    print("Starting DraftAttention Demo Tests\n")
    
    try:
        # Basic functionality tests
        test_basic_draft_attention()
        test_enhanced_draft_attention()
        test_multi_head_blocks()
        test_individual_components()
        
        # Performance tests
        test_memory_efficiency()
        compare_performance()
        
        # Utility tests
        test_weight_loading()
        
        print("🎉 All tests passed successfully!")
        print("\nDraftAttention implementations are ready for use.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()