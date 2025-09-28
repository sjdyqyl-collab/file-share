#!/usr/bin/env python3
"""
Demo script for testing DraftAttention and AdaptiveDraftAttention implementations.

This script demonstrates:
1. Basic usage of DraftAttention
2. Advanced features of AdaptiveDraftAttention
3. Performance comparison
4. Memory usage analysis
"""

import torch
import time
import psutil
import os
from draft_attention import DraftAttention
from adaptive_draft_attention import AdaptiveDraftAttention


def print_tensor_shapes(tensor_dict):
    """Helper function to print tensor shapes."""
    for name, tensor in tensor_dict.items():
        if tensor is not None:
            print(f"  {name}: {tensor.shape}")
        else:
            print(f"  {name}: None")


def memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def demo_basic_draft_attention():
    """Demonstrate basic DraftAttention functionality."""
    print("=" * 60)
    print("DEMO 1: Basic DraftAttention")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 614400  # 768p video with 128 frames
    hidden_size = 6144
    num_heads = 48
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len:,}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of heads: {num_heads}")
    print()
    
    # Initialize model
    model = DraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        pooling_kernel=(8, 16),
        sparsity_ratio=0.9,  # 90% sparsity
        frame_height=48,
        frame_width=80,
        max_frames=128
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    print("Input tensor shapes:")
    print_tensor_shapes({"x": x})
    print()
    
    # Run inference
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        start_memory = memory_usage()
        
        output, attention_weights = model(x, output_attentions=True)
        
        end_time = time.time()
        end_memory = memory_usage()
        
    print("Output tensor shapes:")
    print_tensor_shapes({
        "output": output,
        "attention_weights": attention_weights
    })
    print()
    
    print("Performance metrics:")
    print(f"  Inference time: {end_time - start_time:.3f}s")
    print(f"  Memory usage: {start_memory:.1f}MB → {end_memory:.1f}MB")
    print(f"  Memory increase: {end_memory - start_memory:.1f}MB")
    print()
    
    print("Sparsity statistics:")
    stats = model.get_sparsity_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


def demo_adaptive_draft_attention():
    """Demonstrate AdaptiveDraftAttention with advanced features."""
    print("=" * 60)
    print("DEMO 2: AdaptiveDraftAttention")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 614400
    hidden_size = 6144
    num_heads = 48
    
    # Initialize adaptive model
    model = AdaptiveDraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        pooling_kernels=[(4, 4), (8, 8), (16, 16)],
        min_sparsity=0.5,
        max_sparsity=0.95,
        use_multi_scale=True,
        use_temporal_gate=True,
        use_quantization=False,  # Set to True for quantization
        frame_height=48,
        frame_width=80,
        max_frames=128
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    print("Input tensor shapes:")
    print_tensor_shapes({"x": x})
    print()
    
    # Run inference
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        start_memory = memory_usage()
        
        output, attention_weights, stats = model(x, output_attentions=True)
        
        end_time = time.time()
        end_memory = memory_usage()
        
    print("Output tensor shapes:")
    print_tensor_shapes({
        "output": output,
        "attention_weights": attention_weights
    })
    print()
    
    print("Performance metrics:")
    print(f"  Inference time: {end_time - start_time:.3f}s")
    print(f"  Memory usage: {start_memory:.1f}MB → {end_memory:.1f}MB")
    print(f"  Memory increase: {end_memory - start_memory:.1f}MB")
    print()
    
    print("Adaptive statistics:")
    for key, value in stats.items():
        if value is not None and hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    print("\nAdaptive features enabled:")
    adaptive_stats = model.get_adaptive_stats()
    for key, value in adaptive_stats.items():
        print(f"  {key}: {value}")
    print()


def demo_memory_comparison():
    """Compare memory usage between dense and sparse attention."""
    print("=" * 60)
    print("DEMO 3: Memory Comparison")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 61440  # Smaller for memory comparison
    hidden_size = 1024
    num_heads = 16
    
    # Dense attention (simulated)
    def dense_attention_memory():
        # Full attention matrix: [B, L, L]
        attention_matrix = torch.randn(batch_size, seq_len, seq_len)
        return attention_matrix.numel() * 4 / 1024 / 1024  # MB
    
    # Sparse attention
    def sparse_attention_memory(sparsity_ratio=0.9):
        # Sparse attention: [B, L, L * (1-sparsity)]
        sparse_elements = seq_len * seq_len * (1 - sparsity_ratio)
        return sparse_elements * 4 / 1024 / 1024  # MB
    
    print("Memory usage comparison (MB):")
    print(f"  Dense attention: {dense_attention_memory():.1f}MB")
    print(f"  50% sparse: {sparse_attention_memory(0.5):.1f}MB")
    print(f"  75% sparse: {sparse_attention_memory(0.75):.1f}MB")
    print(f"  90% sparse: {sparse_attention_memory(0.9):.1f}MB")
    print(f"  95% sparse: {sparse_attention_memory(0.95):.1f}MB")
    print()


def demo_shape_validation():
    """Validate tensor shapes throughout computation."""
    print("=" * 60)
    print("DEMO 4: Shape Validation")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 4800  # Reduced for clarity
    hidden_size = 1024
    num_heads = 16
    
    model = DraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        pooling_kernel=(8, 16),
        sparsity_ratio=0.9,
        frame_height=12,
        frame_width=20,
        max_frames=20
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    print("Shape validation:")
    print(f"  Input: {x.shape}")
    
    # Projections
    q = model.q_proj(x)
    k = model.k_proj(x)
    v = model.v_proj(x)
    print(f"  After projections: {q.shape}, {k.shape}, {v.shape}")
    
    # Multi-head reshape
    q = q.view(batch_size, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)
    print(f"  Multi-head: {q.shape}, {k.shape}, {v.shape}")
    
    # Draft attention
    q_avg = q.mean(dim=1)
    k_avg = k.mean(dim=1)
    print(f"  For draft: {q_avg.shape}, {k_avg.shape}")
    
    # Expected draft size
    reduction_factor = 8 * 16
    draft_size = seq_len // reduction_factor
    print(f"  Expected draft size: {draft_size} ({seq_len}/{reduction_factor})")
    
    # Output
    output, _ = model(x)
    print(f"  Output: {output.shape}")
    print()


if __name__ == "__main__":
    print("DraftAttention Demo")
    print("==================")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    try:
        demo_basic_draft_attention()
        demo_adaptive_draft_attention()
        demo_memory_comparison()
        demo_shape_validation()
        
        print("=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()