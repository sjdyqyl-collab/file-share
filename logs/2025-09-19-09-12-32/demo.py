"""
Demo script to test DraftAttention and AdaptiveDraftAttention implementations.
"""

import torch
import numpy as np
from draft_attention import DraftAttention, DraftAttentionConfig
from improved_draft_attention import AdaptiveDraftAttention, AdaptiveDraftAttentionConfig


def test_basic_draft_attention():
    """Test basic DraftAttention functionality."""
    print("=== Testing Basic DraftAttention ===")
    
    # Configuration
    config = DraftAttentionConfig(
        sparsity_ratio=0.75,
        pooling_kernel=(8, 16),
        fallback_steps=3
    )
    
    # Initialize model
    model = DraftAttention(**config.to_dict())
    
    # Create test input
    B, n_heads, seq_len, d_head = 2, 8, 1024, 64
    query = torch.randn(B, n_heads, seq_len, d_head)
    key = torch.randn(B, n_heads, seq_len, d_head)
    value = torch.randn(B, n_heads, seq_len, d_head)
    
    # Test forward pass
    print(f"Input shape: {query.shape}")
    
    # Test with fallback (dense attention)
    result_fallback = model(query, key, value, step=1)
    print(f"Fallback output shape: {result_fallback['output'].shape}")
    
    # Test with sparse attention
    result_sparse = model(query, key, value, step=5)
    print(f"Sparse output shape: {result_sparse['output'].shape}")
    
    # Test weight saving/loading
    model.save_weights("/tmp/draft_attention_test.pt")
    model.load_weights("/tmp/draft_attention_test.pt")
    
    print("✓ Basic DraftAttention tests passed\n")


def test_adaptive_draft_attention():
    """Test AdaptiveDraftAttention with advanced features."""
    print("=== Testing AdaptiveDraftAttention ===")
    
    # Configuration
    config = AdaptiveDraftAttentionConfig(
        base_sparsity=0.75,
        min_sparsity=0.5,
        max_sparsity=0.95,
        adaptive_kernels=[(4, 8), (8, 16), (16, 32)],
        use_quantization=True,
        quantization_bits=8
    )
    
    # Initialize model
    model = AdaptiveDraftAttention(**config.to_dict())
    
    # Create test input
    B, n_heads, seq_len, d_head = 2, 8, 2048, 64
    query = torch.randn(B, n_heads, seq_len, d_head)
    key = torch.randn(B, n_heads, seq_len, d_head)
    value = torch.randn(B, n_heads, seq_len, d_head)
    
    print(f"Input shape: {query.shape}")
    
    # Test forward pass with metadata
    result = model(query, key, value, step=25, total_steps=50, return_attention=True)
    print(f"Output shape: {result['output'].shape}")
    print(f"Metadata: {result['metadata']}")
    
    # Test efficiency stats
    stats = model.get_efficiency_stats()
    print(f"Efficiency stats: {stats}")
    
    print("✓ AdaptiveDraftAttention tests passed\n")


def test_error_bounds():
    """Test theoretical error bounds."""
    print("=== Testing Error Bounds ===")
    
    # Create reference and draft attention
    B, n_heads, seq_len, d_head = 1, 1, 256, 64
    
    # Dense attention
    query = torch.randn(B, n_heads, seq_len, d_head)
    key = torch.randn(B, n_heads, seq_len, d_head)
    value = torch.randn(B, n_heads, seq_len, d_head)
    
    scale = 1.0 / np.sqrt(d_head)
    dense_attention = torch.matmul(query, key.transpose(-2, -1)) * scale
    dense_attention = F.softmax(dense_attention, dim=-1)
    
    # Draft attention
    model = DraftAttention(sparsity_ratio=0.75)
    draft_attention, _ = model._compute_draft_attention(query, key, value)
    
    # Compute error bounds
    # Upsample draft attention for comparison
    g = draft_attention.shape[-1]
    upsampled_draft = F.interpolate(
        draft_attention.view(B * n_heads, g, g, 1).permute(0, 3, 1, 2),
        size=(seq_len, seq_len),
        mode='nearest'
    ).permute(0, 2, 3, 1).view(B, n_heads, seq_len, seq_len)
    
    # Frobenius norm error
    error = torch.norm(dense_attention - upsampled_draft, p='fro')
    relative_error = error / torch.norm(dense_attention, p='fro')
    
    print(f"Frobenius norm error: {error.item():.4f}")
    print(f"Relative error: {relative_error.item():.4f}")
    print("✓ Error bounds test completed\n")


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("=== Testing Memory Efficiency ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = "cuda"
    
    # Large input to test memory savings
    B, n_heads, seq_len, d_head = 1, 8, 4096, 64
    
    # Create inputs on GPU
    query = torch.randn(B, n_heads, seq_len, d_head, device=device)
    key = torch.randn(B, n_heads, seq_len, d_head, device=device)
    value = torch.randn(B, n_heads, seq_len, d_head, device=device)
    
    # Test dense attention memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    scale = 1.0 / np.sqrt(d_head)
    dense_attention = torch.matmul(query, key.transpose(-2, -1)) * scale
    dense_attention = F.softmax(dense_attention, dim=-1)
    dense_output = torch.matmul(dense_attention, value)
    
    dense_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Test draft attention memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = AdaptiveDraftAttention(
        base_sparsity=0.75,
        use_quantization=True,
        quantization_bits=8
    ).to(device)
    
    with torch.no_grad():
        draft_result = model(query, key, value, step=25, total_steps=50)
    
    draft_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"Dense attention memory: {dense_memory:.2f} MB")
    print(f"Draft attention memory: {draft_memory:.2f} MB")
    print(f"Memory reduction: {((dense_memory - draft_memory) / dense_memory * 100):.1f}%")
    print("✓ Memory efficiency test completed\n")


def test_distributed_setup():
    """Test distributed setup (simulation)."""
    print("=== Testing Distributed Setup ===")
    
    # Simulate distributed setup
    config = AdaptiveDraftAttentionConfig(
        distributed=True,
        world_size=4,
        rank=0
    )
    
    model = AdaptiveDraftAttention(**config.to_dict())
    stats = model.get_efficiency_stats()
    
    print(f"Distributed config: {stats}")
    print("✓ Distributed setup test completed\n")


if __name__ == "__main__":
    print("Starting DraftAttention Demo...\n")
    
    try:
        test_basic_draft_attention()
        test_adaptive_draft_attention()
        test_error_bounds()
        test_memory_efficiency()
        test_distributed_setup()
        
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()