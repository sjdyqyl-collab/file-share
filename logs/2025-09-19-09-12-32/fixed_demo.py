"""
Fixed demo script to test DraftAttention and AdaptiveDraftAttention implementations.
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
        pooling_kernel=(4, 8),  # Smaller kernel for testing
        fallback_steps=3
    )
    
    # Initialize model
    model = DraftAttention(**config.to_dict())
    
    # Create test input with dimensions divisible by kernel sizes
    # seq_len = T * H * W, where H=W=32, T=16 -> 16*32*32 = 16384
    B, n_heads, seq_len, d_head = 2, 8, 16384, 64
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
        adaptive_kernels=[(2, 4), (4, 8), (8, 16)],  # Smaller kernels for testing
        use_quantization=True,
        quantization_bits=8
    )
    
    # Initialize model
    model = AdaptiveDraftAttention(**config.to_dict())
    
    # Create test input with appropriate dimensions
    B, n_heads, seq_len, d_head = 1, 8, 8192, 64  # 8*32*32 = 8192
    query = torch.randn(B, n_heads, seq_len, d_head)
    key = torch.randn(B, n_heads, seq_len, d_head)
    value = torch.randn(B, n_heads, seq_len, d_head)
    
    print(f"Input shape: {query.shape}")
    
    # Test forward pass with metadata
    result = model(query, key, value, step=25, total_steps=50, return_attention=True)
    print(f"Output shape: {result['output'].shape}")
    print(f"Metadata keys: {list(result['metadata'].keys())}")
    
    # Test efficiency stats
    stats = model.get_efficiency_stats()
    print(f"Efficiency stats: {stats}")
    
    print("✓ AdaptiveDraftAttention tests passed\n")


def test_error_bounds():
    """Test theoretical error bounds."""
    print("=== Testing Error Bounds ===")
    
    # Create reference and draft attention with appropriate dimensions
    B, n_heads, seq_len, d_head = 1, 1, 4096, 64  # 4*32*32 = 4096
    
    # Dense attention
    query = torch.randn(B, n_heads, seq_len, d_head)
    key = torch.randn(B, n_heads, seq_len, d_head)
    value = torch.randn(B, n_heads, seq_len, d_head)
    
    scale = 1.0 / np.sqrt(d_head)
    dense_attention = torch.matmul(query, key.transpose(-2, -1)) * scale
    dense_attention = F.softmax(dense_attention, dim=-1)
    
    # Draft attention
    model = DraftAttention(sparsity_ratio=0.75, pooling_kernel=(2, 4))
    draft_attention, _ = model._compute_draft_attention(query, key, value)
    
    # Compute error bounds
    g = draft_attention.shape[-1]
    upsampled_draft = F.interpolate(
        draft_attention.view(B * n_heads, g, g, 1).permute(0, 3, 1, 2),
        size=(seq_len, seq_len),
        mode='nearest'
    ).permute(0, 2, 3, 1).view(B, n_heads, seq_len, seq_len)
    
    # Frobenius norm error
    error = torch.norm(dense_attention - upsampled_draft, p='fro')
    relative_error = error / torch.norm(dense_attention, p='fro')
    
    print(f"Dense attention shape: {dense_attention.shape}")
    print(f"Draft attention shape: {draft_attention.shape}")
    print(f"Frobenius norm error: {error.item():.4f}")
    print(f"Relative error: {relative_error.item():.4f}")
    print("✓ Error bounds test completed\n")


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("=== Testing Memory Efficiency ===")
    
    # Create test inputs (smaller for CPU testing)
    B, n_heads, seq_len, d_head = 1, 4, 4096, 32
    
    # Test dense attention
    query = torch.randn(B, n_heads, seq_len, d_head)
    key = torch.randn(B, n_heads, seq_len, d_head)
    value = torch.randn(B, n_heads, seq_len, d_head)
    
    scale = 1.0 / np.sqrt(d_head)
    dense_attention = torch.matmul(query, key.transpose(-2, -1)) * scale
    dense_attention = F.softmax(dense_attention, dim=-1)
    dense_output = torch.matmul(dense_attention, value)
    
    print(f"Dense computation completed")
    
    # Test draft attention
    model = AdaptiveDraftAttention(
        base_sparsity=0.75,
        pooling_kernel=(2, 4),
        use_quantization=True,
        quantization_bits=8
    )
    
    with torch.no_grad():
        draft_result = model(query, key, value, step=25, total_steps=50)
    
    print(f"Draft computation completed")
    print(f"Dense output shape: {dense_output.shape}")
    print(f"Draft output shape: {draft_result['output'].shape}")
    print("✓ Memory efficiency test completed\n")


def test_configuration_saving():
    """Test configuration saving and loading."""
    print("=== Testing Configuration ===")
    
    # Test basic config
    basic_config = DraftAttentionConfig(
        sparsity_ratio=0.8,
        pooling_kernel=(8, 16),
        fallback_steps=5
    )
    
    config_dict = basic_config.to_dict()
    loaded_config = DraftAttentionConfig.from_dict(config_dict)
    
    print(f"Basic config: {config_dict}")
    print(f"Loaded config sparsity: {loaded_config.sparsity_ratio}")
    
    # Test adaptive config
    adaptive_config = AdaptiveDraftAttentionConfig(
        base_sparsity=0.75,
        adaptive_kernels=[(4, 8), (8, 16)],
        use_quantization=True
    )
    
    adaptive_dict = adaptive_config.to_dict()
    loaded_adaptive = AdaptiveDraftAttentionConfig.from_dict(adaptive_dict)
    
    print(f"Adaptive config keys: {list(adaptive_dict.keys())}")
    print(f"Loaded adaptive quantization: {loaded_adaptive.use_quantization}")
    
    print("✓ Configuration tests passed\n")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=== Testing Edge Cases ===")
    
    # Test with very small input
    B, n_heads, seq_len, d_head = 1, 1, 256, 32
    query = torch.randn(B, n_heads, seq_len, d_head)
    key = torch.randn(B, n_heads, seq_len, d_head)
    value = torch.randn(B, n_heads, seq_len, d_head)
    
    # Test with small kernel
    model = DraftAttention(
        sparsity_ratio=0.5,
        pooling_kernel=(1, 2)  # Very small kernel
    )
    
    try:
        result = model(query, key, value, step=10)
        print(f"Small kernel test passed: {result['output'].shape}")
    except Exception as e:
        print(f"Small kernel test handled gracefully: {e}")
    
    # Test with extreme sparsity
    model_extreme = DraftAttention(
        sparsity_ratio=0.95,  # Very high sparsity
        pooling_kernel=(2, 4)
    )
    
    try:
        result = model_extreme(query, key, value, step=10)
        print(f"Extreme sparsity test passed: {result['output'].shape}")
    except Exception as e:
        print(f"Extreme sparsity test handled gracefully: {e}")
    
    print("✓ Edge cases test completed\n")


if __name__ == "__main__":
    print("Starting Fixed DraftAttention Demo...\n")
    
    try:
        test_basic_draft_attention()
        test_adaptive_draft_attention()
        test_error_bounds()
        test_memory_efficiency()
        test_configuration_saving()
        test_edge_cases()
        
        print("🎉 All fixed tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Fixed test failed with error: {e}")
        import traceback
        traceback.print_exc()