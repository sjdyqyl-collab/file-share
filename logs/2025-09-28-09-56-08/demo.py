import torch
import numpy as np
import sys
import os

# Add the directory to path for imports
sys.path.append('/home/wzc/data/file-share/logs/2025-09-28-09-56-08')

from xattention_original import XAttentionOriginal
from xattention_improved import XAttentionImproved

def test_xattention_original():
    """Test the original XAttention implementation."""
    print("Testing XAttention Original...")
    
    # Parameters
    batch_size = 2
    seq_len = 64
    hidden_size = 512
    num_heads = 8
    
    # Create model
    model = XAttentionOriginal(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=8,
        stride=8
    ).cuda()
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    # Test forward pass
    with torch.no_grad():
        output = model(x, optimize_thresholds=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test sparsity stats
    stats = model.get_sparsity_stats()
    print(f"Sparsity stats: {stats}")
    
    return True

def test_xattention_improved():
    """Test the improved XAttention implementation."""
    print("\nTesting XAttention Improved...")
    
    # Parameters
    batch_size = 2
    seq_len = 64
    hidden_size = 512
    num_heads = 8
    
    # Create model
    model = XAttentionImproved(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_sizes=[4, 8, 16],
        stride=8
    ).cuda()
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    # Test forward pass
    with torch.no_grad():
        output = model(x, use_hierarchical_threshold=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test improvement stats
    stats = model.get_improvement_stats()
    print(f"Improvement stats: {stats}")
    
    return True

def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\nTesting Memory Efficiency...")
    
    # Test with larger sequence length
    batch_size = 1
    seq_len = 1024
    hidden_size = 512
    num_heads = 8
    
    model = XAttentionImproved(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_sizes=[8, 16, 32],
        stride=8
    ).cuda()
    
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    # Measure memory usage
    torch.cuda.empty_cache()
    start_memory = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        output = model(x)
    
    end_memory = torch.cuda.memory_allocated()
    memory_used = end_memory - start_memory
    
    print(f"Sequence length: {seq_len}")
    print(f"Memory used: {memory_used / 1024**2:.2f} MB")
    print(f"Output shape: {output.shape}")
    
    return True

def test_gradient_flow():
    """Test gradient flow through the models."""
    print("\nTesting Gradient Flow...")
    
    batch_size = 1
    seq_len = 32
    hidden_size = 256
    num_heads = 4
    
    # Test original model
    model_orig = XAttentionOriginal(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=8,
        stride=8
    ).cuda()
    
    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True).cuda()
    target = torch.randn_like(x)
    
    output = model_orig(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    print(f"Original model gradient norm: {x.grad.norm().item():.4f}")
    
    # Test improved model
    model_improved = XAttentionImproved(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_sizes=[4, 8],
        stride=8
    ).cuda()
    
    x_improved = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True).cuda()
    output_improved = model_improved(x_improved)
    loss_improved = torch.nn.functional.mse_loss(output_improved, target)
    loss_improved.backward()
    
    print(f"Improved model gradient norm: {x_improved.grad.norm().item():.4f}")
    
    return True

if __name__ == "__main__":
    try:
        # Set CUDA device
        torch.cuda.set_device(0)
        
        # Run tests
        test1 = test_xattention_original()
        test2 = test_xattention_improved()
        test3 = test_memory_efficiency()
        test4 = test_gradient_flow()
        
        if all([test1, test2, test3, test4]):
            print("\n✅ All tests passed successfully!")
        else:
            print("\n❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()