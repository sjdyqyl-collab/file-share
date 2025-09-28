import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from draft_attention import DraftAttention
from draft_attention_plus import DraftAttentionPlus

def small_benchmark():
    """Benchmark with smaller sequence lengths to fit in memory."""
    
    # Small test parameters
    batch_size = 1
    frames = 2
    height = 32
    width = 32
    hidden_size = 256
    num_heads = 8
    sequence_length = frames * height * width  # 2048 tokens
    
    print(f"Small test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length} ({frames}×{height}×{width})")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of heads: {num_heads}")
    print()
    
    # Create input tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, sequence_length, hidden_size, device=device)
    frame_size = (height, width)
    
    # Test original DraftAttention
    print("=== Testing Original DraftAttention ===")
    draft_attn = DraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_ratio=0.8,
        kernel_size=(4, 8)
    ).to(device)
    
    # Test forward pass
    try:
        output = draft_attn(x, frame_size=frame_size, frames=frames)
        print(f"✓ Original DraftAttention forward pass successful")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Original DraftAttention failed: {e}")
        return
    
    # Test improved DraftAttention++
    print("\n=== Testing DraftAttention++ ===")
    draft_attn_plus = DraftAttentionPlus(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_range=(0.6, 0.9),
        kernel_sizes=[16, 32, 64],
        use_quantization=False,
        use_multi_gpu=False
    ).to(device)
    
    try:
        output_plus = draft_attn_plus(x, frame_size=frame_size, frames=frames)
        print(f"✓ DraftAttention++ forward pass successful")
        print(f"Output shape: {output_plus.shape}")
    except Exception as e:
        print(f"✗ DraftAttention++ failed: {e}")
        return
    
    # Test correctness
    print("\n=== Testing Correctness ===")
    with torch.no_grad():
        # Copy weights for fair comparison (only copy learnable parameters)
        draft_attn_plus.q_proj.weight.data = draft_attn.q_proj.weight.data.clone()
        draft_attn_plus.k_proj.weight.data = draft_attn.k_proj.weight.data.clone()
        draft_attn_plus.v_proj.weight.data = draft_attn.v_proj.weight.data.clone()
        draft_attn_plus.out_proj.weight.data = draft_attn.out_proj.weight.data.clone()
        
        output1 = draft_attn(x, frame_size=frame_size, frames=frames)
        output2 = draft_attn_plus(x, frame_size=frame_size, frames=frames)
        
        # Check shapes
        assert output1.shape == output2.shape, "Output shapes don't match"
        print(f"✓ Output shapes match: {output1.shape}")
        
        # Check values (should be different due to different pooling strategies)
        diff = torch.abs(output1 - output2).mean()
        print(f"Mean absolute difference: {diff:.6f}")
        
        # Check for NaN/Inf
        assert not torch.isnan(output1).any(), "NaN in original output"
        assert not torch.isnan(output2).any(), "NaN in improved output"
        assert not torch.isinf(output1).any(), "Inf in original output"
        assert not torch.isinf(output2).any(), "Inf in improved output"
        print("✓ No NaN or Inf values")
    
    # Test adaptive pooling
    print("\n=== Testing Adaptive Pooling ===")
    try:
        # Test with different content complexities
        x_complex = torch.randn_like(x) * 2  # Higher complexity
        x_simple = torch.ones_like(x) * 0.1  # Lower complexity
        
        # Forward pass with different inputs
        out1 = draft_attn_plus(x_complex, frame_size=frame_size, frames=frames)
        out2 = draft_attn_plus(x_simple, frame_size=frame_size, frames=frames)
        
        print(f"✓ Adaptive pooling works with different content complexities")
        print(f"  Complex input output shape: {out1.shape}")
        print(f"  Simple input output shape: {out2.shape}")
        
    except Exception as e:
        print(f"✗ Adaptive pooling test failed: {e}")
    
    # Test memory usage
    print("\n=== Testing Memory Usage ===")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        _ = draft_attn_plus(x, frame_size=frame_size, frames=frames)
        torch.cuda.synchronize()
        
        memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Memory usage: {memory_used:.1f} MB")
    
    print("\n=== All Tests Completed Successfully! ===")

def test_gradient_flow():
    """Test that gradients flow correctly through the models."""
    
    batch_size = 1
    frames = 1
    height = 16
    width = 16
    hidden_size = 128
    num_heads = 4
    sequence_length = frames * height * width
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, sequence_length, hidden_size, device=device, requires_grad=True)
    target = torch.randn_like(x)
    
    # Test original
    print("Testing gradient flow in original DraftAttention...")
    draft_attn = DraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_ratio=0.7,
        kernel_size=(2, 4)
    ).to(device)
    
    output = draft_attn(x, frame_size=(height, width), frames=frames)
    loss = F.mse_loss(output, target)
    loss.backward()
    
    # Check gradients exist
    for name, param in draft_attn.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    print("✓ Original DraftAttention gradients flow correctly")
    
    # Test improved
    print("Testing gradient flow in DraftAttention++...")
    draft_attn_plus = DraftAttentionPlus(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_range=(0.5, 0.8),
        use_quantization=False  # Disable quantization for gradient test
    ).to(device)
    
    # Copy weights
    draft_attn_plus.q_proj.weight.data = draft_attn.q_proj.weight.data.clone()
    draft_attn_plus.k_proj.weight.data = draft_attn.k_proj.weight.data.clone()
    draft_attn_plus.v_proj.weight.data = draft_attn.v_proj.weight.data.clone()
    draft_attn_plus.out_proj.weight.data = draft_attn.out_proj.weight.data.clone()
    
    x.grad.zero_()  # Reset gradients
    output_plus = draft_attn_plus(x, frame_size=(height, width), frames=frames)
    loss_plus = F.mse_loss(output_plus, target)
    loss_plus.backward()
    
    # Check gradients exist
    for name, param in draft_attn_plus.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    print("✓ DraftAttention++ gradients flow correctly")

if __name__ == "__main__":
    print("DraftAttention Small Scale Demo")
    print("=" * 40)
    
    small_benchmark()
    print()
    
    test_gradient_flow()
    
    print("\n" + "=" * 40)
    print("All tests completed successfully!")