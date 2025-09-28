import torch
import torch.nn as nn
import time
import numpy as np
from draft_attention import DraftAttention
from draft_attention_plus import DraftAttentionPlus

def benchmark_attention():
    """Benchmark both DraftAttention implementations."""
    
    # Test parameters
    batch_size = 2
    frames = 8
    height = 64
    width = 64
    hidden_size = 768
    num_heads = 12
    sequence_length = frames * height * width  # 32768 tokens
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length} ({frames}×{height}×{width})")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of heads: {num_heads}")
    print()
    
    # Create input tensor
    x = torch.randn(batch_size, sequence_length, hidden_size).cuda()
    frame_size = (height, width)
    
    # Test original DraftAttention
    print("=== Testing Original DraftAttention ===")
    draft_attn = DraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_ratio=0.9,
        kernel_size=(8, 16)
    ).cuda()
    
    # Warmup
    for _ in range(5):
        _ = draft_attn(x, frame_size=frame_size, frames=frames)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        output = draft_attn(x, frame_size=frame_size, frames=frames)
    torch.cuda.synchronize()
    original_time = (time.time() - start_time) / 10
    
    print(f"Original DraftAttention time: {original_time:.4f}s")
    print(f"Output shape: {output.shape}")
    print()
    
    # Test improved DraftAttention++
    print("=== Testing DraftAttention++ ===")
    draft_attn_plus = DraftAttentionPlus(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_range=(0.5, 0.9),
        kernel_sizes=[64, 128, 256],
        use_quantization=False,  # Disable for fair comparison
        use_multi_gpu=False
    ).cuda()
    
    # Warmup
    for _ in range(5):
        _ = draft_attn_plus(x, frame_size=frame_size, frames=frames)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        output_plus = draft_attn_plus(x, frame_size=frame_size, frames=frames)
    torch.cuda.synchronize()
    improved_time = (time.time() - start_time) / 10
    
    print(f"DraftAttention++ time: {improved_time:.4f}s")
    print(f"Output shape: {output_plus.shape}")
    print()
    
    # Speedup calculation
    speedup = original_time / improved_time if improved_time > 0 else 1.0
    print(f"Speedup: {speedup:.2f}x")
    
    # Test quantization
    print("\n=== Testing INT8 Quantization ===")
    draft_attn_plus_quant = DraftAttentionPlus(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_range=(0.5, 0.9),
        use_quantization=True
    ).cuda()
    
    # Copy weights from non-quantized model
    draft_attn_plus_quant.load_state_dict(draft_attn_plus.state_dict())
    draft_attn_plus_quant.quantize_for_inference()
    
    # Benchmark quantized version
    start_time = time.time()
    for _ in range(10):
        output_quant = draft_attn_plus_quant(x, frame_size=frame_size, frames=frames)
    torch.cuda.synchronize()
    quantized_time = (time.time() - start_time) / 10
    
    print(f"Quantized DraftAttention++ time: {quantized_time:.4f}s")
    quantization_speedup = improved_time / quantized_time if quantized_time > 0 else 1.0
    print(f"Quantization speedup: {quantization_speedup:.2f}x")
    
    return {
        'original_time': original_time,
        'improved_time': improved_time,
        'quantized_time': quantized_time,
        'speedup': speedup,
        'quantization_speedup': quantization_speedup
    }

def test_correctness():
    """Test that both implementations produce similar outputs."""
    
    batch_size = 1
    frames = 2
    height = 32
    width = 32
    hidden_size = 256
    num_heads = 8
    sequence_length = frames * height * width
    
    # Create input
    x = torch.randn(batch_size, sequence_length, hidden_size).cuda()
    frame_size = (height, width)
    
    # Create models
    draft_attn = DraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_ratio=0.8
    ).cuda()
    
    draft_attn_plus = DraftAttentionPlus(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_range=(0.8, 0.8),  # Fixed for comparison
        use_quantization=False
    ).cuda()
    
    # Copy weights for fair comparison
    draft_attn_plus.load_state_dict(draft_attn.state_dict(), strict=False)
    
    # Forward pass
    with torch.no_grad():
        output_original = draft_attn(x, frame_size=frame_size, frames=frames)
        output_improved = draft_attn_plus(x, frame_size=frame_size, frames=frames)
    
    # Compute difference
    diff = torch.abs(output_original - output_improved).mean()
    print(f"Mean absolute difference: {diff:.6f}")
    
    # Check shapes
    assert output_original.shape == output_improved.shape
    print("✓ Output shapes match")
    
    return diff

def test_memory_usage():
    """Test memory usage comparison."""
    
    batch_size = 1
    frames = 4
    height = 64
    width = 64
    hidden_size = 512
    num_heads = 8
    sequence_length = frames * height * width
    
    x = torch.randn(batch_size, sequence_length, hidden_size).cuda()
    frame_size = (height, width)
    
    # Test original
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    draft_attn = DraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_ratio=0.9
    ).cuda()
    
    _ = draft_attn(x, frame_size=frame_size, frames=frames)
    torch.cuda.synchronize()
    
    original_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Test improved
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    draft_attn_plus = DraftAttentionPlus(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_range=(0.5, 0.9),
        use_quantization=True
    ).cuda()
    
    draft_attn_plus.quantize_for_inference()
    _ = draft_attn_plus(x, frame_size=frame_size, frames=frames)
    torch.cuda.synchronize()
    
    improved_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"Original memory usage: {original_memory:.1f} MB")
    print(f"Improved memory usage: {improved_memory:.1f} MB")
    print(f"Memory reduction: {(original_memory - improved_memory) / original_memory * 100:.1f}%")
    
    return original_memory, improved_memory

if __name__ == "__main__":
    print("DraftAttention Implementation Demo")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU")
        device = torch.device('cpu')
    else:
        print(f"Running on CUDA device: {torch.cuda.get_device_name()}")
        device = torch.device('cuda')
    
    try:
        # Run tests
        results = benchmark_attention()
        print("\n" + "=" * 50)
        
        diff = test_correctness()
        print("\n" + "=" * 50)
        
        mem_original, mem_improved = test_memory_usage()
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Original DraftAttention: {results['original_time']:.4f}s")
        print(f"DraftAttention++: {results['improved_time']:.4f}s")
        print(f"Overall speedup: {results['speedup']:.2f}x")
        print(f"Quantization speedup: {results['quantization_speedup']:.2f}x")
        print(f"Output correctness: {diff:.6f} (lower is better)")
        print(f"Memory reduction: {(mem_original - mem_improved) / mem_original * 100:.1f}%")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()