import torch
import torch.nn as nn
from draft_attention import DraftAttention
from adaptive_draft_attention import AdaptiveDraftAttention
import time

def test_draft_attention():
    """Test the original DraftAttention implementation."""
    print("=== Testing DraftAttention ===")
    
    # Parameters
    batch_size = 2
    hidden_size = 512
    num_heads = 8
    num_frames = 8
    frame_height = 32  # 512p latent
    frame_width = 48   # 512p latent
    sequence_length = num_frames * frame_height * frame_width
    
    # Create model
    model = DraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_ratio=0.9,
        pooling_kernel=(8, 16),
        block_size=128
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create input
    x = torch.randn(batch_size, sequence_length, hidden_size, device=device)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(x, frame_size=(frame_height, frame_width), num_frames=num_frames, timestep_ratio=0.5)
        end_time = time.time()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Device: {device}")
    print(f"Sparsity stats: {model.get_sparsity_stats()}")
    print("✓ DraftAttention test passed\n")
    
    return output

def test_adaptive_draft_attention():
    """Test the improved AdaptiveDraftAttention implementation."""
    print("=== Testing AdaptiveDraftAttention ===")
    
    # Parameters
    batch_size = 2
    hidden_size = 512
    num_heads = 8
    num_frames = 8
    frame_height = 32  # 512p latent
    frame_width = 48   # 512p latent
    sequence_length = num_frames * frame_height * frame_width
    
    # Create model
    model = AdaptiveDraftAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        sparsity_ratio=0.9,
        pooling_kernels=[(4, 8), (8, 16), (16, 32)],
        block_size=128,
        use_full_attention_steps=0.1,
        adaptive_threshold=True,
        head_specific_sparsity=True
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create input
    x = torch.randn(batch_size, sequence_length, hidden_size, device=device)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        # Test different timestep ratios
        for timestep_ratio in [0.05, 0.3, 0.7, 1.0]:
            start_time = time.time()
            output = model(x, frame_size=(frame_height, frame_width), num_frames=num_frames, timestep_ratio=timestep_ratio)
            end_time = time.time()
            
            print(f"Timestep ratio {timestep_ratio:.2f}:")
            print(f"  Output shape: {output.shape}")
            print(f"  Inference time: {(end_time - start_time)*1000:.2f} ms")
    
    print(f"Device: {device}")
    print(f"Adaptive sparsity stats: {model.get_sparsity_stats()}")
    print("✓ AdaptiveDraftAttention test passed\n")
    
    return output

def test_memory_efficiency():
    """Test memory efficiency comparison."""
    print("=== Testing Memory Efficiency ===")
    
    # Parameters
    batch_size = 1
    hidden_size = 768
    num_heads = 12
    num_frames = 16  # Larger sequence
    frame_height = 48  # 768p latent
    frame_width = 80   # 768p latent
    sequence_length = num_frames * frame_height * frame_width
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Standard attention
    class StandardAttention(nn.Module):
        def __init__(self, hidden_size, num_heads):
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        def forward(self, x):
            return self.attn(x, x, x)[0]
    
    standard = StandardAttention(hidden_size, num_heads).to(device)
    draft = DraftAttention(hidden_size, num_heads, sparsity_ratio=0.9).to(device)
    adaptive = AdaptiveDraftAttention(hidden_size, num_heads, sparsity_ratio=0.9).to(device)
    
    x = torch.randn(batch_size, sequence_length, hidden_size, device=device)
    
    # Measure memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = standard(x)
        standard_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = draft(x, frame_size=(frame_height, frame_width), num_frames=num_frames, timestep_ratio=0.5)
        draft_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = adaptive(x, frame_size=(frame_height, frame_width), num_frames=num_frames, timestep_ratio=0.5)
        adaptive_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"Memory usage:")
        print(f"  Standard attention: {standard_memory:.1f} MB")
        print(f"  DraftAttention: {draft_memory:.1f} MB")
        print(f"  AdaptiveDraftAttention: {adaptive_memory:.1f} MB")
        print(f"  Memory reduction: {(standard_memory - draft_memory)/standard_memory*100:.1f}%")
    
    print("✓ Memory efficiency test completed\n")

if __name__ == "__main__":
    print("DraftAttention Implementation Demo")
    print("=" * 50)
    
    try:
        # Test basic functionality
        output1 = test_draft_attention()
        output2 = test_adaptive_draft_attention()
        test_memory_efficiency()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()