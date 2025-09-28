import torch
import sys
import os

# Add the current directory to path to import xattention
sys.path.append('/home/wzc/data/file-share/logs/2025-09-28-09-33-25')

from xattention import XAttention, ImprovedXAttention

def test_basic_functionality():
    """Test basic functionality of XAttention classes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test XAttention
    print("\n=== Testing XAttention ===")
    model = XAttention(
        hidden_size=256,
        num_heads=8,
        block_size=8,
        stride=4,
        threshold=0.8,
        device=device
    )
    
    # Create test input
    batch_size, seq_len = 1, 128
    hidden_states = torch.randn(batch_size, seq_len, 256, device=device)
    
    print(f"Input shape: {hidden_states.shape}")
    
    with torch.no_grad():
        output = model(hidden_states)
    
    print(f"Output shape: {output.shape}")
    print("✓ XAttention test passed!")
    
    # Test ImprovedXAttention
    print("\n=== Testing ImprovedXAttention ===")
    improved_model = ImprovedXAttention(
        hidden_size=256,
        num_heads=8,
        block_size=8,
        stride=4,
        threshold=0.8,
        device=device,
        enable_adaptive_stride=False,  # Disable for testing
        enable_multi_pattern=False,    # Disable for testing
        enable_auto_threshold=False    # Disable for testing
    )
    
    with torch.no_grad():
        improved_output = improved_model(hidden_states)
    
    print(f"Improved output shape: {improved_output.shape}")
    print("✓ ImprovedXAttention test passed!")
    
    # Test with different sequence lengths
    print("\n=== Testing different sequence lengths ===")
    for seq_len in [64, 256, 512]:
        test_input = torch.randn(1, seq_len, 256, device=device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"Seq_len {seq_len}: Input {test_input.shape} -> Output {test_output.shape}")
    
    print("\n=== All tests completed successfully! ===")

if __name__ == "__main__":
    test_basic_functionality()