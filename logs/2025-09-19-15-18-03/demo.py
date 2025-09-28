"""
Demo script to test DraftAttention implementations.
"""

import torch
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from draft_attention import DraftAttention
from enhanced_draft_attention import EnhancedDraftAttention


def test_draft_attention():
    """Test the basic DraftAttention implementation."""
    print("Testing DraftAttention...")
    
    # Model parameters
    batch_size = 2
    seq_len = 48 * 80  # HunyuanVideo latent size
    dim = 512
    height, width = 48, 80
    
    # Create model
    model = DraftAttention(
        dim=dim,
        num_heads=8,
        sparsity_ratio=0.75,
        pooling_kernel=(8, 16),
        use_full_attention_steps=0.25
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Test forward pass
    with torch.no_grad():
        # Test at different denoising steps
        for step_ratio in [0.1, 0.5, 0.9]:
            output = model(x, height=height, width=width, step_ratio=step_ratio)
            assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
            print(f"  Step ratio {step_ratio}: Success - Output shape {output.shape}")
    
    print("DraftAttention test passed!")


def test_enhanced_draft_attention():
    """Test the EnhancedDraftAttention implementation."""
    print("\nTesting EnhancedDraftAttention...")
    
    # Model parameters
    batch_size = 2
    seq_len = 48 * 80  # HunyuanVideo latent size
    dim = 512
    height, width = 48, 80
    
    # Create model
    model = EnhancedDraftAttention(
        dim=dim,
        num_heads=8,
        base_sparsity_ratio=0.75,
        scales=[(4, 8), (8, 16), (16, 32)],
        quantization_bits=8,
        use_motion_aware=True
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Test forward pass
    with torch.no_grad():
        # Test at different denoising steps
        for step_ratio in [0.1, 0.5, 0.9]:
            output = model(x, height=height, width=width, step_ratio=step_ratio)
            assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
            print(f"  Step ratio {step_ratio}: Success - Output shape {output.shape}")
    
    # Test quantization settings
    model.set_quantization_bits(4)
    model.set_sparsity_ratio(0.9)
    print("  Quantization and sparsity settings updated successfully")
    
    print("EnhancedDraftAttention test passed!")


def test_weight_loading():
    """Test weight loading and saving functionality."""
    print("\nTesting weight loading/saving...")
    
    dim = 512
    model = DraftAttention(dim=dim, num_heads=8)
    
    # Save weights
    temp_path = "/tmp/draft_attention_test.pth"
    model.save_weights(temp_path)
    
    # Create new model and load weights
    new_model = DraftAttention(dim=dim, num_heads=8)
    state_dict = torch.load(temp_path)
    new_model.load_weights(state_dict)
    
    # Verify weights loaded
    for param1, param2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(param1, param2), "Weights not loaded correctly"
    
    # Clean up
    os.remove(temp_path)
    print("  Weight loading/saving test passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("DraftAttention Demo")
    print("=" * 50)
    
    try:
        test_draft_attention()
        test_enhanced_draft_attention()
        test_weight_loading()
        
        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()