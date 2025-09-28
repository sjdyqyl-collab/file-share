"""
Simple test to verify DraftAttention implementations work correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the current directory to path for imports
sys.path.insert(0, '/home/wzc/data/file-share/logs/2025-09-19-15-18-03')

from draft_attention import DraftAttention
from enhanced_draft_attention import EnhancedDraftAttention


def test_basic_functionality():
    """Test basic functionality with simple inputs."""
    print("Testing basic DraftAttention functionality...")
    
    # Test parameters
    batch_size = 1
    seq_len = 32 * 32  # Small test size
    dim = 128
    height, width = 32, 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test DraftAttention
    model = DraftAttention(
        dim=dim,
        num_heads=4,
        sparsity_ratio=0.5,
        pooling_kernel=(4, 4),
        use_full_attention_steps=0.0  # Always use draft attention
    ).to(device)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Test forward pass
    with torch.no_grad():
        output = model(x, height=height, width=width, step_ratio=0.5)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch"
        print("✓ DraftAttention basic test passed")
    
    return True


def test_enhanced_basic():
    """Test basic EnhancedDraftAttention functionality."""
    print("\nTesting basic EnhancedDraftAttention functionality...")
    
    # Test parameters
    batch_size = 1
    seq_len = 32 * 32  # Small test size
    dim = 128
    height, width = 32, 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test EnhancedDraftAttention
    model = EnhancedDraftAttention(
        dim=dim,
        num_heads=4,
        base_sparsity_ratio=0.5,
        scales=[(2, 2), (4, 4)],
        quantization_bits=8,
        use_motion_aware=False  # Disable for basic test
    ).to(device)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Test forward pass
    with torch.no_grad():
        output = model(x, height=height, width=width, step_ratio=0.5)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch"
        print("✓ EnhancedDraftAttention basic test passed")
    
    return True


def test_weight_functions():
    """Test weight loading and saving."""
    print("\nTesting weight loading/saving...")
    
    dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DraftAttention(dim=dim, num_heads=2).to(device)
    
    # Test save
    test_path = "/tmp/test_weights.pth"
    model.save_weights(test_path)
    
    # Test load
    new_model = DraftAttention(dim=dim, num_heads=2).to(device)
    new_model.load_weights(torch.load(test_path))
    
    # Verify
    with torch.no_grad():
        x = torch.randn(1, 100, dim, device=device)
        out1 = model(x, height=10, width=10, step_ratio=0.5)
        out2 = new_model(x, height=10, width=10, step_ratio=0.5)
        
        assert torch.allclose(out1, out2, atol=1e-6), "Weight loading failed"
        print("✓ Weight loading/saving test passed")
    
    # Cleanup
    os.remove(test_path)
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("DraftAttention Implementation Test")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_enhanced_basic()
        test_weight_functions()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)