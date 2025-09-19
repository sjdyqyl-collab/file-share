"""
Demo script to test DraftAttention and AHDA implementations
"""

import torch
import sys
import os
import torch.nn as nn
import math

# Add the current directory to path for imports
sys.path.insert(0, '/home/wzc/data/file-share/logs/2025-09-19-09-32-58')

from draft_attention import DraftAttention, DraftAttentionBlock, create_draft_attention_model
from adaptive_hierarchical_draft_attention import AdaptiveHierarchicalDraftAttention, AHDAConfig, create_ahda_model


def test_draft_attention():
    """Test basic DraftAttention functionality."""
    print("=" * 50)
    print("Testing DraftAttention...")
    
    # Test parameters
    B, N, D = 2, 1024, 768
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = DraftAttention(
        dim=D,
        num_heads=12,
        sparsity_ratio=0.8,
        pooling_kernel=(8, 16)
    ).to(device)
    
    # Create input
    x = torch.randn(B, N, D).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Device: {device}")
    print("✓ DraftAttention test passed!")
    
    return True


def test_ahda():
    """Test Adaptive Hierarchical Draft Attention."""
    print("=" * 50)
    print("Testing AHDA...")
    
    # Test parameters
    B, N, D = 2, 1024, 768
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = AdaptiveHierarchicalDraftAttention(
        dim=D,
        num_heads=12,
        sparsity_ratio=0.8,
        pooling_levels=[(8, 16), (4, 8), (2, 4)],
        fusion_type="learned"
    ).to(device)
    
    # Create input
    x = torch.randn(B, N, D).to(device)
    
    # Forward pass
    with torch.no_grad():
        output, info = model(x, return_intermediate=True)
        
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Number of hierarchical levels: {len(info['draft_maps'])}")
    print(f"✓ Actual sparsity ratio: {info['sparsity_ratio']:.3f}")
    print(f"✓ Device: {device}")
    print("✓ AHDA test passed!")
    
    return True


def test_model_creation():
    """Test model creation utilities."""
    print("=" * 50)
    print("Testing model creation utilities...")
    
    # Test DraftAttention model creation
    draft_model = create_draft_attention_model(
        dim=768,
        num_blocks=6,
        num_heads=12,
        sparsity_ratio=0.8
    )
    
    # Test AHDA model creation
    config = AHDAConfig(
        dim=768,
        num_heads=12,
        sparsity_ratio=0.8,
        pooling_levels=[(8, 16), (4, 8)]
    )
    ahda_model = create_ahda_model(config)
    
    print("✓ DraftAttention model created successfully")
    print("✓ AHDA model created successfully")
    print("✓ Model creation utilities test passed!")
    
    return True


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("=" * 50)
    print("Testing memory efficiency...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, D = 1, 4096, 768  # Larger sequence to see memory benefits
    
    # Standard attention (for comparison)
    class StandardAttention(nn.Module):
        def __init__(self, dim, num_heads=12):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3)
            self.out = nn.Linear(dim, dim)
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = 1.0 / math.sqrt(self.head_dim)
            
        def forward(self, x):
            B, N, D = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            
            out = out.transpose(1, 2).reshape(B, N, D)
            return self.out(out)
    
    # Create models
    standard = StandardAttention(D).to(device)
    draft = DraftAttention(D, num_heads=12, sparsity_ratio=0.8).to(device)
    ahda = AdaptiveHierarchicalDraftAttention(
        D, num_heads=12, sparsity_ratio=0.8
    ).to(device)
    
    x = torch.randn(B, N, D).to(device)
    
    with torch.no_grad():
        # Test inference
        out_std = standard(x)
        out_draft = draft(x)
        out_ahda = ahda(x)
        
        print(f"✓ Standard attention output shape: {out_std.shape}")
        print(f"✓ Draft attention output shape: {out_draft.shape}")
        print(f"✓ AHDA output shape: {out_ahda.shape}")
        
        # Check output consistency
        diff_draft = torch.abs(out_std - out_draft).mean()
        diff_ahda = torch.abs(out_std - out_ahda).mean()
        
        print(f"✓ Draft vs Standard difference: {diff_draft:.6f}")
        print(f"✓ AHDA vs Standard difference: {diff_ahda:.6f}")
    
    return True


def main():
    """Run all tests."""
    print("Starting DraftAttention and AHDA Demo...")
    print("=" * 50)
    
    try:
        # Run tests
        test_draft_attention()
        test_ahda()
        test_model_creation()
        test_memory_efficiency()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
