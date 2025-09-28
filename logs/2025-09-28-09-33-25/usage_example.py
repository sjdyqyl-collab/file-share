"""
XAttention Usage Examples and Documentation

This file demonstrates how to use the XAttention and ImprovedXAttention classes
for efficient long-context attention in Transformer models.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to path
sys.path.append('/home/wzc/data/file-share/logs/2025-09-28-09-33-25')

from xattention import XAttention, ImprovedXAttention

def example_basic_usage():
    """Basic usage example of XAttention."""
    print("=== Basic XAttention Usage ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = XAttention(
        hidden_size=768,      # d_model dimension
        num_heads=12,         # Number of attention heads
        block_size=16,        # Block size for sparse attention
        stride=8,            # Stride for antidiagonal sampling
        threshold=0.9,       # Selection threshold
        max_seq_len=4096,    # Maximum supported sequence length
        device=device
    )
    
    # Create sample input
    batch_size = 2
    seq_len = 1024
    hidden_states = torch.randn(batch_size, seq_len, 768, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model(hidden_states)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Basic usage completed")

def example_improved_usage():
    """Advanced usage with ImprovedXAttention."""
    print("\n=== Improved XAttention Usage ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize improved model with all features
    model = ImprovedXAttention(
        hidden_size=768,
        num_heads=12,
        block_size=16,
        stride=8,
        threshold=0.9,
        max_seq_len=4096,
        device=device,
        enable_adaptive_stride=True,    # Enable adaptive stride selection
        enable_multi_pattern=True,      # Enable multi-pattern scoring
        enable_auto_threshold=True      # Enable automatic threshold optimization
    )
    
    # Test with different sequence lengths
    for seq_len in [512, 1024, 2048]:
        hidden_states = torch.randn(1, seq_len, 768, device=device)
        
        with torch.no_grad():
            output = model(hidden_states)
        
        print(f"Seq_len {seq_len}: {hidden_states.shape} -> {output.shape}")
    
    print("✓ Improved usage completed")

def example_model_integration():
    """Example of integrating XAttention into a Transformer model."""
    print("\n=== Model Integration Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class SimpleTransformerBlock(nn.Module):
        def __init__(self, hidden_size, num_heads, use_improved=False):
            super().__init__()
            if use_improved:
                self.attention = ImprovedXAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    block_size=16,
                    stride=8,
                    threshold=0.9,
                    device=device
                )
            else:
                self.attention = XAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    block_size=16,
                    stride=8,
                    threshold=0.9,
                    device=device
                )
            
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, 4 * hidden_size),
                nn.GELU(),
                nn.Linear(4 * hidden_size, hidden_size)
            )
        
        def forward(self, x):
            # Attention with residual connection
            x = x + self.attention(self.norm1(x))
            # FFN with residual connection
            x = x + self.ffn(self.norm2(x))
            return x
    
    # Create model
    model = SimpleTransformerBlock(hidden_size=512, num_heads=8, use_improved=True)
    
    # Test input
    x = torch.randn(2, 256, 512, device=device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Transformer block: {x.shape} -> {output.shape}")
    print("✓ Model integration completed")

def example_parameter_tuning():
    """Example of tuning XAttention parameters."""
    print("\n=== Parameter Tuning Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Different configurations for different use cases
    configs = [
        {
            "name": "High Accuracy",
            "block_size": 8,
            "stride": 4,
            "threshold": 0.95
        },
        {
            "name": "Balanced",
            "block_size": 16,
            "stride": 8,
            "threshold": 0.9
        },
        {
            "name": "High Speed",
            "block_size": 32,
            "stride": 16,
            "threshold": 0.8
        }
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration:")
        model = XAttention(
            hidden_size=512,
            num_heads=8,
            block_size=config["block_size"],
            stride=config["stride"],
            threshold=config["threshold"],
            device=device
        )
        
        x = torch.randn(1, 512, 512, device=device)
        with torch.no_grad():
            output = model(x)
        
        print(f"  Config: {config}")
        print(f"  Output shape: {output.shape}")

def example_saving_loading():
    """Example of saving and loading model weights."""
    print("\n=== Saving and Loading Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and save model
    model = XAttention(hidden_size=256, num_heads=4, device=device)
    
    # Save weights
    torch.save(model.state_dict(), '/tmp/xattention_weights.pth')
    print("✓ Model weights saved")
    
    # Load weights into new model
    new_model = XAttention(hidden_size=256, num_heads=4, device=device)
    new_model.load_weights(torch.load('/tmp/xattention_weights.pth'))
    print("✓ Model weights loaded")
    
    # Verify loading worked
    test_input = torch.randn(1, 64, 256, device=device)
    with torch.no_grad():
        output1 = model(test_input)
        output2 = new_model(test_input)
    
    # Check if outputs are similar
    diff = torch.abs(output1 - output2).mean()
    print(f"Output difference after loading: {diff:.8f}")
    print("✓ Saving and loading verified")

def example_long_context():
    """Example of using XAttention for long sequences."""
    print("\n=== Long Context Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with very long sequence
    model = XAttention(
        hidden_size=384,
        num_heads=6,
        block_size=16,
        stride=8,
        threshold=0.85,
        max_seq_len=16384,
        device=device
    )
    
    # Test with different long sequence lengths
    for seq_len in [2048, 4096, 8192]:
        x = torch.randn(1, seq_len, 384, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"Long sequence {seq_len}: {x.shape} -> {output.shape}")
    
    print("✓ Long context test completed")

if __name__ == "__main__":
    print("XAttention Usage Examples")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    example_basic_usage()
    example_improved_usage()
    example_model_integration()
    example_parameter_tuning()
    example_saving_loading()
    example_long_context()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nFor more information, see the XAttention paper:")
    print("XAttention: Block Sparse Attention with Antidiagonal Scoring")