"""
Utility functions for XAttention implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import os
import time

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def compute_attention_sparsity(attention_matrix: torch.Tensor) -> Dict[str, float]:
    """
    Compute sparsity metrics for attention matrix
    
    Args:
        attention_matrix: Attention weights (B, H, L, L)
        
    Returns:
        Dictionary with sparsity metrics
    """
    with torch.no_grad():
        # Flatten attention matrix
        flat_attn = attention_matrix.flatten()
        
        # Compute statistics
        mean_weight = flat_attn.mean().item()
        std_weight = flat_attn.std().item()
        
        # Compute sparsity (fraction of near-zero weights)
        threshold = 1e-4
        sparse_mask = (attention_matrix < threshold).float()
        sparsity_ratio = sparse_mask.mean().item()
        
        # Compute attention entropy
        entropy = -torch.sum(attention_matrix * torch.log(attention_matrix + 1e-8), dim=-1)
        avg_entropy = entropy.mean().item()
        
        return {
            'mean_weight': mean_weight,
            'std_weight': std_weight,
            'sparsity_ratio': sparsity_ratio,
            'avg_entropy': avg_entropy
        }


def visualize_attention_pattern(
    attention_matrix: torch.Tensor,
    title: str = "Attention Pattern",
    save_path: str = None
) -> None:
    """
    Visualize attention pattern
    
    Args:
        attention_matrix: Attention weights (L, L) or (H, L, L)
        title: Plot title
        save_path: Optional path to save the plot
    """
    if not HAS_VIZ:
        print("Visualization libraries not available. Skipping plot.")
        return
    
    if len(attention_matrix.shape) == 3:
        # Average across heads
        attention_matrix = attention_matrix.mean(0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_matrix.cpu().numpy(),
        cmap='Blues',
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False
    )
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def benchmark_attention_speed(
    attention_module: nn.Module,
    input_shape: Tuple[int, int, int],
    num_iterations: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark attention module speed
    
    Args:
        attention_module: Attention module to benchmark
        input_shape: (B, L, D) input shape
        num_iterations: Number of benchmark iterations
        device: Device to run on
        
    Returns:
        Dictionary with timing results
    """
    attention_module = attention_module.to(device)
    attention_module.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Warmup
    for _ in range(10):
        _ = attention_module(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time the forward pass
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = attention_module(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    avg_time = total_time / num_iterations
    
    # Compute FLOPs (approximate)
    B, L, D = input_shape
    H = attention_module.num_heads
    d_h = D // H
    
    # Standard attention: 2 * B * H * L * L * d_h FLOPs
    baseline_flops = 2 * B * H * L * L * d_h
    
    return {
        'total_time': total_time,
        'avg_time_ms': avg_time * 1000,
        'baseline_flops': baseline_flops,
        'throughput_samples_per_sec': B * num_iterations / total_time
    }


def compare_attention_methods(
    baseline_module: nn.Module,
    xattention_module: nn.Module,
    input_shape: Tuple[int, int, int],
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Compare baseline and XAttention methods
    
    Args:
        baseline_module: Baseline attention module
        xattention_module: XAttention module
        input_shape: (B, L, D) input shape
        device: Device to run on
        
    Returns:
        Comparison results
    """
    B, L, D = input_shape
    
    # Create test input
    test_input = torch.randn(*input_shape, device=device)
    
    # Get outputs
    with torch.no_grad():
        baseline_output = baseline_module(test_input)
        xattention_output = xattention_module(test_input)
    
    # Compute accuracy metrics
    mse_loss = nn.MSELoss()(baseline_output, xattention_output).item()
    
    # Compute cosine similarity
    cos_sim = nn.CosineSimilarity(dim=-1)
    similarity = cos_sim(
        baseline_output.flatten(),
        xattention_output.flatten()
    ).mean().item()
    
    # Benchmark speeds
    baseline_stats = benchmark_attention_speed(baseline_module, input_shape, device=device)
    xattention_stats = benchmark_attention_speed(xattention_module, input_shape, device=device)
    
    # Compute speedup
    speedup = baseline_stats['avg_time_ms'] / xattention_stats['avg_time_ms']
    
    # Get XAttention sparsity
    if hasattr(xattention_module, 'get_sparsity_stats'):
        sparsity_stats = xattention_module.get_sparsity_stats()
    else:
        sparsity_stats = {}
    
    return {
        'accuracy': {
            'mse': mse_loss,
            'cosine_similarity': similarity
        },
        'performance': {
            'baseline_time_ms': baseline_stats['avg_time_ms'],
            'xattention_time_ms': xattention_stats['avg_time_ms'],
            'speedup': speedup
        },
        'sparsity': sparsity_stats
    }


def create_attention_mask(
    seq_len: int,
    mask_type: str = 'causal',
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create attention mask
    
    Args:
        seq_len: Sequence length
        mask_type: Type of mask ('causal', 'full', 'random')
        device: Device to create mask on
        
    Returns:
        Attention mask tensor
    """
    if mask_type == 'causal':
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    elif mask_type == 'full':
        mask = torch.ones(seq_len, seq_len, device=device)
    elif mask_type == 'random':
        mask = (torch.rand(seq_len, seq_len, device=device) > 0.3).float()
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    return mask


def validate_implementation():
    """Validate the implementation with basic tests"""
    print("Validating XAttention implementation...")
    
    # Test parameters
    B, L, D = 2, 64, 128
    num_heads = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create modules
    from baseline_attention import BaselineAttention
    from xattention import XAttention
    
    baseline = BaselineAttention(D, num_heads).to(device)
    xattention = XAttention(D, num_heads, block_size=8, stride=8, threshold=0.9).to(device)
    
    # Test forward pass
    x = torch.randn(B, L, D, device=device)
    
    try:
        baseline_out = baseline(x)
        xattention_out = xattention(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Baseline output shape: {baseline_out.shape}")
        print(f"  XAttention output shape: {xattention_out.shape}")
        
        # Test sparsity stats
        stats = xattention.get_sparsity_stats()
        print(f"  Sparsity stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def generate_random_input(
    batch_size: int,
    seq_len: int,
    dim: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """Generate random input tensor"""
    return torch.randn(batch_size, seq_len, dim, device=device)


def save_model_checkpoint(
    model: nn.Module,
    filepath: str,
    additional_info: dict = None
) -> None:
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_heads': getattr(model, 'num_heads', None),
            'head_dim': getattr(model, 'head_dim', None),
            'block_size': getattr(model, 'block_size', None),
            'stride': getattr(model, 'stride', None),
            'threshold': getattr(model, 'threshold', None)
        }
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model_checkpoint(
    model: nn.Module,
    filepath: str,
    device: str = 'cuda'
) -> dict:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filepath}")
    return checkpoint


if __name__ == "__main__":
    validate_implementation()