"""
Utility functions for XAttention implementations.
"""

import torch
import torch.nn as nn
import json
import os
from typing import Dict, Any, Optional
import hashlib


def save_model_weights(model: nn.Module, filepath: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Save model weights and metadata.
    
    Args:
        model: PyTorch model
        filepath: Path to save weights
        metadata: Additional metadata to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save state dict
    torch.save(model.state_dict(), filepath)
    
    # Save metadata
    if metadata is not None:
        metadata_path = filepath.replace('.pth', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Create checksum
    with open(filepath, 'rb') as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    
    checksum_path = filepath.replace('.pth', '_checksum.txt')
    with open(checksum_path, 'w') as f:
        f.write(checksum)
    
    print(f"Model weights saved to {filepath}")
    print(f"Checksum: {checksum}")


def load_model_weights(model: nn.Module, filepath: str, strict: bool = True) -> Dict[str, Any]:
    """
    Load model weights and metadata.
    
    Args:
        model: PyTorch model
        filepath: Path to load weights from
        strict: Whether to strictly enforce key matching
        
    Returns:
        dict with metadata if available
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")
    
    # Verify checksum
    checksum_path = filepath.replace('.pth', '_checksum.txt')
    if os.path.exists(checksum_path):
        with open(checksum_path, 'r') as f:
            expected_checksum = f.read().strip()
        
        with open(filepath, 'rb') as f:
            actual_checksum = hashlib.md5(f.read()).hexdigest()
        
        if expected_checksum != actual_checksum:
            print(f"Warning: Checksum mismatch!")
            print(f"Expected: {expected_checksum}")
            print(f"Actual: {actual_checksum}")
    
    # Load weights
    state_dict = torch.load(filepath, map_location='cpu')
    model.load_state_dict(state_dict, strict=strict)
    
    # Load metadata
    metadata = {}
    metadata_path = filepath.replace('.pth', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    print(f"Model weights loaded from {filepath}")
    return metadata


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, float]:
    """
    Compare two models by computing parameter differences.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        dict with comparison metrics
    """
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    if set(params1.keys()) != set(params2.keys()):
        return {'error': 'Model architectures do not match'}
    
    differences = {}
    total_diff = 0.0
    param_count = 0
    
    for name in params1.keys():
        diff = torch.abs(params1[name] - params2[name]).mean().item()
        differences[name] = diff
        total_diff += diff
        param_count += 1
    
    return {
        'average_difference': total_diff / param_count,
        'max_difference': max(differences.values()),
        'parameter_differences': differences
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def profile_memory_usage(model: nn.Module, input_tensor: torch.Tensor, device: str = 'cuda'):
    """
    Profile memory usage of a model.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor for profiling
        device: Device to use for profiling
        
    Returns:
        dict with memory usage statistics
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    stats = {}
    if device == 'cuda':
        stats.update({
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
            'memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2
        })
    
    # Estimate model memory
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    stats['model_memory_mb'] = model_memory
    
    return stats


def create_config_file(config: Dict[str, Any], filepath: str):
    """
    Create a configuration file for XAttention models.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config_file(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        configuration dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    return config


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate XAttention model configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['dim', 'num_heads']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required key: {key}")
            return False
    
    # Validate values
    if config['dim'] <= 0:
        print("dim must be positive")
        return False
    
    if config['num_heads'] <= 0:
        print("num_heads must be positive")
        return False
    
    if config['dim'] % config['num_heads'] != 0:
        print("dim must be divisible by num_heads")
        return False
    
    return True


if __name__ == "__main__":
    # Example usage
    from xattention_original import XAttentionOriginal
    from xattention_improved import XAttentionImproved
    
    # Create models
    original = XAttentionOriginal(dim=256, num_heads=8)
    improved = XAttentionImproved(dim=256, num_heads=8)
    
    # Test utilities
    print("=== Parameter Count ===")
    print("Original:", count_parameters(original))
    print("Improved:", count_parameters(improved))
    
    print("\n=== Save/Load Test ===")
    # Save weights
    save_model_weights(original, "/tmp/xattention_original_weights.pth", 
                       {"model_type": "XAttentionOriginal", "version": "1.0"})
    
    # Load weights
    metadata = load_model_weights(original, "/tmp/xattention_original_weights.pth")
    print("Loaded metadata:", metadata)
    
    print("\n=== Configuration Test ===")
    config = {
        "dim": 512,
        "num_heads": 8,
        "block_size": 16,
        "threshold": 0.85,
        "use_dynamic_threshold": True
    }
    
    create_config_file(config, "/tmp/xattention_config.json")
    loaded_config = load_config_file("/tmp/xattention_config.json")
    print("Loaded config:", loaded_config)
    print("Valid config:", validate_model_config(loaded_config))