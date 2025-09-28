"""
Comprehensive demo and testing script for DraftAttention and HADA frameworks.
"""

import torch
import torch.nn as nn
import time
import json
from draft_attention import DraftAttention, DraftAttentionConfig
from hada_framework import HADAFramework, HADAConfig


def benchmark_attention(model, input_tensor, frame_size, num_frames, num_runs=10, **kwargs):
    """Benchmark attention model performance."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        _ = model(input_tensor, frame_size, num_frames, **kwargs)
    
    # Synchronize CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_tensor, frame_size, num_frames, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time, output


def test_draft_attention():
    """Test DraftAttention implementation."""
    print("=" * 60)
    print("Testing DraftAttention Implementation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test configurations
    configs = [
        {"name": "Small", "B": 1, "N": 32 * 8 * 4, "D": 512, "frames": 4, "H": 32, "W": 8 * 16},
        {"name": "Medium", "B": 2, "N": 64 * 16 * 8, "D": 768, "frames": 8, "H": 64, "W": 16 * 16},
        {"name": "Large", "B": 1, "N": 128 * 32 * 16, "D": 1024, "frames": 16, "H": 128, "W": 32 * 16},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration:")
        print(f"  Batch size: {config['B']}")
        print(f"  Sequence length: {config['N']}")
        print(f"  Hidden dim: {config['D']}")
        print(f"  Frames: {config['frames']}")
        print(f"  Frame size: {config['H']}x{config['W']}")
        
        # Create model
        model = DraftAttention(
            dim=config["D"],
            num_heads=12,
            sparsity_ratio=0.1,
            device=device
        ).to(device)
        
        # Create input
        x = torch.randn(config["B"], config["N"], config["D"], device=device)
        frame_size = (config["H"], config["W"])
        
        # Test forward pass
        try:
            with torch.no_grad():
                output = model(x, frame_size, config["frames"])
            
            # Benchmark
            avg_time, _ = benchmark_attention(model, x, frame_size, config["frames"])
            
            print(f"  ✓ Forward pass successful")
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Average time: {avg_time*1000:.2f}ms")
            
            results.append({
                "config": config["name"],
                "input_shape": list(x.shape),
                "output_shape": list(output.shape),
                "avg_time_ms": avg_time * 1000,
                "status": "success"
            })
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                "config": config["name"],
                "error": str(e),
                "status": "failed"
            })
    
    return results


def test_hada_framework():
    """Test HADA framework implementation."""
    print("\n" + "=" * 60)
    print("Testing HADA Framework Implementation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test configurations
    configs = [
        {"name": "Small", "B": 1, "N": 32 * 8 * 4, "D": 512, "frames": 4, "H": 32, "W": 8 * 16},
        {"name": "Medium", "B": 1, "N": 64 * 16 * 8, "D": 768, "frames": 8, "H": 64, "W": 16 * 16},
    ]
    
    # Test different feature combinations
    feature_configs = [
        {"name": "Baseline", "quant": False, "motion": False, "dynamic": False},
        {"name": "+Quantization", "quant": True, "motion": False, "dynamic": False},
        {"name": "+Motion", "quant": False, "motion": True, "dynamic": False},
        {"name": "+Dynamic", "quant": False, "motion": False, "dynamic": True},
        {"name": "Full HADA", "quant": True, "motion": True, "dynamic": True},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration:")
        
        for feature_config in feature_configs:
            print(f"  Features: {feature_config['name']}")
            
            # Create HADA model
            hada = HADAFramework(
                dim=config["D"],
                num_heads=12,
                num_layers=28,
                use_quantization=feature_config["quant"],
                use_motion_guidance=feature_config["motion"],
                use_dynamic_sparsity=feature_config["dynamic"],
                device=device
            ).to(device)
            
            # Create input
            x = torch.randn(config["B"], config["N"], config["D"], device=device)
            frame_size = (config["H"], config["W"])
            
            try:
                # Test forward pass
                with torch.no_grad():
                    output = hada(x, frame_size, config["frames"], layer_idx=14)
                
                # Benchmark
                avg_time, _ = benchmark_attention(
                    hada, x, frame_size, config["frames"], 
                    layer_idx=14
                )
                
                print(f"    ✓ Output shape: {output.shape}")
                print(f"    ✓ Average time: {avg_time*1000:.2f}ms")
                
                results.append({
                    "config": config["name"],
                    "features": feature_config["name"],
                    "input_shape": list(x.shape),
                    "output_shape": list(output.shape),
                    "avg_time_ms": avg_time * 1000,
                    "status": "success"
                })
                
            except Exception as e:
                print(f"    ✗ Error: {str(e)}")
                results.append({
                    "config": config["name"],
                    "features": feature_config["name"],
                    "error": str(e),
                    "status": "failed"
                })
    
    return results


def compare_performance():
    """Compare performance between DraftAttention and HADA."""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test configuration
    B, N, D = 1, 64 * 16 * 8, 768
    frame_size = (64, 16 * 16)
    num_frames = 8
    
    # Create inputs
    x = torch.randn(B, N, D, device=device)
    
    # DraftAttention
    draft_model = DraftAttention(dim=D, num_heads=12, sparsity_ratio=0.1, device=device).to(device)
    
    # HADA
    hada_model = HADAFramework(
        dim=D, num_heads=12, use_quantization=True, 
        use_motion_guidance=True, use_dynamic_sparsity=True, device=device
    ).to(device)
    
    # Benchmark
    draft_time, _ = benchmark_attention(draft_model, x, frame_size, num_frames)
    hada_time, _ = benchmark_attention(hada_model, x, frame_size, num_frames, layer_idx=14)
    
    print(f"DraftAttention average time: {draft_time*1000:.2f}ms")
    print(f"HADA average time: {hada_time*1000:.2f}ms")
    print(f"Speedup: {draft_time/hada_time:.2f}x")
    
    return {
        "draft_time_ms": draft_time * 1000,
        "hada_time_ms": hada_time * 1000,
        "speedup": draft_time / hada_time
    }


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\n" + "=" * 60)
    print("Memory Efficiency Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return None
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test configuration
    B, N, D = 1, 128 * 32 * 16, 768
    frame_size = (128, 32 * 16)
    num_frames = 16
    
    # Create inputs
    x = torch.randn(B, N, D, device=device)
    
    # Test DraftAttention
    torch.cuda.reset_peak_memory_stats()
    draft_model = DraftAttention(dim=D, num_heads=12, sparsity_ratio=0.1, device=device).to(device)
    
    with torch.no_grad():
        _ = draft_model(x, frame_size, num_frames)
    
    draft_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Test HADA
    torch.cuda.reset_peak_memory_stats()
    hada_model = HADAFramework(
        dim=D, num_heads=12, use_quantization=True, device=device
    ).to(device)
    
    with torch.no_grad():
        _ = hada_model(x, frame_size, num_frames, layer_idx=14)
    
    hada_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"DraftAttention peak memory: {draft_memory:.1f} MB")
    print(f"HADA peak memory: {hada_memory:.1f} MB")
    print(f"Memory reduction: {draft_memory/hada_memory:.2f}x")
    
    return {
        "draft_memory_mb": draft_memory,
        "hada_memory_mb": hada_memory,
        "memory_reduction": draft_memory / hada_memory
    }


def save_results(results, filename):
    """Save test results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    """Main testing function."""
    print("DraftAttention and HADA Framework Testing")
    print("=" * 60)
    
    # Run all tests
    all_results = {}
    
    # Test DraftAttention
    try:
        draft_results = test_draft_attention()
        all_results["draft_attention"] = draft_results
    except Exception as e:
        print(f"DraftAttention test failed: {e}")
        all_results["draft_attention"] = {"error": str(e)}
    
    # Test HADA
    try:
        hada_results = test_hada_framework()
        all_results["hada_framework"] = hada_results
    except Exception as e:
        print(f"HADA framework test failed: {e}")
        all_results["hada_framework"] = {"error": str(e)}
    
    # Performance comparison
    try:
        perf_results = compare_performance()
        all_results["performance_comparison"] = perf_results
    except Exception as e:
        print(f"Performance comparison failed: {e}")
        all_results["performance_comparison"] = {"error": str(e)}
    
    # Memory efficiency
    try:
        memory_results = test_memory_efficiency()
        if memory_results:
            all_results["memory_efficiency"] = memory_results
    except Exception as e:
        print(f"Memory efficiency test failed: {e}")
        all_results["memory_efficiency"] = {"error": str(e)}
    
    # Save results
    save_results(all_results, "/home/wzc/data/file-share/logs/2025-09-19-16-57-59/test_results.json")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("Results saved to: /home/wzc/data/file-share/logs/2025-09-19-16-57-59/test_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()