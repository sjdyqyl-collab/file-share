"""
Comprehensive Performance Testing for XAttention Implementation
Tests baseline attention vs XAttention vs XAttentionOptimized
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import numpy as np
import gc
from typing import Dict, List, Tuple
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_attention import BaselineAttention, BaselineAttentionFlash
from xattention import XAttention, XAttentionOptimized


class PerformanceTester:
    """Comprehensive performance testing suite"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def generate_test_samples(self) -> List[Dict]:
        """Generate test samples based on paper experiments"""
        samples = []
        
        # Language modeling tasks
        language_configs = [
            {"seq_len": 512, "dim": 512, "heads": 8, "name": "Language-512"},
            {"seq_len": 1024, "dim": 768, "heads": 12, "name": "Language-1K"},
            {"seq_len": 2048, "dim": 1024, "heads": 16, "name": "Language-2K"},
            {"seq_len": 4096, "dim": 1024, "heads": 16, "name": "Language-4K"},
            {"seq_len": 8192, "dim": 1024, "heads": 16, "name": "Language-8K"},
            {"seq_len": 16384, "dim": 1024, "heads": 16, "name": "Language-16K"},
            {"seq_len": 32768, "dim": 1024, "heads": 16, "name": "Language-32K"},
            {"seq_len": 65536, "dim": 1024, "heads": 16, "name": "Language-64K"},
        ]
        
        # Video understanding tasks
        video_configs = [
            {"seq_len": 1024, "dim": 768, "heads": 12, "name": "Video-1K"},
            {"seq_len": 4096, "dim": 1024, "heads": 16, "name": "Video-4K"},
            {"seq_len": 8192, "dim": 1024, "heads": 16, "name": "Video-8K"},
            {"seq_len": 16384, "dim": 1024, "heads": 16, "name": "Video-16K"},
        ]
        
        # Generation tasks
        generation_configs = [
            {"seq_len": 2048, "dim": 1024, "heads": 16, "name": "Generation-2K"},
            {"seq_len": 8192, "dim": 1024, "heads": 16, "name": "Generation-8K"},
            {"seq_len": 16384, "dim": 1024, "heads": 16, "name": "Generation-16K"},
            {"seq_len": 32768, "dim": 1024, "heads": 16, "name": "Generation-32K"},
        ]
        
        # Combine all configurations
        all_configs = language_configs + video_configs + generation_configs
        
        for config in all_configs:
            # Skip very large sequences on CPU
            if self.device == 'cpu' and config["seq_len"] > 16384:
                continue
                
            samples.append({
                "batch_size": 1,  # Single batch for consistent testing
                "seq_len": config["seq_len"],
                "dim": config["dim"],
                "num_heads": config["heads"],
                "name": config["name"]
            })
        
        return samples
    
    def create_models(self, config: Dict) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Create models with consistent weights"""
        dim = config["dim"]
        num_heads = config["num_heads"]
        
        # Create baseline
        baseline = BaselineAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
            causal=True
        ).to(self.device)
        
        # Create XAttention
        xattention = XAttention(
            dim=dim,
            num_heads=num_heads,
            block_size=8,
            stride=8,
            threshold=0.9,
            qkv_bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
            causal=True,
            use_dynamic_threshold=True
        ).to(self.device)
        
        # Create XAttentionOptimized
        xattention_opt = XAttentionOptimized(
            dim=dim,
            num_heads=num_heads,
            block_size=8,
            stride=8,
            threshold=0.9,
            qkv_bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
            causal=True
        ).to(self.device)
        
        # Ensure consistent weights across all models
        state_dict = baseline.save_weights()
        xattention.load_weights(state_dict)
        xattention_opt.load_weights(state_dict)
        
        return baseline, xattention, xattention_opt
    
    def measure_runtime(self, model: nn.Module, x: torch.Tensor, num_runs: int = 10) -> Dict:
        """Measure runtime statistics"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(x)
            end.record()
            torch.cuda.synchronize()
            
            avg_time = start.elapsed_time(end) / num_runs  # milliseconds
            
        else:
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000  # milliseconds
        
        return {"avg_time_ms": avg_time}
    
    def measure_accuracy(self, model: nn.Module, x: torch.Tensor) -> Dict:
        """Measure output accuracy characteristics"""
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        # Basic statistics
        stats = {
            "mean": float(output.mean()),
            "std": float(output.std()),
            "min": float(output.min()),
            "max": float(output.max()),
            "norm": float(torch.norm(output))
        }
        
        # For XAttention, get sparsity stats
        if hasattr(model, 'get_sparsity_stats'):
            sparsity_stats = model.get_sparsity_stats()
            stats.update(sparsity_stats)
        
        return stats
    
    def run_single_test(self, config: Dict) -> Dict:
        """Run complete test for a single configuration"""
        print(f"Testing {config['name']}: {config['seq_len']} tokens, {config['dim']} dim")
        
        # Create test input
        B = config["batch_size"]
        L = config["seq_len"]
        D = config["dim"]
        
        x = torch.randn(B, L, D, device=self.device)
        
        # Create models
        baseline, xattention, xattention_opt = self.create_models(config)
        
        # Measure runtimes
        baseline_runtime = self.measure_runtime(baseline, x)
        xattention_runtime = self.measure_runtime(xattention, x)
        xattention_opt_runtime = self.measure_runtime(xattention_opt, x)
        
        # Measure accuracy
        baseline_output = self.measure_accuracy(baseline, x)
        xattention_output = self.measure_accuracy(xattention, x)
        xattention_opt_output = self.measure_accuracy(xattention_opt, x)
        
        # Compute ratios
        runtime_ratios = {
            "xattention_vs_baseline": baseline_runtime["avg_time_ms"] / xattention_runtime["avg_time_ms"],
            "xattention_opt_vs_baseline": baseline_runtime["avg_time_ms"] / xattention_opt_runtime["avg_time_ms"],
            "xattention_opt_vs_xattention": xattention_runtime["avg_time_ms"] / xattention_opt_runtime["avg_time_ms"]
        }
        
        # Compute output similarity (using norm difference as proxy)
        baseline_norm = baseline_output["norm"]
        xattention_norm_diff = abs(baseline_norm - xattention_output["norm"]) / baseline_norm
        xattention_opt_norm_diff = abs(baseline_norm - xattention_opt_output["norm"]) / baseline_norm
        
        output_ratios = {
            "xattention_vs_baseline": 1.0 - xattention_norm_diff,
            "xattention_opt_vs_baseline": 1.0 - xattention_opt_norm_diff
        }
        
        result = {
            "config": config,
            "runtimes": {
                "baseline_ms": baseline_runtime["avg_time_ms"],
                "xattention_ms": xattention_runtime["avg_time_ms"],
                "xattention_opt_ms": xattention_opt_runtime["avg_time_ms"]
            },
            "runtime_ratios": runtime_ratios,
            "output_stats": {
                "baseline": baseline_output,
                "xattention": xattention_output,
                "xattention_opt": xattention_opt_output
            },
            "output_ratios": output_ratios
        }
        
        # Cleanup
        del baseline, xattention, xattention_opt, x
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return result
    
    def run_all_tests(self) -> Dict:
        """Run all tests and compute aggregate statistics"""
        print("Starting comprehensive performance testing...")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Generate test samples
        samples = self.generate_test_samples()
        print(f"Generated {len(samples)} test configurations")
        
        results = []
        
        for sample in samples:
            try:
                result = self.run_single_test(sample)
                results.append(result)
                print(f"✓ Completed {sample['name']}")
            except Exception as e:
                print(f"❌ Failed {sample['name']}: {e}")
                continue
        
        # Compute geometric means
        xattention_speedups = [r["runtime_ratios"]["xattention_vs_baseline"] for r in results]
        xattention_opt_speedups = [r["runtime_ratios"]["xattention_opt_vs_baseline"] for r in results]
        
        xattention_accuracy = [r["output_ratios"]["xattention_vs_baseline"] for r in results]
        xattention_opt_accuracy = [r["output_ratios"]["xattention_opt_vs_baseline"] for r in results]
        
        def geometric_mean(values):
            return np.exp(np.mean(np.log(values)))
        
        summary = {
            "paper_title": "XAttention: Block Sparse Attention with Antidiagonal Scoring",
            "device": self.device,
            "total_tests": len(results),
            "geometric_means": {
                "xattention_speedup": geometric_mean(xattention_speedups),
                "xattention_opt_speedup": geometric_mean(xattention_opt_speedups),
                "xattention_accuracy": geometric_mean(xattention_accuracy),
                "xattention_opt_accuracy": geometric_mean(xattention_opt_accuracy)
            },
            "detailed_results": results
        }
        
        return summary


def main():
    """Main testing function"""
    tester = PerformanceTester()
    results = tester.run_all_tests()
    
    # Save results
    output_path = "/home/wzc/data/file-share/logs/2025-09-18-12-41-29/performance_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTING SUMMARY")
    print("=" * 60)
    print(f"Paper: {results['paper_title']}")
    print(f"Device: {results['device']}")
    print(f"Total tests: {results['total_tests']}")
    print()
    print("Geometric Mean Performance:")
    print(f"  XAttention Speedup: {results['geometric_means']['xattention_speedup']:.2f}x")
    print(f"  XAttentionOptimized Speedup: {results['geometric_means']['xattention_opt_speedup']:.2f}x")
    print(f"  XAttention Accuracy: {results['geometric_means']['xattention_accuracy']:.4f}")
    print(f"  XAttentionOptimized Accuracy: {results['geometric_means']['xattention_opt_accuracy']:.4f}")
    print()
    print(f"Results saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    main()