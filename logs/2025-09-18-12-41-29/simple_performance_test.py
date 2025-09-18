"""
Simple Performance Testing for XAttention Implementation
Focuses on core functionality without complex weight sharing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import math
from typing import Dict, List, Tuple
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_attention import BaselineAttention
from xattention import XAttention, XAttentionOptimized


class SimplePerformanceTester:
    """Simple performance testing suite"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def generate_test_samples(self) -> List[Dict]:
        """Generate test samples"""
        samples = []
        
        # Test configurations based on paper experiments
        configs = [
            {"seq_len": 512, "dim": 512, "heads": 8, "name": "Small-512"},
            {"seq_len": 1024, "dim": 768, "heads": 12, "name": "Medium-1K"},
            {"seq_len": 2048, "dim": 1024, "heads": 16, "name": "Large-2K"},
            {"seq_len": 4096, "dim": 1024, "heads": 16, "name": "XL-4K"},
        ]
        
        # Add larger configs for GPU
        if self.device == 'cuda':
            configs.extend([
                {"seq_len": 8192, "dim": 1024, "heads": 16, "name": "XXL-8K"},
                {"seq_len": 16384, "dim": 1024, "heads": 16, "name": "Huge-16K"},
            ])
        
        for config in configs:
            samples.append({
                "batch_size": 1,
                "seq_len": config["seq_len"],
                "dim": config["dim"],
                "num_heads": config["heads"],
                "name": config["name"]
            })
        
        return samples
    
    def create_models(self, config: Dict) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Create models with same initialization"""
        dim = config["dim"]
        num_heads = config["num_heads"]
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create models
        baseline = BaselineAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
            causal=True
        ).to(self.device)
        
        xattention = XAttention(
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
        
        return baseline, xattention, xattention_opt
    
    def measure_runtime(self, model: nn.Module, x: torch.Tensor, num_runs: int = 3) -> Dict:
        """Measure runtime"""
        model.eval()
        
        # Warmup
        try:
            with torch.no_grad():
                for _ in range(2):
                    _ = model(x)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return {"avg_time_ms": None, "error": "OOM"}
            else:
                raise
        
        # Measure time
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
            
            avg_time = start.elapsed_time(end) / num_runs
            
        else:
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(x)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / num_runs * 1000
        
        return {"avg_time_ms": avg_time}
    
    def measure_accuracy(self, model: nn.Module, x: torch.Tensor) -> Dict:
        """Measure output characteristics"""
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        stats = {
            "mean": float(output.mean()),
            "std": float(output.std()),
            "norm": float(torch.norm(output)),
            "shape": list(output.shape)
        }
        
        if hasattr(model, 'get_sparsity_stats'):
            sparsity_stats = model.get_sparsity_stats()
            stats.update(sparsity_stats)
        
        return stats
    
    def run_single_test(self, config: Dict) -> Dict:
        """Run test for single configuration"""
        print(f"Testing {config['name']}: {config['seq_len']} tokens")
        
        B, L, D = config["batch_size"], config["seq_len"], config["dim"]
        
        try:
            # Create input
            torch.manual_seed(123)
            x = torch.randn(B, L, D, device=self.device)
            
            # Create models
            baseline, xattn, xattn_opt = self.create_models(config)
            
            # Measure runtimes
            baseline_result = self.measure_runtime(baseline, x)
            if baseline_result["avg_time_ms"] is None:
                return {"config": config, "error": "OOM"}
            
            xattn_result = self.measure_runtime(xattn, x)
            xattn_opt_result = self.measure_runtime(xattn_opt, x)
            
            # Measure outputs
            baseline_stats = self.measure_accuracy(baseline, x)
            xattn_stats = self.measure_accuracy(xattn, x)
            xattn_opt_stats = self.measure_accuracy(xattn_opt, x)
            
            # Compute ratios
            xattn_speedup = baseline_result["avg_time_ms"] / xattn_result["avg_time_ms"]
            xattn_opt_speedup = baseline_result["avg_time_ms"] / xattn_opt_result["avg_time_ms"]
            
            # Output similarity
            baseline_norm = baseline_stats["norm"]
            xattn_similarity = 1.0 - abs(baseline_norm - xattn_stats["norm"]) / baseline_norm
            xattn_opt_similarity = 1.0 - abs(baseline_norm - xattn_opt_stats["norm"]) / baseline_norm
            
            result = {
                "config": config,
                "runtimes": {
                    "baseline_ms": baseline_result["avg_time_ms"],
                    "xattention_ms": xattn_result["avg_time_ms"],
                    "xattention_opt_ms": xattn_opt_result["avg_time_ms"]
                },
                "speedups": {
                    "xattention_vs_baseline": xattn_speedup,
                    "xattention_opt_vs_baseline": xattn_opt_speedup
                },
                "accuracy": {
                    "xattention_vs_baseline": xattn_similarity,
                    "xattention_opt_vs_baseline": xattn_opt_similarity
                },
                "sparsity": xattn_stats.get("sparsity", 0.0)
            }
            
            return result
            
        except Exception as e:
            return {"config": config, "error": str(e)}
    
    def run_all_tests(self) -> Dict:
        """Run all tests"""
        print("Starting performance testing...")
        print(f"Device: {self.device}")
        print("=" * 50)
        
        samples = self.generate_test_samples()
        print(f"Testing {len(samples)} configurations")
        
        results = []
        
        for sample in samples:
            try:
                result = self.run_single_test(sample)
                if "error" not in result:
                    results.append(result)
                    print(f"✓ {sample['name']}: {result['speedups']['xattention_vs_baseline']:.2f}x speedup")
                else:
                    print(f"⚠ {sample['name']}: {result['error']}")
            except Exception as e:
                print(f"❌ {sample['name']}: {e}")
        
        # Compute geometric means
        if results:
            def geom_mean(values):
                return math.exp(sum(math.log(v) for v in values) / len(values))
            
            xattn_speedups = [r["speedups"]["xattention_vs_baseline"] for r in results]
            xattn_opt_speedups = [r["speedups"]["xattention_opt_vs_baseline"] for r in results]
            xattn_accuracies = [r["accuracy"]["xattention_vs_baseline"] for r in results]
            xattn_opt_accuracies = [r["accuracy"]["xattention_opt_vs_baseline"] for r in results]
            
            summary = {
                "paper_title": "XAttention: Block Sparse Attention with Antidiagonal Scoring",
                "device": self.device,
                "total_tests": len(results),
                "geometric_means": {
                    "xattention_speedup": geom_mean(xattn_speedups),
                    "xattention_opt_speedup": geom_mean(xattn_opt_speedups),
                    "xattention_accuracy": geom_mean(xattn_accuracies),
                    "xattention_opt_accuracy": geom_mean(xattn_opt_accuracies)
                },
                "detailed_results": results
            }
        else:
            summary = {
                "paper_title": "XAttention: Block Sparse Attention with Antidiagonal Scoring",
                "device": self.device,
                "total_tests": 0,
                "error": "No successful tests"
            }
        
        return summary


def main():
    """Main function"""
    tester = SimplePerformanceTester()
    results = tester.run_all_tests()
    
    # Save results
    output_path = "/home/wzc/data/file-share/logs/2025-09-18-12-41-29/simple_performance_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Paper: {results['paper_title']}")
    print(f"Device: {results['device']}")
    print(f"Successful tests: {results['total_tests']}")
    
    if results['total_tests'] > 0:
        gm = results['geometric_means']
        print(f"\nGeometric Mean Performance:")
        print(f"  XAttention Speedup: {gm['xattention_speedup']:.2f}x")
        print(f"  XAttentionOptimized Speedup: {gm['xattention_opt_speedup']:.2f}x")
        print(f"  XAttention Accuracy: {gm['xattention_accuracy']:.4f}")
        print(f"  XAttentionOptimized Accuracy: {gm['xattention_opt_accuracy']:.4f}")
    
    print(f"\nResults saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()