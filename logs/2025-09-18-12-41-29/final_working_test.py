"""
Final Working Performance Test for XAttention
"""

import torch
import torch.nn as nn
import time
import json
import math
from typing import Dict, List
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_attention import BaselineAttention
from xattention_fixed import XAttentionFixed, XAttentionOptimizedFixed


class FinalPerformanceTester:
    """Final working performance tester"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_test_samples(self) -> List[Dict]:
        """Generate test samples"""
        samples = []
        
        configs = [
            {"seq_len": 512, "dim": 512, "heads": 8, "name": "Small"},
            {"seq_len": 1024, "dim": 768, "heads": 12, "name": "Medium"},
            {"seq_len": 2048, "dim": 1024, "heads": 16, "name": "Large"},
            {"seq_len": 4096, "dim": 1024, "heads": 16, "name": "XL"},
        ]
        
        if self.device == 'cuda':
            configs.extend([
                {"seq_len": 8192, "dim": 1024, "heads": 16, "name": "XXL"},
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
    
    def create_models(self, config: Dict):
        """Create models"""
        dim = config["dim"]
        num_heads = config["num_heads"]
        
        torch.manual_seed(42)
        
        baseline = BaselineAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
            causal=True
        ).to(self.device)
        
        xattn = XAttentionFixed(
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
        
        xattn_opt = XAttentionOptimizedFixed(
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
        
        return baseline, xattn, xattn_opt
    
    def measure_runtime(self, model, x, num_runs=3):
        """Measure runtime"""
        model.eval()
        
        try:
            # Warmup
            with torch.no_grad():
                for _ in range(2):
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
                
                avg_time = start.elapsed_time(end) / num_runs
            else:
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = model(x)
                end = time.perf_counter()
                avg_time = (end - start) / num_runs * 1000
            
            return {"avg_time_ms": avg_time, "success": True}
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return {"avg_time_ms": None, "success": False, "error": "OOM"}
            else:
                raise
    
    def run_test(self, config):
        """Run single test"""
        print(f"Testing {config['name']} ({config['seq_len']} tokens)")
        
        B, L, D = config["batch_size"], config["seq_len"], config["dim"]
        
        try:
            x = torch.randn(B, L, D, device=self.device)
            baseline, xattn, xattn_opt = self.create_models(config)
            
            # Test baseline
            baseline_result = self.measure_runtime(baseline, x)
            if not baseline_result["success"]:
                return {"config": config, "error": baseline_result["error"]}
            
            # Test XAttention
            xattn_result = self.measure_runtime(xattn, x)
            if not xattn_result["success"]:
                return {"config": config, "error": xattn_result["error"]}
            
            # Test XAttentionOptimized
            xattn_opt_result = self.measure_runtime(xattn_opt, x)
            if not xattn_opt_result["success"]:
                return {"config": config, "error": xattn_opt_result["error"]}
            
            # Measure outputs
            with torch.no_grad():
                baseline_out = baseline(x)
                xattn_out = xattn(x)
                xattn_opt_out = xattn_opt(x)
            
            # Compute similarity
            baseline_norm = torch.norm(baseline_out).item()
            xattn_sim = 1.0 - abs(baseline_norm - torch.norm(xattn_out).item()) / baseline_norm
            xattn_opt_sim = 1.0 - abs(baseline_norm - torch.norm(xattn_opt_out).item()) / baseline_norm
            
            # Get sparsity
            sparsity_stats = xattn.get_sparsity_stats()
            
            return {
                "config": config,
                "runtimes": {
                    "baseline_ms": baseline_result["avg_time_ms"],
                    "xattention_ms": xattn_result["avg_time_ms"],
                    "xattention_opt_ms": xattn_opt_result["avg_time_ms"]
                },
                "speedups": {
                    "xattention_vs_baseline": baseline_result["avg_time_ms"] / xattn_result["avg_time_ms"],
                    "xattention_opt_vs_baseline": baseline_result["avg_time_ms"] / xattn_opt_result["avg_time_ms"]
                },
                "accuracy": {
                    "xattention_vs_baseline": max(0.0, xattn_sim),
                    "xattention_opt_vs_baseline": max(0.0, xattn_opt_sim)
                },
                "sparsity": sparsity_stats.get("sparsity", 0.0)
            }
            
        except Exception as e:
            return {"config": config, "error": str(e)}
    
    def run_all_tests(self):
        """Run all tests"""
        print("Running final performance tests...")
        print(f"Device: {self.device}")
        print("=" * 40)
        
        samples = self.generate_test_samples()
        results = []
        
        for sample in samples:
            try:
                result = self.run_test(sample)
                if "error" not in result:
                    results.append(result)
                    print(f"✓ {sample['name']}: {result['speedups']['xattention_vs_baseline']:.2f}x speedup")
                else:
                    print(f"⚠ {sample['name']}: {result['error']}")
            except Exception as e:
                print(f"❌ {sample['name']}: {e}")
        
        if results:
            # Compute geometric means
            def geom_mean(values):
                return math.exp(sum(math.log(max(v, 1e-6)) for v in values) / len(values))
            
            xattn_speedups = [r["speedups"]["xattention_vs_baseline"] for r in results]
            xattn_opt_speedups = [r["speedups"]["xattention_opt_vs_baseline"] for r in results]
            xattn_accuracies = [r["accuracy"]["xattention_vs_baseline"] for r in results]
            xattn_opt_accuracies = [r["accuracy"]["xattention_opt_vs_baseline"] for r in results]
            
            summary = {
                "paper_title": "XAttention: Block Sparse Attention with Antidiagonal Scoring",
                "device": self.device,
                "total_tests": len(results),
                "tested_configs": [r["config"]["name"] for r in results],
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
    tester = FinalPerformanceTester()
    results = tester.run_all_tests()
    
    # Save results
    output_path = "/home/wzc/data/file-share/logs/2025-09-18-12-41-29/final_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 40)
    print("FINAL PERFORMANCE RESULTS")
    print("=" * 40)
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