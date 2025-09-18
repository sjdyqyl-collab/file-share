"""
Fixed Performance Testing for XAttention Implementation
Handles state dict mismatches properly
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


class FixedPerformanceTester:
    """Fixed performance testing suite that handles state dict mismatches"""
    
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
        ]
        
        # Video understanding tasks
        video_configs = [
            {"seq_len": 1024, "dim": 768, "heads": 12, "name": "Video-1K"},
            {"seq_len": 4096, "dim": 1024, "heads": 16, "name": "Video-4K"},
            {"seq_len": 8192, "dim": 1024, "heads": 16, "name": "Video-8K"},
        ]
        
        # Generation tasks
        generation_configs = [
            {"seq_len": 2048, "dim": 1024, "heads": 16, "name": "Generation-2K"},
            {"seq_len": 4096, "dim": 1024, "heads": 16, "name": "Generation-4K"},
            {"seq_len": 8192, "dim": 1024, "heads": 16, "name": "Generation-8K"},
        ]
        
        # Add larger sequences for GPU testing
        if self.device == 'cuda':
            large_configs = [
                {"seq_len": 16384, "dim": 1024, "heads": 16, "name": "Large-16K"},
                {"seq_len": 32768, "dim": 1024, "heads": 16, "name": "Large-32K"},
                {"seq_len": 65536, "dim": 1024, "heads": 16, "name": "Large-64K"},
            ]
            all_configs = language_configs + video_configs + generation_configs + large_configs
        else:
            all_configs = language_configs + video_configs + generation_configs
        
        for config in all_configs:
            samples.append({
                "batch_size": 1,
                "seq_len": config["seq_len"],
                "dim": config["dim"],
                "num_heads": config["heads"],
                "name": config["name"]
            })
        
        return samples
    
    def create_models(self, config: Dict) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Create models with consistent weights by copying compatible parameters"""
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
        
        # Copy compatible weights manually
        with torch.no_grad():
            # Copy QKV weights
            xattention.qkv.weight.data = baseline.qkv.weight.data.clone()
            xattention.qkv.bias.data = baseline.qkv.bias.data.clone()
            
            xattention_opt.qkv.weight.data = baseline.qkv.weight.data.clone()
            xattention_opt.qkv.bias.data = baseline.qkv.bias.data.clone()
            
            # Copy projection weights
            xattention.proj.weight.data = baseline.proj.weight.data.clone()
            xattention.proj.bias.data = baseline.proj.bias.data.clone()
            
            xattention_opt.proj.weight.data = baseline.proj.weight.data.clone()
            xattention_opt.proj.bias.data = baseline.proj.bias.data.clone()
        
        return baseline, xattention, xattention_opt
    
    def measure_runtime(self, model: nn.Module, x: torch.Tensor, num_runs: int = 5) -> Dict:
        """Measure runtime statistics with memory considerations"""
        model.eval()
        
        # Memory check for large sequences
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except:
                pass
        
        # Warmup
        with torch.no_grad():
            try:
                for _ in range(2):
                    _ = model(x)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    return {"avg_time_ms": float('inf'), "oom": True}
                else:
                    raise
        
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
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000
        
        return {"avg_time_ms": avg_time, "oom": False}
    
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
            "norm": float(torch.norm(output)),
            "shape": list(output.shape)
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
        
        try:
            x = torch.randn(B, L, D, device=self.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return {"config": config, "error": "OOM", "oom": True}
            else:
                raise
        
        # Create models
        baseline, xattention, xattention_opt = self.create_models(config)
        
        # Measure runtimes
        baseline_runtime = self.measure_runtime(baseline, x)
        
        if baseline_runtime.get("oom", False):
            result = {"config": config, "error": "OOM", "oom": True}
        else:
            xattention_runtime = self.measure_runtime(xattention, x)
            xattention_opt_runtime = self.measure_runtime(xattention_opt, x)
            
            # Skip if any model OOM
            if any(r.get("oom", False) for r in [baseline_runtime, xattention_runtime, xattention_opt_runtime]):
                result = {"config": config, "error": "OOM", "oom": True}
            else:
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
                
                # Compute output similarity
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
                if not result.get("oom", False):
                    results.append(result)
                    print(f"✓ Completed {sample['name']}")
                else:
                    print(f"⚠ Skipped {sample['name']} (OOM)")
            except Exception as e:
                print(f"❌ Failed {sample['name']}: {e}")
                continue
        
        if not results:
            return {
                "paper_title": "XAttention: Block Sparse Attention with Antidiagonal Scoring",
                "device": self.device,
                "total_tests": 0,
                "error": "No successful tests",
                "geometric_means": {},
                "detailed_results": []
            }
        
        # Compute geometric means
        xattention_speedups = [r["runtime_ratios"]["xattention_vs_baseline"] for r in results]
        xattention_opt_speedups = [r["runtime_ratios"]["xattention_opt_vs_baseline"] for r in results]
        
        xattention_accuracy = [r["output_ratios"]["xattention_vs_baseline"] for r in results]
        xattention_opt_accuracy = [r["output_ratios"]["xattention_opt_vs_baseline"] for r in results]
        
        def geometric_mean(values):
            return float(np.exp(np.mean(np.log(np.array(values)))))
        
        summary = {
            "paper_title": "XAttention: Block Sparse Attention with Antidiagonal Scoring",
            "device": self.device,
            "total_tests": len(results),
            "tested_configs": [r["config"]["name"] for r in results],
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
    tester = FixedPerformanceTester()
    results = tester.run_all_tests()
    
    # Save results
    output_path = "/home/wzc/data/file-share/logs/2025-09-18-12-41-29/final_performance_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTING SUMMARY")
    print("=" * 60)
    print(f"Paper: {results['paper_title']}")
    print(f"Device: {results['device']}")
    print(f"Successful tests: {results['total_tests']}")
    
    if results['total_tests'] > 0:
        print()
        print("Geometric Mean Performance:")
        print(f"  XAttention Speedup: {results['geometric_means']['xattention_speedup']:.2f}x")
        print(f"  XAttentionOptimized Speedup: {results['geometric_means']['xattention_opt_speedup']:.2f}x")
        print(f"  XAttention Accuracy: {results['geometric_means']['xattention_accuracy']:.4f}")
        print(f"  XAttentionOptimized Accuracy: {results['geometric_means']['xattention_opt_accuracy']:.4f}")
    else:
        print("No successful tests completed")
    
    print()
    print(f"Results saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    main()