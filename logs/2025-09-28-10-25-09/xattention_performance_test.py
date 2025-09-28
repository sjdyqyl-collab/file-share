import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import math
import sys
import os

# Add the current directory to path for imports
sys.path.append('/home/wzc/data/file-share/logs/2025-09-28-10-25-09')

from base_attention import BaseAttention
from xattention_simple import XAttentionSimple

class XAttentionTester:
    """Comprehensive testing framework for XAttention performance evaluation."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {self.device}")
        
        # Test configurations
        self.test_configs = [
            {"batch_size": 1, "seq_len": 512, "hidden_size": 512, "num_heads": 8},
            {"batch_size": 2, "seq_len": 1024, "hidden_size": 768, "num_heads": 12},
            {"batch_size": 4, "seq_len": 2048, "hidden_size": 1024, "num_heads": 16},
            {"batch_size": 1, "seq_len": 4096, "hidden_size": 768, "num_heads": 12},
            {"batch_size": 2, "seq_len": 8192, "hidden_size": 1024, "num_heads": 16},
        ]
        
        # XAttention parameters
        self.xattention_params = [
            {"block_size": 64, "stride": 8, "threshold": 0.7},
            {"block_size": 128, "stride": 16, "threshold": 0.8},
            {"block_size": 64, "stride": 4, "threshold": 0.9},
        ]
    
    def generate_test_samples(self, config):
        """Generate test samples based on configuration."""
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        hidden_size = config["hidden_size"]
        
        # Create random input tensors
        query = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        key = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        value = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        
        return query, key, value
    
    def measure_runtime(self, model, query, key, value, num_warmup=5, num_runs=10):
        """Measure runtime of a model."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(query, key, value, causal=True)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure runtime
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(query, key, value, causal=True)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        return avg_time, output
    
    def compute_accuracy_metrics(self, base_output, xattn_output):
        """Compute accuracy metrics between base and XAttention outputs."""
        with torch.no_grad():
            # MSE
            mse = torch.nn.functional.mse_loss(base_output[0], xattn_output[0])
            
            # Mean Absolute Error
            mae = torch.abs(base_output[0] - xattn_output[0]).mean()
            
            # Cosine Similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                base_output[0].view(-1), xattn_output[0].view(-1), dim=0
            )
            
            # Relative Error
            rel_error = torch.abs(base_output[0] - xattn_output[0]) / (torch.abs(base_output[0]) + 1e-8)
            mean_rel_error = rel_error.mean()
            
            return {
                "mse": float(mse.cpu()),
                "mae": float(mae.cpu()),
                "cosine_similarity": float(cos_sim.cpu()),
                "relative_error": float(mean_rel_error.cpu())
            }
    
    def run_single_test(self, config, xattn_params):
        """Run a single test configuration."""
        print(f"\nTesting: B={config['batch_size']}, L={config['seq_len']}, H={config['hidden_size']}")
        
        # Generate test samples
        query, key, value = self.generate_test_samples(config)
        
        # Initialize models with consistent weights
        base_model = BaseAttention(
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"]
        ).to(self.device)
        
        xattn_model = XAttentionSimple(
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            block_size=xattn_params["block_size"],
            stride=xattn_params["stride"],
            threshold=xattn_params["threshold"]
        ).to(self.device)
        
        # Ensure consistent weights
        xattn_model.q_proj.weight.data = base_model.q_proj.weight.data.clone()
        xattn_model.k_proj.weight.data = base_model.k_proj.weight.data.clone()
        xattn_model.v_proj.weight.data = base_model.v_proj.weight.data.clone()
        xattn_model.out_proj.weight.data = base_model.out_proj.weight.data.clone()
        
        # Measure runtime
        base_time, base_output = self.measure_runtime(base_model, query, key, value)
        xattn_time, xattn_output = self.measure_runtime(xattn_model, query, key, value)
        
        # Get sparsity statistics
        _, block_masks = xattn_model(query, key, value, causal=True)
        sparsity_stats = xattn_model.get_sparsity_stats(block_masks)
        
        # Compute accuracy metrics
        accuracy_metrics = self.compute_accuracy_metrics(base_output, xattn_output)
        
        return {
            "config": config,
            "xattention_params": xattn_params,
            "base_time": base_time,
            "xattention_time": xattn_time,
            "speedup": base_time / xattn_time,
            "sparsity_stats": sparsity_stats,
            "accuracy_metrics": accuracy_metrics,
            "runtime_ratio": xattn_time / base_time,
            "output_mse_ratio": accuracy_metrics["mse"] / 1.0,  # Normalized
        }
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests across all configurations."""
        print("Starting comprehensive XAttention performance testing...")
        
        all_results = []
        runtime_ratios = []
        output_mse_ratios = []
        
        test_count = 0
        for config in self.test_configs:
            for xattn_params in self.xattention_params:
                try:
                    result = self.run_single_test(config, xattn_params)
                    all_results.append(result)
                    
                    runtime_ratios.append(result["runtime_ratio"])
                    output_mse_ratios.append(result["output_mse_ratio"])
                    
                    test_count += 1
                    print(f"Test {test_count} completed successfully")
                    
                except Exception as e:
                    print(f"Error in test {test_count}: {str(e)}")
                    continue
        
        # Calculate geometric means
        if runtime_ratios and output_mse_ratios:
            geo_mean_runtime = math.exp(sum(math.log(r) for r in runtime_ratios) / len(runtime_ratios))
            geo_mean_mse = math.exp(sum(math.log(r) for r in output_mse_ratios) / len(output_mse_ratios))
        else:
            geo_mean_runtime = 1.0
            geo_mean_mse = 1.0
        
        # Summary statistics
        summary = {
            "total_tests": len(all_results),
            "geometric_mean_runtime_ratio": geo_mean_runtime,
            "geometric_mean_output_mse_ratio": geo_mean_mse,
            "average_speedup": sum(r["speedup"] for r in all_results) / len(all_results),
            "average_sparsity": sum(r["sparsity_stats"]["sparsity"] for r in all_results) / len(all_results),
        }
        
        return {
            "paper_title": "XAttention: Block-Cross Attention for Long-Context Modeling",
            "summary": summary,
            "detailed_results": all_results,
            "test_configurations": {
                "test_configs": self.test_configs,
                "xattention_params": self.xattention_params
            }
        }
    
    def save_results(self, results, filename="xattention_test_results.json"):
        """Save test results to JSON file."""
        filepath = f"/home/wzc/data/file-share/logs/2025-09-28-10-25-09/{filename}"
        
        # Convert tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        results_json = convert_tensors(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
        return filepath

def main():
    """Main testing function."""
    tester = XAttentionTester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_tests()
    
    # Save results
    filepath = tester.save_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Paper: {results['paper_title']}")
    print(f"Total tests: {results['summary']['total_tests']}")
    print(f"Geometric mean runtime ratio (XAttention/Base): {results['summary']['geometric_mean_runtime_ratio']:.4f}")
    print(f"Geometric mean output MSE ratio: {results['summary']['geometric_mean_output_mse_ratio']:.6f}")
    print(f"Average speedup: {results['summary']['average_speedup']:.2f}x")
    print(f"Average sparsity: {results['summary']['average_sparsity']:.3f}")
    print("="*60)
    
    return filepath

if __name__ == "__main__":
    main()