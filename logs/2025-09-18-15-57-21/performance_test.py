#!/usr/bin/env python3
"""
Comprehensive performance testing for DraftAttention methods.
Tests three methods:
1. Baseline: Standard full attention (initial method)
2. DraftAttention: Original proposed method
3. AdaptiveDraftAttention: Enhanced method
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import numpy as np
from typing import Dict, List, Tuple
import os


class BaselineAttention(nn.Module):
    """Standard full attention as the baseline initial method."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        """
        Standard full attention computation.
        
        Args:
            x: Input tensor (B, N, D)
            
        Returns:
            Output tensor (B, N, D)
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Standard full attention
        scores = torch.bmm(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn_weights, V)
        return self.out_proj(out)


class DraftAttention(nn.Module):
    """Original DraftAttention method from the paper."""
    
    def __init__(self, hidden_dim: int, sparsity_ratio: float = 0.75):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sparsity_ratio = sparsity_ratio
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, pool_factor: int = 8):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, N, D)
            pool_factor: Pooling factor for draft attention
            
        Returns:
            Output tensor (B, N, D)
        """
        B, N, D = x.shape
        
        # Ensure N is divisible by pool_factor
        actual_N = (N // pool_factor) * pool_factor
        if actual_N < N:
            x = x[:, :actual_N, :]
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Draft attention computation
        draft_len = actual_N // pool_factor
        
        # Pool queries and keys
        q_draft = Q.view(B, draft_len, pool_factor, D).mean(dim=2)
        k_draft = K.view(B, draft_len, pool_factor, D).mean(dim=2)
        
        # Compute draft attention
        draft_scores = torch.bmm(q_draft, k_draft.transpose(-2, -1)) / (D ** 0.5)
        draft_attn = F.softmax(draft_scores, dim=-1)
        
        # Create sparsity mask
        num_keep = int(self.sparsity_ratio * draft_len * draft_len)
        flat_attn = draft_attn.view(B, -1)
        _, top_indices = torch.topk(flat_attn, num_keep, dim=-1)
        
        mask = torch.zeros_like(flat_attn)
        mask.scatter_(1, top_indices, 1.0)
        mask = mask.view(B, draft_len, draft_len)
        
        # Expand mask to full resolution
        mask_full = mask.repeat_interleave(pool_factor, dim=1).repeat_interleave(pool_factor, dim=2)
        mask_full = mask_full[:, :N, :N]
        
        # Compute sparse attention
        full_scores = torch.bmm(Q, K.transpose(-2, -1)) / (D ** 0.5)
        masked_scores = full_scores.masked_fill(mask_full == 0, float('-inf'))
        attn_weights = F.softmax(masked_scores, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn_weights, V)
        return self.out_proj(out)


class AdaptiveDraftAttention(nn.Module):
    """Enhanced AdaptiveDraftAttention method."""
    
    def __init__(self, hidden_dim: int, base_sparsity: float = 0.75):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_sparsity = base_sparsity
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Adaptive components
        self.complexity_fc = nn.Linear(hidden_dim, 1)
        self.timestep_fc = nn.Linear(1, 1)
        
    def forward(self, x, timestep: float = 0.5, min_pool: int = 4, max_pool: int = 16):
        """
        Forward pass with adaptive improvements.
        
        Args:
            x: Input tensor (B, N, D)
            timestep: Current denoising step [0, 1]
            min_pool: Minimum pooling factor
            max_pool: Maximum pooling factor
            
        Returns:
            Output tensor (B, N, D)
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Adaptive pooling factor
        content_complexity = torch.sigmoid(self.complexity_fc(Q.mean(dim=1))).mean()
        pool_factor = int(min_pool + (max_pool - min_pool) * (1 - content_complexity.item()))
        pool_factor = max(min_pool, min(pool_factor, N // min_pool))
        
        # Ensure N is divisible by pool_factor
        actual_N = (N // pool_factor) * pool_factor
        if actual_N < N:
            Q = Q[:, :actual_N, :]
            K = K[:, :actual_N, :]
            V = V[:, :actual_N, :]
        
        # Dynamic sparsity
        timestep_tensor = torch.tensor([[timestep]], device=x.device, dtype=x.dtype)
        sparsity_factor = torch.sigmoid(self.timestep_fc(timestep_tensor)).item()
        dynamic_sparsity = self.base_sparsity * (0.8 + 0.2 * sparsity_factor)
        
        # Draft attention
        draft_len = actual_N // pool_factor
        q_draft = Q.view(B, draft_len, pool_factor, D).mean(dim=2)
        k_draft = K.view(B, draft_len, pool_factor, D).mean(dim=2)
        
        draft_scores = torch.bmm(q_draft, k_draft.transpose(-2, -1)) / (D ** 0.5)
        draft_attn = F.softmax(draft_scores, dim=-1)
        
        # Create sparsity mask
        num_keep = int(dynamic_sparsity * draft_len * draft_len)
        flat_attn = draft_attn.view(B, -1)
        _, top_indices = torch.topk(flat_attn, num_keep, dim=-1)
        
        mask = torch.zeros_like(flat_attn)
        mask.scatter_(1, top_indices, 1.0)
        mask = mask.view(B, draft_len, draft_len)
        
        # Expand mask
        mask_full = mask.repeat_interleave(pool_factor, dim=1).repeat_interleave(pool_factor, dim=2)
        mask_full = mask_full[:, :N, :N]
        
        # Compute sparse attention
        full_scores = torch.bmm(Q, K.transpose(-2, -1)) / (D ** 0.5)
        masked_scores = full_scores.masked_fill(mask_full == 0, float('-inf'))
        attn_weights = F.softmax(masked_scores, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn_weights, V)
        return self.out_proj(out)


class PerformanceTester:
    """Comprehensive performance testing class."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {self.device}")
        
    def generate_test_samples(self) -> List[Dict]:
        """Generate diverse test samples based on paper specifications."""
        samples = []
        
        # Video diffusion typical configurations
        configs = [
            # Small-scale tests
            {"batch": 1, "seq_len": 256, "hidden_dim": 256, "name": "small_256"},
            {"batch": 2, "seq_len": 512, "hidden_dim": 512, "name": "medium_512"},
            {"batch": 1, "seq_len": 1024, "hidden_dim": 512, "name": "large_1024"},
            
            # Video-specific configurations
            {"batch": 1, "seq_len": 4096, "hidden_dim": 768, "name": "video_4k"},
            {"batch": 1, "seq_len": 8192, "hidden_dim": 1024, "name": "video_8k"},
            
            # Edge cases
            {"batch": 4, "seq_len": 128, "hidden_dim": 256, "name": "batch_4x128"},
            {"batch": 1, "seq_len": 2048, "hidden_dim": 256, "name": "long_seq"},
        ]
        
        for config in configs:
            # Generate random input
            x = torch.randn(config["batch"], config["seq_len"], config["hidden_dim"])
            
            samples.append({
                "config": config,
                "input": x,
                "expected_shape": (config["batch"], config["seq_len"], config["hidden_dim"])
            })
        
        return samples
    
    def test_method(self, model_class, test_input, **kwargs) -> Dict:
        """Test a single method and return performance metrics."""
        model = model_class(hidden_dim=test_input.shape[-1])
        model.to(self.device)
        model.eval()
        
        test_input = test_input.to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(test_input, **kwargs)
        
        # Time measurement
        times = []
        outputs = []
        
        num_runs = 10
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(test_input, **kwargs)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            outputs.append(output.cpu())
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "output": outputs[0],  # First output for comparison
            "output_shape": tuple(outputs[0].shape)
        }
    
    def compute_geometric_mean(self, ratios: List[float]) -> float:
        """Compute geometric mean of ratios."""
        return np.exp(np.mean(np.log(ratios)))
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive performance tests."""
        print("=== Comprehensive Performance Testing ===\n")
        
        test_samples = self.generate_test_samples()
        results = {
            "paper_title": "DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance",
            "device": str(self.device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": []
        }
        
        all_runtime_ratios = []
        all_output_ratios = []
        
        for sample in test_samples:
            config = sample["config"]
            test_input = sample["input"]
            expected_shape = sample["expected_shape"]
            
            print(f"Testing {config['name']}: {config['batch']}x{config['seq_len']}x{config['hidden_dim']}")
            
            # Test all three methods
            baseline_result = self.test_method(BaselineAttention, test_input)
            draft_result = self.test_method(DraftAttention, test_input, pool_factor=8)
            adaptive_result = self.test_method(AdaptiveDraftAttention, test_input, timestep=0.5)
            
            # Verify output shapes
            assert baseline_result["output_shape"] == expected_shape
            assert draft_result["output_shape"] == expected_shape
            assert adaptive_result["output_shape"] == expected_shape
            
            # Compute ratios
            draft_speedup = baseline_result["avg_time"] / draft_result["avg_time"]
            adaptive_speedup = baseline_result["avg_time"] / adaptive_result["avg_time"]
            
            # Compute output similarity (using L2 norm)
            baseline_output = baseline_result["output"]
            draft_output = draft_result["output"]
            adaptive_output = adaptive_result["output"]
            
            draft_similarity = 1.0 - torch.norm(baseline_output - draft_output) / torch.norm(baseline_output)
            adaptive_similarity = 1.0 - torch.norm(baseline_output - adaptive_output) / torch.norm(baseline_output)
            
            # Store ratios for geometric mean
            all_runtime_ratios.extend([draft_speedup, adaptive_speedup])
            all_output_ratios.extend([draft_similarity.item(), adaptive_similarity.item()])
            
            test_result = {
                "config_name": config["name"],
                "input_shape": expected_shape,
                "baseline": {
                    "avg_time_ms": baseline_result["avg_time"] * 1000,
                    "std_time_ms": baseline_result["std_time"] * 1000
                },
                "draft_attention": {
                    "avg_time_ms": draft_result["avg_time"] * 1000,
                    "std_time_ms": draft_result["std_time"] * 1000,
                    "speedup": draft_speedup,
                    "output_similarity": draft_similarity.item()
                },
                "adaptive_draft_attention": {
                    "avg_time_ms": adaptive_result["avg_time"] * 1000,
                    "std_time_ms": adaptive_result["std_time"] * 1000,
                    "speedup": adaptive_speedup,
                    "output_similarity": adaptive_similarity.item()
                }
            }
            
            results["tests"].append(test_result)
            
            print(f"  Baseline: {baseline_result['avg_time']*1000:.2f}ms")
            print(f"  Draft: {draft_result['avg_time']*1000:.2f}ms ({draft_speedup:.2f}x)")
            print(f"  Adaptive: {adaptive_result['avg_time']*1000:.2f}ms ({adaptive_speedup:.2f}x)")
            print()
        
        # Compute geometric means
        draft_runtime_ratios = [t["draft_attention"]["speedup"] for t in results["tests"]]
        adaptive_runtime_ratios = [t["adaptive_draft_attention"]["speedup"] for t in results["tests"]]
        draft_output_ratios = [t["draft_attention"]["output_similarity"] for t in results["tests"]]
        adaptive_output_ratios = [t["adaptive_draft_attention"]["output_similarity"] for t in results["tests"]]
        
        results["summary"] = {
            "geometric_mean_draft_runtime_ratio": self.compute_geometric_mean(draft_runtime_ratios),
            "geometric_mean_adaptive_runtime_ratio": self.compute_geometric_mean(adaptive_runtime_ratios),
            "geometric_mean_draft_output_ratio": self.compute_geometric_mean(draft_output_ratios),
            "geometric_mean_adaptive_output_ratio": self.compute_geometric_mean(adaptive_output_ratios),
            "total_tests": len(results["tests"])
        }
        
        return results


def main():
    """Main testing function."""
    tester = PerformanceTester()
    results = tester.run_comprehensive_test()
    
    # Save results
    output_path = "/home/wzc/data/file-share/logs/2025-09-18-15-57-21/performance_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=== Test Summary ===")
    print(f"DraftAttention geometric mean speedup: {results['summary']['geometric_mean_draft_runtime_ratio']:.2f}x")
    print(f"AdaptiveDraftAttention geometric mean speedup: {results['summary']['geometric_mean_adaptive_runtime_ratio']:.2f}x")
    print(f"DraftAttention geometric mean output similarity: {results['summary']['geometric_mean_draft_output_ratio']:.4f}")
    print(f"AdaptiveDraftAttention geometric mean output similarity: {results['summary']['geometric_mean_adaptive_output_ratio']:.4f}")
    print(f"\nResults saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    main()