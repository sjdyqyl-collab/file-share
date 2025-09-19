#!/usr/bin/env python3
"""
Comprehensive testing framework for DraftAttention methods.
Tests performance, accuracy, and runtime improvements.
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
    """Standard full attention as baseline for comparison."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        """Full attention computation."""
        B, N, D = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Full attention computation
        scores = torch.bmm(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.bmm(attn_weights, V)
        
        return self.out_proj(out)


class DraftAttention(nn.Module):
    """DraftAttention implementation as described in the paper."""
    
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
        """Draft attention computation."""
        B, N, D = x.shape
        
        # Ensure divisibility
        actual_N = (N // pool_factor) * pool_factor
        if actual_N < N:
            x = x[:, :actual_N, :]
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Draft attention
        draft_len = actual_N // pool_factor
        q_draft = Q.view(B, draft_len, pool_factor, D).mean(dim=2)
        k_draft = K.view(B, draft_len, pool_factor, D).mean(dim=2)
        
        draft_scores = torch.bmm(q_draft, k_draft.transpose(-2, -1)) / (D ** 0.5)
        draft_attn = F.softmax(draft_scores, dim=-1)
        
        # Create sparsity mask
        num_keep = int(self.sparsity_ratio * draft_len * draft_len)
        flat_attn = draft_attn.view(B, -1)
        _, top_indices = torch.topk(flat_attn, num_keep, dim=-1)
        
        mask = torch.zeros_like(flat_attn)
        mask.scatter_(1, top_indices, 1.0)
        mask = mask.view(B, draft_len, draft_len)
        
        # Expand mask
        mask_full = mask.repeat_interleave(pool_factor, dim=1).repeat_interleave(pool_factor, dim=2)
        mask_full = mask_full[:, :N, :N]
        
        # Sparse attention
        full_scores = torch.bmm(Q, K.transpose(-2, -1)) / (D ** 0.5)
        masked_scores = full_scores.masked_fill(mask_full == 0, float('-inf'))
        attn_weights = F.softmax(masked_scores, dim=-1)
        out = torch.bmm(attn_weights, V)
        
        return self.out_proj(out)


class AdaptiveDraftAttention(nn.Module):
    """Enhanced AdaptiveDraftAttention with improvements."""
    
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
        """Adaptive attention computation."""
        B, N, D = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Adaptive pooling
        content_complexity = torch.sigmoid(self.complexity_fc(Q.mean(dim=1))).mean()
        pool_factor = int(min_pool + (max_pool - min_pool) * (1 - content_complexity.item()))
        pool_factor = max(min_pool, min(pool_factor, N // min_pool))
        
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
        
        # Sparse attention
        full_scores = torch.bmm(Q, K.transpose(-2, -1)) / (D ** 0.5)
        masked_scores = full_scores.masked_fill(mask_full == 0, float('-inf'))
        attn_weights = F.softmax(masked_scores, dim=-1)
        out = torch.bmm(attn_weights, V)
        
        return self.out_proj(out)


class TestFramework:
    """Comprehensive testing framework."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def create_test_samples(self) -> List[Dict]:
        """Create diverse test samples based on paper scenarios."""
        samples = []
        
        # Video generation scenarios from the paper
        scenarios = [
            {
                'name': 'HunyuanVideo_768p_128f',
                'batch_size': 1,
                'seq_len': 128 * 48 * 64,  # 128 frames, 48x64 patches
                'hidden_dim': 768,
                'description': 'HunyuanVideo 768p resolution'
            },
            {
                'name': 'Wan2.1_768p_80f',
                'batch_size': 1,
                'seq_len': 80 * 48 * 64,   # 80 frames, 48x64 patches
                'hidden_dim': 768,
                'description': 'Wan2.1 768p resolution'
            },
            {
                'name': 'Wan2.1_512p_80f',
                'batch_size': 1,
                'seq_len': 80 * 32 * 40,   # 80 frames, 32x40 patches
                'hidden_dim': 512,
                'description': 'Wan2.1 512p resolution'
            },
            {
                'name': 'Small_Test',
                'batch_size': 2,
                'seq_len': 256,
                'hidden_dim': 256,
                'description': 'Small test case for quick validation'
            },
            {
                'name': 'Medium_Test',
                'batch_size': 1,
                'seq_len': 1024,
                'hidden_dim': 512,
                'description': 'Medium test case'
            }
        ]
        
        for scenario in scenarios:
            x = torch.randn(
                scenario['batch_size'],
                scenario['seq_len'],
                scenario['hidden_dim'],
                device=self.device
            )
            samples.append({
                'scenario': scenario,
                'input': x,
                'expected_shape': x.shape
            })
        
        return samples
    
    def measure_runtime(self, model, x, num_runs: int = 10) -> Dict:
        """Measure runtime with warm-up."""
        model.eval()
        
        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        
        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        avg_time = (time.time() - start_time) / num_runs
        return {
            'avg_time': avg_time,
            'total_time': avg_time * num_runs
        }
    
    def compute_accuracy_metrics(self, baseline_output, method_output) -> Dict:
        """Compute accuracy metrics between outputs."""
        with torch.no_grad():
            # MSE
            mse = torch.mean((baseline_output - method_output) ** 2).item()
            
            # MAE
            mae = torch.mean(torch.abs(baseline_output - method_output)).item()
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(
                baseline_output.view(-1), 
                method_output.view(-1), 
                dim=0
            ).item()
            
            # Relative error
            relative_error = (mae / torch.mean(torch.abs(baseline_output)).item()) * 100
            
            return {
                'mse': mse,
                'mae': mae,
                'cosine_similarity': cos_sim,
                'relative_error_percent': relative_error
            }
    
    def run_comprehensive_tests(self) -> Dict:
        """Run all tests and collect results."""
        print("🔬 Starting comprehensive DraftAttention testing...")
        
        samples = self.create_test_samples()
        all_results = {
            'paper_title': 'DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance',
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'tests': []
        }
        
        for sample in samples:
            scenario = sample['scenario']
            x = sample['input']
            
            print(f"\n📊 Testing scenario: {scenario['name']}")
            print(f"   Shape: {x.shape}, Device: {self.device}")
            
            # Initialize models with consistent weights
            hidden_dim = scenario['hidden_dim']
            
            baseline = BaselineAttention(hidden_dim).to(self.device)
            draft = DraftAttention(hidden_dim, sparsity_ratio=0.75).to(self.device)
            adaptive = AdaptiveDraftAttention(hidden_dim, base_sparsity=0.75).to(self.device)
            
            # Copy weights for fair comparison
            with torch.no_grad():
                draft.q_proj.weight.data = baseline.q_proj.weight.data.clone()
                draft.k_proj.weight.data = baseline.k_proj.weight.data.clone()
                draft.v_proj.weight.data = baseline.v_proj.weight.data.clone()
                draft.out_proj.weight.data = baseline.out_proj.weight.data.clone()
                
                adaptive.q_proj.weight.data = baseline.q_proj.weight.data.clone()
                adaptive.k_proj.weight.data = baseline.k_proj.weight.data.clone()
                adaptive.v_proj.weight.data = baseline.v_proj.weight.data.clone()
                adaptive.out_proj.weight.data = baseline.out_proj.weight.data.clone()
            
            # Measure runtimes
            baseline_runtime = self.measure_runtime(baseline, x)
            draft_runtime = self.measure_runtime(draft, x)
            adaptive_runtime = self.measure_runtime(adaptive, x)
            
            # Compute outputs for accuracy
            with torch.no_grad():
                baseline_output = baseline(x)
                draft_output = draft(x, pool_factor=8)
                adaptive_output = adaptive(x, timestep=0.5)
            
            # Compute accuracy metrics
            draft_accuracy = self.compute_accuracy_metrics(baseline_output, draft_output)
            adaptive_accuracy = self.compute_accuracy_metrics(baseline_output, adaptive_output)
            
            # Store results
            test_result = {
                'scenario': scenario,
                'runtimes': {
                    'baseline': baseline_runtime,
                    'draft_attention': draft_runtime,
                    'adaptive_draft_attention': adaptive_runtime
                },
                'speedup_ratios': {
                    'draft_vs_baseline': baseline_runtime['avg_time'] / draft_runtime['avg_time'],
                    'adaptive_vs_baseline': baseline_runtime['avg_time'] / adaptive_runtime['avg_time'],
                    'adaptive_vs_draft': draft_runtime['avg_time'] / adaptive_runtime['avg_time']
                },
                'accuracy': {
                    'draft_attention': draft_accuracy,
                    'adaptive_draft_attention': adaptive_accuracy
                },
                'output_shapes': {
                    'baseline': list(baseline_output.shape),
                    'draft_attention': list(draft_output.shape),
                    'adaptive_draft_attention': list(adaptive_output.shape)
                }
            }
            
            all_results['tests'].append(test_result)
            
            # Print summary
            print(f"   Baseline: {baseline_runtime['avg_time']*1000:.2f}ms")
            print(f"   Draft: {draft_runtime['avg_time']*1000:.2f}ms ({test_result['speedup_ratios']['draft_vs_baseline']:.2f}x)")
            print(f"   Adaptive: {adaptive_runtime['avg_time']*1000:.2f}ms ({test_result['speedup_ratios']['adaptive_vs_baseline']:.2f}x)")
            print(f"   Draft Accuracy - MSE: {draft_accuracy['mse']:.2e}, Cosine: {draft_accuracy['cosine_similarity']:.4f}")
            print(f"   Adaptive Accuracy - MSE: {adaptive_accuracy['mse']:.2e}, Cosine: {adaptive_accuracy['cosine_similarity']:.4f}")
        
        # Compute geometric means
        draft_speedups = [t['speedup_ratios']['draft_vs_baseline'] for t in all_results['tests']]
        adaptive_speedups = [t['speedup_ratios']['adaptive_vs_baseline'] for t in all_results['tests']]
        
        draft_cosines = [t['accuracy']['draft_attention']['cosine_similarity'] for t in all_results['tests']]
        adaptive_cosines = [t['accuracy']['adaptive_draft_attention']['cosine_similarity'] for t in all_results['tests']]
        
        all_results['summary'] = {
            'geometric_mean_speedup': {
                'draft_vs_baseline': float(np.prod(draft_speedups) ** (1/len(draft_speedups))),
                'adaptive_vs_baseline': float(np.prod(adaptive_speedups) ** (1/len(adaptive_speedups)))
            },
            'geometric_mean_cosine_similarity': {
                'draft_attention': float(np.prod(draft_cosines) ** (1/len(draft_cosines))),
                'adaptive_draft_attention': float(np.prod(adaptive_cosines) ** (1/len(adaptive_cosines)))
            },
            'total_test_scenarios': len(all_results['tests'])
        }
        
        return all_results


def main():
    """Main testing function."""
    framework = TestFramework()
    results = framework.run_comprehensive_tests()
    
    # Save results
    output_path = '/home/wzc/data/file-share/logs/2025-09-18-15-57-21/test_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Testing completed!")
    print(f"📁 Results saved to: {output_path}")
    
    # Print summary
    summary = results['summary']
    print(f"\n📈 Summary:")
    print(f"   DraftAttention Speedup: {summary['geometric_mean_speedup']['draft_vs_baseline']:.2f}x")
    print(f"   AdaptiveDraftAttention Speedup: {summary['geometric_mean_speedup']['adaptive_vs_baseline']:.2f}x")
    print(f"   DraftAttention Cosine Similarity: {summary['geometric_mean_cosine_similarity']['draft_attention']:.4f}")
    print(f"   AdaptiveDraftAttention Cosine Similarity: {summary['geometric_mean_cosine_similarity']['adaptive_draft_attention']:.4f}")


if __name__ == "__main__":
    main()