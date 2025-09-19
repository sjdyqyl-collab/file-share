#!/usr/bin/env python3
"""
Comprehensive testing script for XAttention implementations.
Tests baseline, original, and improved methods with consistent weights.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import sys
import os
from typing import Dict, List, Tuple

# Add the current directory to path for imports
sys.path.append('/home/wzc/data/file-share/logs/2025-09-19-10-22-22')

from xattention_base import XAttentionBase
from xattention_original import XAttentionOriginal
from xattention_improved import XAttentionImproved


class BaselineAttention(nn.Module):
    """Standard full attention baseline for comparison."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        B, L, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute full attention
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        result = {'output': out}
        if return_attention:
            result['attention_weights'] = attn_weights
            
        return result
    
    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)


class PerformanceTester:
    """Comprehensive performance testing for XAttention methods."""
    
    def __init__(self, device: str = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.results = {}
        
    def generate_test_samples(self, configs: List[Dict]) -> List[Dict]:
        """Generate test samples based on paper configurations."""
        samples = []
        
        # Test configurations based on paper experiments
        test_configs = [
            # Language modeling tasks
            {'batch_size': 2, 'seq_len': 512, 'dim': 256, 'num_heads': 8, 'task': 'language'},
            {'batch_size': 1, 'seq_len': 1024, 'dim': 512, 'num_heads': 8, 'task': 'language'},
            {'batch_size': 1, 'seq_len': 2048, 'dim': 768, 'num_heads': 12, 'task': 'language'},
            
            # Video understanding tasks
            {'batch_size': 2, 'seq_len': 1024, 'dim': 512, 'num_heads': 8, 'task': 'video'},
            {'batch_size': 1, 'seq_len': 2048, 'dim': 768, 'num_heads': 12, 'task': 'video'},
            
            # Generation tasks
            {'batch_size': 1, 'seq_len': 4096, 'dim': 1024, 'num_heads': 16, 'task': 'generation'},
            {'batch_size': 1, 'seq_len': 8192, 'dim': 1024, 'num_heads': 16, 'task': 'generation'},
            
            # Long sequence tasks
            {'batch_size': 1, 'seq_len': 16384, 'dim': 1024, 'num_heads': 16, 'task': 'long_sequence'},
        ]
        
        for config in test_configs:
            # Generate random input
            x = torch.randn(
                config['batch_size'], 
                config['seq_len'], 
                config['dim'],
                device=self.device
            )
            
            samples.append({
                'input': x,
                'config': config,
                'name': f"{config['task']}_L{config['seq_len']}"
            })
            
        return samples
    
    def benchmark_model(self, model: nn.Module, x: torch.Tensor, num_iterations: int = 10) -> Dict:
        """Benchmark a single model."""
        model.eval()
        model.to(self.device)
        x = x.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        
        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time forward pass
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        # Measure memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(x)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            peak_memory = 0
        
        return {
            'avg_time': avg_time,
            'peak_memory_mb': peak_memory,
            'throughput_tps': x.size(1) / avg_time  # tokens per second
        }
    
    def test_accuracy(self, model: nn.Module, x: torch.Tensor, baseline_output: torch.Tensor) -> Dict:
        """Test accuracy against baseline."""
        model.eval()
        model.to(self.device)
        x = x.to(self.device)
        baseline_output = baseline_output.to(self.device)
        
        with torch.no_grad():
            output = model(x)
        
        # Compute similarity metrics
        if isinstance(output, dict):
            output_tensor = output['output']
        else:
            output_tensor = output
            
        mse = torch.nn.functional.mse_loss(output_tensor, baseline_output).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            output_tensor.flatten(), 
            baseline_output.flatten(), 
            dim=0
        ).item()
        
        # Compute sparsity if attention weights are available
        sparsity = 0.0
        if isinstance(output, dict) and 'attention_weights' in output:
            attn_weights = output['attention_weights']
            if attn_weights is not None:
                sparsity = (attn_weights.abs() < 1e-6).float().mean().item()
        
        return {
            'mse': mse,
            'cosine_similarity': cos_sim,
            'sparsity': sparsity
        }
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive testing across all methods."""
        print("Starting comprehensive XAttention testing...")
        
        # Generate test samples
        samples = self.generate_test_samples([])
        
        # Initialize models with consistent configuration
        dim = 512
        num_heads = 8
        
        models = {
            'baseline': BaselineAttention(dim, num_heads),
            'original': XAttentionOriginal(dim, num_heads, block_size=8, stride=8, threshold=0.9),
            'improved': XAttentionImproved(
                dim, num_heads, 
                default_block_size=8, 
                strides=[4, 8, 16],
                default_threshold=0.9,
                use_adaptive_warmup=True,
                use_multi_scale=True,
                use_content_adaptive=True,
                use_gradient_optimization=True
            )
        }
        
        # Initialize weights consistently
        self._initialize_consistent_weights(models, dim, num_heads)
        
        results = {
            'paper_title': 'XAttention: Sparse Attention with Antidiagonal Pattern Selection',
            'device': str(self.device),
            'test_samples': len(samples),
            'results': {}
        }
        
        # Test each sample
        for sample_idx, sample in enumerate(samples):
            print(f"\nTesting sample {sample_idx + 1}/{len(samples)}: {sample['name']}")
            
            x = sample['input']
            sample_results = {}
            
            # Test baseline (full attention)
            print("  Testing baseline full attention...")
            baseline_results = self.benchmark_model(models['baseline'], x)
            with torch.no_grad():
                baseline_output = models['baseline'](x)['output']
            sample_results['baseline'] = baseline_results
            
            # Test original XAttention
            print("  Testing original XAttention...")
            original_results = self.benchmark_model(models['original'], x)
            original_accuracy = self.test_accuracy(models['original'], x, baseline_output)
            sample_results['original'] = {**original_results, **original_accuracy}
            
            # Test improved XAttention
            print("  Testing improved XAttention...")
            improved_results = self.benchmark_model(models['improved'], x)
            improved_accuracy = self.test_accuracy(models['improved'], x, baseline_output)
            sample_results['improved'] = {**improved_results, **improved_accuracy}
            
            # Compute ratios
            sample_results['ratios'] = self._compute_ratios(sample_results)
            
            results['results'][sample['name']] = sample_results
        
        # Compute geometric means
        results['summary'] = self._compute_summary_statistics(results['results'])
        
        return results
    
    def _initialize_consistent_weights(self, models: Dict, dim: int, num_heads: int):
        """Initialize all models with consistent weights."""
        # Create a reference model to copy weights from
        ref_model = BaselineAttention(dim, num_heads)
        ref_state = ref_model.state_dict()
        
        # Load consistent weights into all models
        for name, model in models.items():
            model_state = model.state_dict()
            
            # Copy compatible weights
            for key in model_state.keys():
                if key in ref_state and model_state[key].shape == ref_state[key].shape:
                    model_state[key] = ref_state[key]
                elif 'q_proj' in key and 'weight' in key:
                    # Initialize query projection
                    nn.init.xavier_uniform_(model_state[key])
                elif 'k_proj' in key and 'weight' in key:
                    # Initialize key projection
                    nn.init.xavier_uniform_(model_state[key])
                elif 'v_proj' in key and 'weight' in key:
                    # Initialize value projection
                    nn.init.xavier_uniform_(model_state[key])
                elif 'out_proj' in key and 'weight' in key:
                    # Initialize output projection
                    nn.init.xavier_uniform_(model_state[key])
            
            model.load_state_dict(model_state)
    
    def _compute_ratios(self, sample_results: Dict) -> Dict:
        """Compute runtime and accuracy ratios."""
        baseline_time = sample_results['baseline']['avg_time']
        
        ratios = {}
        
        for method in ['original', 'improved']:
            if method in sample_results:
                method_time = sample_results[method]['avg_time']
                ratios[f'{method}_speedup'] = baseline_time / method_time
                ratios[f'{method}_mse_ratio'] = sample_results[method]['mse'] / 1e-6  # Normalize
                ratios[f'{method}_cosine_similarity'] = sample_results[method]['cosine_similarity']
                ratios[f'{method}_sparsity'] = sample_results[method]['sparsity']
        
        return ratios
    
    def _compute_summary_statistics(self, all_results: Dict) -> Dict:
        """Compute geometric means across all samples."""
        summary = {
            'original': {'speedup': [], 'mse_ratio': [], 'cosine_similarity': [], 'sparsity': []},
            'improved': {'speedup': [], 'mse_ratio': [], 'cosine_similarity': [], 'sparsity': []}
        }
        
        # Collect all ratios
        for sample_name, sample_results in all_results.items():
            if 'ratios' in sample_results:
                for method in ['original', 'improved']:
                    if f'{method}_speedup' in sample_results['ratios']:
                        summary[method]['speedup'].append(sample_results['ratios'][f'{method}_speedup'])
                        summary[method]['mse_ratio'].append(sample_results['ratios'][f'{method}_mse_ratio'])
                        summary[method]['cosine_similarity'].append(sample_results['ratios'][f'{method}_cosine_similarity'])
                        summary[method]['sparsity'].append(sample_results['ratios'][f'{method}_sparsity'])
        
        # Compute geometric means
        def geometric_mean(values):
            if not values:
                return 0.0
            values = np.array(values)
            values = values[values > 0]  # Filter out zeros
            if len(values) == 0:
                return 0.0
            return np.exp(np.mean(np.log(values)))
        
        final_summary = {}
        for method in ['original', 'improved']:
            final_summary[method] = {
                'geometric_mean_speedup': geometric_mean(summary[method]['speedup']),
                'geometric_mean_mse_ratio': geometric_mean(summary[method]['mse_ratio']),
                'mean_cosine_similarity': np.mean(summary[method]['cosine_similarity']),
                'mean_sparsity': np.mean(summary[method]['sparsity']),
                'sample_count': len(summary[method]['speedup'])
            }
        
        return final_summary


def main():
    """Main testing function."""
    tester = PerformanceTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Save results
    output_path = '/home/wzc/data/file-share/logs/2025-09-19-10-22-22/test_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, torch.Tensor) else x)
    
    print(f"\nTesting completed! Results saved to {output_path}")
    
    # Print summary
    print("\n=== Summary Statistics ===")
    for method in ['original', 'improved']:
        if method in results['summary']:
            stats = results['summary'][method]
            print(f"\n{method.upper()} XAttention:")
            print(f"  Geometric Mean Speedup: {stats['geometric_mean_speedup']:.2f}x")
            print(f"  Mean Cosine Similarity: {stats['mean_cosine_similarity']:.4f}")
            print(f"  Mean Sparsity: {stats['mean_sparsity']*100:.2f}%")
            print(f"  Samples tested: {stats['sample_count']}")
    
    return output_path


if __name__ == "__main__":
    main()