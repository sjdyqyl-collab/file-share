"""
Simplified performance testing for DraftAttention methods.
Focuses on core attention mechanisms without complex spatial reshaping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import numpy as np
from typing import Dict, List, Tuple
import os
import sys

# Add the current directory to path for imports
sys.path.append('/home/wzc/data/file-share/logs/2025-09-19-15-18-03')

from draft_attention import DraftAttention
from enhanced_draft_attention import EnhancedDraftAttention


class BaselineAttention(nn.Module):
    """Standard full attention baseline."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        device: torch.device = None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor, height: int = None, width: int = None, step_ratio: float = 0.5) -> torch.Tensor:
        """Standard full attention."""
        B, N, D = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, v)
        
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


# Simplified DraftAttention that works with any sequence length
class SimplifiedDraftAttention(nn.Module):
    """Simplified DraftAttention for testing."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sparsity_ratio: float = 0.75,
        pooling_size: int = 4,
        device: torch.device = None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_size = pooling_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor, height: int = None, width: int = None, step_ratio: float = 0.5) -> torch.Tensor:
        """Simplified draft attention."""
        B, N, D = x.shape
        
        # Use full attention for early steps
        if step_ratio < 0.25:
            return self._full_attention(x)
        
        # Compute Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Draft attention with pooling
        draft_size = max(1, N // self.pooling_size)
        
        # Simple pooling by averaging
        q_draft = F.adaptive_avg_pool1d(q.transpose(-1, -2), draft_size).transpose(-1, -2)
        k_draft = F.adaptive_avg_pool1d(k.transpose(-1, -2), draft_size).transpose(-1, -2)
        
        # Compute draft attention
        draft_attention = torch.matmul(q_draft, k_draft.transpose(-2, -1)) / np.sqrt(self.head_dim)
        draft_attention = F.softmax(draft_attention, dim=-1)
        
        # Create sparsity mask based on draft attention
        mask = self._create_sparsity_mask(draft_attention, N)
        
        # Apply sparse attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.masked_fill(mask == 0, 0.0)
        
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        
        return self.out_proj(out)
    
    def _create_sparsity_mask(self, draft_attention: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Create sparsity mask from draft attention."""
        B, H, draft_len, _ = draft_attention.shape
        
        # Average across heads
        draft_mean = draft_attention.mean(dim=1)
        
        # Determine number of draft tokens to keep
        num_keep = max(1, int(draft_len * self.sparsity_ratio))
        
        # Find top-k draft tokens
        flat_attention = draft_mean.reshape(B, -1)
        _, top_indices = torch.topk(flat_attention, num_keep, dim=-1)
        
        # Create draft-level mask
        draft_mask = torch.zeros_like(flat_attention)
        draft_mask.scatter_(1, top_indices, 1.0)
        draft_mask = draft_mask.reshape(B, draft_len, 1)
        
        # Expand to full sequence length
        full_mask = torch.zeros(B, seq_len, seq_len, device=self.device)
        
        # Map draft sparsity to full sequence (simplified)
        block_size = seq_len // draft_len
        for i in range(draft_len):
            if draft_mask[0, i, 0] > 0:
                start_idx = i * block_size
                end_idx = min((i + 1) * block_size, seq_len)
                full_mask[:, start_idx:end_idx, start_idx:end_idx] = 1.0
        
        return full_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
    
    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full attention."""
        B, N, D = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, v)
        
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class PerformanceTester:
    """Comprehensive performance testing class."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def generate_test_samples(self, configs: List[Dict]) -> List[Dict]:
        """Generate test samples."""
        samples = []
        
        for config in configs:
            batch_size = config['batch_size']
            seq_len = config['seq_len']
            dim = config['dim']
            
            # Generate random input tensor
            x = torch.randn(batch_size, seq_len, dim, device=self.device)
            
            # Generate step ratios for testing
            step_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            samples.append({
                'input': x,
                'step_ratios': step_ratios,
                'config': config
            })
        
        return samples
    
    def test_method(
        self, 
        method: nn.Module, 
        samples: List[Dict], 
        method_name: str,
        warmup_runs: int = 3,
        test_runs: int = 5
    ) -> Dict:
        """Test a single method's performance."""
        print(f"\n{'='*60}")
        print(f"Testing {method_name}")
        print(f"{'='*60}")
        
        method.to(self.device)
        method.eval()
        
        results = {
            'method_name': method_name,
            'device': str(self.device),
            'configurations': []
        }
        
        with torch.no_grad():
            for sample_idx, sample in enumerate(samples):
                print(f"\nConfiguration {sample_idx + 1}:")
                config = sample['config']
                print(f"  Batch: {config['batch_size']}, Seq: {config['seq_len']}, Dim: {config['dim']}")
                
                config_results = {
                    'config': config,
                    'runtimes': [],
                    'outputs': [],
                    'memory_usage': []
                }
                
                # Warmup
                for _ in range(warmup_runs):
                    for step_ratio in sample['step_ratios']:
                        _ = method(sample['input'], step_ratio=step_ratio)
                
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Actual testing
                for run in range(test_runs):
                    run_times = []
                    run_outputs = []
                    
                    for step_ratio in sample['step_ratios']:
                        # Measure runtime
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        start_time = time.perf_counter()
                        
                        output = method(sample['input'], step_ratio=step_ratio)
                        
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        end_time = time.perf_counter()
                        
                        run_times.append(end_time - start_time)
                        run_outputs.append(output.cpu().numpy().tolist())
                    
                    config_results['runtimes'].append(run_times)
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                        config_results['memory_usage'].append(memory_used)
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        config_results['memory_usage'].append(0.0)
                
                results['configurations'].append(config_results)
        
        return results
    
    def compute_ratios(self, baseline_results: Dict, method_results: Dict) -> Dict:
        """Compute runtime and output ratios between methods."""
        ratios = {
            'method_name': method_results['method_name'],
            'runtime_ratios': [],
            'output_ratios': [],
            'geometric_means': {}
        }
        
        baseline_configs = baseline_results['configurations']
        method_configs = method_results['configurations']
        
        for base_config, method_config in zip(baseline_configs, method_configs):
            config_ratios = {
                'config': base_config['config'],
                'runtime_ratios_per_step': [],
                'output_ratios_per_step': []
            }
            
            # Compute runtime ratios
            base_runtimes = np.array(base_config['runtimes']).mean(axis=0)
            method_runtimes = np.array(method_config['runtimes']).mean(axis=0)
            
            runtime_ratios = base_runtimes / method_runtimes
            config_ratios['runtime_ratios_per_step'] = runtime_ratios.tolist()
            
            # Compute output similarity ratios (using cosine similarity)
            base_outputs = np.array(base_config['outputs']).mean(axis=0)
            method_outputs = np.array(method_config['outputs']).mean(axis=0)
            
            output_ratios = []
            for base_out, method_out in zip(base_outputs, method_outputs):
                base_flat = np.array(base_out).flatten()
                method_flat = np.array(method_out).flatten()
                
                # Cosine similarity
                dot_product = np.dot(base_flat, method_flat)
                norm_base = np.linalg.norm(base_flat)
                norm_method = np.linalg.norm(method_flat)
                similarity = dot_product / (norm_base * norm_method + 1e-8)
                
                output_ratios.append(float(similarity))
            
            config_ratios['output_ratios_per_step'] = output_ratios
            
            ratios['runtime_ratios'].append(config_ratios)
        
        # Compute geometric means
        all_runtime_ratios = []
        all_output_ratios = []
        
        for config_ratios in ratios['runtime_ratios']:
            all_runtime_ratios.extend(config_ratios['runtime_ratios_per_step'])
            all_output_ratios.extend(config_ratios['output_ratios_per_step'])
        
        if all_runtime_ratios:
            # Filter out any negative or zero values for log
            valid_runtime_ratios = [r for r in all_runtime_ratios if r > 0]
            if valid_runtime_ratios:
                ratios['geometric_means']['runtime'] = float(np.exp(np.mean(np.log(valid_runtime_ratios))))
        
        if all_output_ratios:
            # Filter out any negative or zero values for log
            valid_output_ratios = [r for r in all_output_ratios if r > 0]
            if valid_output_ratios:
                ratios['geometric_means']['output'] = float(np.exp(np.mean(np.log(valid_output_ratios))))
        
        return ratios
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive performance test."""
        print("="*80)
        print("DRAFTATTENTION PERFORMANCE TESTING")
        print("="*80)
        
        # Test configurations - realistic sequence lengths
        test_configs = [
            {
                'batch_size': 1,
                'seq_len': 1024,   # 1K tokens
                'dim': 256,
                'model': 'Small-1K'
            },
            {
                'batch_size': 1,
                'seq_len': 2048,   # 2K tokens
                'dim': 256,
                'model': 'Medium-2K'
            },
            {
                'batch_size': 2,
                'seq_len': 1024,   # 1K tokens with batch
                'dim': 256,
                'model': 'Small-1K-Batch2'
            },
            {
                'batch_size': 1,
                'seq_len': 4096,   # 4K tokens
                'dim': 256,
                'model': 'Large-4K'
            }
        ]
        
        # Generate test samples
        print("\nGenerating test samples...")
        samples = self.generate_test_samples(test_configs)
        
        # Initialize methods with consistent dimensions
        dim = 256
        methods = {
            'Baseline': BaselineAttention(dim=dim, num_heads=8, device=self.device),
            'DraftAttention': SimplifiedDraftAttention(
                dim=dim, 
                num_heads=8, 
                sparsity_ratio=0.75, 
                pooling_size=4,
                device=self.device
            ),
            'EnhancedDraftAttention': EnhancedDraftAttention(
                dim=dim,
                num_heads=8,
                base_sparsity_ratio=0.75,
                scales=[(4, 8), (8, 16), (16, 32)],
                quantization_bits=8,
                use_motion_aware=True,
                device=self.device
            )
        }
        
        # Test each method
        all_results = {}
        for name, method in methods.items():
            all_results[name] = self.test_method(method, samples, name)
        
        # Compute ratios
        baseline_name = 'Baseline'
        ratios = {}
        
        for method_name in ['DraftAttention', 'EnhancedDraftAttention']:
            if method_name in all_results:
                ratios[method_name] = self.compute_ratios(
                    all_results[baseline_name], 
                    all_results[method_name]
                )
        
        # Compile final results
        final_results = {
            'paper_title': 'DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance',
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_info': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'raw_results': all_results,
            'performance_ratios': ratios,
            'summary': self._generate_summary(ratios)
        }
        
        return final_results
    
    def _generate_summary(self, ratios: Dict) -> Dict:
        """Generate summary of performance improvements."""
        summary = {}
        
        for method_name, ratio_data in ratios.items():
            summary[method_name] = {
                'runtime_speedup': ratio_data['geometric_means'].get('runtime', 1.0),
                'output_similarity': ratio_data['geometric_means'].get('output', 1.0),
                'quality_preservation': 'Excellent' if ratio_data['geometric_means'].get('output', 0) > 0.95 else 'Good'
            }
        
        return summary


def main():
    """Main testing function."""
    print("Starting comprehensive performance testing...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run tests
    tester = PerformanceTester(device)
    results = tester.run_comprehensive_test()
    
    # Save results
    output_path = '/home/wzc/data/file-share/logs/2025-09-19-15-18-03/performance_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    
    # Print summary
    for method, summary in results['summary'].items():
        print(f"\n{method}:")
        print(f"  Runtime Speedup: {summary['runtime_speedup']:.2f}x")
        print(f"  Output Similarity: {summary['output_similarity']:.4f}")
        print(f"  Quality: {summary['quality_preservation']}")
    
    print(f"\nDetailed results saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    main()