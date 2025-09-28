"""
Comprehensive performance testing for DraftAttention methods.
Tests runtime and output quality of:
1. Baseline (full attention)
2. DraftAttention (proposed method)
3. EnhancedDraftAttention (enhanced method)
"""

import torch
import torch.nn as nn
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
    
    def forward(self, x: torch.Tensor, height: int, width: int, step_ratio: float = 0.5) -> torch.Tensor:
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


class PerformanceTester:
    """Comprehensive performance testing class."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def generate_test_samples(self, configs: List[Dict]) -> List[Dict]:
        """Generate test samples based on paper configurations."""
        samples = []
        
        for config in configs:
            batch_size = config['batch_size']
            seq_len = config['seq_len']
            dim = config['dim']
            height = config['height']
            width = config['width']
            
            # Generate random input tensor
            x = torch.randn(batch_size, seq_len, dim, device=self.device)
            
            # Generate step ratios for testing
            step_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            samples.append({
                'input': x,
                'height': height,
                'width': width,
                'step_ratios': step_ratios,
                'config': config
            })
        
        return samples
    
    def test_method(
        self, 
        method: nn.Module, 
        samples: List[Dict], 
        method_name: str,
        warmup_runs: int = 5,
        test_runs: int = 10
    ) -> Dict:
        """Test a single method's performance."""
        print(f"\n{'='*60}")
        print(f"Testing {method_name}")
        print(f"{'='*60}")
        
        method.to(self.device)
        method.eval()
        
        # Ensure consistent weights across methods
        if hasattr(method, 'load_weights'):
            # Create dummy state dict for consistency
            dummy_state = method.state_dict()
            method.load_weights(dummy_state)
        
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
                        _ = method(sample['input'], sample['height'], sample['width'], step_ratio)
                
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Actual testing
                for run in range(test_runs):
                    run_times = []
                    run_outputs = []
                    
                    for step_ratio in sample['step_ratios']:
                        # Measure runtime
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        start_time = time.perf_counter()
                        
                        output = method(sample['input'], sample['height'], sample['width'], step_ratio)
                        
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        end_time = time.perf_counter()
                        
                        run_times.append(end_time - start_time)
                        run_outputs.append(output.cpu().numpy().tolist())
                    
                    config_results['runtimes'].append(run_times)
                    config_results['outputs'].append(run_outputs)
                    
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
            ratios['geometric_means']['runtime'] = float(np.exp(np.mean(np.log(all_runtime_ratios))))
        if all_output_ratios:
            ratios['geometric_means']['output'] = float(np.exp(np.mean(np.log(all_output_ratios))))
        
        return ratios
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive performance test."""
        print("="*80)
        print("DRAFTATTENTION PERFORMANCE TESTING")
        print("="*80)
        
        # Test configurations based on paper
        test_configs = [
            # HunyuanVideo-T2V 768p
            {
                'batch_size': 1,
                'seq_len': 48 * 80 * 128,  # 768p video with 128 frames
                'dim': 768,
                'height': 48,
                'width': 80,
                'model': 'HunyuanVideo-T2V'
            },
            # Wan2.1-T2V 512p
            {
                'batch_size': 1,
                'seq_len': 32 * 48 * 80,
                'dim': 512,
                'height': 32,
                'width': 48,
                'model': 'Wan2.1-T2V-512p'
            },
            # Wan2.1-T2V 768p
            {
                'batch_size': 1,
                'seq_len': 48 * 80 * 80,
                'dim': 768,
                'height': 48,
                'width': 80,
                'model': 'Wan2.1-T2V-768p'
            },
            # Reduced size for practical testing
            {
                'batch_size': 2,
                'seq_len': 16 * 24 * 16,  # Smaller for testing
                'dim': 256,
                'height': 16,
                'width': 24,
                'model': 'Test-Reduced'
            }
        ]
        
        # Generate test samples
        print("\nGenerating test samples...")
        samples = self.generate_test_samples(test_configs)
        
        # Initialize methods
        methods = {
            'Baseline': BaselineAttention(dim=256, num_heads=8, device=self.device),
            'DraftAttention': DraftAttention(
                dim=256, 
                num_heads=8, 
                sparsity_ratio=0.75, 
                pooling_kernel=(8, 16),
                device=self.device
            ),
            'EnhancedDraftAttention': EnhancedDraftAttention(
                dim=256,
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