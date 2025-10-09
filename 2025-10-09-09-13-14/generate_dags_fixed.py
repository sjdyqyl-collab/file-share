#!/usr/bin/env python3
"""
Generate complete DAGs for both baseline and FA Pool configurations
"""

import graphviz
from typing import Dict, List, Tuple
import os

class DAGGenerator:
    def __init__(self):
        self.colors = {
            'input': '#E8F4FD',
            'output': '#D1E7DD',
            'computation': '#F8F9FA',
            'communication': '#FFF3CD',
            'aggregation': '#D4EDDA',
            'routing': '#FFE6CC'
        }
    
    def create_baseline_dag(self) -> str:
        """Create baseline DAG with TP=8, PP=2"""
        dot = graphviz.Digraph('baseline_model', 
                             comment='4-layer Dense Model - Baseline Configuration',
                             graph_attr={
                                 'rankdir': 'TB',
                                 'splines': 'ortho',
                                 'nodesep': '0.5',
                                 'ranksep': '1.0'
                             })
        
        # Set node defaults
        dot.attr('node', shape='rectangle', style='filled', fillcolor=self.colors['computation'])
        
        # Input processing
        dot.node('input', 'Model Input\\nInput: [batch_size, seq_len]\\nOutput: [batch_size, seq_len, 4096]', 
                shape='ellipse', fillcolor=self.colors['input'])
        
        # Embedding layer (replicated across all TP ranks)
        dot.node('embedding', 'Embedding Layer\\nInput: [batch_size, seq_len]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: [0-7]', 
                fillcolor=self.colors['computation'])
        
        # Positional encoding
        dot.node('pos_encoding', 'RoPE Positional Encoding\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: [0-7]', 
                fillcolor=self.colors['computation'])
        
        # Pipeline Stage 0 (Layers 0-1)
        self._add_pipeline_stage(dot, 'stage0', 'Pipeline Stage 0\\nLayers: 0-1\\nGPUs: [0-3]', 
                               ['gpu_0', 'gpu_1', 'gpu_2', 'gpu_3'], 0)
        
        # Pipeline Stage 1 (Layers 2-3)  
        self._add_pipeline_stage(dot, 'stage1', 'Pipeline Stage 1\\nLayers: 2-3\\nGPUs: [4-7]', 
                               ['gpu_4', 'gpu_5', 'gpu_6', 'gpu_7'], 2)
        
        # Output layer
        dot.node('output_layer', 'Output Projection\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 50256]\\nGPUs: [4-7]', 
                fillcolor=self.colors['computation'])
        
        # Model output
        dot.node('output', 'Model Output\\nInput: [batch_size, seq_len, 50256]\\nOutput: [batch_size, seq_len, 50256]', 
                shape='ellipse', fillcolor=self.colors['output'])
        
        # Add connections
        dot.edge('input', 'embedding')
        dot.edge('embedding', 'pos_encoding')
        dot.edge('pos_encoding', 'stage0_layer0_input')
        dot.edge('stage0_layer1_output', 'stage1_layer2_input')
        dot.edge('stage1_layer3_output', 'output_layer')
        dot.edge('output_layer', 'output')
        
        return dot.source
    
    def _add_pipeline_stage(self, dot, prefix, label, devices, layer_offset):
        """Add a complete pipeline stage with tensor parallelism"""
        
        # Stage input
        dot.node(f'{prefix}_input', f'{label}\\nInput Processing', 
                shape='parallelogram', fillcolor=self.colors['routing'])
        
        # Add Layer 0
        self._add_transformer_layer(dot, f'{prefix}_layer{layer_offset}', layer_offset, devices)
        
        # Add Layer 1  
        self._add_transformer_layer(dot, f'{prefix}_layer{layer_offset+1}', layer_offset+1, devices)
        
        # Connect layers
        dot.edge(f'{prefix}_input', f'{prefix}_layer{layer_offset}_input')
        dot.edge(f'{prefix}_layer{layer_offset}_output', f'{prefix}_layer{layer_offset+1}_input')
        
    def _add_transformer_layer(self, dot, prefix, layer_id, devices):
        """Add a complete transformer layer with tensor parallelism"""
        
        # Layer input
        dot.node(f'{prefix}_input', f'Layer {layer_id} Input\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]', 
                shape='parallelogram', fillcolor=self.colors['routing'])
        
        # RMSNorm 1
        dot.node(f'{prefix}_norm1', f'RMSNorm 1\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}', 
                fillcolor=self.colors['computation'])
        
        # Multi-head attention (split across devices)
        self._add_attention_block(dot, prefix, devices)
        
        # Residual connection 1
        dot.node(f'{prefix}_res1', f'Residual Add 1\\nInput: [batch_size, seq_len, 4096], [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}', 
                fillcolor=self.colors['aggregation'])
        
        # RMSNorm 2
        dot.node(f'{prefix}_norm2', f'RMSNorm 2\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}', 
                fillcolor=self.colors['computation'])
        
        # FFN (MLP)
        self._add_ffn_block(dot, prefix, devices)
        
        # Residual connection 2
        dot.node(f'{prefix}_res2', f'Residual Add 2\\nInput: [batch_size, seq_len, 4096], [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}', 
                fillcolor=self.colors['aggregation'])
        
        # Layer output
        dot.node(f'{prefix}_output', f'Layer {layer_id} Output\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]', 
                shape='parallelogram', fillcolor=self.colors['routing'])
        
        # Connections
        dot.edge(f'{prefix}_input', f'{prefix}_norm1')
        dot.edge(f'{prefix}_norm1', f'{prefix}_attn_input')
        dot.edge(f'{prefix}_attn_output', f'{prefix}_res1')
        dot.edge(f'{prefix}_input', f'{prefix}_res1')  # Residual connection
        dot.edge(f'{prefix}_res1', f'{prefix}_norm2')
        dot.edge(f'{prefix}_norm2', f'{prefix}_ffn_input')
        dot.edge(f'{prefix}_ffn_output', f'{prefix}_res2')
        dot.edge(f'{prefix}_res1', f'{prefix}_res2')  # Residual connection
        dot.edge(f'{prefix}_res2', f'{prefix}_output')
        
    def _add_attention_block(self, dot, prefix, devices):
        """Add multi-head attention with tensor parallelism"""
        
        # Input for attention
        dot.node(f'{prefix}_attn_input', f'Attention Input\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]', 
                shape='parallelogram', fillcolor=self.colors['routing'])
        
        # Q, K, V projections (column parallel)
        dot.node(f'{prefix}_q_proj', f'Q Projection\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}\\nType: Column-Parallel', 
                fillcolor=self.colors['computation'])
        dot.node(f'{prefix}_k_proj', f'K Projection\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}\\nType: Column-Parallel', 
                fillcolor=self.colors['computation'])
        dot.node(f'{prefix}_v_proj', f'V Projection\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}\\nType: Column-Parallel', 
                fillcolor=self.colors['computation'])
        
        # Reshape for multi-head
        dot.node(f'{prefix}_reshape_q', f'Reshape Q\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 32, 128]\\nGPUs: {devices}', 
                fillcolor=self.colors['computation'])
        dot.node(f'{prefix}_reshape_k', f'Reshape K\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 32, 128]\\nGPUs: {devices}', 
                fillcolor=self.colors['computation'])
        dot.node(f'{prefix}_reshape_v', f'Reshape V\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 32, 128]\\nGPUs: {devices}', 
                fillcolor=self.colors['computation'])
        
        # Attention computation
        dot.node(f'{prefix}_attention', f'Multi-Head Attention\\nInput: [batch_size, seq_len, 32, 128]\\nOutput: [batch_size, seq_len, 32, 128]\\nGPUs: {devices}\\nType: Distributed FlashAttention', 
                fillcolor=self.colors['computation'])
        
        # Reshape back
        dot.node(f'{prefix}_reshape_out', f'Reshape Output\\nInput: [batch_size, seq_len, 32, 128]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}', 
                fillcolor=self.colors['computation'])
        
        # Output projection (row parallel)
        dot.node(f'{prefix}_out_proj', f'Output Projection\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}\\nType: Row-Parallel', 
                fillcolor=self.colors['computation'])
        
        # Connections
        dot.edge(f'{prefix}_attn_input', f'{prefix}_q_proj')
        dot.edge(f'{prefix}_attn_input', f'{prefix}_k_proj')
        dot.edge(f'{prefix}_attn_input', f'{prefix}_v_proj')
        dot.edge(f'{prefix}_q_proj', f'{prefix}_reshape_q')
        dot.edge(f'{prefix}_k_proj', f'{prefix}_reshape_k')
        dot.edge(f'{prefix}_v_proj', f'{prefix}_reshape_v')
        dot.edge(f'{prefix}_reshape_q', f'{prefix}_attention')
        dot.edge(f'{prefix}_reshape_k', f'{prefix}_attention')
        dot.edge(f'{prefix}_reshape_v', f'{prefix}_attention')
        dot.edge(f'{prefix}_attention', f'{prefix}_reshape_out')
        dot.edge(f'{prefix}_reshape_out', f'{prefix}_out_proj')
        
    def _add_ffn_block(self, dot, prefix, devices):
        """Add FFN block with tensor parallelism"""
        
        # Input for FFN
        dot.node(f'{prefix}_ffn_input', f'FFN Input\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 4096]', 
                shape='parallelogram', fillcolor=self.colors['routing'])
        
        # First linear (column parallel)
        dot.node(f'{prefix}_ffn_up', f'FFN Up Projection\\nInput: [batch_size, seq_len, 4096]\\nOutput: [batch_size, seq_len, 16384]\\nGPUs: {devices}\\nType: Column-Parallel', 
                fillcolor=self.colors['computation'])
        
        # GELU activation
        dot.node(f'{prefix}_gelu', f'GELU Activation\\nInput: [batch_size, seq_len, 16384]\\nOutput: [batch_size, seq_len, 16384]\\nGPUs: {devices}', 
                fillcolor=self.colors['computation'])
        
        # Second linear (row parallel)
        dot.node(f'{prefix}_ffn_down', f'FFN Down Projection\\nInput: [batch_size, seq_len, 16384]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {devices}\\nType: Row-Parallel', 
                fillcolor=self.colors['computation'])
        
        # Connections
        dot.edge(f'{prefix}_ffn_input', f'{prefix}_ffn_up')
        dot.edge(f'{prefix}_ffn_up', f'{prefix}_gelu')
        dot.edge(f'{prefix}_gelu', f'{prefix}_ffn_down')
        
    def create_fa_pool_dag(self, sequence_length=16384) -> str:
        """Create FA Pool DAG for specific sequence length"""
        dot = graphviz.Digraph('fa_pool_model', 
                             comment=f'FA Pool Model SeqLen_{sequence_length}',
                             graph_attr={
                                 'rankdir': 'TB',
                                 'splines': 'ortho',
                                 'nodesep': '0.5',
                                 'ranksep': '1.0'
                             })
        
        # Determine pool GPUs based on sequence length
        if sequence_length <= 4096:
            pool_gpus = 0
            pool_devices = []
        elif sequence_length <= 8192:
            pool_gpus = 8
            pool_devices = [f'gpu_{i}' for i in range(8, 16)]
        elif sequence_length <= 16384:
            pool_gpus = 16
            pool_devices = [f'gpu_{i}' for i in range(8, 24)]
        elif sequence_length <= 32768:
            pool_gpus = 24
            pool_devices = [f'gpu_{i}' for i in range(8, 32)]
        else:
            pool_gpus = 32
            pool_devices = [f'gpu_{i}' for i in range(8, 40)]
        
        base_devices = [f'gpu_{i}' for i in range(8)]
        
        # Set node defaults
        dot.attr('node', shape='rectangle', style='filled', fillcolor=self.colors['computation'])
        
        # Input processing
        dot.node('input', f'Model Input\\nInput: [batch_size, seq_len={sequence_length}]\\nOutput: [batch_size, seq_len, 4096]', 
                shape='ellipse', fillcolor=self.colors['input'])
        
        # Embedding layer
        dot.node('embedding', f'Embedding Layer\\nInput: [batch_size, seq_len={sequence_length}]\\nOutput: [batch_size, seq_len, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['computation'])
        
        # Positional encoding
        dot.node('pos_encoding', f'RoPE Positional Encoding\\nInput: [batch_size, seq_len={sequence_length}, 4096]\\nOutput: [batch_size, seq_len={sequence_length}, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['computation'])
        
        # Add all 4 layers with attention pool
        for layer_id in range(4):
            self._add_fa_pool_layer(dot, f'layer{layer_id}', layer_id, base_devices, pool_devices, sequence_length, pool_gpus)
        
        # Output layer
        dot.node('output_layer', f'Output Projection\\nInput: [batch_size, seq_len={sequence_length}, 4096]\\nOutput: [batch_size, seq_len={sequence_length}, 50256]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['computation'])
        
        # Model output
        dot.node('output', f'Model Output\\nInput: [batch_size, seq_len={sequence_length}, 50256]\\nOutput: [batch_size, seq_len={sequence_length}, 50256]', 
                shape='ellipse', fillcolor=self.colors['output'])
        
        # Connections
        dot.edge('input', 'embedding')
        dot.edge('embedding', 'pos_encoding')
        dot.edge('pos_encoding', 'layer0_input')
        dot.edge('layer0_output', 'layer1_input')
        dot.edge('layer1_output', 'layer2_input')
        dot.edge('layer2_output', 'layer3_input')
        dot.edge('layer3_output', 'output_layer')
        dot.edge('output_layer', 'output')
        
        return dot.source
    
    def _add_fa_pool_layer(self, dot, prefix, layer_id, base_devices, pool_devices, seq_len, pool_gpus):
        """Add FA Pool layer with separate attention and FFN computation"""
        
        # Layer input
        dot.node(f'{prefix}_input', f'Layer {layer_id} Input\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]', 
                shape='parallelogram', fillcolor=self.colors['routing'])
        
        # RMSNorm 1
        dot.node(f'{prefix}_norm1', f'RMSNorm 1\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['computation'])
        
        # KV Cache broadcast to attention pool
        if pool_gpus > 0:
            dot.node(f'{prefix}_kv_broadcast', f'KV Cache Broadcast\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nFrom: {base_devices}\\nTo: {pool_devices}', 
                    shape='ellipse', fillcolor=self.colors['communication'])
        
        # Attention computation
        if pool_gpus > 0:
            self._add_fa_pool_attention(dot, prefix, base_devices, pool_devices, seq_len, pool_gpus)
        else:
            self._add_attention_block(dot, prefix, base_devices)
        
        # Residual connection 1
        dot.node(f'{prefix}_res1', f'Residual Add 1\\nInput: [batch_size, seq_len={seq_len}, 4096], [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['aggregation'])
        
        # RMSNorm 2
        dot.node(f'{prefix}_norm2', f'RMSNorm 2\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['computation'])
        
        # FFN computation (on base layer)
        self._add_ffn_block(dot, prefix, base_devices)
        
        # Residual connection 2
        dot.node(f'{prefix}_res2', f'Residual Add 2\\nInput: [batch_size, seq_len={seq_len}, 4096], [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['aggregation'])
        
        # Layer output
        dot.node(f'{prefix}_output', f'Layer {layer_id} Output\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]', 
                shape='parallelogram', fillcolor=self.colors['routing'])
        
        # Connections
        dot.edge(f'{prefix}_input', f'{prefix}_norm1')
        if pool_gpus > 0:
            dot.edge(f'{prefix}_norm1', f'{prefix}_kv_broadcast')
            dot.edge(f'{prefix}_kv_broadcast', f'{prefix}_attn_input')
            dot.edge(f'{prefix}_attn_output', f'{prefix}_result_agg')
            dot.edge(f'{prefix}_result_agg', f'{prefix}_res1')
        else:
            dot.edge(f'{prefix}_norm1', f'{prefix}_attn_input')
            dot.edge(f'{prefix}_attn_output', f'{prefix}_res1')
        dot.edge(f'{prefix}_input', f'{prefix}_res1')  # Residual connection
        dot.edge(f'{prefix}_res1', f'{prefix}_norm2')
        dot.edge(f'{prefix}_norm2', f'{prefix}_ffn_input')
        dot.edge(f'{prefix}_ffn_output', f'{prefix}_res2')
        dot.edge(f'{prefix}_res1', f'{prefix}_res2')  # Residual connection
        dot.edge(f'{prefix}_res2', f'{prefix}_output')
        
    def _add_fa_pool_attention(self, dot, prefix, base_devices, pool_devices, seq_len, pool_gpus):
        """Add FA Pool attention computation"""
        
        # Calculate block size
        block_size = (seq_len + pool_gpus - 1) // pool_gpus
        
        # Attention input
        dot.node(f'{prefix}_attn_input', f'Attention Input\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]', 
                shape='parallelogram', fillcolor=self.colors['routing'])
        
        # Q, K, V projections on base layer
        dot.node(f'{prefix}_q_proj', f'Q Projection\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['computation'])
        dot.node(f'{prefix}_k_proj', f'K Projection\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['computation'])
        dot.node(f'{prefix}_v_proj', f'V Projection\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['computation'])
        
        # Split into blocks for parallel processing
        for i, gpu in enumerate(pool_devices):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, seq_len)
            block_len = end_idx - start_idx
            
            dot.node(f'{prefix}_block_{i}', f'FlashAttention Block {i}\\nGPU: {gpu}\\nInput: [batch_size, block_len={block_len}, 4096]\\nOutput: [batch_size, block_len={block_len}, 4096]\\nQ: [batch_size, block_len={block_len}, 4096]\\nK/V: [batch_size, seq_len={seq_len}, 4096]', 
                    fillcolor=self.colors['computation'])
        
        # Result aggregation
        dot.node(f'{prefix}_result_agg', f'Result Aggregation\\nInput: {pool_gpus} x [batch_size, block_len, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nGPUs: {pool_devices}\\nType: Concatenation', 
                shape='ellipse', fillcolor=self.colors['communication'])
        
        # Output projection on base layer
        dot.node(f'{prefix}_out_proj', f'Output Projection\\nInput: [batch_size, seq_len={seq_len}, 4096]\\nOutput: [batch_size, seq_len={seq_len}, 4096]\\nGPUs: {base_devices}', 
                fillcolor=self.colors['computation'])
        
        # Connections
        dot.edge(f'{prefix}_attn_input', f'{prefix}_q_proj')
        dot.edge(f'{prefix}_attn_input', f'{prefix}_k_proj')
        dot.edge(f'{prefix}_attn_input', f'{prefix}_v_proj')
        
        # Distribute to attention pool
        for i, gpu in enumerate(pool_devices):
            dot.edge(f'{prefix}_q_proj', f'{prefix}_block_{i}')
            dot.edge(f'{prefix}_k_proj', f'{prefix}_block_{i}')
            dot.edge(f'{prefix}_v_proj', f'{prefix}_block_{i}')
            dot.edge(f'{prefix}_block_{i}', f'{prefix}_result_agg')
        
        dot.edge(f'{prefix}_result_agg', f'{prefix}_out_proj')
        
    def generate_all_dags(self):
        """Generate all DAGs and save to files"""
        
        # Create output directory
        os.makedirs('/home/wzc/data/file-share/2025-10-09-09-13-14', exist_ok=True)
        
        # Generate baseline DAG
        baseline_dot = self.create_baseline_dag()
        with open('/home/wzc/data/file-share/2025-10-09-09-13-14/baseline_model.dot', 'w') as f:
            f.write(baseline_dot)
        
        # Generate FA Pool DAGs for different sequence lengths
        configs = [
            ('fa_pool_4k', 4096),    # No pool GPUs
            ('fa_pool_8k', 8192),    # 8 pool GPUs  
            ('fa_pool_16k', 16384),  # 16 pool GPUs
            ('fa_pool_32k', 32768),  # 32 pool GPUs
        ]
        
        dag_files = [
            '/home/wzc/data/file-share/2025-10-09-09-13-14/baseline_model.dot'
        ]
        
        for name, seq_len in configs:
            fa_pool_dot = self.create_fa_pool_dag(seq_len)
            filename = f'/home/wzc/data/file-share/2025-10-09-09-13-14/{name}.dot'
            with open(filename, 'w') as f:
                f.write(fa_pool_dot)
            dag_files.append(filename)
        
        return dag_files

if __name__ == '__main__':
    generator = DAGGenerator()
    dag_files = generator.generate_all_dags()
    
    # Generate SVG files
    for dot_file in dag_files:
        svg_file = dot_file.replace('.dot', '.svg')
        os.system(f'dot -Tsvg {dot_file} -o {svg_file}')
    
    print("Generated DAG files:")
    for f in dag_files:
        print(f)
        
    # Save JSON with paths
    import json
    with open('/home/wzc/data/file-share/2025-10-09-09-13-14/dag_paths.json', 'w') as f:
        json.dump({
            'baseline': '/home/wzc/data/file-share/2025-10-09-09-13-14/baseline_model.dot',
            'fa_pool_4k': '/home/wzc/data/file-share/2025-10-09-09-13-14/fa_pool_4k.dot',
            'fa_pool_8k': '/home/wzc/data/file-share/2025-10-09-09-13-14/fa_pool_8k.dot',
            'fa_pool_16k': '/home/wzc/data/file-share/2025-10-09-09-13-14/fa_pool_16k.dot',
            'fa_pool_32k': '/home/wzc/data/file-share/2025-10-09-09-13-14/fa_pool_32k.dot'
        }, f, indent=2)