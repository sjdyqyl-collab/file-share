#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

# Create FA Pool DAG with dynamic resource allocation
dot = Digraph(comment='FA Pool DAG: Dynamic Parallel Strategy (8 Base + 32 Pool GPUs)')
dot.attr(rankdir='TB', size='30,40')

# Global attributes
dot.attr('node', fontname='Arial', fontsize='10')

# Define GPU clusters
with dot.subgraph(name='cluster_base_layer') as base_layer:
    base_layer.attr(label='Base Layer (8 GPUs - Model Backbone)', style='rounded', bgcolor='lightblue')
    
    # Input embedding
    base_layer.node('embed_input', 'Input Embedding\nGPU 0\nInput: [batch_size=?, seq_len=?, vocab_size=?]\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
                   shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Sequence length threshold check
    base_layer.node('seq_check', 'Sequence Length Check\nGPU 0\nThreshold: 4096 tokens\nDecision: Base vs Pool', 
                   shape='diamond', style='filled', fillcolor='gold')
    
    # Base layer FFN computations (distributed across 8 GPUs)
    for layer_id in range(4):
        for gpu_id in range(8):
            # FFN Layer 1 (column parallel)
            base_layer.node(f'layer{layer_id}_ffn1_base_gpu{gpu_id}', 
                          f'FFN Layer 1\nBase GPU {gpu_id}\nLayer {layer_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, ffn_hidden_size=2048]', 
                          shape='rectangle', style='filled', fillcolor='lightblue')
            
            # FFN Layer 2 (row parallel)
            base_layer.node(f'layer{layer_id}_ffn2_base_gpu{gpu_id}', 
                          f'FFN Layer 2\nBase GPU {gpu_id}\nLayer {layer_id}\nInput: [batch_size=?, seq_len=?, ffn_hidden_size=2048]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                          shape='rectangle', style='filled', fillcolor='lightblue')
            
            # Residual connections
            base_layer.node(f'layer{layer_id}_res_base_gpu{gpu_id}', 
                          f'Residual Add\nBase GPU {gpu_id}\nLayer {layer_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512], [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                          shape='parallelogram', style='filled', fillcolor='lightgray')
            
            # All-reduce for tensor parallelism
            base_layer.node(f'layer{layer_id}_allreduce_base', 
                          f'All-Reduce\nAcross Base GPUs 0-7\nLayer {layer_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512]×8\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
                          shape='ellipse', style='dashed', fillcolor='orange')

# Attention Pool clusters (dynamic allocation)
with dot.subgraph(name='cluster_attention_pool') as attention_pool:
    attention_pool.attr(label='Attention Pool (0-32 GPUs - Dynamic)', style='rounded', bgcolor='lightyellow')
    
    # Pool size calculation
    attention_pool.node('pool_calc', 'Pool Size Calculation\nGPU 0\nFormula: min(ceil(seq_len/2048), 32)\nOutput: pool_size (1-32)', 
                       shape='ellipse', style='filled', fillcolor='gold')
    
    # Dynamic GPU allocation
    attention_pool.node('gpu_alloc', 'Dynamic GPU Allocation\nBased on pool_size\nAllocate GPUs 8-39', 
                       shape='diamond', style='filled', fillcolor='gold')
    
    # Attention computation across pool GPUs
    for pool_gpu_id in range(32):
        actual_gpu = pool_gpu_id + 8  # Offset by base GPUs
        
        # Block size calculation
        attention_pool.node(f'block_calc_gpu{actual_gpu}', 
                          f'Block Size Calculation\nPool GPU {actual_gpu}\nFormula: b = ceil(n/p)\nOutput: block_size', 
                          shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # QKV projection (distributed)
        attention_pool.node(f'qkv_pool_gpu{actual_gpu}', 
                          f'QKV Projection\nPool GPU {actual_gpu}\nInput: [batch_size=?, block_size=?, hidden_size=4096]\nOutput: [batch_size=?, block_size=?, heads=32, d_k=128]', 
                          shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Attention computation
        attention_pool.node(f'attn_pool_gpu{actual_gpu}', 
                          f'Scaled Dot-Product Attention\nPool GPU {actual_gpu}\nInput: [batch_size=?, block_size=?, heads=32, d_k=128]\nOutput: [batch_size=?, block_size=?, heads=32, d_k=128]', 
                          shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Output projection
        attention_pool.node(f'out_proj_pool_gpu{actual_gpu}', 
                          f'Output Projection\nPool GPU {actual_gpu}\nInput: [batch_size=?, block_size=?, heads=32, d_k=128]\nOutput: [batch_size=?, block_size=?, hidden_size=4096]', 
                          shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # KV cache sharing
        attention_pool.node(f'kv_cache_gpu{actual_gpu}', 
                          f'KV Cache Share\nPool GPU {actual_gpu}\nInput: [batch_size=?, seq_len=?, heads=32, d_k=128]\nOutput: [batch_size=?, seq_len=?, heads=32, d_k=128]', 
                          shape='ellipse', style='dashed', fillcolor='purple')

# Result aggregation
with dot.subgraph(name='cluster_aggregation') as aggregation:
    aggregation.attr(label='Result Aggregation', style='rounded', bgcolor='lightpink')
    
    # Hierarchical reduction
    aggregation.node('hier_reduce', 'Hierarchical Reduction\nAcross Pool GPUs\nInput: [batch_size=?, seq_len=?, hidden_size=4096]×pool_size\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
                    shape='ellipse', style='dashed', fillcolor='orange')
    
    # Overlap with FFN
    aggregation.node('overlap', 'Overlap Execution\nAttention + FFN\nGPU 0-7 & 8-39', 
                    shape='diamond', style='filled', fillcolor='gold')

# Output layer
with dot.subgraph(name='cluster_output') as output_cluster:
    output_cluster.attr(label='Output Layer', style='rounded', bgcolor='lightgreen')
    
    output_cluster.node('final_norm', 'Layer Normalization\nGPU 0\nInput: [batch_size=?, seq_len=?, hidden_size=4096]\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
    
    output_cluster.node('output_proj', 'Output Projection\nGPU 0\nInput: [batch_size=?, seq_len=?, hidden_size=4096]\nOutput: [batch_size=?, seq_len=?, vocab_size=?]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')

# Connect the DAG
# Input flow
dot.edge('embed_input', 'seq_check')

# Sequence length decision
dot.edge('seq_check', 'pool_calc', label='seq_len > 4096')
dot.edge('seq_check', 'layer0_ffn1_base_gpu0', label='seq_len <= 4096')

# Pool calculation and allocation
dot.edge('pool_calc', 'gpu_alloc')

# Attention pool connections
for pool_gpu_id in range(32):
    actual_gpu = pool_gpu_id + 8
    dot.edge('gpu_alloc', f'block_calc_gpu{actual_gpu}')
    dot.edge(f'block_calc_gpu{actual_gpu}', f'qkv_pool_gpu{actual_gpu}')
    dot.edge(f'qkv_pool_gpu{actual_gpu}', f'kv_cache_gpu{actual_gpu}')
    dot.edge(f'kv_cache_gpu{actual_gpu}', f'attn_pool_gpu{actual_gpu}')
    dot.edge(f'attn_pool_gpu{actual_gpu}', f'out_proj_pool_gpu{actual_gpu}')
    dot.edge(f'out_proj_pool_gpu{actual_gpu}', 'hier_reduce')

# Base layer connections (simplified for all 4 layers)
for layer_id in range(4):
    for gpu_id in range(8):
        if layer_id == 0:
            dot.edge('seq_check', f'layer{layer_id}_ffn1_base_gpu{gpu_id}')
        else:
            # Connect from previous layer's all-reduce
            dot.edge(f'layer{layer_id-1}_allreduce_base', f'layer{layer_id}_ffn1_base_gpu{gpu_id}')
        
        dot.edge(f'layer{layer_id}_ffn1_base_gpu{gpu_id}', f'layer{layer_id}_ffn2_base_gpu{gpu_id}')
        dot.edge(f'layer{layer_id}_ffn2_base_gpu{gpu_id}', f'layer{layer_id}_res_base_gpu{gpu_id}')
        dot.edge(f'layer{layer_id}_res_base_gpu{gpu_id}', f'layer{layer_id}_allreduce_base')

# Communication between attention pool and base layer
dot.edge('hier_reduce', 'overlap')
dot.edge('layer3_allreduce_base', 'overlap')
dot.edge('overlap', 'final_norm')
dot.edge('final_norm', 'output_proj')

# Add communication arrows for data flow
dot.edge('embed_input', 'qkv_pool_gpu8', style='dashed', label='broadcast')
for pool_gpu_id in range(1, 32):
    actual_gpu = pool_gpu_id + 8
    dot.edge('qkv_pool_gpu8', f'qkv_pool_gpu{actual_gpu}', style='dashed', label='KV cache')

# Save files
with open('/home/wzc/data/file-share/logs/2025-10-09-11-30-42/fa_pool_dag.dot', 'w') as f:
    f.write(dot.source)

dot.render('/home/wzc/data/file-share/logs/2025-10-09-11-30-42/fa_pool_dag', format='svg', cleanup=False)

print("FA Pool DAG generated successfully!")
print("Files saved:")
print("- /home/wzc/data/file-share/logs/2025-10-09-11-30-42/fa_pool_dag.dot")
print("- /home/wzc/data/file-share/logs/2025-10-09-11-30-42/fa_pool_dag.svg")