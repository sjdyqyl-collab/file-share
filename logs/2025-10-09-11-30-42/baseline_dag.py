#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

# Create baseline DAG with TP=8, PP=2 (16 GPUs total)
dot = Digraph(comment='Baseline DAG: TP=8, PP=2 (16 GPUs)')
dot.attr(rankdir='TB', size='20,30')

# Global attributes
dot.attr('node', fontname='Arial', fontsize='10')

# Define GPU clusters for pipeline stages
with dot.subgraph(name='cluster_stage0') as stage0:
    stage0.attr(label='Pipeline Stage 0 (Layers 0-1)', style='rounded', bgcolor='lightblue')
    
    # Embedding layer (GPU 0-7)
    stage0.node('embed_input', 'Input Embedding\nInput: [batch_size=?, seq_len=?, vocab_size=?]\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
                shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Layer 0 - Multi-Head Attention across GPUs 0-7
    for gpu_id in range(8):
        # QKV projection (column parallel)
        stage0.node(f'layer0_qkv_gpu{gpu_id}', 
                   f'QKV Projection\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=4096]\nOutput: [batch_size=?, seq_len=?, heads=4, d_k=128]', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        # Attention computation
        stage0.node(f'layer0_attn_gpu{gpu_id}', 
                   f'Scaled Dot-Product Attention\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=?, seq_len=?, heads=4, d_k=128]', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Output projection (row parallel)
        stage0.node(f'layer0_out_gpu{gpu_id}', 
                   f'Output Projection\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        # FFN layer 1 (column parallel)
        stage0.node(f'layer0_ffn1_gpu{gpu_id}', 
                   f'FFN Layer 1\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, ffn_hidden_size=2048]', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        
        # FFN layer 2 (row parallel)
        stage0.node(f'layer0_ffn2_gpu{gpu_id}', 
                   f'FFN Layer 2\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, ffn_hidden_size=2048]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Residual connections
        stage0.node(f'layer0_res1_gpu{gpu_id}', 
                   f'Residual Add 1\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512], [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='parallelogram', style='filled', fillcolor='lightgray')
        
        stage0.node(f'layer0_res2_gpu{gpu_id}', 
                   f'Residual Add 2\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512], [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='parallelogram', style='filled', fillcolor='lightgray')
    
    # Communication nodes for tensor parallelism
    stage0.node('layer0_allreduce1', 'All-Reduce\nAcross GPUs 0-7\nInput: [batch_size=?, seq_len=?, hidden_size=512]×8\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
               shape='ellipse', style='dashed', fillcolor='orange')
    stage0.node('layer0_allreduce2', 'All-Reduce\nAcross GPUs 0-7\nInput: [batch_size=?, seq_len=?, hidden_size=512]×8\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
               shape='ellipse', style='dashed', fillcolor='orange')

with dot.subgraph(name='cluster_stage1') as stage1:
    stage1.attr(label='Pipeline Stage 1 (Layers 2-3)', style='rounded', bgcolor='lightgreen')
    
    # Similar structure for layers 2-3 on GPUs 8-15
    for gpu_id in range(8, 16):
        actual_gpu = gpu_id - 8
        # Layer 2 (similar to layer 0)
        stage1.node(f'layer2_qkv_gpu{gpu_id}', 
                   f'QKV Projection\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=4096]\nOutput: [batch_size=?, seq_len=?, heads=4, d_k=128]', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        stage1.node(f'layer2_attn_gpu{gpu_id}', 
                   f'Scaled Dot-Product Attention\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=?, seq_len=?, heads=4, d_k=128]', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
        
        stage1.node(f'layer2_out_gpu{gpu_id}', 
                   f'Output Projection\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        stage1.node(f'layer2_ffn1_gpu{gpu_id}', 
                   f'FFN Layer 1\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, ffn_hidden_size=2048]', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        
        stage1.node(f'layer2_ffn2_gpu{gpu_id}', 
                   f'FFN Layer 2\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, ffn_hidden_size=2048]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        
        stage1.node(f'layer2_res1_gpu{gpu_id}', 
                   f'Residual Add 1\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512], [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='parallelogram', style='filled', fillcolor='lightgray')
        
        stage1.node(f'layer2_res2_gpu{gpu_id}', 
                   f'Residual Add 2\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512], [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='parallelogram', style='filled', fillcolor='lightgray')
        
        # Layer 3 (similar structure)
        stage1.node(f'layer3_qkv_gpu{gpu_id}', 
                   f'QKV Projection\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=4096]\nOutput: [batch_size=?, seq_len=?, heads=4, d_k=128]', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        stage1.node(f'layer3_attn_gpu{gpu_id}', 
                   f'Scaled Dot-Product Attention\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=?, seq_len=?, heads=4, d_k=128]', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
        
        stage1.node(f'layer3_out_gpu{gpu_id}', 
                   f'Output Projection\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        stage1.node(f'layer3_ffn1_gpu{gpu_id}', 
                   f'FFN Layer 1\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, ffn_hidden_size=2048]', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        
        stage1.node(f'layer3_ffn2_gpu{gpu_id}', 
                   f'FFN Layer 2\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, ffn_hidden_size=2048]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        
        stage1.node(f'layer3_res1_gpu{gpu_id}', 
                   f'Residual Add 1\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512], [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='parallelogram', style='filled', fillcolor='lightgray')
        
        stage1.node(f'layer3_res2_gpu{gpu_id}', 
                   f'Residual Add 2\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512], [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, hidden_size=512]', 
                   shape='parallelogram', style='filled', fillcolor='lightgray')
    
    stage1.node('layer2_allreduce1', 'All-Reduce\nAcross GPUs 8-15\nInput: [batch_size=?, seq_len=?, hidden_size=512]×8\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
               shape='ellipse', style='dashed', fillcolor='orange')
    stage1.node('layer2_allreduce2', 'All-Reduce\nAcross GPUs 8-15\nInput: [batch_size=?, seq_len=?, hidden_size=512]×8\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
               shape='ellipse', style='dashed', fillcolor='orange')

# Output layer
with dot.subgraph(name='cluster_output') as output_cluster:
    output_cluster.attr(label='Output Layer', style='rounded', bgcolor='lightpink')
    
    for gpu_id in range(8, 16):
        output_cluster.node(f'output_proj_gpu{gpu_id}', 
                          f'Output Projection\nGPU {gpu_id}\nInput: [batch_size=?, seq_len=?, hidden_size=512]\nOutput: [batch_size=?, seq_len=?, vocab_size=?]', 
                          shape='rectangle', style='filled', fillcolor='lightgreen')
    
    output_cluster.node('output_allreduce', 'All-Reduce\nAcross GPUs 8-15\nInput: [batch_size=?, seq_len=?, vocab_size=?]×8\nOutput: [batch_size=?, seq_len=?, vocab_size=?]', 
                       shape='ellipse', style='dashed', fillcolor='orange')

# Pipeline communication
stage0.node('pipeline_send', 'Send to Stage 1\nFrom GPUs 0-7 to GPUs 8-15\nInput: [batch_size=?, seq_len=?, hidden_size=4096]\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
           shape='ellipse', style='dashed', fillcolor='purple')

stage1.node('pipeline_recv', 'Receive from Stage 0\nFrom GPUs 0-7 to GPUs 8-15\nInput: [batch_size=?, seq_len=?, hidden_size=4096]\nOutput: [batch_size=?, seq_len=?, hidden_size=4096]', 
           shape='ellipse', style='dashed', fillcolor='purple')

# Connect the DAG
# Input to embedding
dot.edge('embed_input', 'layer0_qkv_gpu0')

# Layer 0 connections
for gpu_id in range(8):
    dot.edge('embed_input', f'layer0_qkv_gpu{gpu_id}')
    dot.edge(f'layer0_qkv_gpu{gpu_id}', f'layer0_attn_gpu{gpu_id}')
    dot.edge(f'layer0_attn_gpu{gpu_id}', f'layer0_out_gpu{gpu_id}')
    dot.edge(f'layer0_out_gpu{gpu_id}', f'layer0_res1_gpu{gpu_id}')
    dot.edge(f'layer0_res1_gpu{gpu_id}', f'layer0_ffn1_gpu{gpu_id}')
    dot.edge(f'layer0_ffn1_gpu{gpu_id}', f'layer0_ffn2_gpu{gpu_id}')
    dot.edge(f'layer0_ffn2_gpu{gpu_id}', f'layer0_res2_gpu{gpu_id}')
    
    # Connect to all-reduce
    dot.edge(f'layer0_res1_gpu{gpu_id}', 'layer0_allreduce1')
    dot.edge(f'layer0_res2_gpu{gpu_id}', 'layer0_allreduce2')

# Pipeline stage connections
dot.edge('layer0_allreduce2', 'pipeline_send')
dot.edge('pipeline_send', 'pipeline_recv')

# Layer 2 connections (after pipeline receive)
for gpu_id in range(8, 16):
    actual_gpu = gpu_id - 8
    dot.edge('pipeline_recv', f'layer2_qkv_gpu{gpu_id}')
    dot.edge(f'layer2_qkv_gpu{gpu_id}', f'layer2_attn_gpu{gpu_id}')
    dot.edge(f'layer2_attn_gpu{gpu_id}', f'layer2_out_gpu{gpu_id}')
    dot.edge(f'layer2_out_gpu{gpu_id}', f'layer2_res1_gpu{gpu_id}')
    dot.edge(f'layer2_res1_gpu{gpu_id}', f'layer2_ffn1_gpu{gpu_id}')
    dot.edge(f'layer2_ffn1_gpu{gpu_id}', f'layer2_ffn2_gpu{gpu_id}')
    dot.edge(f'layer2_ffn2_gpu{gpu_id}', f'layer2_res2_gpu{gpu_id}')
    
    # Connect to all-reduce
    dot.edge(f'layer2_res1_gpu{gpu_id}', 'layer2_allreduce1')
    dot.edge(f'layer2_res2_gpu{gpu_id}', 'layer2_allreduce2')

# Layer 3 connections
for gpu_id in range(8, 16):
    dot.edge('layer2_allreduce2', f'layer3_qkv_gpu{gpu_id}')
    dot.edge(f'layer3_qkv_gpu{gpu_id}', f'layer3_attn_gpu{gpu_id}')
    dot.edge(f'layer3_attn_gpu{gpu_id}', f'layer3_out_gpu{gpu_id}')
    dot.edge(f'layer3_out_gpu{gpu_id}', f'layer3_res1_gpu{gpu_id}')
    dot.edge(f'layer3_res1_gpu{gpu_id}', f'layer3_ffn1_gpu{gpu_id}')
    dot.edge(f'layer3_ffn1_gpu{gpu_id}', f'layer3_ffn2_gpu{gpu_id}')
    dot.edge(f'layer3_ffn2_gpu{gpu_id}', f'layer3_res2_gpu{gpu_id}')
    
    # Connect to output
    dot.edge(f'layer3_res2_gpu{gpu_id}', f'output_proj_gpu{gpu_id}')
    dot.edge(f'output_proj_gpu{gpu_id}', 'output_allreduce')

# Save files
with open('/home/wzc/data/file-share/logs/2025-10-09-11-30-42/baseline_dag.dot', 'w') as f:
    f.write(dot.source)

dot.render('/home/wzc/data/file-share/logs/2025-10-09-11-30-42/baseline_dag', format='svg', cleanup=False)

print("Baseline DAG generated successfully!")
print("Files saved:")
print("- /home/wzc/data/file-share/logs/2025-10-09-11-30-42/baseline_dag.dot")
print("- /home/wzc/data/file-share/logs/2025-10-09-11-30-42/baseline_dag.svg")