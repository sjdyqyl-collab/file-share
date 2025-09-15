#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

# Create baseline DAG with TP=8, PP=2
dot = Digraph(comment='Baseline: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)')
dot.attr(rankdir='TB', size='20,20')

# Define cluster for pipeline stage 0 (layers 0-1)
with dot.subgraph(name='cluster_pipeline_stage_0') as c0:
    c0.attr(label='Pipeline Stage 0 (GPU 0-7)', style='rounded', color='blue')
    
    # Input for stage 0
    c0.node('input_0', 'Input\n[batch_size=1024, seq_len=?, D=8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Layer 0 - MHA (TP=8)
    # QKV projections (column parallel)
    for i in range(8):
        c0.node(f'layer0_qkv_proj_{i}', f'QKV Projection GPU{i}\nInput: [batch_size=1024, seq_len=?, D=8192]\nOutput: [batch_size=1024, seq_len=?, heads=2, d_k=512]', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Attention computation per head
    for i in range(8):
        c0.node(f'layer0_attention_{i}', f'Attention GPU{i}\nInput: [batch_size=1024, seq_len=?, heads=2, d_k=512]\nOutput: [batch_size=1024, seq_len=?, heads=2, d_k=512]', 
                shape='rectangle', style='filled', fillcolor='yellow')
    
    # Output projection (row parallel)
    for i in range(8):
        c0.node(f'layer0_out_proj_{i}', f'Output Projection GPU{i}\nInput: [batch_size=1024, seq_len=?, heads=2, d_k=512]\nOutput: [batch_size=1024, seq_len=?, D=1024]', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # All-reduce for attention output
    c0.node('layer0_allreduce', 'All-Reduce\n[batch_size=1024, seq_len=?, D=8192]', shape='parallelogram', style='filled', fillcolor='orange')
    
    # Residual add
    c0.node('layer0_residual', 'Residual Add\n[batch_size=1024, seq_len=?, D=8192]', shape='parallelogram', style='filled', fillcolor='pink')
    
    # Layer norm
    c0.node('layer0_layernorm', 'Layer Norm\n[batch_size=1024, seq_len=?, D=8192]', shape='rectangle', style='filled', fillcolor='lightgray')
    
    # MLP (TP=8)
    # First linear (column parallel)
    for i in range(8):
        c0.node(f'layer0_mlp_linear1_{i}', f'MLP Linear1 GPU{i}\nInput: [batch_size=1024, seq_len=?, D=8192]\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=4096]', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Activation
    c0.node('layer0_mlp_activation', 'GELU\n[batch_size=1024, seq_len=?, ffn_hidden=32768]', shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Second linear (row parallel)
    for i in range(8):
        c0.node(f'layer0_mlp_linear2_{i}', f'MLP Linear2 GPU{i}\nInput: [batch_size=1024, seq_len=?, ffn_hidden=4096]\nOutput: [batch_size=1024, seq_len=?, D=1024]', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # All-reduce for MLP output
    c0.node('layer0_mlp_allreduce', 'All-Reduce\n[batch_size=1024, seq_len=?, D=8192]', shape='parallelogram', style='filled', fillcolor='orange')
    
    # MLP residual add
    c0.node('layer0_mlp_residual', 'Residual Add\n[batch_size=1024, seq_len=?, D=8192]', shape='parallelogram', style='filled', fillcolor='pink')
    
    # Pipeline send to stage 1
    c0.node('pipeline_send_0_1', 'Send to Stage 1\n[batch_size=1024, seq_len=?, D=8192]', shape='ellipse', style='filled', fillcolor='cyan')

# Define cluster for pipeline stage 1 (layers 2)
with dot.subgraph(name='cluster_pipeline_stage_1') as c1:
    c1.attr(label='Pipeline Stage 1 (GPU 8-15)', style='rounded', color='red')
    
    # Receive from stage 0
    c1.node('pipeline_recv_1_0', 'Receive from Stage 0\n[batch_size=1024, seq_len=?, D=8192]', shape='ellipse', style='filled', fillcolor='cyan')
    
    # Layer 1 - MHA (TP=8)
    # QKV projections (column parallel)
    for i in range(8):
        c1.node(f'layer1_qkv_proj_{i}', f'QKV Projection GPU{i+8}\nInput: [batch_size=1024, seq_len=?, D=8192]\nOutput: [batch_size=1024, seq_len=?, heads=2, d_k=512]', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Attention computation per head
    for i in range(8):
        c1.node(f'layer1_attention_{i}', f'Attention GPU{i+8}\nInput: [batch_size=1024, seq_len=?, heads=2, d_k=512]\nOutput: [batch_size=1024, seq_len=?, heads=2, d_k=512]', 
                shape='rectangle', style='filled', fillcolor='yellow')
    
    # Output projection (row parallel)
    for i in range(8):
        c1.node(f'layer1_out_proj_{i}', f'Output Projection GPU{i+8}\nInput: [batch_size=1024, seq_len=?, heads=2, d_k=512]\nOutput: [batch_size=1024, seq_len=?, D=1024]', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # All-reduce for attention output
    c1.node('layer1_allreduce', 'All-Reduce\n[batch_size=1024, seq_len=?, D=8192]', shape='parallelogram', style='filled', fillcolor='orange')
    
    # Residual add
    c1.node('layer1_residual', 'Residual Add\n[batch_size=1024, seq_len=?, D=8192]', shape='parallelogram', style='filled', fillcolor='pink')
    
    # Layer norm
    c1.node('layer1_layernorm', 'Layer Norm\n[batch_size=1024, seq_len=?, D=8192]', shape='rectangle', style='filled', fillcolor='lightgray')
    
    # MLP (TP=8)
    # First linear (column parallel)
    for i in range(8):
        c1.node(f'layer1_mlp_linear1_{i}', f'MLP Linear1 GPU{i+8}\nInput: [batch_size=1024, seq_len=?, D=8192]\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=4096]', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Activation
    c1.node('layer1_mlp_activation', 'GELU\n[batch_size=1024, seq_len=?, ffn_hidden=32768]', shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Second linear (row parallel)
    for i in range(8):
        c1.node(f'layer1_mlp_linear2_{i}', f'MLP Linear2 GPU{i+8}\nInput: [batch_size=1024, seq_len=?, ffn_hidden=4096]\nOutput: [batch_size=1024, seq_len=?, D=1024]', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # All-reduce for MLP output
    c1.node('layer1_mlp_allreduce', 'All-Reduce\n[batch_size=1024, seq_len=?, D=8192]', shape='parallelogram', style='filled', fillcolor='orange')
    
    # MLP residual add
    c1.node('layer1_mlp_residual', 'Residual Add\n[batch_size=1024, seq_len=?, D=8192]', shape='parallelogram', style='filled', fillcolor='pink')
    
    # Final output
    c1.node('final_output', 'Final Output\n[batch_size=1024, seq_len=?, D=8192]', shape='ellipse', style='filled', fillcolor='lightblue')

# Connections for pipeline stage 0
for i in range(8):
    dot.edge('input_0', f'layer0_qkv_proj_{i}')
    dot.edge(f'layer0_qkv_proj_{i}', f'layer0_attention_{i}')
    dot.edge(f'layer0_attention_{i}', f'layer0_out_proj_{i}')
    dot.edge(f'layer0_out_proj_{i}', 'layer0_allreduce')

dot.edge('layer0_allreduce', 'layer0_residual')
dot.edge('input_0', 'layer0_residual')  # Residual connection
dot.edge('layer0_residual', 'layer0_layernorm')

for i in range(8):
    dot.edge('layer0_layernorm', f'layer0_mlp_linear1_{i}')
    dot.edge(f'layer0_mlp_linear1_{i}', 'layer0_mlp_activation')
    dot.edge('layer0_mlp_activation', f'layer0_mlp_linear2_{i}')
    dot.edge(f'layer0_mlp_linear2_{i}', 'layer0_mlp_allreduce')

dot.edge('layer0_mlp_allreduce', 'layer0_mlp_residual')
dot.edge('layer0_residual', 'layer0_mlp_residual')  # Residual connection
dot.edge('layer0_mlp_residual', 'pipeline_send_0_1')

# Pipeline communication
dot.edge('pipeline_send_0_1', 'pipeline_recv_1_0')

# Connections for pipeline stage 1
dot.edge('pipeline_recv_1_0', f'layer1_qkv_proj_0')
for i in range(8):
    dot.edge('pipeline_recv_1_0', f'layer1_qkv_proj_{i}')
    dot.edge(f'layer1_qkv_proj_{i}', f'layer1_attention_{i}')
    dot.edge(f'layer1_attention_{i}', f'layer1_out_proj_{i}')
    dot.edge(f'layer1_out_proj_{i}', 'layer1_allreduce')

dot.edge('layer1_allreduce', 'layer1_residual')
dot.edge('pipeline_recv_1_0', 'layer1_residual')  # Residual connection
dot.edge('layer1_residual', 'layer1_layernorm')

for i in range(8):
    dot.edge('layer1_layernorm', f'layer1_mlp_linear1_{i}')
    dot.edge(f'layer1_mlp_linear1_{i}', 'layer1_mlp_activation')
    dot.edge('layer1_mlp_activation', f'layer1_mlp_linear2_{i}')
    dot.edge(f'layer1_mlp_linear2_{i}', 'layer1_mlp_allreduce')

dot.edge('layer1_mlp_allreduce', 'layer1_mlp_residual')
dot.edge('layer1_residual', 'layer1_mlp_residual')  # Residual connection
dot.edge('layer1_mlp_residual', 'final_output')

# Save the DAG
dot.render('/home/wzc/data/file-share/2025-09-12-16-52-38/baseline_dag', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-12-16-52-38/baseline_dag.dot')

print("Baseline DAG generated successfully!")
print("Files saved:")
print("- /home/wzc/data/file-share/2025-09-12-16-52-38/baseline_dag.svg")
print("- /home/wzc/data/file-share/2025-09-12-16-52-38/baseline_dag.dot")