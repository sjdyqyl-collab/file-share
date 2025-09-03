import graphviz
from graphviz import Digraph

# Create baseline DAG with TP=8, PP=2
dot = Digraph(comment='Baseline: TP=8, PP=2')
dot.attr(rankdir='TB', size='20,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation

# Model parameters
batch_size = 1024
seq_len = 2048
hidden_size = 8192  # 16 heads * 512 head_dim
ffn_hidden_size = 32768
num_layers = 16

# Pipeline stages: 2 stages, 8 layers each
# Stage 0: GPUs 0-7 (layers 0-7)
# Stage 1: GPUs 8-15 (layers 8-15)

# Input
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input')
    c.node('input', f'Input\\n({batch_size}, {seq_len}, {hidden_size})', shape='ellipse', fillcolor='lightblue')

# Stage 0: Layers 0-7 on GPUs 0-7
for layer_idx in range(8):
    gpu_id = layer_idx % 8
    
    # LayerNorm 1
    dot.node(f'ln1_{layer_idx}', f'LayerNorm1\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Multi-Head Attention with TP=8
    # QKV projection (column parallel)
    dot.node(f'qkv_proj_{layer_idx}', f'QKV Projection\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size//8})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Attention computation
    dot.node(f'attn_{layer_idx}', f'Multi-Head Attention\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size//8})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Output projection (row parallel)
    dot.node(f'out_proj_{layer_idx}', f'Output Projection\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Residual add 1
    dot.node(f'residual1_{layer_idx}', f'Residual Add 1\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # LayerNorm 2
    dot.node(f'ln2_{layer_idx}', f'LayerNorm2\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # MLP with TP=8
    # First linear (column parallel)
    dot.node(f'mlp1_{layer_idx}', f'MLP First Linear\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {ffn_hidden_size//8})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Activation
    dot.node(f'gelu_{layer_idx}', f'GELU\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {ffn_hidden_size//8})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Second linear (row parallel)
    dot.node(f'mlp2_{layer_idx}', f'MLP Second Linear\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Residual add 2
    dot.node(f'residual2_{layer_idx}', f'Residual Add 2\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='parallelogram', fillcolor='lightyellow')

# Communication between stages
with dot.subgraph(name='cluster_comm') as c:
    c.attr(label='Pipeline Communication')
    c.node('comm_stage0_to_stage1', f'Pipeline Communication\\nStage 0 → Stage 1\\n({batch_size}, {seq_len}, {hidden_size})', 
           shape='ellipse', fillcolor='lightblue')

# Stage 1: Layers 8-15 on GPUs 8-15
for layer_idx in range(8, 16):
    gpu_id = layer_idx % 8 + 8
    
    # LayerNorm 1
    dot.node(f'ln1_{layer_idx}', f'LayerNorm1\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Multi-Head Attention with TP=8
    dot.node(f'qkv_proj_{layer_idx}', f'QKV Projection\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size//8})', 
             shape='rectangle', fillcolor='lightgreen')
    dot.node(f'attn_{layer_idx}', f'Multi-Head Attention\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size//8})', 
             shape='rectangle', fillcolor='lightgreen')
    dot.node(f'out_proj_{layer_idx}', f'Output Projection\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Residual add 1
    dot.node(f'residual1_{layer_idx}', f'Residual Add 1\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # LayerNorm 2
    dot.node(f'ln2_{layer_idx}', f'LayerNorm2\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # MLP with TP=8
    dot.node(f'mlp1_{layer_idx}', f'MLP First Linear\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {ffn_hidden_size//8})', 
             shape='rectangle', fillcolor='lightgreen')
    dot.node(f'gelu_{layer_idx}', f'GELU\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {ffn_hidden_size//8})', 
             shape='rectangle', fillcolor='lightgreen')
    dot.node(f'mlp2_{layer_idx}', f'MLP Second Linear\\nTP=8\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Residual add 2
    dot.node(f'residual2_{layer_idx}', f'Residual Add 2\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='parallelogram', fillcolor='lightyellow')

# Output
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output')
    c.node('output', f'Output\\n({batch_size}, {seq_len}, {hidden_size})', shape='ellipse', fillcolor='lightblue')

# Connect nodes
# Input to first layer
dot.edge('input', 'ln1_0')

# Stage 0 connections
for layer_idx in range(8):
    dot.edge(f'ln1_{layer_idx}', f'qkv_proj_{layer_idx}')
    dot.edge(f'qkv_proj_{layer_idx}', f'attn_{layer_idx}')
    dot.edge(f'attn_{layer_idx}', f'out_proj_{layer_idx}')
    dot.edge(f'out_proj_{layer_idx}', f'residual1_{layer_idx}')
    
    # Add residual connection
    if layer_idx == 0:
        dot.edge('input', f'residual1_{layer_idx}')
    else:
        dot.edge(f'residual2_{layer_idx-1}', f'residual1_{layer_idx}')
    
    dot.edge(f'residual1_{layer_idx}', f'ln2_{layer_idx}')
    dot.edge(f'ln2_{layer_idx}', f'mlp1_{layer_idx}')
    dot.edge(f'mlp1_{layer_idx}', f'gelu_{layer_idx}')
    dot.edge(f'gelu_{layer_idx}', f'mlp2_{layer_idx}')
    dot.edge(f'mlp2_{layer_idx}', f'residual2_{layer_idx}')
    
    # Add residual connection
    dot.edge(f'residual1_{layer_idx}', f'residual2_{layer_idx}')

# Pipeline communication
dot.edge('residual2_7', 'comm_stage0_to_stage1')

# Stage 1 connections
for layer_idx in range(8, 16):
    dot.edge('comm_stage0_to_stage1', f'ln1_{layer_idx}')
    
    dot.edge(f'ln1_{layer_idx}', f'qkv_proj_{layer_idx}')
    dot.edge(f'qkv_proj_{layer_idx}', f'attn_{layer_idx}')
    dot.edge(f'attn_{layer_idx}', f'out_proj_{layer_idx}')
    dot.edge(f'out_proj_{layer_idx}', f'residual1_{layer_idx}')
    
    # Add residual connection
    if layer_idx == 8:
        dot.edge('comm_stage0_to_stage1', f'residual1_{layer_idx}')
    else:
        dot.edge(f'residual2_{layer_idx-1}', f'residual1_{layer_idx}')
    
    dot.edge(f'residual1_{layer_idx}', f'ln2_{layer_idx}')
    dot.edge(f'ln2_{layer_idx}', f'mlp1_{layer_idx}')
    dot.edge(f'mlp1_{layer_idx}', f'gelu_{layer_idx}')
    dot.edge(f'gelu_{layer_idx}', f'mlp2_{layer_idx}')
    dot.edge(f'mlp2_{layer_idx}', f'residual2_{layer_idx}')
    
    # Add residual connection
    dot.edge(f'residual1_{layer_idx}', f'residual2_{layer_idx}')

# Final output
dot.edge('residual2_15', 'output')

# Save files
dot.render('/home/wzc/data/file-share/2025-09-03-14-50-32/baseline_dag', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-03-14-50-32/baseline_dag.dot')

print("Baseline DAG generated successfully")