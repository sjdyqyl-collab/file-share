import graphviz
from graphviz import Digraph

# Create proposed layer-wise deployment DAG
dot = Digraph(comment='Proposed: Layer-wise Deployment')
dot.attr(rankdir='TB', size='20,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation

# Model parameters
batch_size = 1024
seq_len = 2048
hidden_size = 8192
ffn_hidden_size = 32768
num_layers = 16

# Input
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input')
    c.node('input', f'Input\\n({batch_size}, {seq_len}, {hidden_size})', shape='ellipse', fillcolor='lightblue')

# Layer-wise deployment: 16 layers on 16 GPUs (1 layer per GPU)
for layer_idx in range(16):
    gpu_id = layer_idx
    
    # LayerNorm 1
    dot.node(f'ln1_{layer_idx}', f'LayerNorm1\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # QKV projection (full layer on single GPU)
    dot.node(f'qkv_proj_{layer_idx}', f'QKV Projection\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size*3})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Multi-Head Attention (all heads on single GPU)
    dot.node(f'attn_{layer_idx}', f'Multi-Head Attention\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Output projection
    dot.node(f'out_proj_{layer_idx}', f'Output Projection\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Residual add 1
    dot.node(f'residual1_{layer_idx}', f'Residual Add 1\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # LayerNorm 2
    dot.node(f'ln2_{layer_idx}', f'LayerNorm2\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # MLP First Linear
    dot.node(f'mlp1_{layer_idx}', f'MLP First Linear\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {ffn_hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Activation
    dot.node(f'gelu_{layer_idx}', f'GELU\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {ffn_hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # MLP Second Linear
    dot.node(f'mlp2_{layer_idx}', f'MLP Second Linear\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Residual add 2
    dot.node(f'residual2_{layer_idx}', f'Residual Add 2\\nGPU {gpu_id}\\n({batch_size}, {seq_len}, {hidden_size})', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Communication between layers
    if layer_idx < 15:
        dot.node(f'comm_{layer_idx}_to_{layer_idx+1}', 
                 f'Layer {layer_idx} → {layer_idx+1}\\n({batch_size}, {seq_len}, {hidden_size})', 
                 shape='ellipse', fillcolor='lightblue')

# Output
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output')
    c.node('output', f'Output\\n({batch_size}, {seq_len}, {hidden_size})', shape='ellipse', fillcolor='lightblue')

# Connect nodes
# Input to first layer
dot.edge('input', 'ln1_0')

# Layer connections
for layer_idx in range(16):
    dot.edge(f'ln1_{layer_idx}', f'qkv_proj_{layer_idx}')
    dot.edge(f'qkv_proj_{layer_idx}', f'attn_{layer_idx}')
    dot.edge(f'attn_{layer_idx}', f'out_proj_{layer_idx}')
    dot.edge(f'out_proj_{layer_idx}', f'residual1_{layer_idx}')
    
    # Add residual connection
    if layer_idx == 0:
        dot.edge('input', f'residual1_{layer_idx}')
    else:
        dot.edge(f'comm_{layer_idx-1}_to_{layer_idx}', f'residual1_{layer_idx}')
    
    dot.edge(f'residual1_{layer_idx}', f'ln2_{layer_idx}')
    dot.edge(f'ln2_{layer_idx}', f'mlp1_{layer_idx}')
    dot.edge(f'mlp1_{layer_idx}', f'gelu_{layer_idx}')
    dot.edge(f'gelu_{layer_idx}', f'mlp2_{layer_idx}')
    dot.edge(f'mlp2_{layer_idx}', f'residual2_{layer_idx}')
    
    # Add residual connection
    dot.edge(f'residual1_{layer_idx}', f'residual2_{layer_idx}')
    
    # Communication to next layer
    if layer_idx < 15:
        dot.edge(f'residual2_{layer_idx}', f'comm_{layer_idx}_to_{layer_idx+1}')

# Final output
dot.edge('residual2_15', 'output')

# Save files
dot.render('/home/wzc/data/file-share/2025-09-03-14-50-32/proposed_dag', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-03-14-50-32/proposed_dag.dot')

print("Proposed DAG generated successfully")