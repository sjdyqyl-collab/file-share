import graphviz

# Create a new directed graph for the dense model
dot = graphviz.Digraph('dense_model_dag', comment='Dense Model Layer-wise Deployment on 16 GPUs')
dot.attr(rankdir='TB', size='20,20')

# Define node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
dot.attr('edge', arrowhead='normal')

# Input node
dot.node('input', 'Input\n[batch=1024, seq_len, hidden=8192]', shape='ellipse', fillcolor='lightgreen')

# Add 16 layers, each on a different GPU
for layer in range(1, 17):
    gpu_id = layer - 1  # GPU 0 to 15
    
    # Layer norm 1
    dot.node(f'ln1_{layer}', f'LayerNorm1\n[1024, seq_len, 8192]\nGPU {gpu_id}', fillcolor='lightyellow')
    
    # Multi-head attention
    dot.node(f'mha_q_{layer}', f'MHA Q Linear\n[1024, seq_len, 8192]→[1024, seq_len, 8192]\nGPU {gpu_id}')
    dot.node(f'mha_k_{layer}', f'MHA K Linear\n[1024, seq_len, 8192]→[1024, seq_len, 8192]\nGPU {gpu_id}')
    dot.node(f'mha_v_{layer}', f'MHA V Linear\n[1024, seq_len, 8192]→[1024, seq_len, 8192]\nGPU {gpu_id}')
    dot.node(f'mha_attn_{layer}', f'MHA Attention\n[1024, seq_len, 8192]\nGPU {gpu_id}')
    dot.node(f'mha_out_{layer}', f'MHA Output Linear\n[1024, seq_len, 8192]→[1024, seq_len, 8192]\nGPU {gpu_id}')
    
    # First residual add
    dot.node(f'residual1_{layer}', f'Residual Add 1\n[1024, seq_len, 8192] + [1024, seq_len, 8192]\nGPU {gpu_id}', shape='parallelogram', fillcolor='lightcoral')
    
    # Layer norm 2
    dot.node(f'ln2_{layer}', f'LayerNorm2\n[1024, seq_len, 8192]\nGPU {gpu_id}', fillcolor='lightyellow')
    
    # FFN
    dot.node(f'ffn1_{layer}', f'FFN Linear1\n[1024, seq_len, 8192]→[1024, seq_len, 32768]\nGPU {gpu_id}')
    dot.node(f'gelu_{layer}', f'GELU\n[1024, seq_len, 32768]\nGPU {gpu_id}')
    dot.node(f'ffn2_{layer}', f'FFN Linear2\n[1024, seq_len, 32768]→[1024, seq_len, 8192]\nGPU {gpu_id}')
    
    # Second residual add
    dot.node(f'residual2_{layer}', f'Residual Add 2\n[1024, seq_len, 8192] + [1024, seq_len, 8192]\nGPU {gpu_id}', shape='parallelogram', fillcolor='lightcoral')
    
    # Communication nodes between GPUs
    if layer > 1:
        dot.node(f'comm_{layer-1}_{layer}', f'GPU {layer-2} → GPU {layer-1}\n[1024, seq_len, 8192]', shape='ellipse', fillcolor='lightgreen')

# Output node
dot.node('output', 'Output\n[batch=1024, seq_len, hidden=8192]', shape='ellipse', fillcolor='lightgreen')

# Connect the DAG
# Connect input to first layer
if 1 <= 16:
    dot.edge('input', 'ln1_1')
    dot.edge('ln1_1', 'mha_q_1')
    dot.edge('ln1_1', 'mha_k_1')
    dot.edge('ln1_1', 'mha_v_1')
    dot.edge('mha_q_1', 'mha_attn_1')
    dot.edge('mha_k_1', 'mha_attn_1')
    dot.edge('mha_v_1', 'mha_attn_1')
    dot.edge('mha_attn_1', 'mha_out_1')
    dot.edge('input', 'residual1_1')  # Residual connection
    dot.edge('mha_out_1', 'residual1_1')
    dot.edge('residual1_1', 'ln2_1')
    dot.edge('ln2_1', 'ffn1_1')
    dot.edge('ffn1_1', 'gelu_1')
    dot.edge('gelu_1', 'ffn2_1')
    dot.edge('residual1_1', 'residual2_1')  # Residual connection
    dot.edge('ffn2_1', 'residual2_1')

# Connect layers with communication
for layer in range(2, 17):
    prev_layer = layer - 1
    
    # Communication between GPUs
    dot.edge(f'residual2_{prev_layer}', f'comm_{prev_layer}_{layer}')
    dot.edge(f'comm_{prev_layer}_{layer}', f'ln1_{layer}')
    
    # Layer connections
    dot.edge(f'ln1_{layer}', f'mha_q_{layer}')
    dot.edge(f'ln1_{layer}', f'mha_k_{layer}')
    dot.edge(f'ln1_{layer}', f'mha_v_{layer}')
    dot.edge(f'mha_q_{layer}', f'mha_attn_{layer}')
    dot.edge(f'mha_k_{layer}', f'mha_attn_{layer}')
    dot.edge(f'mha_v_{layer}', f'mha_attn_{layer}')
    dot.edge(f'mha_attn_{layer}', f'mha_out_{layer}')
    dot.edge(f'comm_{prev_layer}_{layer}', f'residual1_{layer}')  # Residual connection
    dot.edge(f'mha_out_{layer}', f'residual1_{layer}')
    dot.edge('residual1_{layer}', f'ln2_{layer}')
    dot.edge(f'ln2_{layer}', f'ffn1_{layer}')
    dot.edge(f'ffn1_{layer}', f'gelu_{layer}')
    dot.edge(f'gelu_{layer}', f'ffn2_{layer}')
    dot.edge('residual1_{layer}', f'residual2_{layer}')  # Residual connection
    dot.edge(f'ffn2_{layer}', f'residual2_{layer}')

# Connect final layer to output
if 16 >= 1:
    dot.edge('residual2_16', 'output')

# Save the DAG
dot.format = 'svg'
dot.render('/home/wzc/data/file-share/submission/dense_model_dag', cleanup=True)
print("Dense model DAG saved to /home/wzc/data/file-share/submission/dense_model_dag.svg")