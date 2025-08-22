import graphviz

# Create a new directed graph
dot = graphviz.Digraph(comment='Helix Dense Transformer DAG with 16-GPU Two-Level Partitioning')
dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')

# Define node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
dot.attr('edge', fontsize='10')

# Input handling
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input Processing', style='rounded', fillcolor='lightgray')
    c.node('input', 'Input Tokens\n(B×L×D)\nAll GPUs', shape='ellipse', fillcolor='lightgreen')
    c.node('embed', 'Embedding Layer\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightblue')
    c.node('pos_enc', 'Positional Encoding\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightblue')

# Layer 1 - First Transformer Layer
with dot.subgraph(name='cluster_layer1') as c:
    c.attr(label='Layer 1', style='rounded', fillcolor='lightyellow')
    
    # LayerNorm 1 - All GPUs
    c.node('ln1_0', 'LayerNorm\n(B×L×D)\nGPU 0-3', shape='rectangle', fillcolor='lightblue')
    c.node('ln1_1', 'LayerNorm\n(B×L×D)\nGPU 4-7', shape='rectangle', fillcolor='lightblue')
    c.node('ln1_2', 'LayerNorm\n(B×L×D)\nGPU 8-11', shape='rectangle', fillcolor='lightblue')
    c.node('ln1_3', 'LayerNorm\n(B×L×D)\nGPU 12-15', shape='rectangle', fillcolor='lightblue')
    
    # Multi-Head Attention with 16-way partitioning (4x4 grid)
    # 4 head groups × 4 dimension slices = 16 partitions
    
    # QKV projections for each partition
    for i in range(4):  # head groups
        for j in range(4):  # dimension slices
            device_id = i * 4 + j
            c.node(f'q_proj_{i}_{j}', f'Q Projection\n(B×L×d_s×h_g)\nGPU {device_id}', 
                   shape='rectangle', fillcolor='lightcoral')
            c.node(f'k_proj_{i}_{j}', f'K Projection\n(B×L×d_s×h_g)\nGPU {device_id}', 
                   shape='rectangle', fillcolor='lightcoral')
            c.node(f'v_proj_{i}_{j}', f'V Projection\n(B×L×d_s×h_g)\nGPU {device_id}', 
                   shape='rectangle', fillcolor='lightcoral')
    
    # Attention computation for each partition
    for i in range(4):
        for j in range(4):
            device_id = i * 4 + j
            c.node(f'attn_{i}_{j}', f'Scaled Dot-Product Attention\n(B×L×d_s×h_g)\nGPU {device_id}', 
                   shape='rectangle', fillcolor='orange')
    
    # Intra-group concatenation (within each head group)
    for i in range(4):
        c.node(f'concat_dim_{i}', f'Dimension Concatenation\n(B×L×d×h_g)\nGPUs {i*4}-{(i+1)*4-1}', 
               shape='parallelogram', fillcolor='lightgreen', style='dashed')
    
    # Final concatenation across head groups
    c.node('concat_heads', 'Head Group Concatenation\n(B×L×D)\nAll GPUs', 
           shape='parallelogram', fillcolor='lightgreen')
    
    # Output projection
    c.node('out_proj', 'Output Projection\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightcoral')
    
    # Residual connection
    c.node('residual1', 'Residual Add\n(B×L×D)\nAll GPUs', shape='ellipse', fillcolor='yellow')
    
    # Second LayerNorm
    c.node('ln2', 'LayerNorm\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightblue')
    
    # MLP with tensor parallelism (16-way)
    # Column-parallel first linear
    for i in range(16):
        c.node(f'mlp_linear1_{i}', f'MLP Linear1\n(B×L×ffn_hidden_size/16)\nGPU {i}', 
               shape='rectangle', fillcolor='lightcoral')
    
    # Activation (local to each GPU)
    for i in range(16):
        c.node(f'mlp_gelu_{i}', f'GELU Activation\n(B×L×ffn_hidden_size/16)\nGPU {i}', 
               shape='rectangle', fillcolor='lightblue')
    
    # Row-parallel second linear
    for i in range(16):
        c.node(f'mlp_linear2_{i}', f'MLP Linear2\n(B×L×D/16)\nGPU {i}', 
               shape='rectangle', fillcolor='lightcoral')
    
    # MLP output aggregation
    c.node('mlp_agg', 'MLP Output Aggregation\n(B×L×D)\nAll GPUs', 
           shape='parallelogram', fillcolor='lightgreen')
    
    # Second residual
    c.node('residual2', 'Residual Add\n(B×L×D)\nAll GPUs', shape='ellipse', fillcolor='yellow')

# Layer 2, 3, 4 (similar structure)
for layer in [2, 3, 4]:
    with dot.subgraph(name=f'cluster_layer{layer}') as c:
        c.attr(label=f'Layer {layer}', style='rounded', fillcolor='lightyellow')
        
        # Similar structure as Layer 1
        c.node(f'ln{layer}_0', f'LayerNorm\n(B×L×D)\nGPU 0-3', shape='rectangle', fillcolor='lightblue')
        c.node(f'ln{layer}_1', f'LayerNorm\n(B×L×D)\nGPU 4-7', shape='rectangle', fillcolor='lightblue')
        c.node(f'ln{layer}_2', f'LayerNorm\n(B×L×D)\nGPU 8-11', shape='rectangle', fillcolor='lightblue')
        c.node(f'ln{layer}_3', f'LayerNorm\n(B×L×D)\nGPU 12-15', shape='rectangle', fillcolor='lightblue')
        
        # MHA for layer
        for i in range(4):
            for j in range(4):
                device_id = i * 4 + j
                c.node(f'q_proj_{layer}_{i}_{j}', f'Q Projection\n(B×L×d_s×h_g)\nGPU {device_id}', 
                       shape='rectangle', fillcolor='lightcoral')
                c.node(f'k_proj_{layer}_{i}_{j}', f'K Projection\n(B×L×d_s×h_g)\nGPU {device_id}', 
                       shape='rectangle', fillcolor='lightcoral')
                c.node(f'v_proj_{layer}_{i}_{j}', f'V Projection\n(B×L×d_s×h_g)\nGPU {device_id}', 
                       shape='rectangle', fillcolor='lightcoral')
                c.node(f'attn_{layer}_{i}_{j}', f'Scaled Dot-Product Attention\n(B×L×d_s×h_g)\nGPU {device_id}', 
                       shape='rectangle', fillcolor='orange')
        
        # Concatenation for layer
        for i in range(4):
            c.node(f'concat_dim_{layer}_{i}', f'Dimension Concatenation\n(B×L×d×h_g)\nGPUs {i*4}-{(i+1)*4-1}', 
                   shape='parallelogram', fillcolor='lightgreen', style='dashed')
        
        c.node(f'concat_heads_{layer}', f'Head Group Concatenation\n(B×L×D)\nAll GPUs', 
               shape='parallelogram', fillcolor='lightgreen')
        c.node(f'out_proj_{layer}', f'Output Projection\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightcoral')
        c.node(f'residual{layer}_1', f'Residual Add\n(B×L×D)\nAll GPUs', shape='ellipse', fillcolor='yellow')
        
        # MLP for layer
        c.node(f'ln{layer}_mlp', f'LayerNorm\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightblue')
        
        for i in range(16):
            c.node(f'mlp_linear1_{layer}_{i}', f'MLP Linear1\n(B×L×ffn_hidden_size/16)\nGPU {i}', 
                   shape='rectangle', fillcolor='lightcoral')
            c.node(f'mlp_gelu_{layer}_{i}', f'GELU Activation\n(B×L×ffn_hidden_size/16)\nGPU {i}', 
                   shape='rectangle', fillcolor='lightblue')
            c.node(f'mlp_linear2_{layer}_{i}', f'MLP Linear2\n(B×L×D/16)\nGPU {i}', 
                   shape='rectangle', fillcolor='lightcoral')
        
        c.node(f'mlp_agg_{layer}', f'MLP Output Aggregation\n(B×L×D)\nAll GPUs', 
               shape='parallelogram', fillcolor='lightgreen')
        c.node(f'residual{layer}_2', f'Residual Add\n(B×L×D)\nAll GPUs', shape='ellipse', fillcolor='yellow')

# Output processing
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Processing', style='rounded', fillcolor='lightgray')
    c.node('final_ln', 'Final LayerNorm\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightblue')
    c.node('lm_head', 'Language Model Head\n(B×L×Vocab)\nAll GPUs', shape='rectangle', fillcolor='lightcoral')
    c.node('output', 'Output Tokens\n(B×L)\nAll GPUs', shape='ellipse', fillcolor='lightgreen')

# Connect the nodes
# Input connections
dot.edge('input', 'embed')
dot.edge('embed', 'pos_enc')

# Layer 1 connections
prev_node = 'pos_enc'
for i in range(4):
    dot.edge(prev_node, f'ln1_{i}')
    
    for j in range(4):
        device_id = i * 4 + j
        dot.edge(f'ln1_{i}', f'q_proj_{i}_{j}')
        dot.edge(f'ln1_{i}', f'k_proj_{i}_{j}')
        dot.edge(f'ln1_{i}', f'v_proj_{i}_{j}')
        dot.edge(f'q_proj_{i}_{j}', f'attn_{i}_{j}')
        dot.edge(f'k_proj_{i}_{j}', f'attn_{i}_{j}')
        dot.edge(f'v_proj_{i}_{j}', f'attn_{i}_{j}')
        dot.edge(f'attn_{i}_{j}', f'concat_dim_{i}')
    
    dot.edge(f'concat_dim_{i}', 'concat_heads')
    
dot.edge('concat_heads', 'out_proj')
dot.edge('out_proj', 'residual1')
dot.edge(prev_node, 'residual1')  # Residual connection
dot.edge('residual1', 'ln2')

# MLP connections
for i in range(16):
    dot.edge('ln2', f'mlp_linear1_{i}')
    dot.edge(f'mlp_linear1_{i}', f'mlp_gelu_{i}')
    dot.edge(f'mlp_gelu_{i}', f'mlp_linear2_{i}')
    dot.edge(f'mlp_linear2_{i}', 'mlp_agg')

dot.edge('mlp_agg', 'residual2')
dot.edge('residual1', 'residual2')  # Residual connection

# Connect layers
prev_node = 'residual2'
for layer in [2, 3, 4]:
    for i in range(4):
        dot.edge(prev_node, f'ln{layer}_{i}')
        
        for j in range(4):
            device_id = i * 4 + j
            dot.edge(f'ln{layer}_{i}', f'q_proj_{layer}_{i}_{j}')
            dot.edge(f'ln{layer}_{i}', f'k_proj_{layer}_{i}_{j}')
            dot.edge(f'ln{layer}_{i}', f'v_proj_{layer}_{i}_{j}')
            dot.edge(f'q_proj_{layer}_{i}_{j}', f'attn_{layer}_{i}_{j}')
            dot.edge(f'k_proj_{layer}_{i}_{j}', f'attn_{layer}_{i}_{j}')
            dot.edge(f'v_proj_{layer}_{i}_{j}', f'attn_{layer}_{i}_{j}')
            dot.edge(f'attn_{layer}_{i}_{j}', f'concat_dim_{layer}_{i}')
        
        dot.edge(f'concat_dim_{layer}_{i}', f'concat_heads_{layer}')
    
    dot.edge(f'concat_heads_{layer}', f'out_proj_{layer}')
    dot.edge(f'out_proj_{layer}', f'residual{layer}_1')
    dot.edge(prev_node, f'residual{layer}_1')
    dot.edge(f'residual{layer}_1', f'ln{layer}_mlp')
    
    for i in range(16):
        dot.edge(f'ln{layer}_mlp', f'mlp_linear1_{layer}_{i}')
        dot.edge(f'mlp_linear1_{layer}_{i}', f'mlp_gelu_{layer}_{i}')
        dot.edge(f'mlp_gelu_{layer}_{i}', f'mlp_linear2_{layer}_{i}')
        dot.edge(f'mlp_linear2_{layer}_{i}', f'mlp_agg_{layer}')
    
    dot.edge(f'mlp_agg_{layer}', f'residual{layer}_2')
    dot.edge(f'residual{layer}_1', f'residual{layer}_2')
    prev_node = f'residual{layer}_2'

# Output connections
dot.edge(prev_node, 'final_ln')
dot.edge('final_ln', 'lm_head')
dot.edge('lm_head', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/submission/helix_dense_dag', format='svg', cleanup=False)
print("Dense transformer DAG saved to /home/wzc/data/file-share/submission/helix_dense_dag.svg")