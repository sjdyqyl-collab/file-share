import graphviz

# Create a new directed graph for MoE model
dot = graphviz.Digraph(comment='Helix MoE Transformer DAG with 16-GPU Two-Level Partitioning')
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

# Layer 1 - First MoE Transformer Layer
with dot.subgraph(name='cluster_layer1') as c:
    c.attr(label='Layer 1 (MoE)', style='rounded', fillcolor='lightyellow')
    
    # LayerNorm 1 - All GPUs
    c.node('ln1_0', 'LayerNorm\n(B×L×D)\nGPU 0-3', shape='rectangle', fillcolor='lightblue')
    c.node('ln1_1', 'LayerNorm\n(B×L×D)\nGPU 4-7', shape='rectangle', fillcolor='lightblue')
    c.node('ln1_2', 'LayerNorm\n(B×L×D)\nGPU 8-11', shape='rectangle', fillcolor='lightblue')
    c.node('ln1_3', 'LayerNorm\n(B×L×D)\nGPU 12-15', shape='rectangle', fillcolor='lightblue')
    
    # Multi-Head Attention with 16-way partitioning (4x4 grid)
    for i in range(4):  # head groups
        for j in range(4):  # dimension slices
            device_id = i * 4 + j
            c.node(f'q_proj_{i}_{j}', f'Q Projection\n(B×L×d_s×h_g)\nGPU {device_id}', 
                   shape='rectangle', fillcolor='lightcoral')
            c.node(f'k_proj_{i}_{j}', f'K Projection\n(B×L×d_s×h_g)\nGPU {device_id}', 
                   shape='rectangle', fillcolor='lightcoral')
            c.node(f'v_proj_{i}_{j}', f'V Projection\n(B×L×d_s×h_g)\nGPU {device_id}', 
                   shape='rectangle', fillcolor='lightcoral')
            c.node(f'attn_{i}_{j}', f'Scaled Dot-Product Attention\n(B×L×d_s×h_g)\nGPU {device_id}', 
                   shape='rectangle', fillcolor='orange')
    
    # Concatenation for MHA
    for i in range(4):
        c.node(f'concat_dim_{i}', f'Dimension Concatenation\n(B×L×d×h_g)\nGPUs {i*4}-{(i+1)*4-1}', 
               shape='parallelogram', fillcolor='lightgreen', style='dashed')
    
    c.node('concat_heads', 'Head Group Concatenation\n(B×L×D)\nAll GPUs', 
           shape='parallelogram', fillcolor='lightgreen')
    c.node('out_proj', 'Output Projection\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightcoral')
    c.node('residual1', 'Residual Add\n(B×L×D)\nAll GPUs', shape='ellipse', fillcolor='yellow')
    
    # Second LayerNorm
    c.node('ln2', 'LayerNorm\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightblue')
    
    # MoE Gate - determines which tokens go to which experts
    c.node('gate', 'MoE Gate\n(B×L×num_experts)\nAll GPUs', shape='parallelogram', fillcolor='purple')
    
    # Expert selection (dashed lines indicate routing)
    c.node('expert_select', 'Expert Selection\n(B×L×top_k)\nAll GPUs', 
           shape='parallelogram', fillcolor='purple', style='dashed')
    
    # 8 Experts distributed across 16 GPUs (2 GPUs per expert)
    expert_gpu_mapping = {
        0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7],
        4: [8, 9], 5: [10, 11], 6: [12, 13], 7: [14, 15]
    }
    
    for expert_id in range(8):
        gpus = expert_gpu_mapping[expert_id]
        
        # Expert-specific MLP with tensor parallelism (2 GPUs per expert)
        for gpu_idx, gpu_id in enumerate(gpus):
            c.node(f'expert{expert_id}_linear1_{gpu_id}', 
                   f'Expert {expert_id} Linear1\n(B×L×ffn_hidden_size/2)\nGPU {gpu_id}', 
                   shape='rectangle', fillcolor='lightcoral')
            c.node(f'expert{expert_id}_gelu_{gpu_id}', 
                   f'Expert {expert_id} GELU\n(B×L×ffn_hidden_size/2)\nGPU {gpu_id}', 
                   shape='rectangle', fillcolor='lightblue')
            c.node(f'expert{expert_id}_linear2_{gpu_id}', 
                   f'Expert {expert_id} Linear2\n(B×L×D/2)\nGPU {gpu_id}', 
                   shape='rectangle', fillcolor='lightcoral')
        
        # Expert output aggregation
        c.node(f'expert{expert_id}_agg', 
               f'Expert {expert_id} Output Aggregation\n(B×L×D)\nGPUs {gpus[0]},{gpus[1]}', 
               shape='parallelogram', fillcolor='lightgreen')
    
    # Expert output aggregation across all experts
    c.node('expert_outputs', 'Expert Outputs Aggregation\n(B×L×D)\nAll GPUs', 
           shape='parallelogram', fillcolor='lightgreen')
    
    # Weighted sum based on gate scores
    c.node('weighted_sum', 'Weighted Sum\n(B×L×D)\nAll GPUs', 
           shape='parallelogram', fillcolor='lightgreen')
    
    # Final residual
    c.node('residual2', 'Residual Add\n(B×L×D)\nAll GPUs', shape='ellipse', fillcolor='yellow')

# Layer 2, 3, 4 (similar MoE structure)
for layer in [2, 3, 4]:
    with dot.subgraph(name=f'cluster_layer{layer}') as c:
        c.attr(label=f'Layer {layer} (MoE)', style='rounded', fillcolor='lightyellow')
        
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
        
        # Concatenation for MHA
        for i in range(4):
            c.node(f'concat_dim_{layer}_{i}', f'Dimension Concatenation\n(B×L×d×h_g)\nGPUs {i*4}-{(i+1)*4-1}', 
                   shape='parallelogram', fillcolor='lightgreen', style='dashed')
        
        c.node(f'concat_heads_{layer}', f'Head Group Concatenation\n(B×L×D)\nAll GPUs', 
               shape='parallelogram', fillcolor='lightgreen')
        c.node(f'out_proj_{layer}', f'Output Projection\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightcoral')
        c.node(f'residual{layer}_1', f'Residual Add\n(B×L×D)\nAll GPUs', shape='ellipse', fillcolor='yellow')
        
        # MoE for layer
        c.node(f'ln{layer}_mlp', f'LayerNorm\n(B×L×D)\nAll GPUs', shape='rectangle', fillcolor='lightblue')
        c.node(f'gate{layer}', f'MoE Gate\n(B×L×num_experts)\nAll GPUs', shape='parallelogram', fillcolor='purple')
        c.node(f'expert_select{layer}', f'Expert Selection\n(B×L×top_k)\nAll GPUs', 
               shape='parallelogram', fillcolor='purple', style='dashed')
        
        for expert_id in range(8):
            gpus = expert_gpu_mapping[expert_id]
            for gpu_idx, gpu_id in enumerate(gpus):
                c.node(f'expert{layer}_{expert_id}_linear1_{gpu_id}', 
                       f'Expert {expert_id} Linear1\n(B×L×ffn_hidden_size/2)\nGPU {gpu_id}', 
                       shape='rectangle', fillcolor='lightcoral')
                c.node(f'expert{layer}_{expert_id}_gelu_{gpu_id}', 
                       f'Expert {expert_id} GELU\n(B×L×ffn_hidden_size/2)\nGPU {gpu_id}', 
                       shape='rectangle', fillcolor='lightblue')
                c.node(f'expert{layer}_{expert_id}_linear2_{gpu_id}', 
                       f'Expert {expert_id} Linear2\n(B×L×D/2)\nGPU {gpu_id}', 
                       shape='rectangle', fillcolor='lightcoral')
            
            c.node(f'expert{layer}_{expert_id}_agg', 
                   f'Expert {expert_id} Output Aggregation\n(B×L×D)\nGPUs {gpus[0]},{gpus[1]}', 
                   shape='parallelogram', fillcolor='lightgreen')
        
        c.node(f'expert_outputs{layer}', f'Expert Outputs Aggregation\n(B×L×D)\nAll GPUs', 
               shape='parallelogram', fillcolor='lightgreen')
        c.node(f'weighted_sum{layer}', f'Weighted Sum\n(B×L×D)\nAll GPUs', 
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
dot.edge('ln2', 'gate')

# MoE connections
expert_gpu_mapping = {
    0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7],
    4: [8, 9], 5: [10, 11], 6: [12, 13], 7: [14, 15]
}

dot.edge('gate', 'expert_select')

for expert_id in range(8):
    gpus = expert_gpu_mapping[expert_id]
    for gpu_id in gpus:
        dot.edge('expert_select', f'expert{expert_id}_linear1_{gpu_id}', style='dashed')
        dot.edge(f'expert{expert_id}_linear1_{gpu_id}', f'expert{expert_id}_gelu_{gpu_id}')
        dot.edge(f'expert{expert_id}_gelu_{gpu_id}', f'expert{expert_id}_linear2_{gpu_id}')
        dot.edge(f'expert{expert_id}_linear2_{gpu_id}', f'expert{expert_id}_agg')
    
    dot.edge(f'expert{expert_id}_agg', 'expert_outputs')

dot.edge('expert_outputs', 'weighted_sum')
dot.edge('weighted_sum', 'residual2')
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
    dot.edge(f'ln{layer}_mlp', f'gate{layer}')
    dot.edge(f'gate{layer}', f'expert_select{layer}')
    
    for expert_id in range(8):
        gpus = expert_gpu_mapping[expert_id]
        for gpu_id in gpus:
            dot.edge(f'expert_select{layer}', f'expert{layer}_{expert_id}_linear1_{gpu_id}', style='dashed')
            dot.edge(f'expert{layer}_{expert_id}_linear1_{gpu_id}', f'expert{layer}_{expert_id}_gelu_{gpu_id}')
            dot.edge(f'expert{layer}_{expert_id}_gelu_{gpu_id}', f'expert{layer}_{expert_id}_linear2_{gpu_id}')
            dot.edge(f'expert{layer}_{expert_id}_linear2_{gpu_id}', f'expert{layer}_{expert_id}_agg')
        
        dot.edge(f'expert{layer}_{expert_id}_agg', f'expert_outputs{layer}')
    
    dot.edge(f'expert_outputs{layer}', f'weighted_sum{layer}')
    dot.edge(f'weighted_sum{layer}', f'residual{layer}_2')
    dot.edge(f'residual{layer}_1', f'residual{layer}_2')
    prev_node = f'residual{layer}_2'

# Output connections
dot.edge(prev_node, 'final_ln')
dot.edge('final_ln', 'lm_head')
dot.edge('lm_head', 'output')

# Save the MoE DAG
dot.render('/home/wzc/data/file-share/submission/helix_moe_dag', format='svg', cleanup=False)
print("MoE transformer DAG saved to /home/wzc/data/file-share/submission/helix_moe_dag.svg")