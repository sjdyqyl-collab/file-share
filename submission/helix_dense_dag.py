import graphviz

# Create DAG for Helix Dense Transformer with 16 GPUs
dot = graphviz.Digraph('Helix_Dense_Transformer', comment='Helix Two-Level Partitioning - Dense Model')
dot.attr(rankdir='TB', size='20,30')

# Define styles
with dot.subgraph(name='cluster_legend') as c:
    c.attr(label='Legend', style='dashed')
    c.node('comp', 'Computation', shape='rectangle')
    c.node('comm', 'Communication', shape='ellipse')
    c.node('route', 'Routing/Aggregation', shape='parallelogram')

# Input processing
dot.node('input', 'Input\nB×L×D', shape='ellipse')

# Layer 1: Multi-Head Attention with Two-Level Partitioning
with dot.subgraph(name='cluster_layer1') as c:
    c.attr(label='Layer 1 - Multi-Head Attention', style='rounded')
    
    # Input distribution across 16 GPUs
    for i in range(4):  # 4 head groups (n=4)
        for j in range(4):  # 4 dimension slices (m=4)
            gpu_id = i * 4 + j
            
            # QKV projection
            c.node(f'q_proj_l1_{i}_{j}', f'Q Projection\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
            c.node(f'k_proj_l1_{i}_{j}', f'K Projection\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
            c.node(f'v_proj_l1_{i}_{j}', f'V Projection\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
            
            # Attention computation
            c.node(f'attn_l1_{i}_{j}', f'Attention\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
            
            # Connections within partition
            c.edge(f'q_proj_l1_{i}_{j}', f'attn_l1_{i}_{j}')
            c.edge(f'k_proj_l1_{i}_{j}', f'attn_l1_{i}_{j}')
            c.edge(f'v_proj_l1_{i}_{j}', f'attn_l1_{i}_{j}')
    
    # Dimension slice concatenation within head groups
    for i in range(4):
        c.node(f'concat_dim_l1_{i}', f'Concatenate\nDimension Slices\nHead Group {i}\nGPU {i*4}-{(i+1)*4-1}', shape='parallelogram')
        for j in range(4):
            c.edge(f'attn_l1_{i}_{j}', f'concat_dim_l1_{i}')
    
    # Head group concatenation
    c.node('concat_heads_l1', 'Concatenate\nAll Head Groups\nAll GPUs', shape='parallelogram')
    for i in range(4):
        c.edge(f'concat_dim_l1_{i}', 'concat_heads_l1')
    
    # Residual connection
    c.node('residual_l1', 'Residual Add\nB×L×D\nAll GPUs', shape='parallelogram')
    c.edge('concat_heads_l1', 'residual_l1')

# Layer 1: MLP
with dot.subgraph(name='cluster_mlp1') as c:
    c.attr(label='Layer 1 - MLP', style='rounded')
    
    # MLP tensor parallelism across 16 GPUs
    for gpu_id in range(16):
        c.node(f'mlp_fc1_l1_{gpu_id}', f'MLP FC1\nB×L×2048\nGPU {gpu_id}', shape='rectangle')
        c.node(f'mlp_gelu_l1_{gpu_id}', f'GELU\nB×L×2048\nGPU {gpu_id}', shape='rectangle')
        c.node(f'mlp_fc2_l1_{gpu_id}', f'MLP FC2\nB×L×D\nGPU {gpu_id}', shape='rectangle')
        
        c.edge(f'mlp_fc1_l1_{gpu_id}', f'mlp_gelu_l1_{gpu_id}')
        c.edge(f'mlp_gelu_l1_{gpu_id}', f'mlp_fc2_l1_{gpu_id}')
    
    # MLP aggregation
    c.node('mlp_agg_l1', 'MLP Output\nAggregation\nB×L×D\nAll GPUs', shape='parallelogram')
    for gpu_id in range(16):
        c.edge(f'mlp_fc2_l1_{gpu_id}', 'mlp_agg_l1')
    
    # MLP residual
    c.node('mlp_residual_l1', 'MLP Residual Add\nB×L×D\nAll GPUs', shape='parallelogram')
    c.edge('mlp_agg_l1', 'mlp_residual_l1')

# Layer 2: Multi-Head Attention (same structure as Layer 1)
with dot.subgraph(name='cluster_layer2') as c:
    c.attr(label='Layer 2 - Multi-Head Attention', style='rounded')
    
    for i in range(4):
        for j in range(4):
            gpu_id = i * 4 + j
            c.node(f'q_proj_l2_{i}_{j}', f'Q Projection\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
            c.node(f'k_proj_l2_{i}_{j}', f'K Projection\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
            c.node(f'v_proj_l2_{i}_{j}', f'V Projection\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
            c.node(f'attn_l2_{i}_{j}', f'Attention\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
            
            c.edge(f'q_proj_l2_{i}_{j}', f'attn_l2_{i}_{j}')
            c.edge(f'k_proj_l2_{i}_{j}', f'attn_l2_{i}_{j}')
            c.edge(f'v_proj_l2_{i}_{j}', f'attn_l2_{i}_{j}')
    
    for i in range(4):
        c.node(f'concat_dim_l2_{i}', f'Concatenate\nDimension Slices\nHead Group {i}\nGPU {i*4}-{(i+1)*4-1}', shape='parallelogram')
        for j in range(4):
            c.edge(f'attn_l2_{i}_{j}', f'concat_dim_l2_{i}')
    
    c.node('concat_heads_l2', 'Concatenate\nAll Head Groups\nAll GPUs', shape='parallelogram')
    for i in range(4):
        c.edge(f'concat_dim_l2_{i}', 'concat_heads_l2')
    
    c.node('residual_l2', 'Residual Add\nB×L×D\nAll GPUs', shape='parallelogram')
    c.edge('concat_heads_l2', 'residual_l2')

# Layer 2: MLP
with dot.subgraph(name='cluster_mlp2') as c:
    c.attr(label='Layer 2 - MLP', style='rounded')
    
    for gpu_id in range(16):
        c.node(f'mlp_fc1_l2_{gpu_id}', f'MLP FC1\nB×L×2048\nGPU {gpu_id}', shape='rectangle')
        c.node(f'mlp_gelu_l2_{gpu_id}', f'GELU\nB×L×2048\nGPU {gpu_id}', shape='rectangle')
        c.node(f'mlp_fc2_l2_{gpu_id}', f'MLP FC2\nB×L×D\nGPU {gpu_id}', shape='rectangle')
        
        c.edge(f'mlp_fc1_l2_{gpu_id}', f'mlp_gelu_l2_{gpu_id}')
        c.edge(f'mlp_gelu_l2_{gpu_id}', f'mlp_fc2_l2_{gpu_id}')
    
    c.node('mlp_agg_l2', 'MLP Output\nAggregation\nB×L×D\nAll GPUs', shape='parallelogram')
    for gpu_id in range(16):
        c.edge(f'mlp_fc2_l2_{gpu_id}', 'mlp_agg_l2')
    
    c.node('mlp_residual_l2', 'MLP Residual Add\nB×L×D\nAll GPUs', shape='parallelogram')
    c.edge('mlp_agg_l2', 'mlp_residual_l2')

# Continue for Layer 3 and 4 (same structure)
for layer in [3, 4]:
    with dot.subgraph(name=f'cluster_layer{layer}') as c:
        c.attr(label=f'Layer {layer} - Multi-Head Attention', style='rounded')
        
        for i in range(4):
            for j in range(4):
                gpu_id = i * 4 + j
                c.node(f'q_proj_l{layer}_{i}_{j}', f'Q Projection\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
                c.node(f'k_proj_l{layer}_{i}_{j}', f'K Projection\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
                c.node(f'v_proj_l{layer}_{i}_{j}', f'V Projection\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
                c.node(f'attn_l{layer}_{i}_{j}', f'Attention\nB×L×d_s\nGPU {gpu_id}', shape='rectangle')
                
                c.edge(f'q_proj_l{layer}_{i}_{j}', f'attn_l{layer}_{i}_{j}')
                c.edge(f'k_proj_l{layer}_{i}_{j}', f'attn_l{layer}_{i}_{j}')
                c.edge(f'v_proj_l{layer}_{i}_{j}', f'attn_l{layer}_{i}_{j}')
        
        for i in range(4):
            c.node(f'concat_dim_l{layer}_{i}', f'Concatenate\nDimension Slices\nHead Group {i}\nGPU {i*4}-{(i+1)*4-1}', shape='parallelogram')
            for j in range(4):
                c.edge(f'attn_l{layer}_{i}_{j}', f'concat_dim_l{layer}_{i}')
        
        c.node(f'concat_heads_l{layer}', f'Concatenate\nAll Head Groups\nAll GPUs', shape='parallelogram')
        for i in range(4):
            c.edge(f'concat_dim_l{layer}_{i}', f'concat_heads_l{layer}')
        
        c.node(f'residual_l{layer}', f'Residual Add\nB×L×D\nAll GPUs', shape='parallelogram')
        c.edge(f'concat_heads_l{layer}', f'residual_l{layer}')

    with dot.subgraph(name=f'cluster_mlp{layer}') as c:
        c.attr(label=f'Layer {layer} - MLP', style='rounded')
        
        for gpu_id in range(16):
            c.node(f'mlp_fc1_l{layer}_{gpu_id}', f'MLP FC1\nB×L×2048\nGPU {gpu_id}', shape='rectangle')
            c.node(f'mlp_gelu_l{layer}_{gpu_id}', f'GELU\nB×L×2048\nGPU {gpu_id}', shape='rectangle')
            c.node(f'mlp_fc2_l{layer}_{gpu_id}', f'MLP FC2\nB×L×D\nGPU {gpu_id}', shape='rectangle')
            
            c.edge(f'mlp_fc1_l{layer}_{gpu_id}', f'mlp_gelu_l{layer}_{gpu_id}')
            c.edge(f'mlp_gelu_l{layer}_{gpu_id}', f'mlp_fc2_l{layer}_{gpu_id}')
        
        c.node(f'mlp_agg_l{layer}', f'MLP Output\nAggregation\nB×L×D\nAll GPUs', shape='parallelogram')
        for gpu_id in range(16):
            c.edge(f'mlp_fc2_l{layer}_{gpu_id}', f'mlp_agg_l{layer}')
        
        c.node(f'mlp_residual_l{layer}', f'MLP Residual Add\nB×L×D\nAll GPUs', shape='parallelogram')
        c.edge(f'mlp_agg_l{layer}', f'mlp_residual_l{layer}')

# Connect layers
for layer in range(1, 5):
    if layer == 1:
        dot.edge('input', 'q_proj_l1_0_0')
        dot.edge('input', 'k_proj_l1_0_0')
        dot.edge('input', 'v_proj_l1_0_0')
        # Need to connect input to all 16 GPUs
        for i in range(4):
            for j in range(4):
                gpu_id = i * 4 + j
                if not (i == 0 and j == 0):
                    dot.edge('input', f'q_proj_l1_{i}_{j}')
                    dot.edge('input', f'k_proj_l1_{i}_{j}')
                    dot.edge('input', f'v_proj_l1_{i}_{j}')
    
    # Connect residual inputs
    if layer == 1:
        dot.edge('input', 'residual_l1')
        dot.edge('residual_l1', 'mlp_fc1_l1_0')
        for gpu_id in range(16):
            if gpu_id != 0:
                dot.edge('residual_l1', f'mlp_fc1_l1_{gpu_id}')
        dot.edge('residual_l1', 'mlp_residual_l1')
    else:
        prev_layer = layer - 1
        dot.edge(f'mlp_residual_l{prev_layer}', f'q_proj_l{layer}_0_0')
        dot.edge(f'mlp_residual_l{prev_layer}', f'k_proj_l{layer}_0_0')
        dot.edge(f'mlp_residual_l{prev_layer}', f'v_proj_l{layer}_0_0')
        
        for i in range(4):
            for j in range(4):
                gpu_id = i * 4 + j
                if not (i == 0 and j == 0):
                    dot.edge(f'mlp_residual_l{prev_layer}', f'q_proj_l{layer}_{i}_{j}')
                    dot.edge(f'mlp_residual_l{prev_layer}', f'k_proj_l{layer}_{i}_{j}')
                    dot.edge(f'mlp_residual_l{prev_layer}', f'v_proj_l{layer}_{i}_{j}')
        
        dot.edge(f'mlp_residual_l{prev_layer}', f'residual_l{layer}')
        dot.edge(f'residual_l{layer}', f'mlp_fc1_l{layer}_0')
        for gpu_id in range(16):
            if gpu_id != 0:
                dot.edge(f'residual_l{layer}', f'mlp_fc1_l{layer}_{gpu_id}')
        dot.edge(f'residual_l{layer}', f'mlp_residual_l{layer}')

# Final output
dot.node('output', 'Output\nB×L×D', shape='ellipse')
dot.edge('mlp_residual_l4', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/submission/helix_dense_dag', format='svg', cleanup=False)
print("Dense Transformer DAG saved to /home/wzc/data/file-share/submission/helix_dense_dag.svg")