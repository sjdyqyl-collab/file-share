import graphviz

def create_proposed_dag():
    dot = graphviz.Digraph('proposed_transformer', comment='Proposed 2-layer Transformer with Two-Level Attention Partitioning')
    dot.attr(rankdir='TB', splines='ortho', compound='true')
    
    # Define subgraphs for each GPU showing 4x4 partitioning
    for gpu_id in range(16):
        head_group = gpu_id // 4
        dim_slice = gpu_id % 4
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as gpu:
            gpu.attr(label=f'GPU {gpu_id}\\nGroup {head_group}, Slice {dim_slice}', style='rounded', color='blue')
    
    # Input layer - broadcast to all GPUs
    dot.node('input', 'Input\n[batch_size=1024, seq_len=?, embedding_dim=8192]', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Broadcast node
    dot.node('broadcast', 'Broadcast Input\n[1024, ?, 8192]\nto all GPUs', 
             shape='parallelogram', style='filled', fillcolor='orange')
    
    # Layer 1 - Two-level partitioned attention
    for gpu_id in range(16):
        head_group = gpu_id // 4
        dim_slice = gpu_id % 4
        heads_start = head_group * 4
        heads_end = heads_start + 3
        dim_start = dim_slice * 128
        dim_end = dim_start + 127
        
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as gpu:
            # Q, K, V projections for specific partition
            dot.node(f'q_proj_l1_g{gpu_id}', 
                     f'Q Projection L1\n[1024, ?, 512]\nHeads {heads_start}-{heads_end}\\nDims {dim_start}-{dim_end}', 
                     shape='rectangle', style='filled', fillcolor='lightblue')
            dot.node(f'k_proj_l1_g{gpu_id}', 
                     f'K Projection L1\n[1024, ?, 512]\nHeads {heads_start}-{heads_end}\\nDims {dim_start}-{dim_end}', 
                     shape='rectangle', style='filled', fillcolor='lightblue')
            dot.node(f'v_proj_l1_g{gpu_id}', 
                     f'V Projection L1\n[1024, ?, 512]\nHeads {heads_start}-{heads_end}\\nDims {dim_start}-{dim_end}', 
                     shape='rectangle', style='filled', fillcolor='lightblue')
            
            # Attention computation for partition
            dot.node(f'attn_l1_g{gpu_id}', 
                     f'Attention L1\n[1024, ?, 4, 128]\nHeads {heads_start}-{heads_end}\\nDims {dim_start}-{dim_end}', 
                     shape='rectangle', style='filled', fillcolor='yellow')
            
            # MLP components - also partitioned
            dot.node(f'mlp_fc_l1_g{gpu_id}', 
                     f'MLP FC L1\n[1024, ?, 8192]\nPartition {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            dot.node(f'mlp_gelu_l1_g{gpu_id}', 
                     f'MLP GELU L1\n[1024, ?, 8192]\nPartition {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            dot.node(f'mlp_proj_l1_g{gpu_id}', 
                     f'MLP Projection L1\n[1024, ?, 8192]\nPartition {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Local residual connections
            dot.node(f'residual_l1_local_{gpu_id}', 
                     f'Local Residual L1\n[1024, ?, 512]\nGPU {gpu_id}', 
                     shape='parallelogram', style='filled', fillcolor='lightgray')
    
    # Layer 1 aggregation nodes
    for head_group in range(4):
        # Concatenate dimension slices within each head group
        dot.node(f'concat_dims_l1_g{head_group}', 
                 f'Concat Dims L1\n[1024, ?, 4, 512]\nHead Group {head_group}', 
                 shape='parallelogram', style='filled', fillcolor='purple')
        
        # Final concatenation across all head groups
    dot.node('concat_heads_l1', 
             'Concat Head Groups L1\n[1024, ?, 16, 512]\\nAll Groups', 
             shape='parallelogram', style='filled', fillcolor='purple')
    
    # Layer 2 - Two-level partitioned attention
    for gpu_id in range(16):
        head_group = gpu_id // 4
        dim_slice = gpu_id % 4
        heads_start = head_group * 4
        heads_end = heads_start + 3
        dim_start = dim_slice * 128
        dim_end = dim_start + 127
        
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as gpu:
            # Q, K, V projections for specific partition
            dot.node(f'q_proj_l2_g{gpu_id}', 
                     f'Q Projection L2\n[1024, ?, 512]\nHeads {heads_start}-{heads_end}\\nDims {dim_start}-{dim_end}', 
                     shape='rectangle', style='filled', fillcolor='lightblue')
            dot.node(f'k_proj_l2_g{gpu_id}', 
                     f'K Projection L2\n[1024, ?, 512]\nHeads {heads_start}-{heads_end}\\nDims {dim_start}-{dim_end}', 
                     shape='rectangle', style='filled', fillcolor='lightblue')
            dot.node(f'v_proj_l2_g{gpu_id}', 
                     f'V Projection L2\n[1024, ?, 512]\nHeads {heads_start}-{heads_end}\\nDims {dim_start}-{dim_end}', 
                     shape='rectangle', style='filled', fillcolor='lightblue')
            
            # Attention computation for partition
            dot.node(f'attn_l2_g{gpu_id}', 
                     f'Attention L2\n[1024, ?, 4, 128]\nHeads {heads_start}-{heads_end}\\nDims {dim_start}-{dim_end}', 
                     shape='rectangle', style='filled', fillcolor='yellow')
            
            # MLP components
            dot.node(f'mlp_fc_l2_g{gpu_id}', 
                     f'MLP FC L2\n[1024, ?, 8192]\nPartition {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            dot.node(f'mlp_gelu_l2_g{gpu_id}', 
                     f'MLP GELU L2\n[1024, ?, 8192]\nPartition {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            dot.node(f'mlp_proj_l2_g{gpu_id}', 
                     f'MLP Projection L2\n[1024, ?, 8192]\nPartition {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Local residual connections
            dot.node(f'residual_l2_local_{gpu_id}', 
                     f'Local Residual L2\n[1024, ?, 512]\nGPU {gpu_id}', 
                     shape='parallelogram', style='filled', fillcolor='lightgray')
    
    # Layer 2 aggregation nodes
    for head_group in range(4):
        dot.node(f'concat_dims_l2_g{head_group}', 
                 f'Concat Dims L2\n[1024, ?, 4, 512]\nHead Group {head_group}', 
                 shape='parallelogram', style='filled', fillcolor='purple')
    
    dot.node('concat_heads_l2', 
             'Concat Head Groups L2\n[1024, ?, 16, 512]\\nAll Groups', 
             shape='parallelogram', style='filled', fillcolor='purple')
    
    # Output layer
    dot.node('output', 'Output\n[batch_size=1024, seq_len=?, embedding_dim=8192]', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connections
    # Input to broadcast
    dot.edge('input', 'broadcast')
    
    # Layer 1 connections
    for gpu_id in range(16):
        head_group = gpu_id // 4
        dim_slice = gpu_id % 4
        
        # Broadcast to projections
        dot.edge('broadcast', f'q_proj_l1_g{gpu_id}')
        dot.edge('broadcast', f'k_proj_l1_g{gpu_id}')
        dot.edge('broadcast', f'v_proj_l1_g{gpu_id}')
        
        # Projections to attention
        dot.edge(f'q_proj_l1_g{gpu_id}', f'attn_l1_g{gpu_id}')
        dot.edge(f'k_proj_l1_g{gpu_id}', f'attn_l1_g{gpu_id}')
        dot.edge(f'v_proj_l1_g{gpu_id}', f'attn_l1_g{gpu_id}')
        
        # Attention to local residual
        dot.edge(f'attn_l1_g{gpu_id}', f'residual_l1_local_{gpu_id}')
        
        # Local residual to MLP (with broadcast)
        dot.edge(f'residual_l1_local_{gpu_id}', f'mlp_fc_l1_g{gpu_id}')
        dot.edge(f'mlp_fc_l1_g{gpu_id}', f'mlp_gelu_l1_g{gpu_id}')
        dot.edge(f'mlp_gelu_l1_g{gpu_id}', f'mlp_proj_l1_g{gpu_id}')
        
        # MLP to dimension concatenation
        dot.edge(f'mlp_proj_l1_g{gpu_id}', f'concat_dims_l1_g{head_group}')
    
    # Dimension concatenation to head concatenation
    for head_group in range(4):
        dot.edge(f'concat_dims_l1_g{head_group}', 'concat_heads_l1')
    
    # Layer 2 connections
    for gpu_id in range(16):
        head_group = gpu_id // 4
        dim_slice = gpu_id % 4
        
        # Concatenated output to projections
        dot.edge('concat_heads_l1', f'q_proj_l2_g{gpu_id}')
        dot.edge('concat_heads_l1', f'k_proj_l2_g{gpu_id}')
        dot.edge('concat_heads_l1', f'v_proj_l2_g{gpu_id}')
        
        # Projections to attention
        dot.edge(f'q_proj_l2_g{gpu_id}', f'attn_l2_g{gpu_id}')
        dot.edge(f'k_proj_l2_g{gpu_id}', f'attn_l2_g{gpu_id}')
        dot.edge(f'v_proj_l2_g{gpu_id}', f'attn_l2_g{gpu_id}')
        
        # Attention to local residual
        dot.edge(f'attn_l2_g{gpu_id}', f'residual_l2_local_{gpu_id}')
        
        # Local residual to MLP
        dot.edge(f'residual_l2_local_{gpu_id}', f'mlp_fc_l2_g{gpu_id}')
        dot.edge(f'mlp_fc_l2_g{gpu_id}', f'mlp_gelu_l2_g{gpu_id}')
        dot.edge(f'mlp_gelu_l2_g{gpu_id}', f'mlp_proj_l2_g{gpu_id}')
        
        # MLP to dimension concatenation
        dot.edge(f'mlp_proj_l2_g{gpu_id}', f'concat_dims_l2_g{head_group}')
    
    # Dimension concatenation to head concatenation
    for head_group in range(4):
        dot.edge(f'concat_dims_l2_g{head_group}', 'concat_heads_l2')
    
    # Final output
    dot.edge('concat_heads_l2', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-12-16-52-38/proposed_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-12-16-52-38/proposed_dag.dot')
    print("Proposed DAG generated successfully!")