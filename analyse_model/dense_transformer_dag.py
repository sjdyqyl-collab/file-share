import graphviz

# Create the DAG for Dense Transformer with Ring Attention + Sequence Parallelism
dot = graphviz.Digraph(comment='Dense Transformer - Ring Attention + Sequence Parallelism', format='svg')
dot.attr(rankdir='TB', size='20,20')

# Define node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
dot.attr('edge', fontsize='10')

# Define 4 GPUs for sequence parallelism and ring attention
P = 4

# Input layer - split across GPUs via sequence parallelism
dot.node('input_total', 'Input\n(B, L, d_model)', shape='ellipse', fillcolor='lightgreen')

# Split input across sequence dimension
dot.node('split', 'Split Sequence\n(B, L/P, d_model)', shape='parallelogram', fillcolor='yellow')
dot.edge('input_total', 'split')

# Each GPU processes its sequence segment
for p in range(P):
    gpu_id = f'GPU_{p}'
    
    # Input for this GPU
    dot.node(f'input_{p}', f'Input Segment {p}\n(B, L/{P}, d_model)\n{gpu_id}', fillcolor='lightcyan')
    dot.edge('split', f'input_{p}')
    
    # Layer 1: Q, K, V projections (parallel across GPUs)
    dot.node(f'qkv_proj_{p}', f'QKV Projection\n(B, L/{P}, d_model) -> (B, L/{P}, 3*d_model)\n{gpu_id}', fillcolor='lightblue')
    dot.edge(f'input_{p}', f'qkv_proj_{p}')
    
    # Split into Q, K, V
    dot.node(f'split_qkv_{p}', f'Split QKV\n(B, L/{P}, d_model) each\n{gpu_id}', shape='parallelogram', fillcolor='yellow')
    dot.edge(f'qkv_proj_{p}', f'split_qkv_{p}')
    
    # Ring Attention stages
    for stage in range(P):
        src_idx = (p - stage) % P
        
        # Compute attention with current K,V block
        dot.node(f'attn_stage_{p}_{stage}', 
                f'Ring Attention Stage {stage}\nQ{p} × K{src_idx} × V{src_idx}\n(B, L/{P}, L/{P}) × (B, L/{P}, d_model)\n{gpu_id}', 
                fillcolor='orange')
        
        if stage == 0:
            dot.edge(f'split_qkv_{p}', f'attn_stage_{p}_{stage}')
        else:
            # Receive K,V from previous stage
            prev_gpu = (p - 1) % P
            dot.node(f'recv_kv_{p}_{stage}', 
                    f'Receive K,V from GPU{prev_gpu}\n(B, L/{P}, d_model) each\n{gpu_id}', 
                    shape='ellipse', fillcolor='lightgreen', style='dashed')
            dot.edge(f'recv_kv_{p}_{stage}', f'attn_stage_{p}_{stage}')
        
        # Send K,V to next GPU
        next_gpu = (p + 1) % P
        dot.node(f'send_kv_{p}_{stage}', 
                f'Send K,V to GPU{next_gpu}\n(B, L/{P}, d_model) each\n{gpu_id}', 
                shape='ellipse', fillcolor='lightgreen', style='dashed')
        dot.edge(f'split_qkv_{p}', f'send_kv_{p}_{stage}')
        
        # Accumulate attention results
        if stage == 0:
            dot.node(f'accum_{p}_{stage}', f'Initialize Output\n(B, L/{P}, d_model)\n{gpu_id}', fillcolor='lightgray')
        else:
            dot.node(f'accum_{p}_{stage}', f'Accumulate Output {stage}\n(B, L/{P}, d_model)\n{gpu_id}', fillcolor='lightgray')
            dot.edge(f'accum_{p}_{stage-1}', f'accum_{p}_{stage}')
        
        dot.edge(f'attn_stage_{p}_{stage}', f'accum_{p}_{stage}')
    
    # Output projection after all ring stages
    dot.node(f'output_proj_{p}', f'Output Projection\n(B, L/{P}, d_model) -> (B, L/{P}, d_model)\n{gpu_id}', fillcolor='lightblue')
    dot.edge(f'accum_{p}_{P-1}', f'output_proj_{p}')
    
    # Residual connection
    dot.node(f'residual_{p}', f'Residual Add\n(B, L/{P}, d_model)\n{gpu_id}', fillcolor='pink')
    dot.edge(f'input_{p}', f'residual_{p}')
    dot.edge(f'output_proj_{p}', f'residual_{p}')
    
    # Layer Norm
    dot.node(f'ln1_{p}', f'Layer Norm 1\n(B, L/{P}, d_model)\n{gpu_id}', fillcolor='lightyellow')
    dot.edge(f'residual_{p}', f'ln1_{p}')
    
    # MLP - Column and Row parallel
    dot.node(f'mlp_col_{p}', f'MLP Column Parallel\n(B, L/{P}, d_model) -> (B, L/{P}, ffn_hidden_size/2)\n{gpu_id}', fillcolor='lightcoral')
    dot.edge(f'ln1_{p}', f'mlp_col_{p}')
    
    dot.node(f'mlp_activation_{p}', f'GELU Activation\n(B, L/{P}, ffn_hidden_size/2)\n{gpu_id}', fillcolor='lightgreen')
    dot.edge(f'mlp_col_{p}', f'mlp_activation_{p}')
    
    dot.node(f'mlp_row_{p}', f'MLP Row Parallel\n(B, L/{P}, ffn_hidden_size/2) -> (B, L/{P}, d_model)\n{gpu_id}', fillcolor='lightcoral')
    dot.edge(f'mlp_activation_{p}', f'mlp_row_{p}')
    
    # Second residual connection
    dot.node(f'residual2_{p}', f'Residual Add 2\n(B, L/{P}, d_model)\n{gpu_id}', fillcolor='pink')
    dot.edge(f'ln1_{p}', f'residual2_{p}')
    dot.edge(f'mlp_row_{p}', f'residual2_{p}')
    
    # Layer Norm 2
    dot.node(f'ln2_{p}', f'Layer Norm 2\n(B, L/{P}, d_model)\n{gpu_id}', fillcolor='lightyellow')
    dot.edge(f'residual2_{p}', f'ln2_{p}')
    
    # Output for this GPU
    dot.node(f'output_seg_{p}', f'Output Segment {p}\n(B, L/{P}, d_model)\n{gpu_id}', fillcolor='lightcyan')
    dot.edge(f'ln2_{p}', f'output_seg_{p}')

# Gather all segments back together
dot.node('gather', 'Gather All Segments\n(B, L, d_model)', shape='parallelogram', fillcolor='yellow')
for p in range(P):
    dot.edge(f'output_seg_{p}', 'gather')

# Final output
dot.node('output_total', 'Final Output\n(B, L, d_model)', shape='ellipse', fillcolor='lightgreen')
dot.edge('gather', 'output_total')

# Save the DAG
dot.render('/home/wzc/data/file-share/submission/dense_transformer_dag', format='svg', cleanup=False)
print("Dense Transformer DAG saved to /home/wzc/data/file-share/submission/dense_transformer_dag.svg")