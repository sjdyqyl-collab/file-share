import graphviz

# Create the DAG for MoE Transformer with Ring Attention + Sequence Parallelism
dot = graphviz.Digraph(comment='MoE Transformer - Ring Attention + Sequence Parallelism', format='svg')
dot.attr(rankdir='TB', size='25,25')

# Define node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
dot.attr('edge', fontsize='10')

# Define 4 GPUs for sequence parallelism and ring attention
P = 4
NUM_EXPERTS = 8
TOP_K = 2

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
    
    # MoE Gate - determines which experts to use
    dot.node(f'gate_{p}', f'MoE Gate\n(B, L/{P}, d_model) -> (B, L/{P}, {NUM_EXPERTS})\n{gpu_id}', fillcolor='purple')
    dot.edge(f'ln1_{p}', f'gate_{p}')
    
    # Expert routing (dashed lines for selection)
    for expert_id in range(NUM_EXPERTS):
        # Expert computation (only active experts shown)
        if expert_id < TOP_K:  # Simplified for visualization
            dot.node(f'expert_{p}_{expert_id}', 
                    f'Expert {expert_id}\n(B, selected_tokens/{TOP_K}, d_model) -> (B, selected_tokens/{TOP_K}, d_model)\n{gpu_id}', 
                    fillcolor='lightcoral')
            
            # Routing connection (dashed)
            dot.edge(f'gate_{p}', f'expert_{p}_{expert_id}', style='dashed', label=f'route tokens')
            
            # Input to expert
            dot.edge(f'ln1_{p}', f'expert_{p}_{expert_id}')
    
    # Expert aggregation
    dot.node(f'expert_agg_{p}', f'Expert Aggregation\nCombine {TOP_K} experts\n(B, L/{P}, d_model)\n{gpu_id}', shape='parallelogram', fillcolor='yellow')
    for expert_id in range(TOP_K):
        dot.edge(f'expert_{p}_{expert_id}', f'expert_agg_{p}')
    
    # Output projection for MoE
    dot.node(f'moe_output_{p}', f'MoE Output Projection\n(B, L/{P}, d_model) -> (B, L/{P}, d_model)\n{gpu_id}', fillcolor='lightblue')
    dot.edge(f'expert_agg_{p}', f'moe_output_{p}')
    
    # Second residual connection
    dot.node(f'residual2_{p}', f'Residual Add 2\n(B, L/{P}, d_model)\n{gpu_id}', fillcolor='pink')
    dot.edge(f'ln1_{p}', f'residual2_{p}')
    dot.edge(f'moe_output_{p}', f'residual2_{p}')
    
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
dot.render('/home/wzc/data/file-share/submission/moe_transformer_dag', format='svg', cleanup=False)
print("MoE Transformer DAG saved to /home/wzc/data/file-share/submission/moe_transformer_dag.svg")