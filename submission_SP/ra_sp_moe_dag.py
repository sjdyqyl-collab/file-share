import graphviz

# Create RA+SP DAG for MoE Transformer with 16 GPUs
dot = graphviz.Digraph(comment='MoE Transformer RA+SP (16 GPUs)', format='svg')
dot.attr(rankdir='TB', size='25,40')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', color='lightblue')  # Communication
dot.attr('node', shape='rectangle', style='filled', color='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', color='lightyellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', color='lightcoral')  # Gate

# Input
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input Layer')
    c.node('input', 'Input\nX: [B, L, d_model]\nGPU: Host', shape='parallelogram')

# Sequence Parallel Split across 16 GPUs
with dot.subgraph(name='cluster_sp_split') as c:
    c.attr(label='Sequence Parallel Split')
    c.node('sp_split', 'Sequence Parallel Split\nX: [B, L/16, d_model]\nGPU: 0-15', shape='parallelogram')

# 4 layers with Ring Attention + Sequence Parallelism + MoE
for layer_id in range(4):
    with dot.subgraph(name=f'cluster_layer{layer_id}') as layer:
        layer.attr(label=f'Layer {layer_id}')
        
        # Ring Attention with Sequence Parallelism
        with dot.subgraph(name=f'cluster_ra_sp{layer_id}') as ra:
            ra.attr(label=f'Ring Attention + Sequence Parallel')
            
            # QKV projections on each GPU
            for gpu_id in range(16):
                ra.node(f'qkv{layer_id}_{gpu_id}', f'QKV Projection\n[Q,K,V]: [B, L/16, d_model]\nGPU: {gpu_id}', shape='rectangle')
            
            # Ring communication stages (16 stages for 16 GPUs)
            for stage in range(16):
                for gpu_id in range(16):
                    src_gpu = (gpu_id - stage) % 16
                    ra.node(f'ring_stage{layer_id}_{stage}_{gpu_id}', 
                           f'Ring Stage {stage}\nCompute: Q_{gpu_id}×K_{src_gpu}×V_{src_gpu}\nGPU: {gpu_id}', 
                           shape='rectangle')
                
                # KV communication
                for gpu_id in range(16):
                    next_gpu = (gpu_id + 1) % 16
                    ra.node(f'kv_send{layer_id}_{stage}_{gpu_id}', 
                           f'Send KV to GPU {next_gpu}\nGPU: {gpu_id}→{next_gpu}', 
                           shape='ellipse')
            
            # Accumulate partial results
            for gpu_id in range(16):
                ra.node(f'accum{layer_id}_{gpu_id}', f'Accumulate Results\nOutput: [B, L/16, d_model]\nGPU: {gpu_id}', shape='parallelogram')
            
            # Output projection
            for gpu_id in range(16):
                ra.node(f'out_proj{layer_id}_{gpu_id}', f'Output Projection\nOutput: [B, L/16, d_model]\nGPU: {gpu_id}', shape='rectangle')
        
        # Residual connection
        for gpu_id in range(16):
            layer.node(f'res_attn{layer_id}_{gpu_id}', f'Residual Add\nInput: [B, L/16, d_model]\nGPU: {gpu_id}', shape='parallelogram')
        
        # MoE - Each GPU has 8 experts, top-2 gating
        with dot.subgraph(name=f'cluster_moe{layer_id}') as moe:
            moe.attr(label=f'Mixture of Experts')
            
            # Gate computation on each GPU
            for gpu_id in range(16):
                moe.node(f'gate{layer_id}_{gpu_id}', f'Gate\nCompute routing scores\nInput: [B, L/16, d_model]\nGPU: {gpu_id}', shape='diamond')
            
            # Expert selection (top-2) on each GPU
            for gpu_id in range(16):
                moe.node(f'select{layer_id}_{gpu_id}', f'Select Top-2 Experts\nGPU: {gpu_id}', shape='parallelogram')
            
            # Expert computation (8 experts per GPU)
            for gpu_id in range(16):
                for exp_id in range(8):
                    moe.node(f'expert{layer_id}_{gpu_id}_{exp_id}', 
                           f'Expert {exp_id}\nOutput: [B, L/16, ffn_dim]\nGPU: {gpu_id}', 
                           shape='rectangle')
                    moe.node(f'expert_act{layer_id}_{gpu_id}_{exp_id}', 
                           f'Expert {exp_id} GELU\nGPU: {gpu_id}', 
                           shape='rectangle')
                    moe.node(f'expert_out{layer_id}_{gpu_id}_{exp_id}', 
                           f'Expert {exp_id} Output\nOutput: [B, L/16, d_model]\nGPU: {gpu_id}', 
                           shape='rectangle')
            
            # Combine expert outputs on each GPU
            for gpu_id in range(16):
                moe.node(f'combine{layer_id}_{gpu_id}', 
                       f'Combine Expert Outputs\nWeighted sum by gate scores\nGPU: {gpu_id}', 
                       shape='parallelogram')
        
        # Residual connection
        for gpu_id in range(16):
            layer.node(f'res_moe{layer_id}_{gpu_id}', f'Residual Add\nInput: [B, L/16, d_model]\nGPU: {gpu_id}', shape='parallelogram')

# Sequence Parallel Gather
with dot.subgraph(name='cluster_sp_gather') as c:
    c.attr(label='Sequence Parallel Gather')
    c.node('sp_gather', 'Sequence Parallel Gather\nX: [B, L, d_model]\nGPU: 0-15 → Host', shape='parallelogram')

# Output
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Layer')
    c.node('output', 'Output\nX: [B, L, d_model]\nGPU: Host', shape='parallelogram')

# Connections
# Input to SP split
dot.edge('input', 'sp_split')

# Layer connections
for layer_id in range(4):
    # Input to QKV projections
    if layer_id == 0:
        for gpu_id in range(16):
            dot.edge('sp_split', f'qkv{layer_id}_{gpu_id}')
    else:
        for gpu_id in range(16):
            dot.edge(f'res_moe{layer_id-1}_{gpu_id}', f'qkv{layer_id}_{gpu_id}')
    
    # Ring attention stages
    for gpu_id in range(16):
        dot.edge(f'qkv{layer_id}_{gpu_id}', f'ring_stage{layer_id}_0_{gpu_id}')
    
    for stage in range(16):
        for gpu_id in range(16):
            # Ring computation stages
            if stage > 0:
                prev_gpu = (gpu_id - 1) % 16
                dot.edge(f'ring_stage{layer_id}_{stage-1}_{gpu_id}', f'ring_stage{layer_id}_{stage}_{gpu_id}')
                dot.edge(f'kv_send{layer_id}_{stage-1}_{prev_gpu}', f'ring_stage{layer_id}_{stage}_{gpu_id}')
            
            # KV communication
            if stage < 15:
                dot.edge(f'ring_stage{layer_id}_{stage}_{gpu_id}', f'kv_send{layer_id}_{stage}_{gpu_id}')
    
    # Accumulate and output projection
    for gpu_id in range(16):
        dot.edge(f'ring_stage{layer_id}_15_{gpu_id}', f'accum{layer_id}_{gpu_id}')
        dot.edge(f'accum{layer_id}_{gpu_id}', f'out_proj{layer_id}_{gpu_id}')
        dot.edge(f'out_proj{layer_id}_{gpu_id}', f'res_attn{layer_id}_{gpu_id}')
        
        # Residual connection for attention
        if layer_id == 0:
            dot.edge('sp_split', f'res_attn{layer_id}_{gpu_id}')
        else:
            dot.edge(f'res_moe{layer_id-1}_{gpu_id}', f'res_attn{layer_id}_{gpu_id}')
        
        # MoE connections
        dot.edge(f'res_attn{layer_id}_{gpu_id}', f'gate{layer_id}_{gpu_id}')
        dot.edge(f'gate{layer_id}_{gpu_id}', f'select{layer_id}_{gpu_id}')
        
        # Expert connections with routing
        for exp_id in range(8):
            dot.edge(f'select{layer_id}_{gpu_id}', f'expert{layer_id}_{gpu_id}_{exp_id}', style='dashed', label=f'if selected')
            dot.edge(f'expert{layer_id}_{gpu_id}_{exp_id}', f'expert_act{layer_id}_{gpu_id}_{exp_id}')
            dot.edge(f'expert_act{layer_id}_{gpu_id}_{exp_id}', f'expert_out{layer_id}_{gpu_id}_{exp_id}')
            dot.edge(f'expert_out{layer_id}_{gpu_id}_{exp_id}', f'combine{layer_id}_{gpu_id}')
        
        dot.edge(f'combine{layer_id}_{gpu_id}', f'res_moe{layer_id}_{gpu_id}')
        dot.edge(f'res_attn{layer_id}_{gpu_id}', f'res_moe{layer_id}_{gpu_id}')  # Residual connection

# Final connections
for gpu_id in range(16):
    dot.edge(f'res_moe3_{gpu_id}', 'sp_gather')

dot.edge('sp_gather', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/submission/ra_sp_moe_dag', format='svg')
print("RA+SP MoE DAG saved to /home/wzc/data/file-share/submission/ra_sp_moe_dag.svg")