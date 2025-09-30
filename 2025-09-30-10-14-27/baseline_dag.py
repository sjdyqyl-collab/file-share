import graphviz

# Create Baseline DAG (TP=8, PP=2)
dot = graphviz.Digraph('Baseline_TP8_PP2', comment='Baseline: Tensor Parallelism=8, Pipeline Parallelism=2')
dot.attr(rankdir='TB', size='20,20')

# Define node shapes
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication

# Input layer
dot.node('input', 'Model Input\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: 0', shape='ellipse')

# Pipeline Stage 1: Layers 1-2 (GPUs 0-7)
for layer in [1, 2]:
    for gpu_id in range(8):
        # Layer Norm
        dot.node(f'ps1_l{layer}_norm_{gpu_id}', f'PS1 L{layer} Layer Norm\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # QKV Projection (Column Parallel)
        dot.node(f'ps1_l{layer}_qkv_{gpu_id}', f'PS1 L{layer} QKV Projection\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Attention Score (with All-reduce)
        dot.node(f'ps1_l{layer}_attn_{gpu_id}', f'PS1 L{layer} Multi-Head Attention\\nInput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nOutput: [batch_size=1, seq_len=2048, hidden=512]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Attention Output Projection (Row Parallel)
        dot.node(f'ps1_l{layer}_attn_proj_{gpu_id}', f'PS1 L{layer} Attention Output Proj\\nInput: [batch_size=1, seq_len=2048, hidden=512]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Attention Residual
        dot.node(f'ps1_l{layer}_attn_res_{gpu_id}', f'PS1 L{layer} Attention Residual\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP Layer Norm
        dot.node(f'ps1_l{layer}_mlp_norm_{gpu_id}', f'PS1 L{layer} MLP Layer Norm\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP First Linear (Column Parallel)
        dot.node(f'ps1_l{layer}_mlp_fc1_{gpu_id}', f'PS1 L{layer} MLP FC1\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, ffn_hidden=1024]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP Activation
        dot.node(f'ps1_l{layer}_mlp_act_{gpu_id}', f'PS1 L{layer} MLP GELU\\nInput: [batch_size=1, seq_len=2048, ffn_hidden=1024]\\nOutput: [batch_size=1, seq_len=2048, ffn_hidden=1024]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP Second Linear (Row Parallel)
        dot.node(f'ps1_l{layer}_mlp_fc2_{gpu_id}', f'PS1 L{layer} MLP FC2\\nInput: [batch_size=1, seq_len=2048, ffn_hidden=1024]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP Residual
        dot.node(f'ps1_l{layer}_mlp_res_{gpu_id}', f'PS1 L{layer} MLP Residual\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')

# Pipeline Stage 2: Layers 3-4 (GPUs 8-15)
for layer in [3, 4]:
    for gpu_id in range(8, 16):
        # Layer Norm
        dot.node(f'ps2_l{layer}_norm_{gpu_id}', f'PS2 L{layer} Layer Norm\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # QKV Projection (Column Parallel)
        dot.node(f'ps2_l{layer}_qkv_{gpu_id}', f'PS2 L{layer} QKV Projection\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Attention Score (with All-reduce)
        dot.node(f'ps2_l{layer}_attn_{gpu_id}', f'PS2 L{layer} Multi-Head Attention\\nInput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nOutput: [batch_size=1, seq_len=2048, hidden=512]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Attention Output Projection (Row Parallel)
        dot.node(f'ps2_l{layer}_attn_proj_{gpu_id}', f'PS2 L{layer} Attention Output Proj\\nInput: [batch_size=1, seq_len=2048, hidden=512]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Attention Residual
        dot.node(f'ps2_l{layer}_attn_res_{gpu_id}', f'PS2 L{layer} Attention Residual\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP Layer Norm
        dot.node(f'ps2_l{layer}_mlp_norm_{gpu_id}', f'PS2 L{layer} MLP Layer Norm\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP First Linear (Column Parallel)
        dot.node(f'ps2_l{layer}_mlp_fc1_{gpu_id}', f'PS2 L{layer} MLP FC1\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, ffn_hidden=1024]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP Activation
        dot.node(f'ps2_l{layer}_mlp_act_{gpu_id}', f'PS2 L{layer} MLP GELU\\nInput: [batch_size=1, seq_len=2048, ffn_hidden=1024]\\nOutput: [batch_size=1, seq_len=2048, ffn_hidden=1024]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP Second Linear (Row Parallel)
        dot.node(f'ps2_l{layer}_mlp_fc2_{gpu_id}', f'PS2 L{layer} MLP FC2\\nInput: [batch_size=1, seq_len=2048, ffn_hidden=1024]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # MLP Residual
        dot.node(f'ps2_l{layer}_mlp_res_{gpu_id}', f'PS2 L{layer} MLP Residual\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')

# Output layer
dot.node('output', 'Model Output\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, vocab_size]\\nGPU: 15', shape='ellipse')

# Communication nodes for tensor parallelism all-reduce
for stage in ['ps1', 'ps2']:
    for layer in [1, 2, 3, 4]:
        for gpu_id in range(8 if stage == 'ps1' else 8, 16):
            # Attention all-reduce
            dot.node(f'{stage}_l{layer}_attn_ar_{gpu_id}', f'{stage.upper()} L{layer} Attention All-Reduce\\nInput: [batch_size=1, seq_len=2048, hidden=512]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='diamond')
            
            # MLP all-reduce  
            dot.node(f'{stage}_l{layer}_mlp_ar_{gpu_id}', f'{stage.upper()} L{layer} MLP All-Reduce\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='diamond')

# Pipeline communication nodes
for layer in [2, 3]:
    dot.node(f'pipeline_l{layer}_send', f'Pipeline L{layer} Send\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPUs: 0-7 → 8-15', shape='diamond')

# Connect the DAG - Pipeline Stage 1
for gpu_id in range(8):
    # Layer 1
    dot.edge('input', f'ps1_l1_norm_{gpu_id}')
    dot.edge(f'ps1_l1_norm_{gpu_id}', f'ps1_l1_qkv_{gpu_id}')
    dot.edge(f'ps1_l1_qkv_{gpu_id}', f'ps1_l1_attn_{gpu_id}')
    dot.edge(f'ps1_l1_attn_{gpu_id}', f'ps1_l1_attn_ar_{gpu_id}')
    dot.edge(f'ps1_l1_attn_ar_{gpu_id}', f'ps1_l1_attn_proj_{gpu_id}')
    dot.edge(f'ps1_l1_attn_proj_{gpu_id}', f'ps1_l1_attn_res_{gpu_id}')
    dot.edge('input', f'ps1_l1_attn_res_{gpu_id}')  # Residual
    
    dot.edge(f'ps1_l1_attn_res_{gpu_id}', f'ps1_l1_mlp_norm_{gpu_id}')
    dot.edge(f'ps1_l1_mlp_norm_{gpu_id}', f'ps1_l1_mlp_fc1_{gpu_id}')
    dot.edge(f'ps1_l1_mlp_fc1_{gpu_id}', f'ps1_l1_mlp_act_{gpu_id}')
    dot.edge(f'ps1_l1_mlp_act_{gpu_id}', f'ps1_l1_mlp_fc2_{gpu_id}')
    dot.edge(f'ps1_l1_mlp_fc2_{gpu_id}', f'ps1_l1_mlp_ar_{gpu_id}')
    dot.edge(f'ps1_l1_mlp_ar_{gpu_id}', f'ps1_l1_mlp_res_{gpu_id}')
    dot.edge(f'ps1_l1_attn_res_{gpu_id}', f'ps1_l1_mlp_res_{gpu_id}')  # Residual
    
    # Layer 2
    dot.edge(f'ps1_l1_mlp_res_{gpu_id}', f'ps1_l2_norm_{gpu_id}')
    dot.edge(f'ps1_l2_norm_{gpu_id}', f'ps1_l2_qkv_{gpu_id}')
    dot.edge(f'ps1_l2_qkv_{gpu_id}', f'ps1_l2_attn_{gpu_id}')
    dot.edge(f'ps1_l2_attn_{gpu_id}', f'ps1_l2_attn_ar_{gpu_id}')
    dot.edge(f'ps1_l2_attn_ar_{gpu_id}', f'ps1_l2_attn_proj_{gpu_id}')
    dot.edge(f'ps1_l2_attn_proj_{gpu_id}', f'ps1_l2_attn_res_{gpu_id}')
    dot.edge(f'ps1_l1_mlp_res_{gpu_id}', f'ps1_l2_attn_res_{gpu_id}')  # Residual
    
    dot.edge(f'ps1_l2_attn_res_{gpu_id}', f'ps1_l2_mlp_norm_{gpu_id}')
    dot.edge(f'ps1_l2_mlp_norm_{gpu_id}', f'ps1_l2_mlp_fc1_{gpu_id}')
    dot.edge(f'ps1_l2_mlp_fc1_{gpu_id}', f'ps1_l2_mlp_act_{gpu_id}')
    dot.edge(f'ps1_l2_mlp_act_{gpu_id}', f'ps1_l2_mlp_fc2_{gpu_id}')
    dot.edge(f'ps1_l2_mlp_fc2_{gpu_id}', f'ps1_l2_mlp_ar_{gpu_id}')
    dot.edge(f'ps1_l2_mlp_ar_{gpu_id}', f'ps1_l2_mlp_res_{gpu_id}')
    dot.edge(f'ps1_l2_attn_res_{gpu_id}', f'ps1_l2_mlp_res_{gpu_id}')  # Residual
    
    # Pipeline communication
    dot.edge(f'ps1_l2_mlp_res_{gpu_id}', 'pipeline_l2_send')

# Connect pipeline communication to Stage 2
for gpu_id in range(8, 16):
    dot.edge('pipeline_l2_send', f'ps2_l3_norm_{gpu_id}')

# Pipeline Stage 2 connections
for gpu_id in range(8, 16):
    # Layer 3
    dot.edge(f'ps2_l3_norm_{gpu_id}', f'ps2_l3_qkv_{gpu_id}')
    dot.edge(f'ps2_l3_qkv_{gpu_id}', f'ps2_l3_attn_{gpu_id}')
    dot.edge(f'ps2_l3_attn_{gpu_id}', f'ps2_l3_attn_ar_{gpu_id}')
    dot.edge(f'ps2_l3_attn_ar_{gpu_id}', f'ps2_l3_attn_proj_{gpu_id}')
    dot.edge(f'ps2_l3_attn_proj_{gpu_id}', f'ps2_l3_attn_res_{gpu_id}')
    dot.edge('pipeline_l2_send', f'ps2_l3_attn_res_{gpu_id}')  # Residual
    
    dot.edge(f'ps2_l3_attn_res_{gpu_id}', f'ps2_l3_mlp_norm_{gpu_id}')
    dot.edge(f'ps2_l3_mlp_norm_{gpu_id}', f'ps2_l3_mlp_fc1_{gpu_id}')
    dot.edge(f'ps2_l3_mlp_fc1_{gpu_id}', f'ps2_l3_mlp_act_{gpu_id}')
    dot.edge(f'ps2_l3_mlp_act_{gpu_id}', f'ps2_l3_mlp_fc2_{gpu_id}')
    dot.edge(f'ps2_l3_mlp_fc2_{gpu_id}', f'ps2_l3_mlp_ar_{gpu_id}')
    dot.edge(f'ps2_l3_mlp_ar_{gpu_id}', f'ps2_l3_mlp_res_{gpu_id}')
    dot.edge(f'ps2_l3_attn_res_{gpu_id}', f'ps2_l3_mlp_res_{gpu_id}')  # Residual
    
    # Layer 4
    dot.edge(f'ps2_l3_mlp_res_{gpu_id}', f'ps2_l4_norm_{gpu_id}')
    dot.edge(f'ps2_l4_norm_{gpu_id}', f'ps2_l4_qkv_{gpu_id}')
    dot.edge(f'ps2_l4_qkv_{gpu_id}', f'ps2_l4_attn_{gpu_id}')
    dot.edge(f'ps2_l4_attn_{gpu_id}', f'ps2_l4_attn_ar_{gpu_id}')
    dot.edge(f'ps2_l4_attn_ar_{gpu_id}', f'ps2_l4_attn_proj_{gpu_id}')
    dot.edge(f'ps2_l4_attn_proj_{gpu_id}', f'ps2_l4_attn_res_{gpu_id}')
    dot.edge(f'ps2_l3_mlp_res_{gpu_id}', f'ps2_l4_attn_res_{gpu_id}')  # Residual
    
    dot.edge(f'ps2_l4_attn_res_{gpu_id}', f'ps2_l4_mlp_norm_{gpu_id}')
    dot.edge(f'ps2_l4_mlp_norm_{gpu_id}', f'ps2_l4_mlp_fc1_{gpu_id}')
    dot.edge(f'ps2_l4_mlp_fc1_{gpu_id}', f'ps2_l4_mlp_act_{gpu_id}')
    dot.edge(f'ps2_l4_mlp_act_{gpu_id}', f'ps2_l4_mlp_fc2_{gpu_id}')
    dot.edge(f'ps2_l4_mlp_fc2_{gpu_id}', f'ps2_l4_mlp_ar_{gpu_id}')
    dot.edge(f'ps2_l4_mlp_ar_{gpu_id}', f'ps2_l4_mlp_res_{gpu_id}')
    dot.edge(f'ps2_l4_attn_res_{gpu_id}', f'ps2_l4_mlp_res_{gpu_id}')  # Residual
    
    # Connect to output
    dot.edge(f'ps2_l4_mlp_res_{gpu_id}', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/2025-09-30-10-14-27/baseline_tp8_pp2', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-30-10-14-27/baseline_tp8_pp2.dot')

print("Baseline TP=8 PP=2 DAG generated successfully!")