import graphviz

# Create the DAG for Helix two-level attention partitioning
dot = graphviz.Digraph('Helix_Two_Level_Partitioning', comment='Two-Level Attention Partitioning DAG')
dot.attr(rankdir='TB', size='20,20')

# Input node
dot.node('input', 'Input Tensor\\nX: [B, L, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')

# Layer 0 - Multi-Head Attention with Two-Level Partitioning
with dot.subgraph(name='cluster_layer0_mha') as c:
    c.attr(label='Layer 0 - Multi-Head Attention', style='rounded', fillcolor='lightyellow')
    
    # Broadcast input to all GPUs
    dot.node('broadcast', 'Broadcast\\nX: [B, L, 8192] → 16 GPUs', shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # 16 GPU partitions for MHA computation
    partitions = []
    for gpu_id in range(16):
        head_group = gpu_id // 4
        dim_slice = gpu_id % 4
        
        # Q projection
        dot.node(f'q_proj_{gpu_id}', f'Q Projection\\nGPU {gpu_id}\\nW_Q: [8192, 512]\\n→ Q: [B, L, 512]', 
                shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # K projection
        dot.node(f'k_proj_{gpu_id}', f'K Projection\\nGPU {gpu_id}\\nW_K: [8192, 512]\\n→ K: [B, L, 512]', 
                shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # V projection
        dot.node(f'v_proj_{gpu_id}', f'V Projection\\nGPU {gpu_id}\\nW_V: [8192, 512]\\n→ V: [B, L, 512]', 
                shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Attention computation
        dot.node(f'attn_{gpu_id}', f'Multi-Head Attention\\nGPU {gpu_id}\\n4 heads × 128 dims\\n→ [B, L, 512]', 
                shape='rectangle', style='filled', fillcolor='lightgoldenrod')
        
        partitions.append((head_group, dim_slice, gpu_id))
    
    # Intra-group concatenation (4 groups of 4 GPUs each)
    for group_id in range(4):
        dot.node(f'intra_concat_{group_id}', 
                f'Intra-Group Concat\\nGroup {group_id}\\nConcat 4×128 dims\\n→ [B, L, 2048]', 
                shape='parallelogram', style='filled', fillcolor='lightsteelblue')
    
    # Final concatenation
    dot.node('final_concat_mha', 'Final Concatenation\\nConcat 4×2048 dims\\n→ [B, L, 8192]', 
            shape='parallelogram', style='filled', fillcolor='lightsteelblue')

# Layer 0 - FFN
with dot.subgraph(name='cluster_layer0_ffn') as c:
    c.attr(label='Layer 0 - Feed Forward Network', style='rounded', fillcolor='lightyellow')
    
    # FFN operations (standard tensor parallelism across 16 GPUs)
    dot.node('ffn_gate_proj_0', 'Gate Projection\\n16-way TP\\nW_gate: [8192, 32768]\\n→ [B, L, 32768]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_up_proj_0', 'Up Projection\\n16-way TP\\nW_up: [8192, 32768]\\n→ [B, L, 32768]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_silu_0', 'SiLU Activation\\n16-way TP\\n→ [B, L, 32768]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_mul_0', 'Element-wise Mul\\n16-way TP\\n→ [B, L, 32768]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_down_proj_0', 'Down Projection\\n16-way TP\\nW_down: [32768, 8192]\\n→ [B, L, 8192]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_all_reduce_0', 'All-Reduce\\n16-way TP\\n→ [B, L, 8192]', 
            shape='parallelogram', style='filled', fillcolor='lightsteelblue')

# Residual connection for layer 0
dot.node('residual_0', 'Residual Add\\nLayer 0\\n→ [B, L, 8192]', 
        shape='parallelogram', style='filled', fillcolor='lightgray')

# Layer 1 - Multi-Head Attention (same structure as Layer 0)
with dot.subgraph(name='cluster_layer1_mha') as c:
    c.attr(label='Layer 1 - Multi-Head Attention', style='rounded', fillcolor='lightyellow')
    
    # Broadcast input to all GPUs
    dot.node('broadcast_1', 'Broadcast\\nX: [B, L, 8192] → 16 GPUs', shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # 16 GPU partitions for MHA computation
    for gpu_id in range(16):
        head_group = gpu_id // 4
        dim_slice = gpu_id % 4
        
        # Q projection
        dot.node(f'q_proj_1_{gpu_id}', f'Q Projection\\nGPU {gpu_id}\\nW_Q: [8192, 512]\\n→ Q: [B, L, 512]', 
                shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # K projection
        dot.node(f'k_proj_1_{gpu_id}', f'K Projection\\nGPU {gpu_id}\\nW_K: [8192, 512]\\n→ K: [B, L, 512]', 
                shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # V projection
        dot.node(f'v_proj_1_{gpu_id}', f'V Projection\\nGPU {gpu_id}\\nW_V: [8192, 512]\\n→ V: [B, L, 512]', 
                shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Attention computation
        dot.node(f'attn_1_{gpu_id}', f'Multi-Head Attention\\nGPU {gpu_id}\\n4 heads × 128 dims\\n→ [B, L, 512]', 
                shape='rectangle', style='filled', fillcolor='lightgoldenrod')
    
    # Intra-group concatenation
    for group_id in range(4):
        dot.node(f'intra_concat_1_{group_id}', 
                f'Intra-Group Concat\\nGroup {group_id}\\nConcat 4×128 dims\\n→ [B, L, 2048]', 
                shape='parallelogram', style='filled', fillcolor='lightsteelblue')
    
    # Final concatenation
    dot.node('final_concat_mha_1', 'Final Concatenation\\nConcat 4×2048 dims\\n→ [B, L, 8192]', 
            shape='parallelogram', style='filled', fillcolor='lightsteelblue')

# Layer 1 - FFN
with dot.subgraph(name='cluster_layer1_ffn') as c:
    c.attr(label='Layer 1 - Feed Forward Network', style='rounded', fillcolor='lightyellow')
    
    # FFN operations
    dot.node('ffn_gate_proj_1', 'Gate Projection\\n16-way TP\\nW_gate: [8192, 32768]\\n→ [B, L, 32768]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_up_proj_1', 'Up Projection\\n16-way TP\\nW_up: [8192, 32768]\\n→ [B, L, 32768]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_silu_1', 'SiLU Activation\\n16-way TP\\n→ [B, L, 32768]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_mul_1', 'Element-wise Mul\\n16-way TP\\n→ [B, L, 32768]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_down_proj_1', 'Down Projection\\n16-way TP\\nW_down: [32768, 8192]\\n→ [B, L, 8192]', 
            shape='rectangle', style='filled', fillcolor='lightpink')
    
    dot.node('ffn_all_reduce_1', 'All-Reduce\\n16-way TP\\n→ [B, L, 8192]', 
            shape='parallelogram', style='filled', fillcolor='lightsteelblue')

# Residual connection for layer 1
dot.node('residual_1', 'Residual Add\\nLayer 1\\n→ [B, L, 8192]', 
        shape='parallelogram', style='filled', fillcolor='lightgray')

# Output
dot.node('output', 'Output\\nY: [B, L, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')

# Connect the nodes
# Input to broadcast
dot.edge('input', 'broadcast')

# Layer 0 MHA connections
for gpu_id in range(16):
    dot.edge('broadcast', f'q_proj_{gpu_id}')
    dot.edge('broadcast', f'k_proj_{gpu_id}')
    dot.edge('broadcast', f'v_proj_{gpu_id}')
    dot.edge(f'q_proj_{gpu_id}', f'attn_{gpu_id}')
    dot.edge(f'k_proj_{gpu_id}', f'attn_{gpu_id}')
    dot.edge(f'v_proj_{gpu_id}', f'attn_{gpu_id}')

# Intra-group concatenation connections
for group_id in range(4):
    start_gpu = group_id * 4
    for gpu_offset in range(4):
        gpu_id = start_gpu + gpu_offset
        dot.edge(f'attn_{gpu_id}', f'intra_concat_{group_id}')

# Final concatenation
for group_id in range(4):
    dot.edge(f'intra_concat_{group_id}', 'final_concat_mha')

# Layer 0 FFN connections
dot.edge('final_concat_mha', 'ffn_gate_proj_0')
dot.edge('final_concat_mha', 'ffn_up_proj_0')
dot.edge('ffn_gate_proj_0', 'ffn_silu_0')
dot.edge('ffn_up_proj_0', 'ffn_mul_0')
dot.edge('ffn_silu_0', 'ffn_mul_0')
dot.edge('ffn_mul_0', 'ffn_down_proj_0')
dot.edge('ffn_down_proj_0', 'ffn_all_reduce_0')

# Residual connection for layer 0
dot.edge('final_concat_mha', 'residual_0')
dot.edge('ffn_all_reduce_0', 'residual_0')

# Layer 1 MHA connections
dot.edge('residual_0', 'broadcast_1')
for gpu_id in range(16):
    dot.edge('broadcast_1', f'q_proj_1_{gpu_id}')
    dot.edge('broadcast_1', f'k_proj_1_{gpu_id}')
    dot.edge('broadcast_1', f'v_proj_1_{gpu_id}')
    dot.edge(f'q_proj_1_{gpu_id}', f'attn_1_{gpu_id}')
    dot.edge(f'k_proj_1_{gpu_id}', f'attn_1_{gpu_id}')
    dot.edge(f'v_proj_1_{gpu_id}', f'attn_1_{gpu_id}')

# Layer 1 intra-group concatenation
for group_id in range(4):
    start_gpu = group_id * 4
    for gpu_offset in range(4):
        gpu_id = start_gpu + gpu_offset
        dot.edge(f'attn_1_{gpu_id}', f'intra_concat_1_{group_id}')

# Layer 1 final concatenation
for group_id in range(4):
    dot.edge(f'intra_concat_1_{group_id}', 'final_concat_mha_1')

# Layer 1 FFN connections
dot.edge('final_concat_mha_1', 'ffn_gate_proj_1')
dot.edge('final_concat_mha_1', 'ffn_up_proj_1')
dot.edge('ffn_gate_proj_1', 'ffn_silu_1')
dot.edge('ffn_up_proj_1', 'ffn_mul_1')
dot.edge('ffn_silu_1', 'ffn_mul_1')
dot.edge('ffn_mul_1', 'ffn_down_proj_1')
dot.edge('ffn_down_proj_1', 'ffn_all_reduce_1')

# Residual connection for layer 1
dot.edge('final_concat_mha_1', 'residual_1')
dot.edge('ffn_all_reduce_1', 'residual_1')

# Output
dot.edge('residual_1', 'output')

# Save the DAG
dot.format = 'svg'
dot.render('/home/wzc/data/file-share/submission/helix_two_level_partitioning')

print("Generated Helix Two-Level Partitioning DAG")