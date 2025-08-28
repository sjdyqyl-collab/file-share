import graphviz

# Create the DAG for baseline: Tensor Parallelism + Pipeline Parallelism
dot = graphviz.Digraph('Baseline_TP_PP', comment='Tensor Parallelism + Pipeline Parallelism DAG')
dot.attr(rankdir='TB', size='20,20')

# Input node
dot.node('input', 'Input Tensor\\nX: [B, L, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')

# Pipeline Stage 0 - GPUs 0-7
with dot.subgraph(name='cluster_stage0') as c:
    c.attr(label='Pipeline Stage 0 - GPUs 0-7', style='rounded', fillcolor='lightyellow')
    
    # Layer 0 - Multi-Head Attention with TP=8
    with dot.subgraph(name='cluster_stage0_mha') as mha:
        mha.attr(label='Layer 0 - MHA (TP=8)', style='rounded', fillcolor='lightcyan')
        
        # Input broadcast to stage 0
        dot.node('broadcast_s0', 'Broadcast\\nX: [B, L, 8192] → 8 GPUs', 
                shape='parallelogram', style='filled', fillcolor='lightgreen')
        
        # Q projections across 8 GPUs
        for gpu_id in range(8):
            dot.node(f'q_proj_s0_{gpu_id}', 
                    f'Q Projection\\nGPU {gpu_id}\\nW_Q: [8192, 1024]\\n→ Q: [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # K projections across 8 GPUs
        for gpu_id in range(8):
            dot.node(f'k_proj_s0_{gpu_id}', 
                    f'K Projection\\nGPU {gpu_id}\\nW_K: [8192, 1024]\\n→ K: [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # V projections across 8 GPUs
        for gpu_id in range(8):
            dot.node(f'v_proj_s0_{gpu_id}', 
                    f'V Projection\\nGPU {gpu_id}\\nW_V: [8192, 1024]\\n→ V: [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Attention computation across 8 GPUs
        for gpu_id in range(8):
            dot.node(f'attn_s0_{gpu_id}', 
                    f'Multi-Head Attention\\nGPU {gpu_id}\\n2 heads × 512 dims\\n→ [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightgoldenrod')
        
        # All-reduce for attention output
        dot.node('all_reduce_attn_s0', 'All-Reduce\\n8 GPUs\\n→ [B, L, 8192]', 
                shape='parallelogram', style='filled', fillcolor='lightsteelblue')
    
    # Layer 0 - FFN with TP=8
    with dot.subgraph(name='cluster_stage0_ffn') as ffn:
        ffn.attr(label='Layer 0 - FFN (TP=8)', style='rounded', fillcolor='lightcyan')
        
        # Gate projection
        for gpu_id in range(8):
            dot.node(f'ffn_gate_s0_{gpu_id}', 
                    f'Gate Projection\\nGPU {gpu_id}\\nW_gate: [8192, 4096]\\n→ [B, L, 4096]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # Up projection
        for gpu_id in range(8):
            dot.node(f'ffn_up_s0_{gpu_id}', 
                    f'Up Projection\\nGPU {gpu_id}\\nW_up: [8192, 4096]\\n→ [B, L, 4096]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # SiLU activation
        for gpu_id in range(8):
            dot.node(f'ffn_silu_s0_{gpu_id}', 
                    f'SiLU\\nGPU {gpu_id}\\n→ [B, L, 4096]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # Element-wise multiplication
        for gpu_id in range(8):
            dot.node(f'ffn_mul_s0_{gpu_id}', 
                    f'Element-wise Mul\\nGPU {gpu_id}\\n→ [B, L, 4096]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # Down projection
        for gpu_id in range(8):
            dot.node(f'ffn_down_s0_{gpu_id}', 
                    f'Down Projection\\nGPU {gpu_id}\\nW_down: [4096, 1024]\\n→ [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # All-reduce for FFN output
        dot.node('all_reduce_ffn_s0', 'All-Reduce\\n8 GPUs\\n→ [B, L, 8192]', 
                shape='parallelogram', style='filled', fillcolor='lightsteelblue')
    
    # Residual connection for layer 0
    dot.node('residual_s0', 'Residual Add\\nLayer 0\\n→ [B, L, 8192]', 
            shape='parallelogram', style='filled', fillcolor='lightgray')

# Pipeline communication between stages
dot.node('pipeline_comm', 'Pipeline Communication\\nStage 0 → Stage 1\\n[B, L, 8192]', 
        shape='parallelogram', style='filled', fillcolor='lightgreen')

# Pipeline Stage 1 - GPUs 8-15
with dot.subgraph(name='cluster_stage1') as c:
    c.attr(label='Pipeline Stage 1 - GPUs 8-15', style='rounded', fillcolor='lightyellow')
    
    # Layer 1 - Multi-Head Attention with TP=8
    with dot.subgraph(name='cluster_stage1_mha') as mha:
        mha.attr(label='Layer 1 - MHA (TP=8)', style='rounded', fillcolor='lightcyan')
        
        # Input broadcast to stage 1
        dot.node('broadcast_s1', 'Broadcast\\nX: [B, L, 8192] → 8 GPUs', 
                shape='parallelogram', style='filled', fillcolor='lightgreen')
        
        # Q projections across 8 GPUs
        for gpu_id in range(8, 16):
            local_id = gpu_id - 8
            dot.node(f'q_proj_s1_{gpu_id}', 
                    f'Q Projection\\nGPU {gpu_id}\\nW_Q: [8192, 1024]\\n→ Q: [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # K projections across 8 GPUs
        for gpu_id in range(8, 16):
            local_id = gpu_id - 8
            dot.node(f'k_proj_s1_{gpu_id}', 
                    f'K Projection\\nGPU {gpu_id}\\nW_K: [8192, 1024]\\n→ K: [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # V projections across 8 GPUs
        for gpu_id in range(8, 16):
            local_id = gpu_id - 8
            dot.node(f'v_proj_s1_{gpu_id}', 
                    f'V Projection\\nGPU {gpu_id}\\nW_V: [8192, 1024]\\n→ V: [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Attention computation across 8 GPUs
        for gpu_id in range(8, 16):
            local_id = gpu_id - 8
            dot.node(f'attn_s1_{gpu_id}', 
                    f'Multi-Head Attention\\nGPU {gpu_id}\\n2 heads × 512 dims\\n→ [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightgoldenrod')
        
        # All-reduce for attention output
        dot.node('all_reduce_attn_s1', 'All-Reduce\\n8 GPUs\\n→ [B, L, 8192]', 
                shape='parallelogram', style='filled', fillcolor='lightsteelblue')
    
    # Layer 1 - FFN with TP=8
    with dot.subgraph(name='cluster_stage1_ffn') as ffn:
        ffn.attr(label='Layer 1 - FFN (TP=8)', style='rounded', fillcolor='lightcyan')
        
        # Gate projection
        for gpu_id in range(8, 16):
            local_id = gpu_id - 8
            dot.node(f'ffn_gate_s1_{gpu_id}', 
                    f'Gate Projection\\nGPU {gpu_id}\\nW_gate: [8192, 4096]\\n→ [B, L, 4096]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # Up projection
        for gpu_id in range(8, 16):
            local_id = gpu_id - 8
            dot.node(f'ffn_up_s1_{gpu_id}', 
                    f'Up Projection\\nGPU {gpu_id}\\nW_up: [8192, 4096]\\n→ [B, L, 4096]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # SiLU activation
        for gpu_id in range(8, 16):
            local_id = gpu_id - 8
            dot.node(f'ffn_silu_s1_{gpu_id}', 
                    f'SiLU\\nGPU {gpu_id}\\n→ [B, L, 4096]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # Element-wise multiplication
        for gpu_id in range(8, 16):
            local_id = gpu_id - 8
            dot.node(f'ffn_mul_s1_{gpu_id}', 
                    f'Element-wise Mul\\nGPU {gpu_id}\\n→ [B, L, 4096]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # Down projection
        for gpu_id in range(8, 16):
            local_id = gpu_id - 8
            dot.node(f'ffn_down_s1_{gpu_id}', 
                    f'Down Projection\\nGPU {gpu_id}\\nW_down: [4096, 1024]\\n→ [B, L, 1024]', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
        
        # All-reduce for FFN output
        dot.node('all_reduce_ffn_s1', 'All-Reduce\\n8 GPUs\\n→ [B, L, 8192]', 
                shape='parallelogram', style='filled', fillcolor='lightsteelblue')
    
    # Residual connection for layer 1
    dot.node('residual_s1', 'Residual Add\\nLayer 1\\n→ [B, L, 8192]', 
            shape='parallelogram', style='filled', fillcolor='lightgray')

# Output
dot.node('output', 'Output\\nY: [B, L, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')

# Connect the nodes
# Input to stage 0
dot.edge('input', 'broadcast_s0')

# Stage 0 MHA connections
for gpu_id in range(8):
    dot.edge('broadcast_s0', f'q_proj_s0_{gpu_id}')
    dot.edge('broadcast_s0', f'k_proj_s0_{gpu_id}')
    dot.edge('broadcast_s0', f'v_proj_s0_{gpu_id}')
    dot.edge(f'q_proj_s0_{gpu_id}', f'attn_s0_{gpu_id}')
    dot.edge(f'k_proj_s0_{gpu_id}', f'attn_s0_{gpu_id}')
    dot.edge(f'v_proj_s0_{gpu_id}', f'attn_s0_{gpu_id}')
    dot.edge(f'attn_s0_{gpu_id}', 'all_reduce_attn_s0')

# Stage 0 FFN connections
dot.edge('all_reduce_attn_s0', 'residual_s0')
for gpu_id in range(8):
    dot.edge('all_reduce_attn_s0', f'ffn_gate_s0_{gpu_id}')
    dot.edge('all_reduce_attn_s0', f'ffn_up_s0_{gpu_id}')
    dot.edge(f'ffn_gate_s0_{gpu_id}', f'ffn_silu_s0_{gpu_id}')
    dot.edge(f'ffn_up_s0_{gpu_id}', f'ffn_mul_s0_{gpu_id}')
    dot.edge(f'ffn_silu_s0_{gpu_id}', f'ffn_mul_s0_{gpu_id}')
    dot.edge(f'ffn_mul_s0_{gpu_id}', f'ffn_down_s0_{gpu_id}')
    dot.edge(f'ffn_down_s0_{gpu_id}', 'all_reduce_ffn_s0')

dot.edge('all_reduce_ffn_s0', 'residual_s0')

# Pipeline communication
dot.edge('residual_s0', 'pipeline_comm')
dot.edge('pipeline_comm', 'broadcast_s1')

# Stage 1 MHA connections
for gpu_id in range(8, 16):
    dot.edge('broadcast_s1', f'q_proj_s1_{gpu_id}')
    dot.edge('broadcast_s1', f'k_proj_s1_{gpu_id}')
    dot.edge('broadcast_s1', f'v_proj_s1_{gpu_id}')
    dot.edge(f'q_proj_s1_{gpu_id}', f'attn_s1_{gpu_id}')
    dot.edge(f'k_proj_s1_{gpu_id}', f'attn_s1_{gpu_id}')
    dot.edge(f'v_proj_s1_{gpu_id}', f'attn_s1_{gpu_id}')
    dot.edge(f'attn_s1_{gpu_id}', 'all_reduce_attn_s1')

# Stage 1 FFN connections
dot.edge('all_reduce_attn_s1', 'residual_s1')
for gpu_id in range(8, 16):
    dot.edge('all_reduce_attn_s1', f'ffn_gate_s1_{gpu_id}')
    dot.edge('all_reduce_attn_s1', f'ffn_up_s1_{gpu_id}')
    dot.edge(f'ffn_gate_s1_{gpu_id}', f'ffn_silu_s1_{gpu_id}')
    dot.edge(f'ffn_up_s1_{gpu_id}', f'ffn_mul_s1_{gpu_id}')
    dot.edge(f'ffn_silu_s1_{gpu_id}', f'ffn_mul_s1_{gpu_id}')
    dot.edge(f'ffn_mul_s1_{gpu_id}', f'ffn_down_s1_{gpu_id}')
    dot.edge(f'ffn_down_s1_{gpu_id}', 'all_reduce_ffn_s1')

dot.edge('all_reduce_ffn_s1', 'residual_s1')
dot.edge('residual_s1', 'output')

# Save the DAG
dot.format = 'svg'
dot.render('/home/wzc/data/file-share/submission/baseline_tp_pp')

print("Generated Baseline TP+PP DAG")