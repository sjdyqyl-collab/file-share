import graphviz

# Create baseline DAG (TP=8, PP=2, 16 GPUs)
dot = graphviz.Digraph('baseline_moe', comment='Baseline MoE Deployment (TP=8, PP=2, 16 GPUs)')
dot.attr(rankdir='TB', size='20,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', fillcolor='orange')  # Communication

# Input node
dot.node('input', 'Input\\n[1024 seqs, 10000 tokens, 8192 dim]\\nAll GPUs', shape='ellipse')

# Pipeline Stage 1 (Layers 1-2) - GPUs 0-7
with dot.subgraph(name='cluster_stage1') as c:
    c.attr(label='Pipeline Stage 1\\nGPUs 0-7', style='dashed')
    
    # Layer 1
    c.node('l1_norm1', 'LayerNorm1\\n[1024×10000×8192]\\nGPUs 0-7', shape='rectangle')
    c.node('l1_mha_qkv', 'MHA QKV\\n[1024×10000×8192]\\nGPUs 0-7 (TP=8)', shape='rectangle')
    c.node('l1_mha_attn', 'MHA Attention\\n[1024×10000×8192]\\nGPUs 0-7 (TP=8)', shape='rectangle')
    c.node('l1_mha_out', 'MHA Output\\n[1024×10000×8192]\\nGPUs 0-7 (TP=8)', shape='rectangle')
    c.node('l1_res1', 'Residual Add1\\n[1024×10000×8192]\\nGPUs 0-7', shape='parallelogram')
    
    c.node('l1_norm2', 'LayerNorm2\\n[1024×10000×8192]\\nGPUs 0-7', shape='rectangle')
    c.node('l1_gate', 'Gate\\n[1024×10000×16 experts]\\nGPUs 0-7', shape='parallelogram')
    
    # Expert 0-3 on GPU 0
    for i in range(4):
        c.node(f'l1_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 0', shape='rectangle')
    
    # Expert 4-7 on GPU 1
    for i in range(4, 8):
        c.node(f'l1_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 1', shape='rectangle')
    
    # Expert 8-11 on GPU 2
    for i in range(8, 12):
        c.node(f'l1_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 2', shape='rectangle')
    
    # Expert 12-15 on GPU 3
    for i in range(12, 16):
        c.node(f'l1_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 3', shape='rectangle')
    
    c.node('l1_agg', 'Expert Aggregation\\n[1024×10000×8192]\\nGPUs 0-7', shape='parallelogram')
    c.node('l1_res2', 'Residual Add2\\n[1024×10000×8192]\\nGPUs 0-7', shape='parallelogram')

# Pipeline Stage 2 (Layers 3-4) - GPUs 8-15
with dot.subgraph(name='cluster_stage2') as c:
    c.attr(label='Pipeline Stage 2\\nGPUs 8-15', style='dashed')
    
    # Communication between stages
    c.node('stage1_to_stage2', 'Pipeline Communication\\n[1024×10000×8192]\\nGPUs 7→8', shape='diamond')
    
    # Layer 3
    c.node('l3_norm1', 'LayerNorm3\\n[1024×10000×8192]\\nGPUs 8-15', shape='rectangle')
    c.node('l3_mha_qkv', 'MHA QKV\\n[1024×10000×8192]\\nGPUs 8-15 (TP=8)', shape='rectangle')
    c.node('l3_mha_attn', 'MHA Attention\\n[1024×10000×8192]\\nGPUs 8-15 (TP=8)', shape='rectangle')
    c.node('l3_mha_out', 'MHA Output\\n[1024×10000×8192]\\nGPUs 8-15 (TP=8)', shape='rectangle')
    c.node('l3_res1', 'Residual Add3\\n[1024×10000×8192]\\nGPUs 8-15', shape='parallelogram')
    
    c.node('l3_norm2', 'LayerNorm4\\n[1024×10000×8192]\\nGPUs 8-15', shape='rectangle')
    c.node('l3_gate', 'Gate\\n[1024×10000×16 experts]\\nGPUs 8-15', shape='parallelogram')
    
    # Expert 0-3 on GPU 8
    for i in range(4):
        c.node(f'l3_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 8', shape='rectangle')
    
    # Expert 4-7 on GPU 9
    for i in range(4, 8):
        c.node(f'l3_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 9', shape='rectangle')
    
    # Expert 8-11 on GPU 10
    for i in range(8, 12):
        c.node(f'l3_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 10', shape='rectangle')
    
    # Expert 12-15 on GPU 11
    for i in range(12, 16):
        c.node(f'l3_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 11', shape='rectangle')
    
    c.node('l3_agg', 'Expert Aggregation\\n[1024×10000×8192]\\nGPUs 8-15', shape='parallelogram')
    c.node('l3_res2', 'Residual Add4\\n[1024×10000×8192]\\nGPUs 8-15', shape='parallelogram')

# Layer 2 (in stage 1)
with dot.subgraph(name='cluster_layer2') as c:
    c.attr(label='Layer 2\\nGPUs 0-7', style='dotted')
    
    c.node('l2_norm1', 'LayerNorm2\\n[1024×10000×8192]\\nGPUs 0-7', shape='rectangle')
    c.node('l2_mha_qkv', 'MHA QKV2\\n[1024×10000×8192]\\nGPUs 0-7 (TP=8)', shape='rectangle')
    c.node('l2_mha_attn', 'MHA Attention2\\n[1024×10000×8192]\\nGPUs 0-7 (TP=8)', shape='rectangle')
    c.node('l2_mha_out', 'MHA Output2\\n[1024×10000×8192]\\nGPUs 0-7 (TP=8)', shape='rectangle')
    c.node('l2_res1', 'Residual Add2\\n[1024×10000×8192]\\nGPUs 0-7', shape='parallelogram')
    
    c.node('l2_norm2', 'LayerNorm2\\n[1024×10000×8192]\\nGPUs 0-7', shape='rectangle')
    c.node('l2_gate', 'Gate2\\n[1024×10000×16 experts]\\nGPUs 0-7', shape='parallelogram')
    
    # Expert 0-3 on GPU 4
    for i in range(4):
        c.node(f'l2_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 4', shape='rectangle')
    
    # Expert 4-7 on GPU 5
    for i in range(4, 8):
        c.node(f'l2_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 5', shape='rectangle')
    
    # Expert 8-11 on GPU 6
    for i in range(8, 12):
        c.node(f'l2_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 6', shape='rectangle')
    
    # Expert 12-15 on GPU 7
    for i in range(12, 16):
        c.node(f'l2_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 7', shape='rectangle')
    
    c.node('l2_agg', 'Expert Aggregation2\\n[1024×10000×8192]\\nGPUs 0-7', shape='parallelogram')
    c.node('l2_res2', 'Residual Add2\\n[1024×10000×8192]\\nGPUs 0-7', shape='parallelogram')

# Layer 4 (in stage 2)
with dot.subgraph(name='cluster_layer4') as c:
    c.attr(label='Layer 4\\nGPUs 8-15', style='dotted')
    
    c.node('l4_norm1', 'LayerNorm4\\n[1024×10000×8192]\\nGPUs 8-15', shape='rectangle')
    c.node('l4_mha_qkv', 'MHA QKV4\\n[1024×10000×8192]\\nGPUs 8-15 (TP=8)', shape='rectangle')
    c.node('l4_mha_attn', 'MHA Attention4\\n[1024×10000×8192]\\nGPUs 8-15 (TP=8)', shape='rectangle')
    c.node('l4_mha_out', 'MHA Output4\\n[1024×10000×8192]\\nGPUs 8-15 (TP=8)', shape='rectangle')
    c.node('l4_res1', 'Residual Add4\\n[1024×10000×8192]\\nGPUs 8-15', shape='parallelogram')
    
    c.node('l4_norm2', 'LayerNorm4\\n[1024×10000×8192]\\nGPUs 8-15', shape='rectangle')
    c.node('l4_gate', 'Gate4\\n[1024×10000×16 experts]\\nGPUs 8-15', shape='parallelogram')
    
    # Expert 0-3 on GPU 12
    for i in range(4):
        c.node(f'l4_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 12', shape='rectangle')
    
    # Expert 4-7 on GPU 13
    for i in range(4, 8):
        c.node(f'l4_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 13', shape='rectangle')
    
    # Expert 8-11 on GPU 14
    for i in range(8, 12):
        c.node(f'l4_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 14', shape='rectangle')
    
    # Expert 12-15 on GPU 15
    for i in range(12, 16):
        c.node(f'l4_exp{i}', f'Expert{i}\\n[1024×10000×32768]\\nGPU 15', shape='rectangle')
    
    c.node('l4_agg', 'Expert Aggregation4\\n[1024×10000×8192]\\nGPUs 8-15', shape='parallelogram')
    c.node('l4_res2', 'Residual Add4\\n[1024×10000×8192]\\nGPUs 8-15', shape='parallelogram')

# Output node
dot.node('output', 'Output\\n[1024 seqs, 10000 tokens, 8192 dim]\\nGPUs 8-15', shape='ellipse')

# Connections
dot.edge('input', 'l1_norm1')
dot.edge('l1_norm1', 'l1_mha_qkv')
dot.edge('l1_mha_qkv', 'l1_mha_attn')
dot.edge('l1_mha_attn', 'l1_mha_out')
dot.edge('l1_mha_out', 'l1_res1')
dot.edge('input', 'l1_res1')  # Residual connection
dot.edge('l1_res1', 'l1_norm2')
dot.edge('l1_norm2', 'l1_gate')

# Connect experts for layer 1
for i in range(16):
    dot.edge('l1_gate', f'l1_exp{i}', style='dashed')
    dot.edge(f'l1_exp{i}', 'l1_agg')

dot.edge('l1_agg', 'l1_res2')
dot.edge('l1_res1', 'l1_res2')  # Residual connection
dot.edge('l1_res2', 'l2_norm1')

# Layer 2 connections
dot.edge('l2_norm1', 'l2_mha_qkv')
dot.edge('l2_mha_qkv', 'l2_mha_attn')
dot.edge('l2_mha_attn', 'l2_mha_out')
dot.edge('l2_mha_out', 'l2_res1')
dot.edge('l1_res2', 'l2_res1')  # Residual connection
dot.edge('l2_res1', 'l2_norm2')
dot.edge('l2_norm2', 'l2_gate')

# Connect experts for layer 2
for i in range(16):
    dot.edge('l2_gate', f'l2_exp{i}', style='dashed')
    dot.edge(f'l2_exp{i}', 'l2_agg')

dot.edge('l2_agg', 'l2_res2')
dot.edge('l2_res1', 'l2_res2')  # Residual connection
dot.edge('l2_res2', 'stage1_to_stage2')

# Pipeline stage 2 connections
dot.edge('stage1_to_stage2', 'l3_norm1')
dot.edge('l3_norm1', 'l3_mha_qkv')
dot.edge('l3_mha_qkv', 'l3_mha_attn')
dot.edge('l3_mha_attn', 'l3_mha_out')
dot.edge('l3_mha_out', 'l3_res1')
dot.edge('stage1_to_stage2', 'l3_res1')  # Residual connection
dot.edge('l3_res1', 'l3_norm2')
dot.edge('l3_norm2', 'l3_gate')

# Connect experts for layer 3
for i in range(16):
    dot.edge('l3_gate', f'l3_exp{i}', style='dashed')
    dot.edge(f'l3_exp{i}', 'l3_agg')

dot.edge('l3_agg', 'l3_res2')
dot.edge('l3_res1', 'l3_res2')  # Residual connection
dot.edge('l3_res2', 'l4_norm1')

# Layer 4 connections
dot.edge('l4_norm1', 'l4_mha_qkv')
dot.edge('l4_mha_qkv', 'l4_mha_attn')
dot.edge('l4_mha_attn', 'l4_mha_out')
dot.edge('l4_mha_out', 'l4_res1')
dot.edge('l3_res2', 'l4_res1')  # Residual connection
dot.edge('l4_res1', 'l4_norm2')
dot.edge('l4_norm2', 'l4_gate')

# Connect experts for layer 4
for i in range(16):
    dot.edge('l4_gate', f'l4_exp{i}', style='dashed')
    dot.edge(f'l4_exp{i}', 'l4_agg')

dot.edge('l4_agg', 'l4_res2')
dot.edge('l4_res1', 'l4_res2')  # Residual connection
dot.edge('l4_res2', 'output')

# Save files
dot.render('/home/wzc/data/file-share/2025-09-08-09-35-54/baseline_moe', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-08-09-35-54/baseline_moe.dot')

print("Baseline DAG generated successfully!")