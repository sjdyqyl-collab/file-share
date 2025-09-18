import graphviz

# Create proposed DAG (EP=64, 64 GPUs, 1 expert per GPU)
dot = graphviz.Digraph('proposed_moe', comment='Proposed Cross-Node Expert Parallelism (EP=64, 64 GPUs)')
dot.attr(rankdir='TB', size='30,30')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', fillcolor='orange')  # Communication

# Input node - distributed across all GPUs
dot.node('input', 'Input\\n[1024 seqs, 10000 tokens, 8192 dim]\\nAll 64 GPUs', shape='ellipse')

# Layer 1
with dot.subgraph(name='cluster_layer1') as c:
    c.attr(label='Layer 1\\n64 GPUs, 1 Expert/GPU', style='dashed')
    
    # Pre-processing (duplicated on all GPUs)
    c.node('l1_norm1', 'LayerNorm1\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l1_mha_qkv', 'MHA QKV1\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l1_mha_attn', 'MHA Attention1\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l1_mha_out', 'MHA Output1\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l1_res1', 'Residual Add1\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')
    
    c.node('l1_norm2', 'LayerNorm2\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l1_gate', 'Gate1\\n[1024×10000×64 experts]\\nAll 64 GPUs', shape='parallelogram')
    
    # Create 64 experts, one per GPU
    for gpu in range(64):
        expert_id = gpu % 16  # 16 experts per layer distributed across 64 GPUs
        c.node(f'l1_exp{gpu}', f'Expert{expert_id}\\n[1024×tokens×32768]\\nGPU {gpu}', shape='rectangle')
    
    # Communication nodes for expert routing
    for gpu in range(64):
        c.node(f'l1_route_{gpu}', f'Token Routing\\n[selected tokens×8192]\\nGPU {gpu}', shape='diamond')
        c.node(f'l1_gather_{gpu}', f'Expert Gathering\\n[1024×10000×8192]\\nGPU {gpu}', shape='diamond')
    
    c.node('l1_agg', 'Expert Aggregation1\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')
    c.node('l1_res2', 'Residual Add1\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')

# Layer 2
with dot.subgraph(name='cluster_layer2') as c:
    c.attr(label='Layer 2\\n64 GPUs, 1 Expert/GPU', style='dashed')
    
    c.node('l2_norm1', 'LayerNorm2\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l2_mha_qkv', 'MHA QKV2\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l2_mha_attn', 'MHA Attention2\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l2_mha_out', 'MHA Output2\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l2_res1', 'Residual Add2\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')
    
    c.node('l2_norm2', 'LayerNorm2\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l2_gate', 'Gate2\\n[1024×10000×64 experts]\\nAll 64 GPUs', shape='parallelogram')
    
    # Create 64 experts, one per GPU
    for gpu in range(64):
        expert_id = gpu % 16
        c.node(f'l2_exp{gpu}', f'Expert{expert_id}\\n[1024×tokens×32768]\\nGPU {gpu}', shape='rectangle')
    
    # Communication nodes for expert routing
    for gpu in range(64):
        c.node(f'l2_route_{gpu}', f'Token Routing\\n[selected tokens×8192]\\nGPU {gpu}', shape='diamond')
        c.node(f'l2_gather_{gpu}', f'Expert Gathering\\n[1024×10000×8192]\\nGPU {gpu}', shape='diamond')
    
    c.node('l2_agg', 'Expert Aggregation2\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')
    c.node('l2_res2', 'Residual Add2\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')

# Layer 3
with dot.subgraph(name='cluster_layer3') as c:
    c.attr(label='Layer 3\\n64 GPUs, 1 Expert/GPU', style='dashed')
    
    c.node('l3_norm1', 'LayerNorm3\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l3_mha_qkv', 'MHA QKV3\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l3_mha_attn', 'MHA Attention3\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l3_mha_out', 'MHA Output3\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l3_res1', 'Residual Add3\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')
    
    c.node('l3_norm2', 'LayerNorm3\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l3_gate', 'Gate3\\n[1024×10000×64 experts]\\nAll 64 GPUs', shape='parallelogram')
    
    # Create 64 experts, one per GPU
    for gpu in range(64):
        expert_id = gpu % 16
        c.node(f'l3_exp{gpu}', f'Expert{expert_id}\\n[1024×tokens×32768]\\nGPU {gpu}', shape='rectangle')
    
    # Communication nodes for expert routing
    for gpu in range(64):
        c.node(f'l3_route_{gpu}', f'Token Routing\\n[selected tokens×8192]\\nGPU {gpu}', shape='diamond')
        c.node(f'l3_gather_{gpu}', f'Expert Gathering\\n[1024×10000×8192]\\nGPU {gpu}', shape='diamond')
    
    c.node('l3_agg', 'Expert Aggregation3\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')
    c.node('l3_res2', 'Residual Add3\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')

# Layer 4
with dot.subgraph(name='cluster_layer4') as c:
    c.attr(label='Layer 4\\n64 GPUs, 1 Expert/GPU', style='dashed')
    
    c.node('l4_norm1', 'LayerNorm4\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l4_mha_qkv', 'MHA QKV4\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l4_mha_attn', 'MHA Attention4\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l4_mha_out', 'MHA Output4\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l4_res1', 'Residual Add4\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')
    
    c.node('l4_norm2', 'LayerNorm4\\n[1024×10000×8192]\\nAll 64 GPUs', shape='rectangle')
    c.node('l4_gate', 'Gate4\\n[1024×10000×64 experts]\\nAll 64 GPUs', shape='parallelogram')
    
    # Create 64 experts, one per GPU
    for gpu in range(64):
        expert_id = gpu % 16
        c.node(f'l4_exp{gpu}', f'Expert{expert_id}\\n[1024×tokens×32768]\\nGPU {gpu}', shape='rectangle')
    
    # Communication nodes for expert routing
    for gpu in range(64):
        c.node(f'l4_route_{gpu}', f'Token Routing\\n[selected tokens×8192]\\nGPU {gpu}', shape='diamond')
        c.node(f'l4_gather_{gpu}', f'Expert Gathering\\n[1024×10000×8192]\\nGPU {gpu}', shape='diamond')
    
    c.node('l4_agg', 'Expert Aggregation4\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')
    c.node('l4_res2', 'Residual Add4\\n[1024×10000×8192]\\nAll 64 GPUs', shape='parallelogram')

# Output node
dot.node('output', 'Output\\n[1024 seqs, 10000 tokens, 8192 dim]\\nAll 64 GPUs', shape='ellipse')

# Connections for Layer 1
dot.edge('input', 'l1_norm1')
dot.edge('l1_norm1', 'l1_mha_qkv')
dot.edge('l1_mha_qkv', 'l1_mha_attn')
dot.edge('l1_mha_attn', 'l1_mha_out')
dot.edge('l1_mha_out', 'l1_res1')
dot.edge('input', 'l1_res1')  # Residual connection
dot.edge('l1_res1', 'l1_norm2')
dot.edge('l1_norm2', 'l1_gate')

# Expert routing and communication for Layer 1
for gpu in range(64):
    dot.edge('l1_gate', f'l1_route_{gpu}')
    dot.edge(f'l1_route_{gpu}', f'l1_exp{gpu}')
    dot.edge(f'l1_exp{gpu}', f'l1_gather_{gpu}')
    dot.edge(f'l1_gather_{gpu}', 'l1_agg')

dot.edge('l1_agg', 'l1_res2')
dot.edge('l1_res1', 'l1_res2')  # Residual connection
dot.edge('l1_res2', 'l2_norm1')

# Connections for Layer 2
dot.edge('l2_norm1', 'l2_mha_qkv')
dot.edge('l2_mha_qkv', 'l2_mha_attn')
dot.edge('l2_mha_attn', 'l2_mha_out')
dot.edge('l2_mha_out', 'l2_res1')
dot.edge('l1_res2', 'l2_res1')  # Residual connection
dot.edge('l2_res1', 'l2_norm2')
dot.edge('l2_norm2', 'l2_gate')

# Expert routing and communication for Layer 2
for gpu in range(64):
    dot.edge('l2_gate', f'l2_route_{gpu}')
    dot.edge(f'l2_route_{gpu}', f'l2_exp{gpu}')
    dot.edge(f'l2_exp{gpu}', f'l2_gather_{gpu}')
    dot.edge(f'l2_gather_{gpu}', 'l2_agg')

dot.edge('l2_agg', 'l2_res2')
dot.edge('l2_res1', 'l2_res2')  # Residual connection
dot.edge('l2_res2', 'l3_norm1')

# Connections for Layer 3
dot.edge('l3_norm1', 'l3_mha_qkv')
dot.edge('l3_mha_qkv', 'l3_mha_attn')
dot.edge('l3_mha_attn', 'l3_mha_out')
dot.edge('l3_mha_out', 'l3_res1')
dot.edge('l2_res2', 'l3_res1')  # Residual connection
dot.edge('l3_res1', 'l3_norm2')
dot.edge('l3_norm2', 'l3_gate')

# Expert routing and communication for Layer 3
for gpu in range(64):
    dot.edge('l3_gate', f'l3_route_{gpu}')
    dot.edge(f'l3_route_{gpu}', f'l3_exp{gpu}')
    dot.edge(f'l3_exp{gpu}', f'l3_gather_{gpu}')
    dot.edge(f'l3_gather_{gpu}', 'l3_agg')

dot.edge('l3_agg', 'l3_res2')
dot.edge('l3_res1', 'l3_res2')  # Residual connection
dot.edge('l3_res2', 'l4_norm1')

# Connections for Layer 4
dot.edge('l4_norm1', 'l4_mha_qkv')
dot.edge('l4_mha_qkv', 'l4_mha_attn')
dot.edge('l4_mha_attn', 'l4_mha_out')
dot.edge('l4_mha_out', 'l4_res1')
dot.edge('l3_res2', 'l4_res1')  # Residual connection
dot.edge('l4_res1', 'l4_norm2')
dot.edge('l4_norm2', 'l4_gate')

# Expert routing and communication for Layer 4
for gpu in range(64):
    dot.edge('l4_gate', f'l4_route_{gpu}')
    dot.edge(f'l4_route_{gpu}', f'l4_exp{gpu}')
    dot.edge(f'l4_exp{gpu}', f'l4_gather_{gpu}')
    dot.edge(f'l4_gather_{gpu}', 'l4_agg')

dot.edge('l4_agg', 'l4_res2')
dot.edge('l4_res1', 'l4_res2')  # Residual connection
dot.edge('l4_res2', 'output')

# Save files
dot.render('/home/wzc/data/file-share/2025-09-08-09-35-54/proposed_moe', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-08-09-35-54/proposed_moe.dot')

print("Proposed DAG generated successfully!")