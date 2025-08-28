import graphviz

# Create MoE baseline DAG for TP=8, PP=2 with 4 experts per layer
dot = graphviz.Digraph('moe_baseline_transformer', comment='MoE Baseline with TP=8, PP=2 and 4 Experts/Layer')
dot.attr(rankdir='TB', size='25,25')

# Input node
dot.node('input', 'Input\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='lightblue')

# Layer 0 - Stage 0 (Devices 0-7)
with dot.subgraph(name='cluster_stage0_moe') as c:
    c.attr(label='Stage 0 (Devices 0-7)\nLayers 0-1', style='dashed', color='blue')
    
    # Layer 0
    c.node('l0_norm', 'LayerNorm\n(B×L×8192)', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('l0_qkv', 'QKV Linear\n(B×L×8192→B×L×24576)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l0_qkv_split', 'Split Heads\n(B×L×24576→16×B×L×512)', shape='parallelogram', style='filled', fillcolor='lightcoral')
    
    # Attention across 8 devices
    for i in range(8):
        c.node(f'l0_attn_{i}', f'Attention Head {i*2}-{i*2+1}\n(B×L×1024)', 
               shape='rectangle', style='filled', fillcolor='lightcyan')
    
    c.node('l0_concat', 'Concat Heads\n(16×B×L×512→B×L×8192)', shape='parallelogram', style='filled', fillcolor='lightcoral')
    c.node('l0_proj', 'Output Projection\n(B×L×8192→B×L×8192)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l0_res1', 'Residual Add\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='orange')
    
    # MoE Layer
    c.node('l0_gate', 'Expert Gate\n(B×L×8192→B×L×4)', shape='parallelogram', style='filled', fillcolor='purple')
    
    # 4 experts distributed across devices
    for expert_id in range(4):
        c.node(f'l0_expert_{expert_id}', f'Expert {expert_id}\n(B×L×8192→B×L×8192)\nDevices {expert_id*2}-{expert_id*2+1}', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
    
    c.node('l0_expert_agg', 'Expert Aggregation\n(B×L×8192)', shape='parallelogram', style='filled', fillcolor='gold')
    c.node('l0_res2', 'Residual Add\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='orange')
    
    # Layer 1
    c.node('l1_norm', 'LayerNorm\n(B×L×8192)', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('l1_qkv', 'QKV Linear\n(B×L×8192→B×L×24576)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l1_qkv_split', 'Split Heads\n(B×L×24576→16×B×L×512)', shape='parallelogram', style='filled', fillcolor='lightcoral')
    
    for i in range(8):
        c.node(f'l1_attn_{i}', f'Attention Head {i*2}-{i*2+1}\n(B×L×1024)', 
               shape='rectangle', style='filled', fillcolor='lightcyan')
    
    c.node('l1_concat', 'Concat Heads\n(16×B×L×512→B×L×8192)', shape='parallelogram', style='filled', fillcolor='lightcoral')
    c.node('l1_proj', 'Output Projection\n(B×L×8192→B×L×8192)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l1_res1', 'Residual Add\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='orange')
    
    # MoE Layer 1
    c.node('l1_gate', 'Expert Gate\n(B×L×8192→B×L×4)', shape='parallelogram', style='filled', fillcolor='purple')
    
    for expert_id in range(4):
        c.node(f'l1_expert_{expert_id}', f'Expert {expert_id}\n(B×L×8192→B×L×8192)\nDevices {expert_id*2}-{expert_id*2+1}', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
    
    c.node('l1_expert_agg', 'Expert Aggregation\n(B×L×8192)', shape='parallelogram', style='filled', fillcolor='gold')
    c.node('l1_res2', 'Residual Add\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='orange')

# Pipeline communication
dot.node('pipeline_comm', 'Pipeline Communication\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='purple')

# Layer 2 - Stage 1 (Devices 8-15)
with dot.subgraph(name='cluster_stage1_moe') as c:
    c.attr(label='Stage 1 (Devices 8-15)\nLayers 2-3', style='dashed', color='red')
    
    # Layer 2
    c.node('l2_norm', 'LayerNorm\n(B×L×8192)', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('l2_qkv', 'QKV Linear\n(B×L×8192→B×L×24576)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l2_qkv_split', 'Split Heads\n(B×L×24576→16×B×L×512)', shape='parallelogram', style='filled', fillcolor='lightcoral')
    
    for i in range(8):
        c.node(f'l2_attn_{i}', f'Attention Head {i*2}-{i*2+1}\n(B×L×1024)', 
               shape='rectangle', style='filled', fillcolor='lightcyan')
    
    c.node('l2_concat', 'Concat Heads\n(16×B×L×512→B×L×8192)', shape='parallelogram', style='filled', fillcolor='lightcoral')
    c.node('l2_proj', 'Output Projection\n(B×L×8192→B×L×8192)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l2_res1', 'Residual Add\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='orange')
    
    # MoE Layer 2
    c.node('l2_gate', 'Expert Gate\n(B×L×8192→B×L×4)', shape='parallelogram', style='filled', fillcolor='purple')
    
    for expert_id in range(4):
        c.node(f'l2_expert_{expert_id}', f'Expert {expert_id}\n(B×L×8192→B×L×8192)\nDevices {8+expert_id*2}-{9+expert_id*2}', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
    
    c.node('l2_expert_agg', 'Expert Aggregation\n(B×L×8192)', shape='parallelogram', style='filled', fillcolor='gold')
    c.node('l2_res2', 'Residual Add\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='orange')
    
    # Layer 3
    c.node('l3_norm', 'LayerNorm\n(B×L×8192)', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('l3_qkv', 'QKV Linear\n(B×L×8192→B×L×24576)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l3_qkv_split', 'Split Heads\n(B×L×24576→16×B×L×512)', shape='parallelogram', style='filled', fillcolor='lightcoral')
    
    for i in range(8):
        c.node(f'l3_attn_{i}', f'Attention Head {i*2}-{i*2+1}\n(B×L×1024)', 
               shape='rectangle', style='filled', fillcolor='lightcyan')
    
    c.node('l3_concat', 'Concat Heads\n(16×B×L×512→B×L×8192)', shape='parallelogram', style='filled', fillcolor='lightcoral')
    c.node('l3_proj', 'Output Projection\n(B×L×8192→B×L×8192)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l3_res1', 'Residual Add\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='orange')
    
    # MoE Layer 3
    c.node('l3_gate', 'Expert Gate\n(B×L×8192→B×L×4)', shape='parallelogram', style='filled', fillcolor='purple')
    
    for expert_id in range(4):
        c.node(f'l3_expert_{expert_id}', f'Expert {expert_id}\n(B×L×8192→B×L×8192)\nDevices {8+expert_id*2}-{9+expert_id*2}', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
    
    c.node('l3_expert_agg', 'Expert Aggregation\n(B×L×8192)', shape='parallelogram', style='filled', fillcolor='gold')
    c.node('l3_res2', 'Residual Add\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='orange')

# Output
dot.node('output', 'Output\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='lightblue')

# Connections
# Input to Layer 0
dot.edge('input', 'l0_norm')
dot.edge('l0_norm', 'l0_qkv')
dot.edge('l0_qkv', 'l0_qkv_split')

# Layer 0 attention
for i in range(8):
    dot.edge('l0_qkv_split', f'l0_attn_{i}')
    dot.edge(f'l0_attn_{i}', 'l0_concat')

dot.edge('l0_concat', 'l0_proj')
dot.edge('l0_proj', 'l0_res1')
dot.edge('input', 'l0_res1')  # Residual

# Layer 0 MoE
dot.edge('l0_res1', 'l0_gate')
for expert_id in range(4):
    dot.edge('l0_gate', f'l0_expert_{expert_id}', style='dashed', label=f'route tokens')
    dot.edge('l0_expert_{expert_id}', 'l0_expert_agg')
dot.edge('l0_expert_agg', 'l0_res2')
dot.edge('l0_res1', 'l0_res2')  # Residual

# Layer 1
dot.edge('l0_res2', 'l1_norm')
dot.edge('l1_norm', 'l1_qkv')
dot.edge('l1_qkv', 'l1_qkv_split')

for i in range(8):
    dot.edge('l1_qkv_split', f'l1_attn_{i}')
    dot.edge(f'l1_attn_{i}', 'l1_concat')

dot.edge('l1_concat', 'l1_proj')
dot.edge('l1_proj', 'l1_res1')
dot.edge('l0_res2', 'l1_res1')  # Residual

# Layer 1 MoE
dot.edge('l1_res1', 'l1_gate')
for expert_id in range(4):
    dot.edge('l1_gate', f'l1_expert_{expert_id}', style='dashed', label=f'route tokens')
    dot.edge('l1_expert_{expert_id}', 'l1_expert_agg')
dot.edge('l1_expert_agg', 'l1_res2')
dot.edge('l1_res1', 'l1_res2')  # Residual

# Pipeline communication
dot.edge('l1_res2', 'pipeline_comm')
dot.edge('pipeline_comm', 'l2_norm')

# Layer 2
dot.edge('l2_norm', 'l2_qkv')
dot.edge('l2_qkv', 'l2_qkv_split')

for i in range(8):
    dot.edge('l2_qkv_split', f'l2_attn_{i}')
    dot.edge(f'l2_attn_{i}', 'l2_concat')

dot.edge('l2_concat', 'l2_proj')
dot.edge('l2_proj', 'l2_res1')
dot.edge('pipeline_comm', 'l2_res1')  # Residual

# Layer 2 MoE
dot.edge('l2_res1', 'l2_gate')
for expert_id in range(4):
    dot.edge('l2_gate', f'l2_expert_{expert_id}', style='dashed', label=f'route tokens')
    dot.edge('l2_expert_{expert_id}', 'l2_expert_agg')
dot.edge('l2_expert_agg', 'l2_res2')
dot.edge('l2_res1', 'l2_res2')  # Residual

# Layer 3
dot.edge('l2_res2', 'l3_norm')
dot.edge('l3_norm', 'l3_qkv')
dot.edge('l3_qkv', 'l3_qkv_split')

for i in range(8):
    dot.edge('l3_qkv_split', f'l3_attn_{i}')
    dot.edge(f'l3_attn_{i}', 'l3_concat')

dot.edge('l3_concat', 'l3_proj')
dot.edge('l3_proj', 'l3_res1')
dot.edge('l2_res2', 'l3_res1')  # Residual

# Layer 3 MoE
dot.edge('l3_res1', 'l3_gate')
for expert_id in range(4):
    dot.edge('l3_gate', f'l3_expert_{expert_id}', style='dashed', label=f'route tokens')
    dot.edge('l3_expert_{expert_id}', 'l3_expert_agg')
dot.edge('l3_expert_agg', 'l3_res2')
dot.edge('l3_res1', 'l3_res2')  # Residual

# Output
dot.edge('l3_res2', 'output')

# Save files
dot.save('/home/wzc/data/file-share/submission/moe_baseline_transformer.dot')
dot.render('/home/wzc/data/file-share/submission/moe_baseline_transformer', format='svg', cleanup=True)

print("MoE Baseline DAG generated successfully")