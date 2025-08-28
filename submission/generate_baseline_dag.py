import graphviz

# Create baseline DAG for TP=8, PP=2
dot = graphviz.Digraph('baseline_dense_transformer', comment='Baseline Dense Transformer with TP=8, PP=2')
dot.attr(rankdir='TB', size='20,20')

# Input node
dot.node('input', 'Input\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='lightblue')

# Layer 0 - Stage 0 (Devices 0-7)
with dot.subgraph(name='cluster_stage0') as c:
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
    
    # FFN
    c.node('l0_ffn_norm', 'LayerNorm\n(B×L×8192)', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('l0_ffn1', 'FFN Up\n(B×L×8192→B×L×32768)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l0_ffn2', 'FFN Down\n(B×L×32768→B×L×8192)', shape='rectangle', style='filled', fillcolor='lightgreen')
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
    
    # FFN
    c.node('l1_ffn_norm', 'LayerNorm\n(B×L×8192)', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('l1_ffn1', 'FFN Up\n(B×L×8192→B×L×32768)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l1_ffn2', 'FFN Down\n(B×L×32768→B×L×8192)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l1_res2', 'Residual Add\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='orange')

# Pipeline communication
dot.node('pipeline_comm', 'Pipeline Communication\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='purple')

# Layer 2 - Stage 1 (Devices 8-15)
with dot.subgraph(name='cluster_stage1') as c:
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
    
    # FFN
    c.node('l2_ffn_norm', 'LayerNorm\n(B×L×8192)', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('l2_ffn1', 'FFN Up\n(B×L×8192→B×L×32768)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l2_ffn2', 'FFN Down\n(B×L×32768→B×L×8192)', shape='rectangle', style='filled', fillcolor='lightgreen')
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
    
    # FFN
    c.node('l3_ffn_norm', 'LayerNorm\n(B×L×8192)', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('l3_ffn1', 'FFN Up\n(B×L×8192→B×L×32768)', shape='rectangle', style='filled', fillcolor='lightgreen')
    c.node('l3_ffn2', 'FFN Down\n(B×L×32768→B×L×8192)', shape='rectangle', style='filled', fillcolor='lightgreen')
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

dot.edge('l0_res1', 'l0_ffn_norm')
dot.edge('l0_ffn_norm', 'l0_ffn1')
dot.edge('l0_ffn1', 'l0_ffn2')
dot.edge('l0_ffn2', 'l0_res2')
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

dot.edge('l1_res1', 'l1_ffn_norm')
dot.edge('l1_ffn_norm', 'l1_ffn1')
dot.edge('l1_ffn1', 'l1_ffn2')
dot.edge('l1_ffn2', 'l1_res2')
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

dot.edge('l2_res1', 'l2_ffn_norm')
dot.edge('l2_ffn_norm', 'l2_ffn1')
dot.edge('l2_ffn1', 'l2_ffn2')
dot.edge('l2_ffn2', 'l2_res2')
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

dot.edge('l3_res1', 'l3_ffn_norm')
dot.edge('l3_ffn_norm', 'l3_ffn1')
dot.edge('l3_ffn1', 'l3_ffn2')
dot.edge('l3_ffn2', 'l3_res2')
dot.edge('l3_res1', 'l3_res2')  # Residual

# Output
dot.edge('l3_res2', 'output')

# Save files
dot.save('/home/wzc/data/file-share/submission/baseline_dense_transformer.dot')
dot.render('/home/wzc/data/file-share/submission/baseline_dense_transformer', format='svg', cleanup=True)

print("Baseline DAG generated successfully")