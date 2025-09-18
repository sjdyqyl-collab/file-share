import graphviz

# Create baseline DAG with Tensor Parallelism (8-way) + Pipeline Parallelism (2-way)
dot = graphviz.Digraph('baseline_dense_transformer', comment='Dense Transformer Baseline (TP=8, PP=2)')
dot.attr(rankdir='TB', size='20,30')

# Global attributes
dot.attr('node', fontname='monospace', fontsize='10')
dot.attr('edge', fontname='monospace', fontsize='8')

# Input node
dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]', 
         shape='ellipse', style='filled', fillcolor='lightblue')

# Layer 0 (Devices 0-7)
with dot.subgraph(name='cluster_layer0') as c:
    c.attr(label='Layer 0 (Devices 0-7)', style='rounded', bgcolor='lightgray')
    
    # LayerNorm 0
    c.node('ln0', 'LayerNorm\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: 0-7', 
           shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # QKV Projection (Column Parallel)
    for i in range(8):
        c.node(f'qkv_proj_{i}', f'QKV Projection\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 3072]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # All-Gather for QKV
    for i in range(8):
        c.node(f'qkv_gather_{i}', f'All-Gather\\nInput: [1024, 10000, 3072]\\nOutput: [1024, 10000, 24576]\\nGPU: {i}', 
               shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Multi-Head Attention
    for i in range(8):
        c.node(f'mha_{i}', f'Multi-Head Attention\\nQ: [1024, 10000, 24576/8]\\nK: [1024, 10000, 24576/8]\\nV: [1024, 10000, 24576/8]\\nOutput: [1024, 10000, 8192]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightpink')
    
    # Output Projection (Row Parallel)
    for i in range(8):
        c.node(f'out_proj_{i}', f'Output Projection\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 1024]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # All-Reduce for Output
    c.node('out_reduce', 'All-Reduce Sum\\nInput: [1024, 10000, 1024] x8\\nOutput: [1024, 10000, 8192]\\nGPU: 0-7', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Residual Add 0
    c.node('res0', 'Residual Add\\nInput1: [1024, 10000, 8192]\\nInput2: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: 0-7', 
           shape='ellipse', style='filled', fillcolor='lightblue')
    
    # LayerNorm 1
    c.node('ln1', 'LayerNorm\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: 0-7', 
           shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # MLP Gate/Up Projection (Column Parallel)
    for i in range(8):
        c.node(f'mlp_gate_up_{i}', f'MLP Gate/Up\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # All-Gather for MLP intermediate
    c.node('mlp_gather', 'All-Gather\\nInput: [1024, 10000, 8192] x8\\nOutput: [1024, 10000, 65536]\\nGPU: 0-7', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # MLP Down Projection (Row Parallel)
    for i in range(8):
        c.node(f'mlp_down_{i}', f'MLP Down\\nInput: [1024, 10000, 4096]\\nOutput: [1024, 10000, 1024]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # All-Reduce for MLP
    c.node('mlp_reduce', 'All-Reduce Sum\\nInput: [1024, 10000, 1024] x8\\nOutput: [1024, 10000, 8192]\\nGPU: 0-7', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Residual Add 1
    c.node('res1', 'Residual Add\\nInput1: [1024, 10000, 8192]\\nInput2: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: 0-7', 
           shape='ellipse', style='filled', fillcolor='lightblue')

# Pipeline communication from Layer 0 to Layer 1
for i in range(8):
    dot.node(f'pipeline_send_{i}', f'Pipeline Send\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {i} -> {i+8}', 
             shape='parallelogram', style='filled', fillcolor='orange')

# Layer 1 (Devices 8-15)
with dot.subgraph(name='cluster_layer1') as c:
    c.attr(label='Layer 1 (Devices 8-15)', style='rounded', bgcolor='lightgray')
    
    # Similar structure as Layer 0 but on devices 8-15
    c.node('ln2', 'LayerNorm\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: 8-15', 
           shape='rectangle', style='filled', fillcolor='lightyellow')
    
    for i in range(8, 16):
        c.node(f'qkv_proj2_{i}', f'QKV Projection\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 3072]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
    
    for i in range(8, 16):
        c.node(f'qkv_gather2_{i}', f'All-Gather\\nInput: [1024, 10000, 3072]\\nOutput: [1024, 10000, 24576]\\nGPU: {i}', 
               shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    for i in range(8, 16):
        c.node(f'mha2_{i}', f'Multi-Head Attention\\nOutput: [1024, 10000, 8192]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightpink')
    
    for i in range(8, 16):
        c.node(f'out_proj2_{i}', f'Output Projection\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 1024]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
    
    c.node('out_reduce2', 'All-Reduce Sum\\nOutput: [1024, 10000, 8192]\\nGPU: 8-15', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    c.node('res2', 'Residual Add\\nOutput: [1024, 10000, 8192]\\nGPU: 8-15', 
           shape='ellipse', style='filled', fillcolor='lightblue')
    
    c.node('ln3', 'LayerNorm\\nOutput: [1024, 10000, 8192]\\nGPU: 8-15', 
           shape='rectangle', style='filled', fillcolor='lightyellow')
    
    for i in range(8, 16):
        c.node(f'mlp_gate_up2_{i}', f'MLP Gate/Up\\nOutput: [1024, 10000, 8192]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
    
    c.node('mlp_gather2', 'All-Gather\\nOutput: [1024, 10000, 65536]\\nGPU: 8-15', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    for i in range(8, 16):
        c.node(f'mlp_down2_{i}', f'MLP Down\\nOutput: [1024, 10000, 1024]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
    
    c.node('mlp_reduce2', 'All-Reduce Sum\\nOutput: [1024, 10000, 8192]\\nGPU: 8-15', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    c.node('res3', 'Residual Add\\nOutput: [1024, 10000, 8192]\\nGPU: 8-15', 
           shape='ellipse', style='filled', fillcolor='lightblue')

# Pipeline communication from Layer 1 to Layer 2
for i in range(8, 16):
    dot.node(f'pipeline_send2_{i}', f'Pipeline Send\\nGPU: {i} -> {i-8}', 
             shape='parallelogram', style='filled', fillcolor='orange')

# Layer 2 (Devices 0-7) - Similar to Layer 0
with dot.subgraph(name='cluster_layer2') as c:
    c.attr(label='Layer 2 (Devices 0-7)', style='rounded', bgcolor='lightgray')
    # Structure identical to layer 0
    c.node('ln4', 'LayerNorm\\nGPU: 0-7', shape='rectangle', style='filled', fillcolor='lightyellow')
    for i in range(8):
        c.node(f'qkv_proj4_{i}', f'QKV Projection\\nGPU: {i}', shape='rectangle', style='filled', fillcolor='lightcoral')
    c.node('out_reduce4', 'All-Reduce Sum\\nGPU: 0-7', shape='parallelogram', style='filled', fillcolor='lightgreen')
    c.node('res4', 'Residual Add\\nGPU: 0-7', shape='ellipse', style='filled', fillcolor='lightblue')
    c.node('ln5', 'LayerNorm\\nGPU: 0-7', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('mlp_reduce4', 'All-Reduce Sum\\nGPU: 0-7', shape='parallelogram', style='filled', fillcolor='lightgreen')
    c.node('res5', 'Residual Add\\nGPU: 0-7', shape='ellipse', style='filled', fillcolor='lightblue')

# Pipeline communication from Layer 2 to Layer 3
for i in range(8):
    dot.node(f'pipeline_send3_{i}', f'Pipeline Send\\nGPU: {i} -> {i+8}', 
             shape='parallelogram', style='filled', fillcolor='orange')

# Layer 3 (Devices 8-15) - Similar to Layer 1
with dot.subgraph(name='cluster_layer3') as c:
    c.attr(label='Layer 3 (Devices 8-15)', style='rounded', bgcolor='lightgray')
    c.node('ln6', 'LayerNorm\\nGPU: 8-15', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('out_reduce6', 'All-Reduce Sum\\nGPU: 8-15', shape='parallelogram', style='filled', fillcolor='lightgreen')
    c.node('res6', 'Residual Add\\nGPU: 8-15', shape='ellipse', style='filled', fillcolor='lightblue')
    c.node('ln7', 'LayerNorm\\nGPU: 8-15', shape='rectangle', style='filled', fillcolor='lightyellow')
    c.node('mlp_reduce6', 'All-Reduce Sum\\nGPU: 8-15', shape='parallelogram', style='filled', fillcolor='lightgreen')
    c.node('res7', 'Residual Add\\nGPU: 8-15', shape='ellipse', style='filled', fillcolor='lightblue')

# Output node
dot.node('output', 'Output\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]', 
         shape='ellipse', style='filled', fillcolor='lightblue')

# Connect the DAG
# Input to Layer 0
dot.edge('input', 'ln0')
for i in range(8):
    dot.edge('ln0', f'qkv_proj_{i}')
    dot.edge(f'qkv_proj_{i}', f'qkv_gather_{i}')
    dot.edge(f'qkv_gather_{i}', f'mha_{i}')
    dot.edge(f'mha_{i}', f'out_proj_{i}')
    dot.edge(f'out_proj_{i}', 'out_reduce')
    dot.edge('out_reduce', 'res0')
    dot.edge('res0', 'ln1')
    dot.edge('ln1', f'mlp_gate_up_{i}')
    dot.edge(f'mlp_gate_up_{i}', 'mlp_gather')
    dot.edge('mlp_gather', f'mlp_down_{i}')
    dot.edge(f'mlp_down_{i}', 'mlp_reduce')
    dot.edge('mlp_reduce', 'res1')
    dot.edge('res1', f'pipeline_send_{i}')

# Pipeline communication
for i in range(8):
    dot.edge(f'pipeline_send_{i}', f'qkv_proj2_{i+8}')

# Layer 1 connections (simplified)
for i in range(8, 16):
    dot.edge(f'qkv_proj2_{i}', f'qkv_gather2_{i}')
    dot.edge(f'qkv_gather2_{i}', f'mha2_{i}')
    dot.edge(f'mha2_{i}', f'out_proj2_{i}')
    dot.edge(f'out_proj2_{i}', 'out_reduce2')
    dot.edge('out_reduce2', 'res2')
    dot.edge('res2', 'ln3')
    dot.edge('ln3', f'mlp_gate_up2_{i}')
    dot.edge(f'mlp_gate_up2_{i}', 'mlp_gather2')
    dot.edge('mlp_gather2', f'mlp_down2_{i}')
    dot.edge(f'mlp_down2_{i}', 'mlp_reduce2')
    dot.edge('mlp_reduce2', 'res3')
    dot.edge('res3', f'pipeline_send2_{i}')

# Continue for layers 2 and 3 (simplified connections)
for i in range(8, 16):
    dot.edge(f'pipeline_send2_{i}', f'qkv_proj4_{i-8}')

for i in range(8):
    dot.edge(f'qkv_proj4_{i}', 'out_reduce4')
    dot.edge('out_reduce4', 'res4')
    dot.edge('res4', 'ln5')
    dot.edge('ln5', 'mlp_reduce4')
    dot.edge('mlp_reduce4', 'res5')
    dot.edge('res5', f'pipeline_send3_{i}')
    dot.edge(f'pipeline_send3_{i}', f'qkv_proj2_{i+8}')

for i in range(8, 16):
    dot.edge(f'qkv_proj2_{i}', 'out_reduce6')
    dot.edge('out_reduce6', 'res6')
    dot.edge('res6', 'ln7')
    dot.edge('ln7', 'mlp_reduce6')
    dot.edge('mlp_reduce6', 'res7')
    dot.edge('res7', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/2025-09-15-11-59-28/baseline_dense_transformer', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-15-11-59-28/baseline_dense_transformer.dot')

print("Baseline DAG generated successfully!")
print("Files saved:")
print("- /home/wzc/data/file-share/2025-09-15-11-59-28/baseline_dense_transformer.svg")
print("- /home/wzc/data/file-share/2025-09-15-11-59-28/baseline_dense_transformer.dot")