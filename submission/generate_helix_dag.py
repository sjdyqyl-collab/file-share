import graphviz

# Create Helix DAG for two-level partitioning across 16 devices
dot = graphviz.Digraph('helix_two_level_partitioning', comment='Helix Two-Level Attention Partitioning (m×n=16)')
dot.attr(rankdir='TB', size='30,30')

# Input node
dot.node('input', 'Input\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='lightblue')

# Global LayerNorm (all devices)
dot.node('global_norm', 'Global LayerNorm\n(B×L×8192)\nAll GPUs', shape='rectangle', style='filled', fillcolor='lightyellow')

# For each layer (0-3)
for layer in range(4):
    with dot.subgraph(name=f'cluster_layer{layer}') as c:
        c.attr(label=f'Layer {layer} - Two-Level Partitioning', style='dashed', color='darkgreen')
        
        # QKV Linear across all devices
        c.node(f'l{layer}_qkv', f'QKV Linear\n(B×L×8192→B×L×24576)\nAll GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # 16 partitions (4 head groups × 4 dimension slices)
        for head_group in range(4):
            for dim_slice in range(4):
                device_id = head_group * 4 + dim_slice
                
                # Partition-specific computation
                with c.subgraph(name=f'cluster_partition_{layer}_{head_group}_{dim_slice}') as pc:
                    pc.attr(label=f'Partition {head_group},{dim_slice} (GPU {device_id})', style='dotted', color='gray')
                    
                    # Local QKV extraction
                    pc.node(f'l{layer}_qkv_{head_group}_{dim_slice}', 
                           f'Extract QKV\n(B×L×1536)\nGPU {device_id}', 
                           shape='parallelogram', style='filled', fillcolor='lightcoral')
                    
                    # Local attention computation
                    pc.node(f'l{layer}_attention_{head_group}_{dim_slice}', 
                           f'Local Attention\n4 heads × 128 dims\n(B×L×512)\nGPU {device_id}', 
                           shape='rectangle', style='filled', fillcolor='lightcyan')
                    
                    # Store partition info
                    pc.node(f'l{layer}_output_{head_group}_{dim_slice}', 
                           f'Partition Output\n(B×L×512)\nGPU {device_id}', 
                           shape='ellipse', style='filled', fillcolor='lightpink')
        
        # Intra-group concatenation (4 devices per head group)
        for head_group in range(4):
            c.node(f'l{layer}_intra_concat_{head_group}', 
                   f'Intra-Group Concat\nHead Group {head_group}\nConcat 4×128 dims\n(B×L×2048)\nGPUs {head_group*4}-{head_group*4+3}', 
                   shape='parallelogram', style='filled', fillcolor='gold')
        
        # Inter-group concatenation (all 4 head groups)
        c.node(f'l{layer}_inter_concat', 
               f'Inter-Group Concat\nConcat 4×2048 dims\n(B×L×8192)\nAll GPUs', 
               shape='parallelogram', style='filled', fillcolor='gold')
        
        # Output projection
        c.node(f'l{layer}_proj', 
               f'Output Projection\n(B×L×8192→B×L×8192)\nAll GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Residual connection
        c.node(f'l{layer}_res1', 
               f'Residual Add\n(B×L×8192)\nAll GPUs', 
               shape='ellipse', style='filled', fillcolor='orange')
        
        # FFN components
        c.node(f'l{layer}_ffn_norm', 
               f'FFN LayerNorm\n(B×L×8192)\nAll GPUs', 
               shape='rectangle', style='filled', fillcolor='lightyellow')
        c.node(f'l{layer}_ffn_up', 
               f'FFN Up\n(B×L×8192→B×L×32768)\nAll GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node(f'l{layer}_ffn_down', 
               f'FFN Down\n(B×L×32768→B×L×8192)\nAll GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node(f'l{layer}_res2', 
               f'Residual Add\n(B×L×8192)\nAll GPUs', 
               shape='ellipse', style='filled', fillcolor='orange')

# Output node
dot.node('output', 'Output\n(B×L×8192)', shape='ellipse', style='filled', fillcolor='lightblue')

# Connections for all layers
for layer in range(4):
    if layer == 0:
        dot.edge('input', 'global_norm')
        prev_output = 'global_norm'
    else:
        prev_output = f'l{layer-1}_res2'
    
    # QKV computation
    dot.edge(prev_output, f'l{layer}_qkv')
    
    # Route to all partitions
    for head_group in range(4):
        for dim_slice in range(4):
            device_id = head_group * 4 + dim_slice
            dot.edge(f'l{layer}_qkv', f'l{layer}_qkv_{head_group}_{dim_slice}')
            dot.edge(f'l{layer}_qkv_{head_group}_{dim_slice}', f'l{layer}_attention_{head_group}_{dim_slice}')
            dot.edge(f'l{layer}_attention_{head_group}_{dim_slice}', f'l{layer}_output_{head_group}_{dim_slice}')
            
            # Intra-group concatenation
            dot.edge(f'l{layer}_output_{head_group}_{dim_slice}', f'l{layer}_intra_concat_{head_group}')
    
    # Inter-group concatenation
    for head_group in range(4):
        dot.edge(f'l{layer}_intra_concat_{head_group}', f'l{layer}_inter_concat')
    
    # Output projection and residual
    dot.edge(f'l{layer}_inter_concat', f'l{layer}_proj')
    dot.edge(f'l{layer}_proj', f'l{layer}_res1')
    if layer == 0:
        dot.edge('global_norm', f'l{layer}_res1')
    else:
        dot.edge(f'l{layer-1}_res2', f'l{layer}_res1')
    
    # FFN
    dot.edge(f'l{layer}_res1', f'l{layer}_ffn_norm')
    dot.edge(f'l{layer}_ffn_norm', f'l{layer}_ffn_up')
    dot.edge(f'l{layer}_ffn_up', f'l{layer}_ffn_down')
    dot.edge(f'l{layer}_ffn_down', f'l{layer}_res2')
    dot.edge(f'l{layer}_res1', f'l{layer}_res2')

# Final output
dot.edge('l3_res2', 'output')

# Save files
dot.save('/home/wzc/data/file-share/submission/helix_two_level_partitioning.dot')
dot.render('/home/wzc/data/file-share/submission/helix_two_level_partitioning', format='svg', cleanup=True)

print("Helix DAG generated successfully")