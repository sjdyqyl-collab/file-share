import graphviz

# Create Ring Attention with Sequence Parallelism DAG
dot = graphviz.Digraph('ring_attention_sequence_parallel', 
                      comment='Ring Attention with Sequence Parallelism (16 devices)')
dot.attr(rankdir='TB', size='30,40')

# Global attributes
dot.attr('node', fontname='monospace', fontsize='9')
dot.attr('edge', fontname='monospace', fontsize='7')

# Input node - sequence split across 16 devices
dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\\nSplit into 16 chunks of 625 tokens', 
         shape='ellipse', style='filled', fillcolor='lightblue')

# Sequence partition nodes for each device
for device_id in range(16):
    dot.node(f'seq_split_{device_id}', 
             f'Sequence Partition\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 625, 8192]\\nGPU: {device_id}', 
             shape='parallelogram', style='filled', fillcolor='lightgreen')

# Process all 4 layers with ring attention
for layer in range(4):
    layer_name = f'layer_{layer}'
    
    with dot.subgraph(name=f'cluster_{layer_name}') as c:
        c.attr(label=f'Layer {layer} (All 16 GPUs)', style='rounded', bgcolor='lightgray')
        
        # LayerNorm for each device
        for device_id in range(16):
            c.node(f'ln_{layer}_{device_id}', 
                   f'LayerNorm\\nInput: [1024, 625, 8192]\\nOutput: [1024, 625, 8192]\\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        # QKV Projection (replicated across all devices)
        for device_id in range(16):
            c.node(f'qkv_proj_{layer}_{device_id}', 
                   f'QKV Projection\\nInput: [1024, 625, 8192]\\nOutput: [1024, 625, 24576]\\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Ring Attention computation - 16 stages
        for stage in range(16):
            for device_id in range(16):
                src_device = (device_id - stage) % 16
                c.node(f'ring_att_stage_{layer}_{stage}_{device_id}', 
                       f'Ring Attention Stage {stage}\\nQ: [1024, 625, 8192]\\nK: [1024, 625, 8192] (from GPU {src_device})\\nV: [1024, 625, 8192] (from GPU {src_device})\\nOutput: [1024, 625, 8192]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightpink')
        
        # K/V communication nodes for ring topology
        for device_id in range(16):
            next_device = (device_id + 1) % 16
            prev_device = (device_id - 1) % 16
            
            # Send K/V to next device
            c.node(f'kv_send_{layer}_{device_id}', 
                   f'Send K/V\\nInput: [1024, 625, 8192]\\nOutput: [1024, 625, 8192]\\nGPU: {device_id} -> {next_device}', 
                   shape='parallelogram', style='filled', fillcolor='orange')
            
            # Receive K/V from previous device
            c.node(f'kv_recv_{layer}_{device_id}', 
                   f'Recv K/V\\nInput: [1024, 625, 8192]\\nOutput: [1024, 625, 8192]\\nGPU: {prev_device} -> {device_id}', 
                   shape='parallelogram', style='filled', fillcolor='lightgreen')
        
        # Accumulate partial attention results
        for device_id in range(16):
            c.node(f'att_accum_{layer}_{device_id}', 
                   f'Accumulate Attention\\nInput: [1024, 625, 8192] x16\\nOutput: [1024, 625, 8192]\\nGPU: {device_id}', 
                   shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Output Projection (replicated)
        for device_id in range(16):
            c.node(f'out_proj_{layer}_{device_id}', 
                   f'Output Projection\\nInput: [1024, 625, 8192]\\nOutput: [1024, 625, 8192]\\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Residual Add 1
        for device_id in range(16):
            c.node(f'res1_{layer}_{device_id}', 
                   f'Residual Add\\nInput1: [1024, 625, 8192]\\nInput2: [1024, 625, 8192]\\nOutput: [1024, 625, 8192]\\nGPU: {device_id}', 
                   shape='ellipse', style='filled', fillcolor='lightblue')
        
        # LayerNorm 2
        for device_id in range(16):
            c.node(f'ln2_{layer}_{device_id}', 
                   f'LayerNorm\\nInput: [1024, 625, 8192]\\nOutput: [1024, 625, 8192]\\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        # MLP Gate/Up Projection
        for device_id in range(16):
            c.node(f'mlp_gate_up_{layer}_{device_id}', 
                   f'MLP Gate/Up\\nInput: [1024, 625, 8192]\\nOutput: [1024, 625, 65536]\\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # MLP Down Projection
        for device_id in range(16):
            c.node(f'mlp_down_{layer}_{device_id}', 
                   f'MLP Down\\nInput: [1024, 625, 65536]\\nOutput: [1024, 625, 8192]\\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Residual Add 2
        for device_id in range(16):
            c.node(f'res2_{layer}_{device_id}', 
                   f'Residual Add\\nInput1: [1024, 625, 8192]\\nInput2: [1024, 625, 8192]\\nOutput: [1024, 625, 8192]\\nGPU: {device_id}', 
                   shape='ellipse', style='filled', fillcolor='lightblue')

# Output aggregation - gather sequence chunks back together
dot.node('output_gather', 
         'Gather Sequence Chunks\\nInput: [1024, 625, 8192] x16\\nOutput: [1024, 10000, 8192]', 
         shape='parallelogram', style='filled', fillcolor='lightgreen')

# Output node
dot.node('output', 'Output\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]', 
         shape='ellipse', style='filled', fillcolor='lightblue')

# Connect the DAG
# Input to sequence partition
for device_id in range(16):
    dot.edge('input', f'seq_split_{device_id}')

# Layer 0 connections
for device_id in range(16):
    # Input to LayerNorm
    dot.edge(f'seq_split_{device_id}', f'ln_0_{device_id}')
    
    # LayerNorm to QKV Projection
    dot.edge(f'ln_0_{device_id}', f'qkv_proj_0_{device_id}')
    
    # Ring attention stages
    for stage in range(16):
        if stage == 0:
            dot.edge(f'qkv_proj_0_{device_id}', f'ring_att_stage_0_0_{device_id}')
        else:
            # Connect from previous stage
            prev_device = (device_id - 1) % 16
            dot.edge(f'kv_recv_0_{device_id}', f'ring_att_stage_0_{stage}_{device_id}')
            dot.edge(f'ring_att_stage_0_{stage-1}_{device_id}', f'ring_att_stage_0_{stage}_{device_id}')
        
        # K/V communication
        if stage < 15:  # Not the last stage
            dot.edge(f'ring_att_stage_0_{stage}_{device_id}', f'kv_send_0_{device_id}')
            next_device = (device_id + 1) % 16
            dot.edge(f'kv_send_0_{device_id}', f'kv_recv_0_{next_device}')
    
    # Accumulate results
    for stage in range(16):
        dot.edge(f'ring_att_stage_0_15_{device_id}', f'att_accum_0_{device_id}')
    
    # Output projection and residual
    dot.edge(f'att_accum_0_{device_id}', f'out_proj_0_{device_id}')
    dot.edge(f'out_proj_0_{device_id}', f'res1_0_{device_id}')
    dot.edge(f'seq_split_{device_id}', f'res1_0_{device_id}')  # Residual connection
    
    # MLP path
    dot.edge(f'res1_0_{device_id}', f'ln2_0_{device_id}')
    dot.edge(f'ln2_0_{device_id}', f'mlp_gate_up_0_{device_id}')
    dot.edge(f'mlp_gate_up_0_{device_id}', f'mlp_down_0_{device_id}')
    dot.edge(f'mlp_down_0_{device_id}', f'res2_0_{device_id}')
    dot.edge(f'res1_0_{device_id}', f'res2_0_{device_id}')  # Residual connection

# Continue for layers 1, 2, 3 (simplified connections)
for layer in range(1, 4):
    for device_id in range(16):
        prev_layer = layer - 1
        dot.edge(f'res2_{prev_layer}_{device_id}', f'ln_{layer}_{device_id}')
        dot.edge(f'ln_{layer}_{device_id}', f'qkv_proj_{layer}_{device_id}')
        
        # Ring attention stages
        for stage in range(16):
            if stage == 0:
                dot.edge(f'qkv_proj_{layer}_{device_id}', f'ring_att_stage_{layer}_0_{device_id}')
            else:
                dot.edge(f'kv_recv_{layer}_{device_id}', f'ring_att_stage_{layer}_{stage}_{device_id}')
                dot.edge(f'ring_att_stage_{layer}_{stage-1}_{device_id}', f'ring_att_stage_{layer}_{stage}_{device_id}')
            
            if stage < 15:
                dot.edge(f'ring_att_stage_{layer}_{stage}_{device_id}', f'kv_send_{layer}_{device_id}')
                next_device = (device_id + 1) % 16
                dot.edge(f'kv_send_{layer}_{device_id}', f'kv_recv_{layer}_{next_device}')
        
        for stage in range(16):
            dot.edge(f'ring_att_stage_{layer}_15_{device_id}', f'att_accum_{layer}_{device_id}')
        
        dot.edge(f'att_accum_{layer}_{device_id}', f'out_proj_{layer}_{device_id}')
        dot.edge(f'out_proj_{layer}_{device_id}', f'res1_{layer}_{device_id}')
        dot.edge(f'res2_{prev_layer}_{device_id}', f'res1_{layer}_{device_id}')
        
        dot.edge(f'res1_{layer}_{device_id}', f'ln2_{layer}_{device_id}')
        dot.edge(f'ln2_{layer}_{device_id}', f'mlp_gate_up_{layer}_{device_id}')
        dot.edge(f'mlp_gate_up_{layer}_{device_id}', f'mlp_down_{layer}_{device_id}')
        dot.edge(f'mlp_down_{layer}_{device_id}', f'res2_{layer}_{device_id}')
        dot.edge(f'res1_{layer}_{device_id}', f'res2_{layer}_{device_id}')

# Final output connections
for device_id in range(16):
    dot.edge(f'res2_3_{device_id}', 'output_gather')
dot.edge('output_gather', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/2025-09-15-11-59-28/ring_attention_sequence_parallel', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-15-11-59-28/ring_attention_sequence_parallel.dot')

print("Ring Attention Sequence Parallel DAG generated successfully!")
print("Files saved:")
print("- /home/wzc/data/file-share/2025-09-15-11-59-28/ring_attention_sequence_parallel.svg")
print("- /home/wzc/data/file-share/2025-09-15-11-59-28/ring_attention_sequence_parallel.dot")