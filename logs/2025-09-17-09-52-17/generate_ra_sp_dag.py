#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_ra_sp_dag():
    """
    Create DAG for Ring Attention + Sequence Parallelism model
    16 GPUs total, each handling 625 tokens of the 10000 token sequence
    """
    
    dot = Digraph(comment='RA+SP Transformer DAG - Ring Attention + Sequence Parallelism')
    dot.attr(rankdir='TB', size='25,25')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node - sequence split across 16 devices
    dot.node('input', 'Input Sequence\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: all GPUs', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Sequence split node
    dot.node('sequence_split', 'Sequence Split\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=625, d_model=8192]\nGPU: 16 devices', 
             shape='diamond', fillcolor='orange')
    
    # Create 4 layers with ring attention
    for layer_num in range(4):
        create_ra_sp_layer(dot, layer_num)
    
    # Sequence gather node
    dot.node('sequence_gather', 'Sequence Gather\nInput: [batch_size=1024, seq_len=625, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 16 devices', 
             shape='diamond', fillcolor='orange')
    
    # Output node
    dot.node('output', 'Output\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: all GPUs', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('input', 'sequence_split')
    dot.edge('sequence_split', 'layer0_input_0')
    
    # Connect layers
    for layer_num in range(3):
        for device_id in range(16):
            dot.edge(f'layer{layer_num}_output_{device_id}', f'layer{layer_num+1}_input_{device_id}')
    
    # Connect final layer to gather
    for device_id in range(16):
        dot.edge(f'layer3_output_{device_id}', 'sequence_gather')
    
    dot.edge('sequence_gather', 'output')
    
    return dot

def create_ra_sp_layer(graph, layer_num):
    """Create nodes for a single RA+SP transformer layer"""
    
    # Create subgraph for this layer
    with graph.subgraph(name=f'cluster_layer{layer_num}') as layer:
        layer.attr(label=f'Layer {layer_num} - Ring Attention + Sequence Parallelism', style='rounded')
        
        # For each device (16 total)
        for device_id in range(16):
            device_cluster = f'cluster_layer{layer_num}_device{device_id}'
            with layer.subgraph(name=device_cluster) as device:
                device.attr(label=f'Device {device_id} - Sequence Chunk {device_id}', style='dashed')
                
                # Input for this device
                device.node(f'layer{layer_num}_input_{device_id}', 
                           f'Layer {layer_num} Input\nInput: [batch_size=1024, seq_len=625, d_model=8192]\nGPU: {device_id}')
                
                # Layer Norm 1
                device.node(f'layer{layer_num}_ln1_{device_id}', 
                           f'Layer {layer_num} LN1\nInput: [batch_size=1024, seq_len=625, d_model=8192]\nOutput: [batch_size=1024, seq_len=625, d_model=8192]\nGPU: {device_id}')
                
                # QKV Projection
                device.node(f'layer{layer_num}_qkv_{device_id}', 
                           f'Layer {layer_num} QKV Projection\nInput: [batch_size=1024, seq_len=625, d_model=8192]\nOutput: [batch_size=1024, seq_len=625, qkv_dim=24576]\nGPU: {device_id}')
                
                # Split into heads
                device.node(f'layer{layer_num}_split_heads_{device_id}', 
                           f'Layer {layer_num} Split Heads\nInput: [batch_size=1024, seq_len=625, qkv_dim=24576]\nOutput: Q: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nK: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nV: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nGPU: {device_id}', 
                           shape='diamond', fillcolor='lightyellow')
                
                # Ring attention stages
                create_ring_attention_nodes(device, layer_num, device_id)
                
                # Concatenate heads
                device.node(f'layer{layer_num}_concat_heads_{device_id}', 
                           f'Layer {layer_num} Concat Heads\nInput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=625, d_model=8192]\nGPU: {device_id}', 
                           shape='diamond', fillcolor='lightyellow')
                
                # Attention output projection
                device.node(f'layer{layer_num}_attn_out_{device_id}', 
                           f'Layer {layer_num} Attention Output\nInput: [batch_size=1024, seq_len=625, d_model=8192]\nOutput: [batch_size=1024, seq_len=625, d_model=8192]\nGPU: {device_id}')
                
                # Residual connection 1
                device.node(f'layer{layer_num}_residual1_{device_id}', 
                           f'Layer {layer_num} Residual Add\nInput: [batch_size=1024, seq_len=625, d_model=8192]\nOutput: [batch_size=1024, seq_len=625, d_model=8192]\nGPU: {device_id}', 
                           shape='ellipse', fillcolor='lightcoral')
                
                # Layer Norm 2
                device.node(f'layer{layer_num}_ln2_{device_id}', 
                           f'Layer {layer_num} LN2\nInput: [batch_size=1024, seq_len=625, d_model=8192]\nOutput: [batch_size=1024, seq_len=625, d_model=8192]\nGPU: {device_id}')
                
                # MLP First Linear
                device.node(f'layer{layer_num}_mlp1_{device_id}', 
                           f'Layer {layer_num} MLP First Linear\nInput: [batch_size=1024, seq_len=625, d_model=8192]\nOutput: [batch_size=1024, seq_len=625, ffn_hidden=32768]\nGPU: {device_id}')
                
                # GELU activation
                device.node(f'layer{layer_num}_gelu_{device_id}', 
                           f'Layer {layer_num} GELU\nInput: [batch_size=1024, seq_len=625, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=625, ffn_hidden=32768]\nGPU: {device_id}')
                
                # MLP Second Linear
                device.node(f'layer{layer_num}_mlp2_{device_id}', 
                           f'Layer {layer_num} MLP Second Linear\nInput: [batch_size=1024, seq_len=625, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=625, d_model=8192]\nGPU: {device_id}')
                
                # Residual connection 2
                device.node(f'layer{layer_num}_residual2_{device_id}', 
                           f'Layer {layer_num} Final Residual\nInput: [batch_size=1024, seq_len=625, d_model=8192]\nOutput: [batch_size=1024, seq_len=625, d_model=8192]\nGPU: {device_id}', 
                           shape='ellipse', fillcolor='lightcoral')
                
                # Connect device nodes
                device.edge(f'layer{layer_num}_input_{device_id}', f'layer{layer_num}_ln1_{device_id}')
                device.edge(f'layer{layer_num}_ln1_{device_id}', f'layer{layer_num}_qkv_{device_id}')
                device.edge(f'layer{layer_num}_qkv_{device_id}', f'layer{layer_num}_split_heads_{device_id}')
                device.edge(f'layer{layer_num}_split_heads_{device_id}', f'layer{layer_num}_ring_stage0_{device_id}')
                device.edge(f'layer{layer_num}_ring_final_{device_id}', f'layer{layer_num}_concat_heads_{device_id}')
                device.edge(f'layer{layer_num}_concat_heads_{device_id}', f'layer{layer_num}_attn_out_{device_id}')
                device.edge(f'layer{layer_num}_attn_out_{device_id}', f'layer{layer_num}_residual1_{device_id}')
                device.edge(f'layer{layer_num}_residual1_{device_id}', f'layer{layer_num}_ln2_{device_id}')
                device.edge(f'layer{layer_num}_ln2_{device_id}', f'layer{layer_num}_mlp1_{device_id}')
                device.edge(f'layer{layer_num}_mlp1_{device_id}', f'layer{layer_num}_gelu_{device_id}')
                device.edge(f'layer{layer_num}_gelu_{device_id}', f'layer{layer_num}_mlp2_{device_id}')
                device.edge(f'layer{layer_num}_mlp2_{device_id}', f'layer{layer_num}_residual2_{device_id}')
                
                # Residual connections
                device.edge(f'layer{layer_num}_input_{device_id}', f'layer{layer_num}_residual1_{device_id}')
                device.edge(f'layer{layer_num}_residual1_{device_id}', f'layer{layer_num}_residual2_{device_id}')

def create_ring_attention_nodes(graph, layer_num, device_id):
    """Create ring attention computation nodes for a single device"""
    
    # Ring attention stages (16 stages total)
    for stage in range(16):
        next_device = (device_id + 1) % 16
        prev_device = (device_id - 1) % 16
        
        if stage == 0:
            # Initial stage - use local KV
            graph.node(f'layer{layer_num}_ring_stage0_{device_id}', 
                      f'Ring Stage 0\nCompute: Q_local × K_local × V_local\nInput Q: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nInput KV: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nGPU: {device_id}')
            
            # Communication node
            graph.node(f'layer{layer_num}_comm_0_{device_id}', 
                      f'Send KV to Device {next_device}\nInput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nGPU: {device_id} → {next_device}', 
                      shape='diamond', fillcolor='yellow', style='dashed')
            
            graph.edge(f'layer{layer_num}_ring_stage0_{device_id}', f'layer{layer_num}_comm_0_{device_id}')
            
        elif stage == 15:
            # Final stage
            graph.node(f'layer{layer_num}_ring_stage{stage}_{device_id}', 
                      f'Ring Stage {stage}\nCompute: Q_local × K_ring × V_ring\nInput Q: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nInput KV: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nGPU: {device_id}')
            
            # Accumulate all partial results
            graph.node(f'layer{layer_num}_ring_final_{device_id}', 
                      f'Attention Accumulation\nInput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nGPU: {device_id}')
            
            graph.edge(f'layer{layer_num}_ring_stage{stage}_{device_id}', f'layer{layer_num}_ring_final_{device_id}')
            
        else:
            # Intermediate stages
            graph.node(f'layer{layer_num}_ring_stage{stage}_{device_id}', 
                      f'Ring Stage {stage}\nCompute: Q_local × K_ring × V_ring\nInput Q: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nInput KV: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nGPU: {device_id}')
            
            # Communication node
            graph.node(f'layer{layer_num}_comm_{stage}_{device_id}', 
                      f'Send KV to Device {next_device}\nInput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=625, heads=16, d_k=512]\nGPU: {device_id} → {next_device}', 
                      shape='diamond', fillcolor='yellow', style='dashed')
            
            graph.edge(f'layer{layer_num}_ring_stage{stage}_{device_id}', f'layer{layer_num}_comm_{stage}_{device_id}')
        
        # Connect stages
        if stage > 0:
            graph.edge(f'layer{layer_num}_comm_{stage-1}_{prev_device}', f'layer{layer_num}_ring_stage{stage}_{device_id}')
        
        # Accumulate partial results
        if stage > 0:
            graph.edge(f'layer{layer_num}_ring_stage{stage}_{device_id}', f'layer{layer_num}_ring_final_{device_id}')

if __name__ == '__main__':
    dag = create_ra_sp_dag()
    
    # Save DOT file
    with open('/home/wzc/data/file-share/2025-09-17-09-52-17/ra_sp_dag.dot', 'w') as f:
        f.write(dag.source)
    
    # Render to SVG
    dag.render('/home/wzc/data/file-share/2025-09-17-09-52-17/ra_sp_dag', format='svg', cleanup=False)
    print("RA+SP DAG generated successfully")