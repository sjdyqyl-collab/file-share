#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_baseline_dag():
    """
    Create DAG for baseline model using Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
    16 GPUs total: 8 GPUs per pipeline stage
    """
    
    dot = Digraph(comment='Baseline Transformer DAG - TP=8, PP=2')
    dot.attr(rankdir='TB', size='20,20')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: all GPUs', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Pipeline Stage 0 (Layers 0-1) - GPUs 0-7
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (Layers 0-1)\nGPUs 0-7', style='rounded')
        
        # Layer 0
        create_layer_nodes(stage0, 0, 'stage0')
        
        # Layer 1
        create_layer_nodes(stage0, 1, 'stage0')
    
    # Pipeline communication between stages
    dot.node('pipeline_comm', 'Pipeline Communication\nSend activations from GPUs 0-7 to GPUs 8-15\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: cross-stage', 
             shape='diamond', fillcolor='yellow')
    
    # Pipeline Stage 1 (Layers 2-3) - GPUs 8-15
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (Layers 2-3)\nGPUs 8-15', style='rounded')
        
        # Layer 2
        create_layer_nodes(stage1, 2, 'stage1')
        
        # Layer 3
        create_layer_nodes(stage1, 3, 'stage1')
    
    # Output node
    dot.node('output', 'Output\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: all GPUs', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('input', 'layer0_input_split')
    dot.edge('layer0_output', 'layer1_input')
    dot.edge('layer1_output', 'pipeline_comm')
    dot.edge('pipeline_comm', 'layer2_input_split')
    dot.edge('layer2_output', 'layer3_input')
    dot.edge('layer3_output', 'output')
    
    return dot

def create_layer_nodes(graph, layer_num, stage_name):
    """Create nodes for a single transformer layer"""
    
    # Input distribution for tensor parallelism
    graph.node(f'layer{layer_num}_input_split', 
               f'Layer {layer_num} Input Split\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=1024]\nGPU: 8 devices', 
               shape='diamond', fillcolor='orange')
    
    # Layer Norm (replicated across TP devices)
    for i in range(8):
        device_id = 0 + i if stage_name == 'stage0' else 8 + i
        graph.node(f'layer{layer_num}_ln1_{i}', 
                   f'Layer {layer_num} LN1\nInput: [batch_size=1024, seq_len=10000, d_model=1024]\nOutput: [batch_size=1024, seq_len=10000, d_model=1024]\nGPU: {device_id}')
    
    # QKV Projection (column parallel)
    for i in range(8):
        device_id = 0 + i if stage_name == 'stage0' else 8 + i
        graph.node(f'layer{layer_num}_qkv_{i}', 
                   f'Layer {layer_num} QKV Projection\nInput: [batch_size=1024, seq_len=10000, d_model=1024]\nOutput: [batch_size=1024, seq_len=10000, qkv_dim=3072]\nGPU: {device_id}')
    
    # QKV split into heads
    graph.node(f'layer{layer_num}_qkv_split', 
               f'Layer {layer_num} QKV Split\nInput: [batch_size=1024, seq_len=10000, qkv_dim=24576]\nOutput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]\nGPU: 8 devices', 
               shape='diamond', fillcolor='orange')
    
    # Multi-head attention computation
    for i in range(8):
        device_id = 0 + i if stage_name == 'stage0' else 8 + i
        graph.node(f'layer{layer_num}_mha_{i}', 
                   f'Layer {layer_num} MHA\nInput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]\nGPU: {device_id}')
    
    # Attention output projection (row parallel)
    for i in range(8):
        device_id = 0 + i if stage_name == 'stage0' else 8 + i
        graph.node(f'layer{layer_num}_attn_out_{i}', 
                   f'Layer {layer_num} Attention Output\nInput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, d_model=1024]\nGPU: {device_id}')
    
    # All-reduce for attention output
    graph.node(f'layer{layer_num}_attn_allreduce', 
               f'Layer {layer_num} Attention All-Reduce\nInput: [batch_size=1024, seq_len=10000, d_model=1024]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8 devices', 
               shape='diamond', fillcolor='yellow')
    
    # Residual connection
    graph.node(f'layer{layer_num}_residual1', 
               f'Layer {layer_num} Residual Add\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: all GPUs', 
               shape='ellipse', fillcolor='lightcoral')
    
    # Layer Norm for MLP (replicated)
    for i in range(8):
        device_id = 0 + i if stage_name == 'stage0' else 8 + i
        graph.node(f'layer{layer_num}_ln2_{i}', 
                   f'Layer {layer_num} LN2\nInput: [batch_size=1024, seq_len=10000, d_model=1024]\nOutput: [batch_size=1024, seq_len=10000, d_model=1024]\nGPU: {device_id}')
    
    # MLP first linear (column parallel)
    for i in range(8):
        device_id = 0 + i if stage_name == 'stage0' else 8 + i
        graph.node(f'layer{layer_num}_mlp1_{i}', 
                   f'Layer {layer_num} MLP First Linear\nInput: [batch_size=1024, seq_len=10000, d_model=1024]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=4096]\nGPU: {device_id}')
    
    # MLP activation
    for i in range(8):
        device_id = 0 + i if stage_name == 'stage0' else 8 + i
        graph.node(f'layer{layer_num}_gelu_{i}', 
                   f'Layer {layer_num} GELU\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=4096]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=4096]\nGPU: {device_id}')
    
    # MLP second linear (row parallel)
    for i in range(8):
        device_id = 0 + i if stage_name == 'stage0' else 8 + i
        graph.node(f'layer{layer_num}_mlp2_{i}', 
                   f'Layer {layer_num} MLP Second Linear\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=4096]\nOutput: [batch_size=1024, seq_len=10000, d_model=1024]\nGPU: {device_id}')
    
    # All-reduce for MLP output
    graph.node(f'layer{layer_num}_mlp_allreduce', 
               f'Layer {layer_num} MLP All-Reduce\nInput: [batch_size=1024, seq_len=10000, d_model=1024]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8 devices', 
               shape='diamond', fillcolor='yellow')
    
    # Final residual connection
    graph.node(f'layer{layer_num}_residual2', 
               f'Layer {layer_num} Final Residual\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: all GPUs', 
               shape='ellipse', fillcolor='lightcoral')
    
    # Connect layer nodes
    if layer_num == 0:
        prev_node = 'layer0_input_split'
    elif layer_num == 2:
        prev_node = 'pipeline_comm'
    else:
        prev_node = f'layer{layer_num-1}_residual2'
    
    for i in range(8):
        graph.edge(prev_node, f'layer{layer_num}_ln1_{i}')
        graph.edge(f'layer{layer_num}_ln1_{i}', f'layer{layer_num}_qkv_{i}')
        graph.edge(f'layer{layer_num}_qkv_{i}', f'layer{layer_num}_qkv_split')
        graph.edge(f'layer{layer_num}_qkv_split', f'layer{layer_num}_mha_{i}')
        graph.edge(f'layer{layer_num}_mha_{i}', f'layer{layer_num}_attn_out_{i}')
        graph.edge(f'layer{layer_num}_attn_out_{i}', f'layer{layer_num}_attn_allreduce')
        graph.edge(f'layer{layer_num}_attn_allreduce', f'layer{layer_num}_residual1')
        graph.edge(f'layer{layer_num}_residual1', f'layer{layer_num}_ln2_{i}')
        graph.edge(f'layer{layer_num}_ln2_{i}', f'layer{layer_num}_mlp1_{i}')
        graph.edge(f'layer{layer_num}_mlp1_{i}', f'layer{layer_num}_gelu_{i}')
        graph.edge(f'layer{layer_num}_gelu_{i}', f'layer{layer_num}_mlp2_{i}')
        graph.edge(f'layer{layer_num}_mlp2_{i}', f'layer{layer_num}_mlp_allreduce')
        graph.edge(f'layer{layer_num}_mlp_allreduce', f'layer{layer_num}_residual2')
    
    # Add edges for residual connections
    for i in range(8):
        graph.edge(prev_node, f'layer{layer_num}_residual1')
        graph.edge(f'layer{layer_num}_residual1', f'layer{layer_num}_residual2')

if __name__ == '__main__':
    dag = create_baseline_dag()
    
    # Save DOT file
    with open('/home/wzc/data/file-share/2025-09-17-09-52-17/baseline_dag.dot', 'w') as f:
        f.write(dag.source)
    
    # Render to SVG
    dag.render('/home/wzc/data/file-share/2025-09-17-09-52-17/baseline_dag', format='svg', cleanup=False)
    print("Baseline DAG generated successfully")