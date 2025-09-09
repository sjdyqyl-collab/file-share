#!/usr/bin/env python3

import graphviz

# Create proposed DAG for 64 GPUs with EP=64, 1 expert per GPU
dot = graphviz.Digraph('proposed_moe', comment='Proposed MoE with EP=64, 1 expert/GPU')
dot.attr(rankdir='TB', size='100,100')

# Define global parameters
batch_size = 1024
seq_len = 10000
token_dim = 8192
num_heads = 16
head_dim = 512
hidden_dim = 32768
num_layers = 4
experts_per_layer = 16
total_gpus = 64

# Color scheme for different layers
layer_colors = {
    0: 'lightblue',
    1: 'lightgreen', 
    2: 'lightyellow',
    3: 'lightcoral'
}

# Input node
dot.node('input', f'Total Input\\nShape: [{batch_size}, {seq_len}, {token_dim}]', 
         shape='ellipse', style='filled', fillcolor='lightgray')

# Process each layer
for layer in range(num_layers):
    color = layer_colors[layer]
    
    # Each layer has 16 experts distributed across 16 GPUs
    # With EP=64, we have 16 experts per layer × 4 layers = 64 GPUs total
    # So each expert gets its own GPU
    
    # First, we need to handle token routing across all GPUs
    # Create routing nodes for each GPU that will handle routing
    
    for expert_id in range(16):
        gpu_id = layer * 16 + expert_id
        
        # MHA for this expert's GPU (can be shared or per-expert)
        mha_name = f'mha_{layer}_{expert_id}'
        dot.node(mha_name, f'MHA\\nGPU:{gpu_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='rectangle', style='filled', fillcolor=color)
        
        # Residual after MHA
        residual1_name = f'residual1_{layer}_{expert_id}'
        dot.node(residual1_name, f'Residual Add\\nGPU:{gpu_id}\\nIn1: [{batch_size}, {seq_len}, {token_dim}]\\nIn2: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='parallelogram', style='filled', fillcolor=color)
        
        # LayerNorm after MHA
        ln1_name = f'ln1_{layer}_{expert_id}'
        dot.node(ln1_name, f'LayerNorm\\nGPU:{gpu_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='rectangle', style='filled', fillcolor=color)
        
        # Gate for expert selection - this runs on all GPUs to determine routing
        gate_name = f'gate_{layer}_{expert_id}'
        dot.node(gate_name, f'Gate\\nGPU:{gpu_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: routing weights',
                shape='diamond', style='filled', fillcolor=color)
        
        # Token sharding - split tokens based on routing decisions
        shard_name = f'shard_{layer}_{expert_id}'
        dot.node(shard_name, f'Token Sharding\\nGPU:{gpu_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: selected tokens',
                shape='parallelogram', style='filled', fillcolor=color)
        
        # Communication: send tokens to expert GPU
        send_name = f'send_{layer}_{expert_id}'
        dot.node(send_name, f'Send Tokens\\nFrom GPU:{gpu_id}\\nTo Expert GPU:{layer*16 + expert_id}\\nShape: [selected_tokens, {token_dim}]',
                shape='ellipse', style='filled', fillcolor='orange')
        
        # Expert computation on dedicated GPU
        expert_name = f'expert_{layer}_{expert_id}'
        
        # First linear layer (column parallel with TP=2 as mentioned in paper)
        linear1_name = f'linear1_{layer}_{expert_id}'
        dot.node(linear1_name, f'Expert {expert_id}\\nLinear1 (Col-TP2)\\nGPU:{layer*16 + expert_id}\\nIn: [tokens, {token_dim}]\\nOut: [tokens, {hidden_dim//2}]',
                shape='rectangle', style='filled', fillcolor=color)
        
        # Activation
        activation_name = f'activation_{layer}_{expert_id}'
        dot.node(activation_name, f'GELU\\nGPU:{layer*16 + expert_id}\\nIn: [tokens, {hidden_dim//2}]\\nOut: [tokens, {hidden_dim//2}]',
                shape='rectangle', style='filled', fillcolor=color)
        
        # Second linear layer (row parallel with TP=2)
        linear2_name = f'linear2_{layer}_{expert_id}'
        dot.node(linear2_name, f'Linear2 (Row-TP2)\\nGPU:{layer*16 + expert_id}\\nIn: [tokens, {hidden_dim//2}]\\nOut: [tokens, {token_dim}]',
                shape='rectangle', style='filled', fillcolor=color)
        
        # Communication: receive processed tokens back
        recv_name = f'recv_{layer}_{expert_id}'
        dot.node(recv_name, f'Receive Results\\nFrom Expert GPU:{layer*16 + expert_id}\\nTo GPU:{gpu_id}\\nShape: [tokens, {token_dim}]',
                shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Combine received tokens
        combine_name = f'combine_{layer}_{expert_id}'
        dot.node(combine_name, f'Combine Results\\nGPU:{gpu_id}\\nIn: multiple [tokens, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='parallelogram', style='filled', fillcolor=color)
        
        # Residual after experts
        residual2_name = f'residual2_{layer}_{expert_id}'
        dot.node(residual2_name, f'Residual Add\\nGPU:{gpu_id}\\nIn1: [{batch_size}, {seq_len}, {token_dim}]\\nIn2: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='parallelogram', style='filled', fillcolor=color)
        
        # LayerNorm after experts
        ln2_name = f'ln2_{layer}_{expert_id}'
        dot.node(ln2_name, f'LayerNorm\\nGPU:{gpu_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='rectangle', style='filled', fillcolor=color)

# Connect the DAG
# Layer 0 connections
for expert_id in range(16):
    gpu_id = expert_id  # GPU 0-15 for layer 0
    
    dot.edge('input', f'mha_0_{expert_id}')
    dot.edge(f'mha_0_{expert_id}', f'residual1_0_{expert_id}')
    dot.edge(f'residual1_0_{expert_id}', f'ln1_0_{expert_id}')
    dot.edge(f'ln1_0_{expert_id}', f'gate_0_{expert_id}')
    dot.edge(f'ln1_0_{expert_id}', f'shard_0_{expert_id}')
    dot.edge(f'gate_0_{expert_id}', f'shard_0_{expert_id}', style='dashed')
    dot.edge(f'shard_0_{expert_id}', f'send_0_{expert_id}')
    dot.edge(f'send_0_{expert_id}', f'linear1_0_{expert_id}')
    dot.edge(f'linear1_0_{expert_id}', f'activation_0_{expert_id}')
    dot.edge(f'activation_0_{expert_id}', f'linear2_0_{expert_id}')
    dot.edge(f'linear2_0_{expert_id}', f'recv_0_{expert_id}')
    dot.edge(f'recv_0_{expert_id}', f'combine_0_{expert_id}')
    dot.edge(f'combine_0_{expert_id}', f'residual2_0_{expert_id}')
    dot.edge(f'residual2_0_{expert_id}', f'ln2_0_{expert_id}')

# Connect between layers
for layer in range(1, num_layers):
    for expert_id in range(16):
        curr_gpu_id = layer * 16 + expert_id
        prev_gpu_id = expert_id  # Simple mapping
        
        dot.edge(f'ln2_{layer-1}_{prev_gpu_id}', f'mha_{layer}_{expert_id}')
        dot.edge(f'mha_{layer}_{expert_id}', f'residual1_{layer}_{expert_id}')
        dot.edge(f'residual1_{layer}_{expert_id}', f'ln1_{layer}_{expert_id}')
        dot.edge(f'ln1_{layer}_{expert_id}', f'gate_{layer}_{expert_id}')
        dot.edge(f'ln1_{layer}_{expert_id}', f'shard_{layer}_{expert_id}')
        dot.edge(f'gate_{layer}_{expert_id}', f'shard_{layer}_{expert_id}', style='dashed')
        dot.edge(f'shard_{layer}_{expert_id}', f'send_{layer}_{expert_id}')
        dot.edge(f'send_{layer}_{expert_id}', f'linear1_{layer}_{expert_id}')
        dot.edge(f'linear1_{layer}_{expert_id}', f'activation_{layer}_{expert_id}')
        dot.edge(f'activation_{layer}_{expert_id}', f'linear2_{layer}_{expert_id}')
        dot.edge(f'linear2_{layer}_{expert_id}', f'recv_{layer}_{expert_id}')
        dot.edge(f'recv_{layer}_{expert_id}', f'combine_{layer}_{expert_id}')
        dot.edge(f'combine_{layer}_{expert_id}', f'residual2_{layer}_{expert_id}')
        dot.edge(f'residual2_{layer}_{expert_id}', f'ln2_{layer}_{expert_id}')

# Output node - collect from all GPUs in final layer
dot.node('output', f'Total Output\\nShape: [{batch_size}, {seq_len}, {token_dim}]', 
         shape='ellipse', style='filled', fillcolor='lightgray')

# Connect final layer outputs
for expert_id in range(16):
    dot.edge(f'ln2_3_{expert_id}', 'output')

# Save the DAG
dot.format = 'svg'
dot.render('/home/wzc/data/file-share/2025-09-08-11-10-54/proposed_moe')

# Also save as dot file
dot.format = 'dot'
dot.render('/home/wzc/data/file-share/2025-09-08-11-10-54/proposed_moe')

print("Proposed DAG generated successfully")