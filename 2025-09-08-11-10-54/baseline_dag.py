#!/usr/bin/env python3

import graphviz

# Create baseline DAG for 16 GPUs with TP=8, PP=2, 4 experts per GPU
dot = graphviz.Digraph('baseline_moe', comment='Baseline MoE with TP=8, PP=2, 4 experts/GPU')
dot.attr(rankdir='TB', size='50,50')

# Define global parameters
batch_size = 1024
seq_len = 10000
token_dim = 8192
num_heads = 16
head_dim = 512
hidden_dim = 32768
num_layers = 4
experts_per_layer = 16
experts_per_gpu = 4
gpus_per_layer = 4  # 16 experts / 4 experts per GPU = 4 GPUs per layer

# Color scheme for different GPU groups
gpu_colors = {
    'layer0': 'lightblue',
    'layer1': 'lightgreen',
    'layer2': 'lightyellow',
    'layer3': 'lightcoral'
}

# Input node
dot.node('input', f'Total Input\\nShape: [{batch_size}, {seq_len}, {token_dim}]', 
         shape='ellipse', style='filled', fillcolor='lightgray')

# Process each layer (4 layers total)
for layer in range(num_layers):
    layer_name = f'layer{layer}'
    
    # For each layer, we have 4 GPUs handling 4 experts each
    for gpu_id in range(4):
        gpu_global_id = layer * 4 + gpu_id
        
        # Multi-Head Attention (shared across experts on same GPU)
        attn_name = f'attn_{layer}_{gpu_id}'
        dot.node(attn_name, f'MHA\\nGPU:{gpu_global_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='rectangle', style='filled', fillcolor=gpu_colors[layer_name])
        
        # Residual connection after attention
        residual1_name = f'residual1_{layer}_{gpu_id}'
        dot.node(residual1_name, f'Residual Add\\nGPU:{gpu_global_id}\\nIn1: [{batch_size}, {seq_len}, {token_dim}]\\nIn2: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='parallelogram', style='filled', fillcolor=gpu_colors[layer_name])
        
        # LayerNorm after attention
        ln1_name = f'ln1_{layer}_{gpu_id}'
        dot.node(ln1_name, f'LayerNorm\\nGPU:{gpu_global_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='rectangle', style='filled', fillcolor=gpu_colors[layer_name])
        
        # Gate for expert selection
        gate_name = f'gate_{layer}_{gpu_id}'
        dot.node(gate_name, f'Gate\\nGPU:{gpu_global_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: routing decisions',
                shape='diamond', style='filled', fillcolor=gpu_colors[layer_name])
        
        # Process 4 experts on this GPU
        for expert_id in range(4):
            expert_global_id = gpu_id * 4 + expert_id
            expert_name = f'expert_{layer}_{gpu_id}_{expert_id}'
            
            # Expert MLP (with tensor parallelism within expert)
            # First linear: token_dim -> hidden_dim (column parallel)
            linear1_name = f'linear1_{layer}_{gpu_id}_{expert_id}'
            dot.node(linear1_name, f'Expert {expert_global_id}\\nLinear1 (Col-Parallel)\\nGPU:{gpu_global_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {hidden_dim//2}]',
                    shape='rectangle', style='filled', fillcolor=gpu_colors[layer_name])
            
            # Activation
            activation_name = f'activation_{layer}_{gpu_id}_{expert_id}'
            dot.node(activation_name, f'GELU\\nGPU:{gpu_global_id}\\nIn: [{batch_size}, {seq_len}, {hidden_dim//2}]\\nOut: [{batch_size}, {seq_len}, {hidden_dim//2}]',
                    shape='rectangle', style='filled', fillcolor=gpu_colors[layer_name])
            
            # Second linear: hidden_dim -> token_dim (row parallel)
            linear2_name = f'linear2_{layer}_{gpu_id}_{expert_id}'
            dot.node(linear2_name, f'Linear2 (Row-Parallel)\\nGPU:{gpu_global_id}\\nIn: [{batch_size}, {seq_len}, {hidden_dim//2}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                    shape='rectangle', style='filled', fillcolor=gpu_colors[layer_name])
            
            # Expert aggregation
            expert_agg_name = f'expert_agg_{layer}_{gpu_id}_{expert_id}'
            dot.node(expert_agg_name, f'Expert {expert_global_id}\\nAggregation\\nGPU:{gpu_global_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                    shape='parallelogram', style='filled', fillcolor=gpu_colors[layer_name])
            
            # Connect gate to expert with dashed line
            dot.edge(gate_name, expert_agg_name, style='dashed', label=f'route to expert {expert_global_id}')
            
            # Connect expert components
            dot.edge(linear1_name, activation_name)
            dot.edge(activation_name, linear2_name)
            dot.edge(linear2_name, expert_agg_name)
        
        # Combine expert outputs
        expert_combine_name = f'expert_combine_{layer}_{gpu_id}'
        dot.node(expert_combine_name, f'Expert Combine\\nGPU:{gpu_global_id}\\nIn: 4×[{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='parallelogram', style='filled', fillcolor=gpu_colors[layer_name])
        
        # Residual connection after experts
        residual2_name = f'residual2_{layer}_{gpu_id}'
        dot.node(residual2_name, f'Residual Add\\nGPU:{gpu_global_id}\\nIn1: [{batch_size}, {seq_len}, {token_dim}]\\nIn2: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='parallelogram', style='filled', fillcolor=gpu_colors[layer_name])
        
        # LayerNorm after experts
        ln2_name = f'ln2_{layer}_{gpu_id}'
        dot.node(ln2_name, f'LayerNorm\\nGPU:{gpu_global_id}\\nIn: [{batch_size}, {seq_len}, {token_dim}]\\nOut: [{batch_size}, {seq_len}, {token_dim}]',
                shape='rectangle', style='filled', fillcolor=gpu_colors[layer_name])

# Connect the layers
# Layer 0 connections
for gpu_id in range(4):
    dot.edge('input', f'attn_0_{gpu_id}')
    dot.edge(f'attn_0_{gpu_id}', f'residual1_0_{gpu_id}')
    dot.edge(f'residual1_0_{gpu_id}', f'ln1_0_{gpu_id}')
    dot.edge(f'ln1_0_{gpu_id}', f'gate_0_{gpu_id}')
    
    # Connect gate to all 4 experts
    for expert_id in range(4):
        dot.edge(f'gate_0_{gpu_id}', f'expert_agg_0_{gpu_id}_{expert_id}')
    
    # Connect expert outputs to combine
    for expert_id in range(4):
        dot.edge(f'expert_agg_0_{gpu_id}_{expert_id}', f'expert_combine_0_{gpu_id}')
    
    dot.edge(f'expert_combine_0_{gpu_id}', f'residual2_0_{gpu_id}')
    dot.edge(f'residual2_0_{gpu_id}', f'ln2_0_{gpu_id}')

# Connect between layers (pipeline parallelism)
for layer in range(1, num_layers):
    for gpu_id in range(4):
        prev_gpu_id = gpu_id  # Simple mapping for now
        dot.edge(f'ln2_{layer-1}_{prev_gpu_id}', f'attn_{layer}_{gpu_id}')
        dot.edge(f'attn_{layer}_{gpu_id}', f'residual1_{layer}_{gpu_id}')
        dot.edge(f'residual1_{layer}_{gpu_id}', f'ln1_{layer}_{gpu_id}')
        dot.edge(f'ln1_{layer}_{gpu_id}', f'gate_{layer}_{gpu_id}')
        
        # Connect gate to all 4 experts
        for expert_id in range(4):
            dot.edge(f'gate_{layer}_{gpu_id}', f'expert_agg_{layer}_{gpu_id}_{expert_id}')
        
        # Connect expert outputs to combine
        for expert_id in range(4):
            dot.edge(f'expert_agg_{layer}_{gpu_id}_{expert_id}', f'expert_combine_{layer}_{gpu_id}')
        
        dot.edge(f'expert_combine_{layer}_{gpu_id}', f'residual2_{layer}_{gpu_id}')
        dot.edge(f'residual2_{layer}_{gpu_id}', f'ln2_{layer}_{gpu_id}')

# Output node - collect from all GPUs in final layer
dot.node('output', f'Total Output\\nShape: [{batch_size}, {seq_len}, {token_dim}]', 
         shape='ellipse', style='filled', fillcolor='lightgray')

# Connect final layer outputs
for gpu_id in range(4):
    dot.edge(f'ln2_3_{gpu_id}', 'output')

# Save the DAG
dot.format = 'svg'
dot.render('/home/wzc/data/file-share/2025-09-08-11-10-54/baseline_moe')

# Also save as dot file
dot.format = 'dot'
dot.render('/home/wzc/data/file-share/2025-09-08-11-10-54/baseline_moe')

print("Baseline DAG generated successfully")