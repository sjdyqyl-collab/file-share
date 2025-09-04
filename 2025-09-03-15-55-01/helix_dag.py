#!/usr/bin/env python3

import graphviz

# Create DAG for Helix method with 16 GPUs (4x4 partitioning)
helix_dag = graphviz.Digraph('Helix_Two_Level_Partitioning', 
                             filename='helix_two_level_partitioning.dot',
                             format='svg')
helix_dag.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')

# Define node styles
helix_dag.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
helix_dag.attr('node', shape='box', style='filled', fillcolor='lightgreen')  # Computation
helix_dag.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation

# Global parameters
batch_size = 1024
seq_len = 2048  # Assuming standard sequence length
hidden_dim = 8192  # 16 heads * 512 head dimension
num_heads = 16
head_dim = 512
mlp_hidden = 32768
layers = 4

# Partitioning parameters
m = 4  # dimension segments
n = 4  # head groups
total_partitions = m * n  # 16 GPUs
heads_per_group = num_heads // n  # 4 heads per group
segment_dim = head_dim // m  # 128 dimensions per segment

# Input layer
helix_dag.node('input', f'Input\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', shape='ellipse')

# Process each layer
for layer in range(layers):
    layer_prefix = f'layer{layer}'
    
    # LayerNorm (replicated across all GPUs)
    helix_dag.node(f'{layer_prefix}_ln1', f'LayerNorm\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', shape='box')
    helix_dag.edge('input' if layer == 0 else f'layer{layer-1}_output', f'{layer_prefix}_ln1')
    
    # Multi-Head Attention with Two-Level Partitioning
    # Create Q, K, V projections for each partition
    for i in range(n):  # head groups
        for j in range(m):  # dimension segments
            gpu_id = i * m + j
            partition_id = f'{i}_{j}'
            
            # Q projection for partition (i,j)
            helix_dag.node(f'{layer_prefix}_q_proj_{partition_id}', 
                         f'Q Proj\\nShape: {batch_size}×{seq_len}×{heads_per_group}×{segment_dim}\\nGPU: {gpu_id}', 
                         shape='box')
            helix_dag.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_q_proj_{partition_id}')
            
            # K projection for partition (i,j)
            helix_dag.node(f'{layer_prefix}_k_proj_{partition_id}', 
                         f'K Proj\\nShape: {batch_size}×{seq_len}×{heads_per_group}×{segment_dim}\\nGPU: {gpu_id}', 
                         shape='box')
            helix_dag.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_k_proj_{partition_id}')
            
            # V projection for partition (i,j)
            helix_dag.node(f'{layer_prefix}_v_proj_{partition_id}', 
                         f'V Proj\\nShape: {batch_size}×{seq_len}×{heads_per_group}×{segment_dim}\\nGPU: {gpu_id}', 
                         shape='box')
            helix_dag.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_v_proj_{partition_id}')
            
            # Attention computation for partition (i,j)
            helix_dag.node(f'{layer_prefix}_attn_{partition_id}', 
                         f'Attention\\nShape: {batch_size}×{seq_len}×{heads_per_group}×{segment_dim}\\nGPU: {gpu_id}', 
                         shape='box')
            helix_dag.edge(f'{layer_prefix}_q_proj_{partition_id}', f'{layer_prefix}_attn_{partition_id}')
            helix_dag.edge(f'{layer_prefix}_k_proj_{partition_id}', f'{layer_prefix}_attn_{partition_id}')
            helix_dag.edge(f'{layer_prefix}_v_proj_{partition_id}', f'{layer_prefix}_attn_{partition_id}')
    
    # Intra-group aggregation (concatenate dimension segments within each head group)
    for i in range(n):  # head groups
        helix_dag.node(f'{layer_prefix}_intra_group_agg_{i}', 
                     f'Intra-Group Concat\\nShape: {batch_size}×{seq_len}×{heads_per_group}×{head_dim}\\nGPU: {i*m}-{(i+1)*m-1}', 
                     shape='parallelogram')
        
        for j in range(m):  # dimension segments
            partition_id = f'{i}_{j}'
            helix_dag.edge(f'{layer_prefix}_attn_{partition_id}', f'{layer_prefix}_intra_group_agg_{i}')
    
    # Inter-group aggregation (concatenate all head groups)
    helix_dag.node(f'{layer_prefix}_inter_group_agg', 
                 f'Inter-Group Concat\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', 
                 shape='parallelogram')
    
    for i in range(n):
        helix_dag.edge(f'{layer_prefix}_intra_group_agg_{i}', f'{layer_prefix}_inter_group_agg')
    
    # Output projection
    helix_dag.node(f'{layer_prefix}_out_proj', 
                 f'Output Projection\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', 
                 shape='box')
    helix_dag.edge(f'{layer_prefix}_inter_group_agg', f'{layer_prefix}_out_proj')
    
    # Residual connection
    helix_dag.node(f'{layer_prefix}_residual1', 
                 f'Residual Add\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', 
                 shape='parallelogram')
    helix_dag.edge('input' if layer == 0 else f'layer{layer-1}_output', f'{layer_prefix}_residual1')
    helix_dag.edge(f'{layer_prefix}_out_proj', f'{layer_prefix}_residual1')
    
    # LayerNorm 2
    helix_dag.node(f'{layer_prefix}_ln2', 
                 f'LayerNorm\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', 
                 shape='box')
    helix_dag.edge(f'{layer_prefix}_residual1', f'{layer_prefix}_ln2')
    
    # MLP with tensor parallelism (column-row parallel)
    # First linear layer (column parallel)
    for gpu_id in range(16):
        helix_dag.node(f'{layer_prefix}_mlp_linear1_{gpu_id}', 
                     f'MLP Linear1\\nShape: {batch_size}×{seq_len}×{mlp_hidden//16}\\nGPU: {gpu_id}', 
                     shape='box')
        helix_dag.edge(f'{layer_prefix}_ln2', f'{layer_prefix}_mlp_linear1_{gpu_id}')
    
    # Gather after first linear
    helix_dag.node(f'{layer_prefix}_mlp_gather', 
                 f'MLP Gather\\nShape: {batch_size}×{seq_len}×{mlp_hidden}\\nGPU: All GPUs', 
                 shape='parallelogram')
    for gpu_id in range(16):
        helix_dag.edge(f'{layer_prefix}_mlp_linear1_{gpu_id}', f'{layer_prefix}_mlp_gather')
    
    # GELU activation
    helix_dag.node(f'{layer_prefix}_gelu', 
                 f'GELU\\nShape: {batch_size}×{seq_len}×{mlp_hidden}\\nGPU: All GPUs', 
                 shape='box')
    helix_dag.edge(f'{layer_prefix}_mlp_gather', f'{layer_prefix}_gelu')
    
    # Second linear layer (row parallel)
    for gpu_id in range(16):
        helix_dag.node(f'{layer_prefix}_mlp_linear2_{gpu_id}', 
                     f'MLP Linear2\\nShape: {batch_size}×{seq_len}×{hidden_dim//16}\\nGPU: {gpu_id}', 
                     shape='box')
        helix_dag.edge(f'{layer_prefix}_gelu', f'{layer_prefix}_mlp_linear2_{gpu_id}')
    
    # All-reduce sum for MLP output
    helix_dag.node(f'{layer_prefix}_mlp_allreduce', 
                 f'MLP All-Reduce\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', 
                 shape='parallelogram')
    for gpu_id in range(16):
        helix_dag.edge(f'{layer_prefix}_mlp_linear2_{gpu_id}', f'{layer_prefix}_mlp_allreduce')
    
    # Final residual connection
    helix_dag.node(f'{layer_prefix}_residual2', 
                 f'Residual Add\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', 
                 shape='parallelogram')
    helix_dag.edge(f'{layer_prefix}_residual1', f'{layer_prefix}_residual2')
    helix_dag.edge(f'{layer_prefix}_mlp_allreduce', f'{layer_prefix}_residual2')
    
    # Layer output
    helix_dag.node(f'{layer_prefix}_output', 
                 f'Layer {layer} Output\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', 
                 shape='ellipse')
    helix_dag.edge(f'{layer_prefix}_residual2', f'{layer_prefix}_output')

# Final output
helix_dag.node('output', f'Final Output\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: All GPUs', shape='ellipse')
helix_dag.edge('layer3_output', 'output')

# Save the DAG
helix_dag.render()