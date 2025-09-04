#!/usr/bin/env python3

import graphviz

# Create DAG for baseline method with TP=8, PP=2
baseline_dag = graphviz.Digraph('Baseline_TP8_PP2', 
                               filename='baseline_tp8_pp2.dot',
                               format='svg')
baseline_dag.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')

# Define node styles
baseline_dag.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
baseline_dag.attr('node', shape='box', style='filled', fillcolor='lightgreen')  # Computation
baseline_dag.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation

# Global parameters
batch_size = 1024
seq_len = 2048
hidden_dim = 8192
num_heads = 16
head_dim = 512
mlp_hidden = 32768
layers = 4

# Pipeline parallelism: 2 stages, 4 layers per stage
# Tensor parallelism: 8 GPUs per stage
# Total: 16 GPUs (8×2)

# Input layer
baseline_dag.node('input', f'Input\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage 0 (GPUs 0-7)', shape='ellipse')

# Process pipeline stages
for stage in range(2):
    stage_start = stage * 2  # 2 layers per stage
    stage_end = stage_start + 2
    stage_gpus = list(range(stage * 8, (stage + 1) * 8))
    
    # Pipeline stage boundary
    if stage > 0:
        baseline_dag.node(f'pipeline_stage_{stage}_input', 
                         f'Pipeline Stage {stage} Input\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage {stage} (GPUs {stage*8}-{(stage+1)*8-1})', 
                         shape='parallelogram')
        baseline_dag.edge(f'layer{stage_start-1}_output', f'pipeline_stage_{stage}_input')
    
    # Process layers within each stage
    for layer in range(stage_start, stage_end):
        layer_prefix = f'layer{layer}'
        prev_node = 'input' if layer == 0 else (f'pipeline_stage_{stage}_input' if layer == stage_start and stage > 0 else f'layer{layer-1}_output')
        
        # LayerNorm 1 (replicated across 8 GPUs in stage)
        baseline_dag.node(f'{layer_prefix}_ln1', 
                         f'LayerNorm\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='box')
        baseline_dag.edge(prev_node, f'{layer_prefix}_ln1')
        
        # Multi-Head Attention with Tensor Parallelism (8-way)
        # Split heads: 16 heads / 8 GPUs = 2 heads per GPU
        heads_per_gpu = num_heads // 8
        
        # Q, K, V projections (tensor parallel)
        for gpu_idx, gpu_id in enumerate(stage_gpus):
            # Q projection for this GPU
            baseline_dag.node(f'{layer_prefix}_q_proj_gpu{gpu_id}', 
                             f'Q Proj\\nShape: {batch_size}×{seq_len}×{heads_per_gpu*head_dim}\\nGPU: {gpu_id}', 
                             shape='box')
            baseline_dag.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_q_proj_gpu{gpu_id}')
            
            # K projection for this GPU
            baseline_dag.node(f'{layer_prefix}_k_proj_gpu{gpu_id}', 
                             f'K Proj\\nShape: {batch_size}×{seq_len}×{heads_per_gpu*head_dim}\\nGPU: {gpu_id}', 
                             shape='box')
            baseline_dag.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_k_proj_gpu{gpu_id}')
            
            # V projection for this GPU
            baseline_dag.node(f'{layer_prefix}_v_proj_gpu{gpu_id}', 
                             f'V Proj\\nShape: {batch_size}×{seq_len}×{heads_per_gpu*head_dim}\\nGPU: {gpu_id}', 
                             shape='box')
            baseline_dag.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_v_proj_gpu{gpu_id}')
            
            # Attention computation for this GPU
            baseline_dag.node(f'{layer_prefix}_attn_gpu{gpu_id}', 
                             f'Attention\\nShape: {batch_size}×{seq_len}×{heads_per_gpu*head_dim}\\nGPU: {gpu_id}', 
                             shape='box')
            baseline_dag.edge(f'{layer_prefix}_q_proj_gpu{gpu_id}', f'{layer_prefix}_attn_gpu{gpu_id}')
            baseline_dag.edge(f'{layer_prefix}_k_proj_gpu{gpu_id}', f'{layer_prefix}_attn_gpu{gpu_id}')
            baseline_dag.edge(f'{layer_prefix}_v_proj_gpu{gpu_id}', f'{layer_prefix}_attn_gpu{gpu_id}')
        
        # Concatenate attention outputs from all GPUs
        baseline_dag.node(f'{layer_prefix}_attn_concat', 
                         f'Attention Concat\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='parallelogram')
        
        for gpu_id in stage_gpus:
            baseline_dag.edge(f'{layer_prefix}_attn_gpu{gpu_id}', f'{layer_prefix}_attn_concat')
        
        # Output projection (tensor parallel)
        for gpu_idx, gpu_id in enumerate(stage_gpus):
            baseline_dag.node(f'{layer_prefix}_out_proj_gpu{gpu_id}', 
                             f'Output Projection\\nShape: {batch_size}×{seq_len}×{hidden_dim//8}\\nGPU: {gpu_id}', 
                             shape='box')
            baseline_dag.edge(f'{layer_prefix}_attn_concat', f'{layer_prefix}_out_proj_gpu{gpu_id}')
        
        # All-reduce for output projection
        baseline_dag.node(f'{layer_prefix}_out_allreduce', 
                         f'Output All-Reduce\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='parallelogram')
        
        for gpu_id in stage_gpus:
            baseline_dag.edge(f'{layer_prefix}_out_proj_gpu{gpu_id}', f'{layer_prefix}_out_allreduce')
        
        # First residual connection
        baseline_dag.node(f'{layer_prefix}_residual1', 
                         f'Residual Add\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='parallelogram')
        baseline_dag.edge(prev_node, f'{layer_prefix}_residual1')
        baseline_dag.edge(f'{layer_prefix}_out_allreduce', f'{layer_prefix}_residual1')
        
        # LayerNorm 2
        baseline_dag.node(f'{layer_prefix}_ln2', 
                         f'LayerNorm\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='box')
        baseline_dag.edge(f'{layer_prefix}_residual1', f'{layer_prefix}_ln2')
        
        # MLP with tensor parallelism
        # First linear layer (column parallel)
        for gpu_idx, gpu_id in enumerate(stage_gpus):
            baseline_dag.node(f'{layer_prefix}_mlp_linear1_gpu{gpu_id}', 
                             f'MLP Linear1\\nShape: {batch_size}×{seq_len}×{mlp_hidden//8}\\nGPU: {gpu_id}', 
                             shape='box')
            baseline_dag.edge(f'{layer_prefix}_ln2', f'{layer_prefix}_mlp_linear1_gpu{gpu_id}')
        
        # Gather after first linear
        baseline_dag.node(f'{layer_prefix}_mlp_gather', 
                         f'MLP Gather\\nShape: {batch_size}×{seq_len}×{mlp_hidden}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='parallelogram')
        
        for gpu_id in stage_gpus:
            baseline_dag.edge(f'{layer_prefix}_mlp_linear1_gpu{gpu_id}', f'{layer_prefix}_mlp_gather')
        
        # GELU activation
        baseline_dag.node(f'{layer_prefix}_gelu', 
                         f'GELU\\nShape: {batch_size}×{seq_len}×{mlp_hidden}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='box')
        baseline_dag.edge(f'{layer_prefix}_mlp_gather', f'{layer_prefix}_gelu')
        
        # Second linear layer (row parallel)
        for gpu_idx, gpu_id in enumerate(stage_gpus):
            baseline_dag.node(f'{layer_prefix}_mlp_linear2_gpu{gpu_id}', 
                             f'MLP Linear2\\nShape: {batch_size}×{seq_len}×{hidden_dim//8}\\nGPU: {gpu_id}', 
                             shape='box')
            baseline_dag.edge(f'{layer_prefix}_gelu', f'{layer_prefix}_mlp_linear2_gpu{gpu_id}')
        
        # All-reduce sum for MLP output
        baseline_dag.node(f'{layer_prefix}_mlp_allreduce', 
                         f'MLP All-Reduce\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='parallelogram')
        
        for gpu_id in stage_gpus:
            baseline_dag.edge(f'{layer_prefix}_mlp_linear2_gpu{gpu_id}', f'{layer_prefix}_mlp_allreduce')
        
        # Final residual connection
        baseline_dag.node(f'{layer_prefix}_residual2', 
                         f'Residual Add\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='parallelogram')
        baseline_dag.edge(f'{layer_prefix}_residual1', f'{layer_prefix}_residual2')
        baseline_dag.edge(f'{layer_prefix}_mlp_allreduce', f'{layer_prefix}_residual2')
        
        # Layer output
        baseline_dag.node(f'{layer_prefix}_output', 
                         f'Layer {layer} Output\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage {stage} (All {len(stage_gpus)} GPUs)', 
                         shape='ellipse')
        baseline_dag.edge(f'{layer_prefix}_residual2', f'{layer_prefix}_output')

# Final output
baseline_dag.node('output', f'Final Output\\nShape: {batch_size}×{seq_len}×{hidden_dim}\\nGPU: Stage 1 (GPUs 8-15)', shape='ellipse')
baseline_dag.edge('layer3_output', 'output')

# Save the DAG
baseline_dag.render()