#!/usr/bin/env python3

import os
from graphviz import Digraph

def generate_baseline_dag():
    """Generate baseline DAG with TP=8, PP=2, 16 GPUs total, 4 experts per GPU"""
    
    dot = Digraph('Baseline_MoE_Deployment')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.2')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Model parameters
    batch_size = 1024
    seq_len = 10000
    token_dim = 8192
    num_heads = 16
    head_dim = 512
    mlp_hidden = 32768
    num_layers = 4
    num_experts_per_layer = 16
    
    # Input node
    dot.node('input', f'Total Input\\nbatch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Pipeline stages
    for stage in [0, 1]:
        with dot.subgraph(name=f'cluster_stage_{stage}') as c:
            c.attr(label=f'Pipeline Stage {stage}', style='dashed', color='black')
            
            # Each stage has 8 GPUs for tensor parallelism
            for tp_rank in range(8):
                gpu_id = stage * 8 + tp_rank
                
                with c.subgraph(name=f'cluster_gpu_{gpu_id}') as gpu_cluster:
                    gpu_cluster.attr(label=f'GPU {gpu_id}', style='rounded,filled', fillcolor='lightgray')
                    
                    # Process 2 layers per stage (4 layers total, 2 per stage)
                    for layer in [stage * 2, stage * 2 + 1]:
                        layer_prefix = f'L{layer}_GPU{gpu_id}'
                        
                        # MHA computation
                        dot.node(f'{layer_prefix}_mha_qkv', 
                                f'MHA QKV Linear\\nINPUT: {batch_size},{seq_len},{token_dim}\\nOUTPUT: {batch_size},{seq_len},{num_heads},{head_dim}*3\\nGPU: {gpu_id}',
                                shape='rectangle', fillcolor='yellow')
                        
                        dot.node(f'{layer_prefix}_mha_attn', 
                                f'MHA Attention\\nINPUT: {batch_size},{seq_len},{num_heads},{head_dim}\\nOUTPUT: {batch_size},{seq_len},{token_dim}\\nGPU: {gpu_id}',
                                shape='rectangle', fillcolor='yellow')
                        
                        dot.node(f'{layer_prefix}_mha_residual', 
                                f'MHA Residual Add\\nINPUT: {batch_size},{seq_len},{token_dim}, {batch_size},{seq_len},{token_dim}\\nOUTPUT: {batch_size},{seq_len},{token_dim}\\nGPU: {gpu_id}',
                                shape='ellipse', fillcolor='lightcoral')
                        
                        # Expert selection (gate)
                        dot.node(f'{layer_prefix}_gate', 
                                f'Expert Gate\\nINPUT: {batch_size},{seq_len},{token_dim}\\nOUTPUT: {batch_size},{seq_len},16\\nGPU: {gpu_id}',
                                shape='diamond', fillcolor='orange')
                        
                        # 4 experts per GPU (16 experts total, 4 per GPU)
                        for expert_idx in range(4):
                            expert_id = (gpu_id * 4) + expert_idx
                            if expert_id < num_experts_per_layer:
                                dot.node(f'{layer_prefix}_expert_{expert_idx}', 
                                        f'Expert {expert_id}\\nINPUT: {batch_size},{seq_len},{token_dim}\\nOUTPUT: {batch_size},{seq_len},{token_dim}\\nGPU: {gpu_id}',
                                        shape='rectangle', fillcolor='lightblue')
                        
                        # Expert aggregation
                        dot.node(f'{layer_prefix}_expert_agg', 
                                f'Expert Aggregation\\nINPUT: 4×({batch_size},{seq_len},{token_dim})\\nOUTPUT: {batch_size},{seq_len},{token_dim}\\nGPU: {gpu_id}',
                                shape='hexagon', fillcolor='lightgreen')
                        
                        # Final residual
                        dot.node(f'{layer_prefix}_final_residual', 
                                f'Final Residual Add\\nINPUT: {batch_size},{seq_len},{token_dim}, {batch_size},{seq_len},{token_dim}\\nOUTPUT: {batch_size},{seq_len},{token_dim}\\nGPU: {gpu_id}',
                                shape='ellipse', fillcolor='lightcoral')
    
    # Add edges
    # Input to first stage
    dot.edge('input', 'L0_GPU0_mha_qkv')
    
    # Connect within each GPU
    for stage in [0, 1]:
        for tp_rank in range(8):
            gpu_id = stage * 8 + tp_rank
            
            for layer in [stage * 2, stage * 2 + 1]:
                layer_prefix = f'L{layer}_GPU{gpu_id}'
                
                # MHA path
                dot.edge(f'{layer_prefix}_mha_qkv', f'{layer_prefix}_mha_attn')
                dot.edge(f'{layer_prefix}_mha_attn', f'{layer_prefix}_mha_residual')
                
                # Expert path
                dot.edge(f'{layer_prefix}_mha_residual', f'{layer_prefix}_gate')
                
                # Connect gate to experts
                for expert_idx in range(4):
                    expert_id = (gpu_id * 4) + expert_idx
                    if expert_id < 16:  # 16 experts per layer
                        dot.edge(f'{layer_prefix}_gate', f'{layer_prefix}_expert_{expert_idx}', style='dashed')
                        dot.edge(f'{layer_prefix}_mha_residual', f'{layer_prefix}_expert_{expert_idx}')
                        dot.edge(f'{layer_prefix}_expert_{expert_idx}', f'{layer_prefix}_expert_agg')
                
                dot.edge(f'{layer_prefix}_expert_agg', f'{layer_prefix}_final_residual')
                dot.edge(f'{layer_prefix}_mha_residual', f'{layer_prefix}_final_residual')
    
    # Connect between layers and stages
    for layer in range(3):  # Connect L0->L1, L1->L2, L2->L3
        next_layer = layer + 1
        
        # Connect final residual of current layer to MHA of next layer
        for tp_rank in range(8):
            curr_gpu = (layer // 2) * 8 + tp_rank
            next_gpu = (next_layer // 2) * 8 + tp_rank
            
            dot.edge(f'L{layer}_GPU{curr_gpu}_final_residual', 
                    f'L{next_layer}_GPU{next_gpu}_mha_qkv')
    
    # Output node
    dot.node('output', f'Total Output\\nbatch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connect final layer to output
    for tp_rank in range(8):
        gpu_id = 1 * 8 + tp_rank  # Last stage
        dot.edge(f'L3_GPU{gpu_id}_final_residual', 'output')
    
    # Save the DAG
    dot.render('/home/wzc/data/file-share/2025-09-09-10-02-47/baseline_moe_deployment', format='svg', cleanup=False)
    dot.save('/home/wzc/data/file-share/2025-09-09-10-02-47/baseline_moe_deployment.dot')
    
    return '/home/wzc/data/file-share/2025-09-09-10-02-47/baseline_moe_deployment.dot'

if __name__ == '__main__':
    generate_baseline_dag()