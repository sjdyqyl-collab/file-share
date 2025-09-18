#!/usr/bin/env python3

import os
from graphviz import Digraph

def generate_proposed_dag():
    """Generate proposed DAG with 64 GPUs, 1 expert per GPU, EP=64"""
    
    dot = Digraph('Proposed_Cross_Node_MoE_Deployment')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='1.5')
    
    # Model parameters
    batch_size = 1024
    seq_len = 10000
    token_dim = 8192
    num_heads = 16
    head_dim = 512
    mlp_hidden = 32768
    num_layers = 4
    num_experts_per_layer = 64
    num_gpus = 64
    
    # Input node
    dot.node('input', f'Total Input\\nbatch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Create nodes for each layer
    for layer in range(num_layers):
        with dot.subgraph(name=f'cluster_layer_{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer} (64 GPUs)', style='dashed', color='blue')
            
            # MHA computation - distributed across all GPUs for tensor parallelism
            for gpu_id in range(num_gpus):
                layer_prefix = f'L{layer}_GPU{gpu_id}'
                
                # MHA components
                dot.node(f'{layer_prefix}_mha_qkv', 
                        f'MHA QKV Linear\\nINPUT: {batch_size},{seq_len},{token_dim}\\nOUTPUT: {batch_size},{seq_len},{num_heads},{head_dim}\\nGPU: {gpu_id}',
                        shape='rectangle', fillcolor='yellow', width='2.5')
                
                dot.node(f'{layer_prefix}_mha_attn', 
                        f'MHA Attention\\nINPUT: {batch_size},{seq_len},{num_heads},{head_dim}\\nOUTPUT: {batch_size},{seq_len},{token_dim}\\nGPU: {gpu_id}',
                        shape='rectangle', fillcolor='yellow', width='2.5')
                
                dot.node(f'{layer_prefix}_mha_residual', 
                        f'MHA Residual\\nINPUT: {batch_size},{seq_len},{token_dim}, {batch_size},{seq_len},{token_dim}\\nOUTPUT: {batch_size},{seq_len},{token_dim}\\nGPU: {gpu_id}',
                        shape='ellipse', fillcolor='lightcoral', width='2.5')
                
                # Expert gate (global routing)
                dot.node(f'{layer_prefix}_gate', 
                        f'Expert Gate\\nINPUT: {batch_size},{seq_len},{token_dim}\\nOUTPUT: {batch_size},{seq_len},64\\nGPU: {gpu_id}',
                        shape='diamond', fillcolor='orange', width='2.5')
                
                # Single expert per GPU
                expert_id = gpu_id + layer * num_gpus
                dot.node(f'{layer_prefix}_expert', 
                        f'Expert {expert_id}\\nINPUT: tokens routed\\nOUTPUT: {batch_size},selected_tokens,{token_dim}\\nGPU: {gpu_id}',
                        shape='rectangle', fillcolor='lightblue', width='2.5')
                
                # Token routing nodes
                dot.node(f'{layer_prefix}_token_split', 
                        f'Token Split\\nINPUT: {batch_size},{seq_len},{token_dim}\\nOUTPUT: routed tokens\\nGPU: {gpu_id}',
                        shape='hexagon', fillcolor='lightgreen', width='2.5')
                
                dot.node(f'{layer_prefix}_token_gather', 
                        f'Token Gather\\nINPUT: processed tokens\\nOUTPUT: {batch_size},{seq_len},{token_dim}\\nGPU: {gpu_id}',
                        shape='hexagon', fillcolor='lightgreen', width='2.5')
                
                # Final residual
                dot.node(f'{layer_prefix}_final_residual', 
                        f'Final Residual\\nINPUT: {batch_size},{seq_len},{token_dim}, {batch_size},{seq_len},{token_dim}\\nOUTPUT: {batch_size},{seq_len},{token_dim}\\nGPU: {gpu_id}',
                        shape='ellipse', fillcolor='lightcoral', width='2.5')
    
    # Add edges
    # Input to first layer
    for gpu_id in range(num_gpus):
        dot.edge('input', f'L0_GPU{gpu_id}_mha_qkv')
    
    # Connect within each layer
    for layer in range(num_layers):
        for gpu_id in range(num_gpus):
            layer_prefix = f'L{layer}_GPU{gpu_id}'
            
            # MHA path
            dot.edge(f'{layer_prefix}_mha_qkv', f'{layer_prefix}_mha_attn')
            dot.edge(f'{layer_prefix}_mha_attn', f'{layer_prefix}_mha_residual')
            
            # Expert routing path
            dot.edge(f'{layer_prefix}_mha_residual', f'{layer_prefix}_gate')
            dot.edge(f'{layer_prefix}_gate', f'{layer_prefix}_token_split', style='dashed')
            dot.edge(f'{layer_prefix}_mha_residual', f'{layer_prefix}_token_split')
            dot.edge(f'{layer_prefix}_token_split', f'{layer_prefix}_expert')
            dot.edge(f'{layer_prefix}_expert', f'{layer_prefix}_token_gather')
            dot.edge(f'{layer_prefix}_token_gather', f'{layer_prefix}_final_residual')
            dot.edge(f'{layer_prefix}_mha_residual', f'{layer_prefix}_final_residual')
    
    # Cross-GPU communication for expert routing
    for layer in range(num_layers):
        # Token routing between GPUs
        for src_gpu in range(num_gpus):
            for dst_gpu in range(num_gpus):
                if src_gpu != dst_gpu:
                    # Communication edges for token routing
                    dot.edge(f'L{layer}_GPU{src_gpu}_token_split', 
                            f'L{layer}_GPU{dst_gpu}_expert', 
                            style='dotted', color='red', 
                            label='cross-node routing')
                    
                    dot.edge(f'L{layer}_GPU{dst_gpu}_expert', 
                            f'L{layer}_GPU{src_gpu}_token_gather', 
                            style='dotted', color='blue', 
                            label='cross-node gathering')
    
    # Connect between layers
    for layer in range(num_layers - 1):
        next_layer = layer + 1
        for gpu_id in range(num_gpus):
            dot.edge(f'L{layer}_GPU{gpu_id}_final_residual', 
                    f'L{next_layer}_GPU{gpu_id}_mha_qkv')
    
    # Output node
    dot.node('output', f'Total Output\\nbatch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}', 
             shape='parallelogram', fillcolor='lightgreen', width='3')
    
    # Connect final layer to output
    for gpu_id in range(num_gpus):
        dot.edge(f'L3_GPU{gpu_id}_final_residual', 'output')
    
    # Save the DAG
    dot.render('/home/wzc/data/file-share/2025-09-09-10-02-47/proposed_cross_node_moe_deployment', format='svg', cleanup=False)
    dot.save('/home/wzc/data/file-share/2025-09-09-10-02-47/proposed_cross_node_moe_deployment.dot')
    
    return '/home/wzc/data/file-share/2025-09-09-10-02-47/proposed_cross_node_moe_deployment.dot'

if __name__ == '__main__':
    generate_proposed_dag()