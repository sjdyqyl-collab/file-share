#!/usr/bin/env python3

import os
from graphviz import Digraph

def create_baseline_dag():
    """Create DAG for baseline TP=8, PP=2 configuration"""
    
    dot = Digraph(comment='Baseline TP=8 PP=2 Configuration DAG')
    dot.attr(rankdir='TB', size='30,20')
    
    # Define colors for different stages
    stage0_color = 'lightblue'
    stage1_color = 'lightgreen'
    comm_color = 'yellow'
    
    # Input node
    dot.node('input', 'Input\n[batch_size=1024, seq_len=10000, token_dim=8192]', 
             shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Process each layer for both stages
    for layer in range(4):
        # Stage 0 (GPUs 0-7)
        with dot.subgraph(name=f'cluster_stage0_layer{layer}') as stage0:
            stage0.attr(label=f'Layer {layer} - Stage 0 (GPUs 0-7)', style='rounded', fillcolor=stage0_color)
            
            # Multi-Head Attention for Stage 0
            stage0.node(f'mha_s0_l{layer}', f'MHA Layer {layer} Stage 0\n[batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPUs: 0-7', 
                       shape='rectangle', style='filled', fillcolor=stage0_color)
            
            # Expert computation for Stage 0 - 8 experts per GPU
            for gpu in range(8):
                experts = f"expert_0-7" if gpu % 2 == 0 else f"expert_8-15"
                stage0.node(f'expert_s0_l{layer}_gpu{gpu}', 
                           f'Experts {experts} Layer {layer}\n[batch_size=1024, token_dim=8192]\nGPU: {gpu}', 
                           shape='rectangle', style='filled', fillcolor=stage0_color)
            
            # Residual connections
            stage0.node(f'residual_s0_l{layer}', f'Residual Add Layer {layer} Stage 0\n[batch_size=1024, seq_len=10000, token_dim=8192]\nGPUs: 0-7', 
                       shape='parallelogram', style='filled', fillcolor=stage0_color)
            
            # Layer normalization
            stage0.node(f'layernorm_s0_l{layer}', f'LayerNorm Layer {layer} Stage 0\n[batch_size=1024, seq_len=10000, token_dim=8192]\nGPUs: 0-7', 
                       shape='rectangle', style='filled', fillcolor=stage0_color)
        
        # Stage 1 (GPUs 8-15)
        with dot.subgraph(name=f'cluster_stage1_layer{layer}') as stage1:
            stage1.attr(label=f'Layer {layer} - Stage 1 (GPUs 8-15)', style='rounded', fillcolor=stage1_color)
            
            # Multi-Head Attention for Stage 1
            stage1.node(f'mha_s1_l{layer}', f'MHA Layer {layer} Stage 1\n[batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPUs: 8-15', 
                       shape='rectangle', style='filled', fillcolor=stage1_color)
            
            # Expert computation for Stage 1 - 8 experts per GPU
            for gpu in range(8, 16):
                experts = f"expert_0-7" if gpu % 2 == 0 else f"expert_8-15"
                stage1.node(f'expert_s1_l{layer}_gpu{gpu}', 
                           f'Experts {experts} Layer {layer}\n[batch_size=1024, token_dim=8192]\nGPU: {gpu}', 
                           shape='rectangle', style='filled', fillcolor=stage1_color)
            
            # Residual connections
            stage1.node(f'residual_s1_l{layer}', f'Residual Add Layer {layer} Stage 1\n[batch_size=1024, seq_len=10000, token_dim=8192]\nGPUs: 8-15', 
                       shape='parallelogram', style='filled', fillcolor=stage1_color)
            
            # Layer normalization
            stage1.node(f'layernorm_s1_l{layer}', f'LayerNorm Layer {layer} Stage 1\n[batch_size=1024, seq_len=10000, token_dim=8192]\nGPUs: 8-15', 
                       shape='rectangle', style='filled', fillcolor=stage1_color)
        
        # Communication between stages
        if layer < 3:  # Not after last layer
            dot.node(f'comm_l{layer}', f'Pipeline Communication\nLayer {layer} → {layer+1}\n[batch_size=1024, seq_len=10000, token_dim=8192]\nAll GPUs', 
                    shape='ellipse', style='filled', fillcolor=comm_color)
    
    # Output node
    dot.node('output', 'Output\n[batch_size=1024, seq_len=10000, token_dim=8192]', 
             shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Connect the nodes
    # Input to first layer
    dot.edge('input', 'mha_s0_l0')
    
    # Layer 0 connections
    dot.edge('mha_s0_l0', 'expert_s0_l0_gpu0')
    dot.edge('mha_s0_l0', 'expert_s0_l0_gpu1')
    dot.edge('mha_s0_l0', 'expert_s0_l0_gpu2')
    dot.edge('mha_s0_l0', 'expert_s0_l0_gpu3')
    dot.edge('mha_s0_l0', 'expert_s0_l0_gpu4')
    dot.edge('mha_s0_l0', 'expert_s0_l0_gpu5')
    dot.edge('mha_s0_l0', 'expert_s0_l0_gpu6')
    dot.edge('mha_s0_l0', 'expert_s0_l0_gpu7')
    
    for gpu in range(8):
        dot.edge(f'expert_s0_l0_gpu{gpu}', f'residual_s0_l0')
    dot.edge('mha_s0_l0', f'residual_s0_l0')
    dot.edge(f'residual_s0_l0', f'layernorm_s0_l0')
    
    # Pipeline communication
    dot.edge(f'layernorm_s0_l0', 'mha_s1_l0')
    
    # Stage 1 connections
    dot.edge('mha_s1_l0', 'expert_s1_l0_gpu8')
    dot.edge('mha_s1_l0', 'expert_s1_l0_gpu9')
    dot.edge('mha_s1_l0', 'expert_s1_l0_gpu10')
    dot.edge('mha_s1_l0', 'expert_s1_l0_gpu11')
    dot.edge('mha_s1_l0', 'expert_s1_l0_gpu12')
    dot.edge('mha_s1_l0', 'expert_s1_l0_gpu13')
    dot.edge('mha_s1_l0', 'expert_s1_l0_gpu14')
    dot.edge('mha_s1_l0', 'expert_s1_l0_gpu15')
    
    for gpu in range(8, 16):
        dot.edge(f'expert_s1_l0_gpu{gpu}', f'residual_s1_l0')
    dot.edge('mha_s1_l0', f'residual_s1_l0')
    dot.edge(f'residual_s1_l0', f'layernorm_s1_l0')
    
    # Continue for remaining layers...
    for layer in range(1, 4):
        if layer == 1:
            prev = 'layernorm_s1_l0'
        elif layer == 2:
            prev = 'layernorm_s1_l1'
        else:
            prev = 'layernorm_s1_l2'
            
        dot.edge(prev, f'mha_s0_l{layer}')
        
        # Stage 0 for this layer
        dot.edge(f'mha_s0_l{layer}', f'expert_s0_l{layer}_gpu0')
        dot.edge(f'mha_s0_l{layer}', f'expert_s0_l{layer}_gpu1')
        dot.edge(f'mha_s0_l{layer}', f'expert_s0_l{layer}_gpu2')
        dot.edge(f'mha_s0_l{layer}', f'expert_s0_l{layer}_gpu3')
        dot.edge(f'mha_s0_l{layer}', f'expert_s0_l{layer}_gpu4')
        dot.edge(f'mha_s0_l{layer}', f'expert_s0_l{layer}_gpu5')
        dot.edge(f'mha_s0_l{layer}', f'expert_s0_l{layer}_gpu6')
        dot.edge(f'mha_s0_l{layer}', f'expert_s0_l{layer}_gpu7')
        
        for gpu in range(8):
            dot.edge(f'expert_s0_l{layer}_gpu{gpu}', f'residual_s0_l{layer}')
        dot.edge(f'mha_s0_l{layer}', f'residual_s0_l{layer}')
        dot.edge(f'residual_s0_l{layer}', f'layernorm_s0_l{layer}')
        
        # Pipeline communication
        if layer < 3:
            dot.edge(f'layernorm_s0_l{layer}', f'mha_s1_l{layer}')
        
        # Stage 1 for this layer
        dot.edge(f'layernorm_s0_l{layer}', f'mha_s1_l{layer}')
        
        dot.edge(f'mha_s1_l{layer}', f'expert_s1_l{layer}_gpu8')
        dot.edge(f'mha_s1_l{layer}', f'expert_s1_l{layer}_gpu9')
        dot.edge(f'mha_s1_l{layer}', f'expert_s1_l{layer}_gpu10')
        dot.edge(f'mha_s1_l{layer}', f'expert_s1_l{layer}_gpu11')
        dot.edge(f'mha_s1_l{layer}', f'expert_s1_l{layer}_gpu12')
        dot.edge(f'mha_s1_l{layer}', f'expert_s1_l{layer}_gpu13')
        dot.edge(f'mha_s1_l{layer}', f'expert_s1_l{layer}_gpu14')
        dot.edge(f'mha_s1_l{layer}', f'expert_s1_l{layer}_gpu15')
        
        for gpu in range(8, 16):
            dot.edge(f'expert_s1_l{layer}_gpu{gpu}', f'residual_s1_l{layer}')
        dot.edge(f'mha_s1_l{layer}', f'residual_s1_l{layer}')
        dot.edge(f'residual_s1_l{layer}', f'layernorm_s1_l{layer}')
    
    # Final output
    dot.edge('layernorm_s1_l3', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/baseline_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/baseline_dag.dot')
    print("Baseline DAG created successfully")