#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_proposed_dag():
    """
    Create DAG for proposed cross-node expert parallelism method
    EP=64, one expert per GPU, 4 layers, 16 experts per layer
    """
    
    dot = Digraph(comment='Proposed Cross-Node Expert Parallelism DAG (EP=64)', format='svg')
    dot.attr(rankdir='TB', size='25,35', ranksep='1.2', nodesep='0.3')
    
    # Input node
    dot.node('input', 'Model Input\n[1024, 10000, 8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Global variables
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192
    ffn_hidden = 32768
    experts_per_layer = 16
    total_gpus = 64
    
    # Create layers 0-3 with one expert per GPU
    for layer_idx in range(4):
        layer_prefix = f"layer_{layer_idx}"
        
        # Calculate GPU ranges for this layer
        gpu_start = layer_idx * 16
        gpu_end = gpu_start + 16
        
        # Add layer boundary
        dot.node(f'{layer_prefix}_boundary', f'Layer {layer_idx} Boundary\nGPUs {gpu_start}-{gpu_end-1}', 
                shape='box', style='dashed', fillcolor='lightgray')
        
        # Gate computation (distributed across all GPUs in layer)
        gate_gpu = gpu_start  # Primary gate GPU
        dot.node(f'{layer_prefix}_gate', f'Gate Network\n[1024, 10000, 16]\nGPU {gate_gpu}', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Token sharding and routing
        dot.node(f'{layer_prefix}_token_shard', f'Token Sharding\n[1024, 10000, 8192] -> [tokens, 8192]\nGPU {gate_gpu}', 
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # Async communication for token routing
        dot.node(f'{layer_prefix}_async_comm', f'Async All-to-All\n[tokens, 8192]\nAll GPUs {gpu_start}-{gpu_end-1}', 
                shape='ellipse', style='filled', fillcolor='lightsteelblue')
        
        # Expert computation (one expert per GPU)
        for expert_idx in range(experts_per_layer):
            expert_gpu = gpu_start + expert_idx
            expert_prefix = f'{layer_prefix}_expert_{expert_idx}'
            
            # Expert input receive
            dot.node(f'{expert_prefix}_receive', 
                    f'Expert {expert_idx} Receive\n[local_tokens, 8192]\nGPU {expert_gpu}', 
                    shape='ellipse', style='filled', fillcolor='lightgreen')
            
            # Expert MLP computation (no tensor parallelism)
            dot.node(f'{expert_prefix}_linear1', 
                    f'Expert {expert_idx} Linear1\n[local_tokens, 8192] -> [local_tokens, 32768]\nGPU {expert_gpu}', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
            
            dot.node(f'{expert_prefix}_gelu', 
                    f'Expert {expert_idx} GELU\n[local_tokens, 32768]\nGPU {expert_gpu}', 
                    shape='rectangle', style='filled', fillcolor='lightpink')
            
            dot.node(f'{expert_prefix}_linear2', 
                    f'Expert {expert_idx} Linear2\n[local_tokens, 32768] -> [local_tokens, 8192]\nGPU {expert_gpu}', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Expert output send
            dot.node(f'{expert_prefix}_send', 
                    f'Expert {expert_idx} Send\n[local_tokens, 8192]\nGPU {expert_gpu}', 
                    shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Token aggregation
        dot.node(f'{layer_prefix}_token_agg', f'Token Aggregation\n[tokens, 8192] -> [1024, 10000, 8192]\nGPU {gate_gpu}', 
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # Async communication back
        dot.node(f'{layer_prefix}_async_back', f'Async All-to-All Back\n[tokens, 8192]\nAll GPUs {gpu_start}-{gpu_end-1}', 
                shape='ellipse', style='filled', fillcolor='lightsteelblue')
        
        # Output reconstruction
        dot.node(f'{layer_prefix}_reconstruct', f'Output Reconstruction\n[1024, 10000, 8192]\nGPU {gate_gpu}', 
                shape='parallelogram', style='filled', fillcolor='lightgreen')
        
        # Residual connection
        dot.node(f'{layer_prefix}_residual', f'Layer {layer_idx} Residual Add\n[1024, 10000, 8192]\nGPU {gate_gpu}', 
                shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Output node
    dot.node('output', 'Model Output\n[1024, 10000, 8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect the DAG
    dot.edge('input', 'layer_0_boundary')
    
    # Layer 0 connections
    dot.edge('layer_0_boundary', 'layer_0_gate')
    dot.edge('layer_0_gate', 'layer_0_token_shard')
    dot.edge('layer_0_token_shard', 'layer_0_async_comm')
    
    # Expert connections for layer 0
    for expert_idx in range(experts_per_layer):
        expert_gpu = expert_idx
        expert_prefix = f'layer_0_expert_{expert_idx}'
        
        # Gate to token routing (dashed)
        dot.edge('layer_0_gate', f'{expert_prefix}_receive', style='dashed')
        dot.edge('layer_0_async_comm', f'{expert_prefix}_receive')
        
        # Expert computation chain
        dot.edge(f'{expert_prefix}_receive', f'{expert_prefix}_linear1')
        dot.edge(f'{expert_prefix}_linear1', f'{expert_prefix}_gelu')
        dot.edge(f'{expert_prefix}_gelu', f'{expert_prefix}_linear2')
        dot.edge(f'{expert_prefix}_linear2', f'{expert_prefix}_send')
        dot.edge(f'{expert_prefix}_send', 'layer_0_async_back')
    
    dot.edge('layer_0_async_back', 'layer_0_token_agg')
    dot.edge('layer_0_token_agg', 'layer_0_reconstruct')
    dot.edge('layer_0_boundary', 'layer_0_residual', style='dotted')
    dot.edge('layer_0_reconstruct', 'layer_0_residual')
    dot.edge('layer_0_residual', 'layer_1_boundary')
    
    # Layer 1 connections
    dot.edge('layer_1_boundary', 'layer_1_gate')
    dot.edge('layer_1_gate', 'layer_1_token_shard')
    dot.edge('layer_1_token_shard', 'layer_1_async_comm')
    
    for expert_idx in range(experts_per_layer):
        expert_gpu = 16 + expert_idx
        expert_prefix = f'layer_1_expert_{expert_idx}'
        
        dot.edge('layer_1_gate', f'{expert_prefix}_receive', style='dashed')
        dot.edge('layer_1_async_comm', f'{expert_prefix}_receive')
        
        dot.edge(f'{expert_prefix}_receive', f'{expert_prefix}_linear1')
        dot.edge(f'{expert_prefix}_linear1', f'{expert_prefix}_gelu')
        dot.edge(f'{expert_prefix}_gelu', f'{expert_prefix}_linear2')
        dot.edge(f'{expert_prefix}_linear2', f'{expert_prefix}_send')
        dot.edge(f'{expert_prefix}_send', 'layer_1_async_back')
    
    dot.edge('layer_1_async_back', 'layer_1_token_agg')
    dot.edge('layer_1_token_agg', 'layer_1_reconstruct')
    dot.edge('layer_1_boundary', 'layer_1_residual', style='dotted')
    dot.edge('layer_1_reconstruct', 'layer_1_residual')
    dot.edge('layer_1_residual', 'layer_2_boundary')
    
    # Layer 2 connections
    dot.edge('layer_2_boundary', 'layer_2_gate')
    dot.edge('layer_2_gate', 'layer_2_token_shard')
    dot.edge('layer_2_token_shard', 'layer_2_async_comm')
    
    for expert_idx in range(experts_per_layer):
        expert_gpu = 32 + expert_idx
        expert_prefix = f'layer_2_expert_{expert_idx}'
        
        dot.edge('layer_2_gate', f'{expert_prefix}_receive', style='dashed')
        dot.edge('layer_2_async_comm', f'{expert_prefix}_receive')
        
        dot.edge(f'{expert_prefix}_receive', f'{expert_prefix}_linear1')
        dot.edge(f'{expert_prefix}_linear1', f'{expert_prefix}_gelu')
        dot.edge(f'{expert_prefix}_gelu', f'{expert_prefix}_linear2')
        dot.edge(f'{expert_prefix}_linear2', f'{expert_prefix}_send')
        dot.edge(f'{expert_prefix}_send', 'layer_2_async_back')
    
    dot.edge('layer_2_async_back', 'layer_2_token_agg')
    dot.edge('layer_2_token_agg', 'layer_2_reconstruct')
    dot.edge('layer_2_boundary', 'layer_2_residual', style='dotted')
    dot.edge('layer_2_reconstruct', 'layer_2_residual')
    dot.edge('layer_2_residual', 'layer_3_boundary')
    
    # Layer 3 connections
    dot.edge('layer_3_boundary', 'layer_3_gate')
    dot.edge('layer_3_gate', 'layer_3_token_shard')
    dot.edge('layer_3_token_shard', 'layer_3_async_comm')
    
    for expert_idx in range(experts_per_layer):
        expert_gpu = 48 + expert_idx
        expert_prefix = f'layer_3_expert_{expert_idx}'
        
        dot.edge('layer_3_gate', f'{expert_prefix}_receive', style='dashed')
        dot.edge('layer_3_async_comm', f'{expert_prefix}_receive')
        
        dot.edge(f'{expert_prefix}_receive', f'{expert_prefix}_linear1')
        dot.edge(f'{expert_prefix}_linear1', f'{expert_prefix}_gelu')
        dot.edge(f'{expert_prefix}_gelu', f'{expert_prefix}_linear2')
        dot.edge(f'{expert_prefix}_linear2', f'{expert_prefix}_send')
        dot.edge(f'{expert_prefix}_send', 'layer_3_async_back')
    
    dot.edge('layer_3_async_back', 'layer_3_token_agg')
    dot.edge('layer_3_token_agg', 'layer_3_reconstruct')
    dot.edge('layer_3_boundary', 'layer_3_residual', style='dotted')
    dot.edge('layer_3_reconstruct', 'layer_3_residual')
    dot.edge('layer_3_residual', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-09-10-35-11/proposed_moe_dag', format='svg', cleanup=True)
    dag.save('/home/wzc/data/file-share/2025-09-09-10-35-11/proposed_moe_dag.dot')
    print("Proposed MoE DAG generated successfully")