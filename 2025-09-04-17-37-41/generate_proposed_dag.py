#!/usr/bin/env python3
"""
Generate proposed cross-node expert parallelism DAG with 1 expert per GPU
"""

import graphviz

def generate_proposed_dag():
    dot = graphviz.Digraph('proposed_moe_dag', comment='Proposed Cross-Node Expert Parallelism DAG')
    dot.attr(rankdir='TB', size='30,40')
    
    # Input node
    dot.node('input', 'Input\n[1024, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Global routing for token distribution
    dot.node('global_router', 'Global Token Router\nRoute tokens to experts\nAll GPUs', 
             shape='parallelogram', style='filled', fillcolor='gold')
    
    # Process each layer with cross-node expert placement
    for layer_id in range(4):
        with dot.subgraph(name=f'layer_{layer_id}') as c:
            c.attr(label=f'MoE Layer {layer_id} - Cross-Node Expert Parallelism')
            
            # Attention across all GPUs (replicated for each expert)
            for expert_offset in range(16):
                gpu_id = layer_id * 16 + expert_offset
                
                # Attention components for this expert's GPU
                c.node(f'attn{layer_id}_q_{gpu_id}', f'Q Linear\n[1024, 8192]→[1024, 8192]\nGPU {gpu_id}', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
                c.node(f'attn{layer_id}_k_{gpu_id}', f'K Linear\n[1024, 8192]→[1024, 8192]\nGPU {gpu_id}', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
                c.node(f'attn{layer_id}_v_{gpu_id}', f'V Linear\n[1024, 8192]→[1024, 8192]\nGPU {gpu_id}', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
                
                c.node(f'attn{layer_id}_score_{gpu_id}', f'Attention Score\n[1024, 1024]\nGPU {gpu_id}', 
                       shape='rectangle', style='filled', fillcolor='yellow')
                c.node(f'attn{layer_id}_out_{gpu_id}', f'Attention Output\n[1024, 8192]\nGPU {gpu_id}', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Residual connection
                c.node(f'residual{layer_id}_{gpu_id}', f'Residual Add\n[1024, 8192]\nGPU {gpu_id}', 
                       shape='parallelogram', style='filled', fillcolor='orange')
                
                # Single expert per GPU
                expert_id = layer_id * 16 + expert_offset
                c.node(f'expert{layer_id}_{expert_id}', f'Expert {expert_id}\nMLP\n[1024, 32768]→[1024, 8192]\nGPU {gpu_id}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Gate for this expert
                c.node(f'gate{layer_id}_{gpu_id}', f'Gate {expert_id}\nSelect tokens\nGPU {gpu_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightpink')
                
                # Token aggregation after expert processing
                c.node(f'token_agg{layer_id}_{gpu_id}', f'Token Aggregation\n[1024, 8192]\nGPU {gpu_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightgray')
                
                # Final residual
                c.node(f'final_residual{layer_id}_{gpu_id}', f'Final Residual\n[1024, 8192]\nGPU {gpu_id}', 
                       shape='parallelogram', style='filled', fillcolor='orange')
    
    # Cross-node communication nodes
    for layer_id in range(4):
        dot.node(f'comm_layer{layer_id}', f'Cross-Node Communication\nLayer {layer_id}\nAsync Token Routing', 
                 shape='ellipse', style='filled', fillcolor='purple')
    
    # Output node
    dot.node('output', 'Output\n[1024, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connections
    # Input to global router
    dot.edge('input', 'global_router')
    
    # Layer 0 connections
    for gpu_id in range(16):
        # Global router distributes tokens
        dot.edge('global_router', f'gate0_{gpu_id}', style='dashed')
        dot.edge('global_router', f'attn0_q_{gpu_id}')
        dot.edge('global_router', f'attn0_k_{gpu_id}')
        dot.edge('global_router', f'attn0_v_{gpu_id}')
        
        # Attention computation
        dot.edge(f'attn0_q_{gpu_id}', f'attn0_score_{gpu_id}')
        dot.edge(f'attn0_k_{gpu_id}', f'attn0_score_{gpu_id}')
        dot.edge(f'attn0_v_{gpu_id}', f'attn0_out_{gpu_id}')
        dot.edge(f'attn0_score_{gpu_id}', f'attn0_out_{gpu_id}')
        
        # Residual connection
        dot.edge('global_router', f'residual0_{gpu_id}')
        dot.edge(f'attn0_out_{gpu_id}', f'residual0_{gpu_id}')
        
        # Gate to expert selection
        dot.edge(f'gate0_{gpu_id}', f'expert0_{gpu_id}', style='dashed')
        dot.edge(f'residual0_{gpu_id}', f'expert0_{gpu_id}')
        dot.edge(f'expert0_{gpu_id}', f'token_agg0_{gpu_id}')
        
        # Expert processing to final residual
        dot.edge(f'token_agg0_{gpu_id}', f'final_residual0_{gpu_id}')
        dot.edge(f'residual0_{gpu_id}', f'final_residual0_{gpu_id}')
        
        # Connect to communication node
        dot.edge(f'final_residual0_{gpu_id}', 'comm_layer0')
    
    # Layer 1 connections
    for gpu_id in range(16, 32):
        actual_id = gpu_id - 16
        
        # Cross-node communication
        dot.edge('comm_layer0', f'gate1_{gpu_id}')
        dot.edge('comm_layer0', f'attn1_q_{gpu_id}')
        dot.edge('comm_layer0', f'attn1_k_{gpu_id}')
        dot.edge('comm_layer0', f'attn1_v_{gpu_id}')
        
        # Attention computation
        dot.edge(f'attn1_q_{gpu_id}', f'attn1_score_{gpu_id}')
        dot.edge(f'attn1_k_{gpu_id}', f'attn1_score_{gpu_id}')
        dot.edge(f'attn1_v_{gpu_id}', f'attn1_out_{gpu_id}')
        dot.edge(f'attn1_score_{gpu_id}', f'attn1_out_{gpu_id}')
        
        # Residual connection
        dot.edge('comm_layer0', f'residual1_{gpu_id}')
        dot.edge(f'attn1_out_{gpu_id}', f'residual1_{gpu_id}')
        
        # Gate to expert
        dot.edge(f'gate1_{gpu_id}', f'expert1_{gpu_id}', style='dashed')
        dot.edge(f'residual1_{gpu_id}', f'expert1_{gpu_id}')
        dot.edge(f'expert1_{gpu_id}', f'token_agg1_{gpu_id}')
        
        # Final residual
        dot.edge(f'token_agg1_{gpu_id}', f'final_residual1_{gpu_id}')
        dot.edge(f'residual1_{gpu_id}', f'final_residual1_{gpu_id}')
        
        # Connect to communication
        dot.edge(f'final_residual1_{gpu_id}', 'comm_layer1')
    
    # Layer 2 connections
    for gpu_id in range(32, 48):
        actual_id = gpu_id - 32
        
        dot.edge('comm_layer1', f'gate2_{gpu_id}')
        dot.edge('comm_layer1', f'attn2_q_{gpu_id}')
        dot.edge('comm_layer1', f'attn2_k_{gpu_id}')
        dot.edge('comm_layer1', f'attn2_v_{gpu_id}')
        
        # Attention computation
        dot.edge(f'attn2_q_{gpu_id}', f'attn2_score_{gpu_id}')
        dot.edge(f'attn2_k_{gpu_id}', f'attn2_score_{gpu_id}')
        dot.edge(f'attn2_v_{gpu_id}', f'attn2_out_{gpu_id}')
        dot.edge(f'attn2_score_{gpu_id}', f'attn2_out_{gpu_id}')
        
        # Residual connection
        dot.edge('comm_layer1', f'residual2_{gpu_id}')
        dot.edge(f'attn2_out_{gpu_id}', f'residual2_{gpu_id}')
        
        # Gate to expert
        dot.edge(f'gate2_{gpu_id}', f'expert2_{gpu_id}', style='dashed')
        dot.edge(f'residual2_{gpu_id}', f'expert2_{gpu_id}')
        dot.edge(f'expert2_{gpu_id}', f'token_agg2_{gpu_id}')
        
        # Final residual
        dot.edge(f'token_agg2_{gpu_id}', f'final_residual2_{gpu_id}')
        dot.edge(f'residual2_{gpu_id}', f'final_residual2_{gpu_id}')
        
        # Connect to communication
        dot.edge(f'final_residual2_{gpu_id}', 'comm_layer2')
    
    # Layer 3 connections
    for gpu_id in range(48, 64):
        actual_id = gpu_id - 48
        
        dot.edge('comm_layer2', f'gate3_{gpu_id}')
        dot.edge('comm_layer2', f'attn3_q_{gpu_id}')
        dot.edge('comm_layer2', f'attn3_k_{gpu_id}')
        dot.edge('comm_layer2', f'attn3_v_{gpu_id}')
        
        # Attention computation
        dot.edge(f'attn3_q_{gpu_id}', f'attn3_score_{gpu_id}')
        dot.edge(f'attn3_k_{gpu_id}', f'attn3_score_{gpu_id}')
        dot.edge(f'attn3_v_{gpu_id}', f'attn3_out_{gpu_id}')
        dot.edge(f'attn3_score_{gpu_id}', f'attn3_out_{gpu_id}')
        
        # Residual connection
        dot.edge('comm_layer2', f'residual3_{gpu_id}')
        dot.edge(f'attn3_out_{gpu_id}', f'residual3_{gpu_id}')
        
        # Gate to expert
        dot.edge(f'gate3_{gpu_id}', f'expert3_{gpu_id}', style='dashed')
        dot.edge(f'residual3_{gpu_id}', f'expert3_{gpu_id}')
        dot.edge(f'expert3_{gpu_id}', f'token_agg3_{gpu_id}')
        
        # Final residual
        dot.edge(f'token_agg3_{gpu_id}', f'final_residual3_{gpu_id}')
        dot.edge(f'residual3_{gpu_id}', f'final_residual3_{gpu_id}')
        
        # Connect to output
        dot.edge(f'final_residual3_{gpu_id}', 'output')
    
    return dot

if __name__ == '__main__':
    dag = generate_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-04-17-37-41/proposed_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-04-17-37-41/proposed_moe_dag.dot')
    print("Proposed DAG generated successfully")