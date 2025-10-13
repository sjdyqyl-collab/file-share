#!/usr/bin/env python3

import os
from graphviz import Digraph

def create_proposed_dag():
    """Create DAG for proposed EP=16 configuration with one expert per GPU"""
    
    dot = Digraph(comment='Proposed EP=16 Cross-Node Expert Parallelism DAG')
    dot.attr(rankdir='TB', size='40,25')
    
    # Define colors
    layer_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    comm_color = 'orange'
    gate_color = 'lightpink'
    
    # Input node
    dot.node('input', 'Input\n[batch_size=1024, seq_len=10000, token_dim=8192]', 
             shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Process each layer
    for layer in range(4):
        layer_color = layer_colors[layer]
        
        with dot.subgraph(name=f'cluster_layer{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer} (16 Experts across 16 GPUs)', style='rounded', fillcolor=layer_color)
            
            # Multi-Head Attention (shared across all GPUs)
            layer_cluster.node(f'mha_l{layer}', f'Multi-Head Attention Layer {layer}\n[batch_size=1024, seq_len=10000, heads=16, d_k=512]\nAll GPUs', 
                              shape='rectangle', style='filled', fillcolor=layer_color)
            
            # Gating mechanism for expert selection
            layer_cluster.node(f'gate_l{layer}', f'Gating Network Layer {layer}\n[batch_size=1024, seq_len=10000, experts=16]\nAll GPUs', 
                              shape='parallelogram', style='filled, dashed', fillcolor=gate_color)
            
            # Expert routing and communication
            layer_cluster.node(f'route_l{layer}', f'Token Routing Layer {layer}\n[batch_size=1024, seq_len=10000, top_k=2]\nAll GPUs', 
                              shape='parallelogram', style='filled', fillcolor=comm_color)
            
            # Individual experts on each GPU
            for expert_id in range(16):
                gpu_id = expert_id
                layer_cluster.node(f'expert_l{layer}_e{expert_id}', 
                                  f'Expert {expert_id} Layer {layer}\n[batch_size=variable, token_dim=8192]\nGPU: {gpu_id}', 
                                  shape='rectangle', style='filled', fillcolor=layer_color)
            
            # Expert aggregation
            layer_cluster.node(f'agg_l{layer}', f'Expert Aggregation Layer {layer}\n[batch_size=1024, seq_len=10000, token_dim=8192]\nAll GPUs', 
                              shape='parallelogram', style='filled', fillcolor=comm_color)
            
            # Residual connections
            layer_cluster.node(f'residual_l{layer}', f'Residual Add Layer {layer}\n[batch_size=1024, seq_len=10000, token_dim=8192]\nAll GPUs', 
                              shape='parallelogram', style='filled', fillcolor=layer_color)
            
            # Layer normalization
            layer_cluster.node(f'layernorm_l{layer}', f'LayerNorm Layer {layer}\n[batch_size=1024, seq_len=10000, token_dim=8192]\nAll GPUs', 
                              shape='rectangle', style='filled', fillcolor=layer_color)
    
    # Output node
    dot.node('output', 'Output\n[batch_size=1024, seq_len=10000, token_dim=8192]', 
             shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Connect the nodes
    # Input to first layer
    dot.edge('input', 'mha_l0')
    
    # Layer 0 connections
    dot.edge('mha_l0', 'gate_l0')
    dot.edge('gate_l0', 'route_l0', style='dashed')
    dot.edge('route_l0', 'expert_l0_e0')
    dot.edge('route_l0', 'expert_l0_e1')
    dot.edge('route_l0', 'expert_l0_e2')
    dot.edge('route_l0', 'expert_l0_e3')
    dot.edge('route_l0', 'expert_l0_e4')
    dot.edge('route_l0', 'expert_l0_e5')
    dot.edge('route_l0', 'expert_l0_e6')
    dot.edge('route_l0', 'expert_l0_e7')
    dot.edge('route_l0', 'expert_l0_e8')
    dot.edge('route_l0', 'expert_l0_e9')
    dot.edge('route_l0', 'expert_l0_e10')
    dot.edge('route_l0', 'expert_l0_e11')
    dot.edge('route_l0', 'expert_l0_e12')
    dot.edge('route_l0', 'expert_l0_e13')
    dot.edge('route_l0', 'expert_l0_e14')
    dot.edge('route_l0', 'expert_l0_e15')
    
    # Connect experts to aggregation
    for expert_id in range(16):
        dot.edge(f'expert_l0_e{expert_id}', 'agg_l0')
    
    dot.edge('agg_l0', 'residual_l0')
    dot.edge('mha_l0', 'residual_l0')
    dot.edge('residual_l0', 'layernorm_l0')
    
    # Continue for remaining layers
    for layer in range(1, 4):
        prev_norm = f'layernorm_l{layer-1}'
        dot.edge(prev_norm, f'mha_l{layer}')
        dot.edge(f'mha_l{layer}', f'gate_l{layer}')
        dot.edge(f'gate_l{layer}', f'route_l{layer}', style='dashed')
        
        # Connect routing to all experts
        for expert_id in range(16):
            dot.edge(f'route_l{layer}', f'expert_l{layer}_e{expert_id}')
        
        # Connect experts to aggregation
        for expert_id in range(16):
            dot.edge(f'expert_l{layer}_e{expert_id}', f'agg_l{layer}')
        
        dot.edge(f'agg_l{layer}', f'residual_l{layer}')
        dot.edge(f'mha_l{layer}', f'residual_l{layer}')
        dot.edge(f'residual_l{layer}', f'layernorm_l{layer}')
    
    # Final output
    dot.edge('layernorm_l3', 'output')
    
    return dot

def create_detailed_proposed_dag():
    """Create a more detailed DAG showing token flow and communication"""
    
    dot = Digraph(comment='Detailed EP=16 Expert Parallelism with Token Flow')
    dot.attr(rankdir='TB', size='50,30')
    
    # Colors
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    
    # Input
    dot.node('input', 'Input Tokens\n[batch_size=1024, seq_len=10000, token_dim=8192]', 
             shape='ellipse', style='filled', fillcolor='lightgray')
    
    for layer in range(4):
        color = colors[layer]
        
        # Layer structure
        with dot.subgraph(name=f'layer{layer}') as layer_sg:
            layer_sg.attr(label=f'Layer {layer}', style='rounded', fillcolor=color)
            
            # MHA computation
            layer_sg.node(f'mha_{layer}', f'MHA\n[1024×10000×8192]\nAll GPUs', 
                          shape='rectangle', style='filled', fillcolor=color)
            
            # Gating computation
            layer_sg.node(f'gate_{layer}', f'Gating Scores\n[1024×10000×16]\nAll GPUs', 
                          shape='parallelogram', style='filled', fillcolor='lightpink')
            
            # Token batching
            layer_sg.node(f'batch_{layer}', f'Token Batching\nby Destination Expert\nAll GPUs', 
                          shape='parallelogram', style='filled', fillcolor='orange')
            
            # Communication nodes
            for gpu in range(16):
                layer_sg.node(f'comm_{layer}_{gpu}', 
                              f'Token Transfer\nto GPU {gpu}\n[variable batch]\nAsync', 
                              shape='ellipse', style='filled', fillcolor='yellow')
            
            # Expert computation nodes
            for gpu in range(16):
                layer_sg.node(f'expert_{layer}_{gpu}', 
                              f'Expert {gpu}\n[variable batch×8192]\nGPU {gpu}', 
                              shape='rectangle', style='filled', fillcolor=color)
            
            # Gather tokens
            layer_sg.node(f'gather_{layer}', f'Gather Results\n[1024×10000×8192]\nAll GPUs', 
                          shape='parallelogram', style='filled', fillcolor='orange')
            
            # Residual and norm
            layer_sg.node(f'residual_{layer}', f'Residual+Norm\n[1024×10000×8192]\nAll GPUs', 
                          shape='rectangle', style='filled', fillcolor=color)
    
    # Output
    dot.node('output', 'Output\n[1024×10000×8192]', 
             shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Connect everything
    dot.edge('input', 'mha_0')
    
    for layer in range(4):
        # MHA to gating
        dot.edge(f'mha_{layer}', f'gate_{layer}')
        
        # Gating to batching
        dot.edge(f'gate_{layer}', f'batch_{layer}', style='dashed')
        
        # Batching to communication
        for gpu in range(16):
            dot.edge(f'batch_{layer}', f'comm_{layer}_{gpu}')
        
        # Communication to experts
        for gpu in range(16):
            dot.edge(f'comm_{layer}_{gpu}', f'expert_{layer}_{gpu}')
        
        # Experts to gather
        for gpu in range(16):
            dot.edge(f'expert_{layer}_{gpu}', f'gather_{layer}')
        
        # Gather to residual
        dot.edge(f'gather_{layer}', f'residual_{layer}')
        dot.edge(f'mha_{layer}', f'residual_{layer}')
        
        # Connect to next layer
        if layer < 3:
            dot.edge(f'residual_{layer}', f'mha_{layer+1}')
        else:
            dot.edge(f'residual_{layer}', 'output')
    
    return dot

if __name__ == "__main__":
    # Create both versions
    dag1 = create_proposed_dag()
    dag1.render('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/proposed_dag', format='svg', cleanup=False)
    dag1.save('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/proposed_dag.dot')
    
    dag2 = create_detailed_proposed_dag()
    dag2.render('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/proposed_detailed_dag', format='svg', cleanup=False)
    dag2.save('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/proposed_detailed_dag.dot')
    
    print("Proposed DAGs created successfully")