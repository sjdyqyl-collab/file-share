#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_proposed_dag():
    """Create proposed DAG with 64 GPUs, EP=64, 1 expert/GPU"""
    dot = Digraph(comment='Proposed MoE Deployment - 64 GPUs (EP=64)')
    dot.attr(rankdir='TB', splines='ortho')
    
    # Global input
    dot.node('input', 'Input\n(batch=1024, seq=10K, hidden=32768)', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Expert placement: 16 experts/layer × 4 layers = 64 experts total
    # Each GPU hosts exactly 1 expert across all layers
    # GPU 0-15: Layer 0 experts 0-15
    # GPU 16-31: Layer 1 experts 0-15  
    # GPU 32-47: Layer 2 experts 0-15
    # GPU 48-63: Layer 3 experts 0-15
    
    # Layer 0: GPUs 0-15, each with 1 expert
    with dot.subgraph(name='cluster_layer_0') as c0:
        c0.attr(label='Layer 0\n16 Experts across GPUs 0-15', style='dashed')
        
        # MHA for layer 0 (replicated across all GPUs for tensor parallelism)
        mha_0 = 'mha_0_all'
        c0.node(mha_0, 'MHA Layer 0\nAll GPUs 0-15\nInput: (1024, 10K, 32768)\nOutput: (1024, 10K, 32768)', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Expert selection gate for layer 0
        gate_0 = 'gate_0_all'
        c0.node(gate_0, 'Gate Layer 0\nAll GPUs\nSelect top-2 experts from 16', 
                shape='parallelogram', style='filled', fillcolor='yellow')
        
        # 16 experts, one per GPU
        for expert_id in range(16):
            gpu_id = expert_id
            expert_name = f'expert_0_{expert_id}_gpu_{gpu_id}'
            c0.node(expert_name, f'Expert {expert_id}\nLayer 0\nGPU {gpu_id}\nInput: (tokens, 32768)\nOutput: (tokens, 32768)', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # FFN aggregation for layer 0
        ffn_0 = 'ffn_0_all'
        c0.node(ffn_0, 'FFN Layer 0\nAll GPUs\nAggregate 2 selected experts', 
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # Residual connection for layer 0
        residual_0 = 'residual_0_all'
        c0.node(residual_0, 'Residual Add\nLayer 0\nAll GPUs\nInput: (1024, 10K, 32768)', 
                shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Layer 1: GPUs 16-31, each with 1 expert
    with dot.subgraph(name='cluster_layer_1') as c1:
        c1.attr(label='Layer 1\n16 Experts across GPUs 16-31', style='dashed')
        
        # MHA for layer 1
        mha_1 = 'mha_1_all'
        c1.node(mha_1, 'MHA Layer 1\nAll GPUs 16-31\nInput: (1024, 10K, 32768)\nOutput: (1024, 10K, 32768)', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Expert selection gate for layer 1
        gate_1 = 'gate_1_all'
        c1.node(gate_1, 'Gate Layer 1\nAll GPUs\nSelect top-2 experts from 16', 
                shape='parallelogram', style='filled', fillcolor='yellow')
        
        # 16 experts, one per GPU
        for expert_id in range(16):
            gpu_id = expert_id + 16
            expert_name = f'expert_1_{expert_id}_gpu_{gpu_id}'
            c1.node(expert_name, f'Expert {expert_id}\nLayer 1\nGPU {gpu_id}\nInput: (tokens, 32768)\nOutput: (tokens, 32768)', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # FFN aggregation for layer 1
        ffn_1 = 'ffn_1_all'
        c1.node(ffn_1, 'FFN Layer 1\nAll GPUs\nAggregate 2 selected experts', 
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # Residual connection for layer 1
        residual_1 = 'residual_1_all'
        c1.node(residual_1, 'Residual Add\nLayer 1\nAll GPUs\nInput: (1024, 10K, 32768)', 
                shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Layer 2: GPUs 32-47, each with 1 expert
    with dot.subgraph(name='cluster_layer_2') as c2:
        c2.attr(label='Layer 2\n16 Experts across GPUs 32-47', style='dashed')
        
        # MHA for layer 2
        mha_2 = 'mha_2_all'
        c2.node(mha_2, 'MHA Layer 2\nAll GPUs 32-47\nInput: (1024, 10K, 32768)\nOutput: (1024, 10K, 32768)', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Expert selection gate for layer 2
        gate_2 = 'gate_2_all'
        c2.node(gate_2, 'Gate Layer 2\nAll GPUs\nSelect top-2 experts from 16', 
                shape='parallelogram', style='filled', fillcolor='yellow')
        
        # 16 experts, one per GPU
        for expert_id in range(16):
            gpu_id = expert_id + 32
            expert_name = f'expert_2_{expert_id}_gpu_{gpu_id}'
            c2.node(expert_name, f'Expert {expert_id}\nLayer 2\nGPU {gpu_id}\nInput: (tokens, 32768)\nOutput: (tokens, 32768)', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # FFN aggregation for layer 2
        ffn_2 = 'ffn_2_all'
        c2.node(ffn_2, 'FFN Layer 2\nAll GPUs\nAggregate 2 selected experts', 
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # Residual connection for layer 2
        residual_2 = 'residual_2_all'
        c2.node(residual_2, 'Residual Add\nLayer 2\nAll GPUs\nInput: (1024, 10K, 32768)', 
                shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Layer 3: GPUs 48-63, each with 1 expert
    with dot.subgraph(name='cluster_layer_3') as c3:
        c3.attr(label='Layer 3\n16 Experts across GPUs 48-63', style='dashed')
        
        # MHA for layer 3
        mha_3 = 'mha_3_all'
        c3.node(mha_3, 'MHA Layer 3\nAll GPUs 48-63\nInput: (1024, 10K, 32768)\nOutput: (1024, 10K, 32768)', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Expert selection gate for layer 3
        gate_3 = 'gate_3_all'
        c3.node(gate_3, 'Gate Layer 3\nAll GPUs\nSelect top-2 experts from 16', 
                shape='parallelogram', style='filled', fillcolor='yellow')
        
        # 16 experts, one per GPU
        for expert_id in range(16):
            gpu_id = expert_id + 48
            expert_name = f'expert_3_{expert_id}_gpu_{gpu_id}'
            c3.node(expert_name, f'Expert {expert_id}\nLayer 3\nGPU {gpu_id}\nInput: (tokens, 32768)\nOutput: (tokens, 32768)', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # FFN aggregation for layer 3
        ffn_3 = 'ffn_3_all'
        c3.node(ffn_3, 'FFN Layer 3\nAll GPUs\nAggregate 2 selected experts', 
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # Residual connection for layer 3
        residual_3 = 'residual_3_all'
        c3.node(residual_3, 'Residual Add\nLayer 3\nAll GPUs\nInput: (1024, 10K, 32768)', 
                shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Global output
    dot.node('output', 'Output\n(batch=1024, seq=10K, hidden=32768)', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Communication nodes for expert routing
    # These represent the cross-GPU communication for token routing
    
    # Layer 0 routing
    route_0 = 'route_0'
    dot.node(route_0, 'Expert Routing\nLayer 0\nCross-GPU Communication\nSend tokens to selected experts', 
             shape='ellipse', style='filled', fillcolor='pink')
    
    # Layer 1 routing  
    route_1 = 'route_1'
    dot.node(route_1, 'Expert Routing\nLayer 1\nCross-GPU Communication\nSend tokens to selected experts', 
             shape='ellipse', style='filled', fillcolor='pink')
    
    # Layer 2 routing
    route_2 = 'route_2'
    dot.node(route_2, 'Expert Routing\nLayer 2\nCross-GPU Communication\nSend tokens to selected experts', 
             shape='ellipse', style='filled', fillcolor='pink')
    
    # Layer 3 routing
    route_3 = 'route_3'
    dot.node(route_3, 'Expert Routing\nLayer 3\nCross-GPU Communication\nSend tokens to selected experts', 
             shape='ellipse', style='filled', fillcolor='pink')
    
    # Add connections
    # Input to first layer
    dot.edge('input', 'mha_0_all')
    dot.edge('mha_0_all', 'gate_0_all')
    dot.edge('gate_0_all', 'route_0')
    
    # Layer 0 expert routing
    for expert_id in range(16):
        gpu_id = expert_id
        dot.edge('route_0', f'expert_0_{expert_id}_gpu_{gpu_id}')
        dot.edge(f'expert_0_{expert_id}_gpu_{gpu_id}', 'ffn_0_all')
    
    dot.edge('ffn_0_all', 'residual_0_all')
    dot.edge('mha_0_all', 'residual_0_all')
    dot.edge('residual_0_all', 'mha_1_all')
    
    # Layer 1 connections
    dot.edge('mha_1_all', 'gate_1_all')
    dot.edge('gate_1_all', 'route_1')
    
    for expert_id in range(16):
        gpu_id = expert_id + 16
        dot.edge('route_1', f'expert_1_{expert_id}_gpu_{gpu_id}')
        dot.edge(f'expert_1_{expert_id}_gpu_{gpu_id}', 'ffn_1_all')
    
    dot.edge('ffn_1_all', 'residual_1_all')
    dot.edge('mha_1_all', 'residual_1_all')
    dot.edge('residual_1_all', 'mha_2_all')
    
    # Layer 2 connections
    dot.edge('mha_2_all', 'gate_2_all')
    dot.edge('gate_2_all', 'route_2')
    
    for expert_id in range(16):
        gpu_id = expert_id + 32
        dot.edge('route_2', f'expert_2_{expert_id}_gpu_{gpu_id}')
        dot.edge(f'expert_2_{expert_id}_gpu_{gpu_id}', 'ffn_2_all')
    
    dot.edge('ffn_2_all', 'residual_2_all')
    dot.edge('mha_2_all', 'residual_2_all')
    dot.edge('residual_2_all', 'mha_3_all')
    
    # Layer 3 connections
    dot.edge('mha_3_all', 'gate_3_all')
    dot.edge('gate_3_all', 'route_3')
    
    for expert_id in range(16):
        gpu_id = expert_id + 48
        dot.edge('route_3', f'expert_3_{expert_id}_gpu_{gpu_id}')
        dot.edge(f'expert_3_{expert_id}_gpu_{gpu_id}', 'ffn_3_all')
    
    dot.edge('ffn_3_all', 'residual_3_all')
    dot.edge('mha_3_all', 'residual_3_all')
    dot.edge('residual_3_all', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-05-16-00-24/proposed_moe_deployment', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-05-16-00-24/proposed_moe_deployment.dot')
    print("Proposed DAG generated successfully")