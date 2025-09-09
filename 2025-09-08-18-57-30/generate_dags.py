#!/usr/bin/env python3

import os
from graphviz import Digraph

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2 across 16 GPUs"""
    dot = Digraph(name='baseline_dag', comment='Baseline: TP=8, PP=2')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\n[B=1024, H=8192]', shape='ellipse', fillcolor='lightgreen')
    
    # Create 16 layers with tensor parallelism
    for layer in range(1, 17):
        stage = 1 if layer <= 8 else 2
        
        # Layer input
        dot.node(f'layer{layer}_input', f'Layer {layer} Input\n[B=1024, H=8192]', 
                shape='ellipse', fillcolor='lightyellow')
        
        if layer == 1:
            dot.edge('input', 'layer1_input')
        else:
            if layer == 9:  # Cross-stage communication
                dot.node(f'comm_stage1_2', 'Stage 1→2\n[B=1024, H=8192]', 
                        shape='parallelogram', fillcolor='orange', style='dashed')
                dot.edge(f'layer{layer-1}_output', f'comm_stage1_2')
                dot.edge(f'comm_stage1_2', f'layer{layer}_input')
            else:
                dot.edge(f'layer{layer-1}_output', f'layer{layer}_input')
        
        # Tensor parallel components for each layer
        for tp in range(8):
            gpu_id = (layer - 1) * 8 + tp if stage == 1 else (layer - 9) * 8 + tp
            
            # Attention components
            dot.node(f'layer{layer}_attn_q_{tp}', f'Q Projection {tp}\n[B=1024, H=1024]\nGPU{gpu_id}', 
                    fillcolor='lightcoral')
            dot.node(f'layer{layer}_attn_k_{tp}', f'K Projection {tp}\n[B=1024, H=1024]\nGPU{gpu_id}', 
                    fillcolor='lightcoral')
            dot.node(f'layer{layer}_attn_v_{tp}', f'V Projection {tp}\n[B=1024, H=1024]\nGPU{gpu_id}', 
                    fillcolor='lightcoral')
            
            dot.node(f'layer{layer}_attn_out_{tp}', f'Attention Output {tp}\n[B=1024, H=1024]\nGPU{gpu_id}', 
                    fillcolor='lightcoral')
            
            # MLP components
            dot.node(f'layer{layer}_mlp_fc1_{tp}', f'MLP FC1 {tp}\n[B=1024, H=4096]\nGPU{gpu_id}', 
                    fillcolor='lightcyan')
            dot.node(f'layer{layer}_mlp_fc2_{tp}', f'MLP FC2 {tp}\n[B=1024, H=1024]\nGPU{gpu_id}', 
                    fillcolor='lightcyan')
            
            # Residual connections
            dot.node(f'layer{layer}_res1_{tp}', f'Residual Add 1 {tp}\n[B=1024, H=1024]\nGPU{gpu_id}', 
                    shape='diamond', fillcolor='lightpink')
            dot.node(f'layer{layer}_res2_{tp}', f'Residual Add 2 {tp}\n[B=1024, H=1024]\nGPU{gpu_id}', 
                    shape='diamond', fillcolor='lightpink')
            
            # Connections within tensor parallel group
            dot.edge(f'layer{layer}_input', f'layer{layer}_attn_q_{tp}')
            dot.edge(f'layer{layer}_input', f'layer{layer}_attn_k_{tp}')
            dot.edge(f'layer{layer}_input', f'layer{layer}_attn_v_{tp}')
            
            # Add communication nodes for tensor parallelism
            dot.node(f'layer{layer}_attn_allreduce_{tp}', f'All-Reduce\n[B=1024, H=1024]\nTP Group', 
                    shape='parallelogram', fillcolor='orange', style='dashed')
            dot.node(f'layer{layer}_mlp_allreduce_{tp}', f'All-Reduce\n[B=1024, H=1024]\nTP Group', 
                    shape='parallelogram', fillcolor='orange', style='dashed')
            
            dot.edge(f'layer{layer}_attn_out_{tp}', f'layer{layer}_res1_{tp}')
            dot.edge(f'layer{layer}_input', f'layer{layer}_res1_{tp}')  # Residual connection
            dot.edge(f'layer{layer}_res1_{tp}', f'layer{layer}_mlp_fc1_{tp}')
            dot.edge(f'layer{layer}_mlp_fc2_{tp}', f'layer{layer}_res2_{tp}')
            dot.edge(f'layer{layer}_res1_{tp}', f'layer{layer}_res2_{tp}')  # Residual connection
            
            if tp == 7:  # Last TP group member
                dot.node(f'layer{layer}_output', f'Layer {layer} Output\n[B=1024, H=8192]', 
                        shape='ellipse', fillcolor='lightyellow')
                dot.edge(f'layer{layer}_res2_{tp}', f'layer{layer}_output')
    
    # Output node
    dot.node('output', 'Final Output\n[B=1024, H=8192]', shape='ellipse', fillcolor='lightgreen')
    dot.edge('layer16_output', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with 1 layer per GPU"""
    dot = Digraph(name='proposed_dag', comment='Proposed: 1 Layer per GPU')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\n[B=1024, H=8192]', shape='ellipse', fillcolor='lightgreen')
    
    # Create 16 layers, each on a separate GPU
    for layer in range(1, 17):
        gpu_id = layer - 1
        
        # Layer input
        if layer == 1:
            dot.node(f'layer{layer}_input', f'Layer {layer} Input\n[B=1024, H=8192]\nGPU{gpu_id}', 
                    shape='ellipse', fillcolor='lightyellow')
            dot.edge('input', f'layer{layer}_input')
        else:
            # Communication between GPUs
            dot.node(f'comm_{layer-1}_{layer}', f'GPU{layer-2}→GPU{gpu_id}\n[B=1024, H=8192]', 
                    shape='parallelogram', fillcolor='orange', style='dashed')
            dot.node(f'layer{layer}_input', f'Layer {layer} Input\n[B=1024, H=8192]\nGPU{gpu_id}', 
                    shape='ellipse', fillcolor='lightyellow')
            dot.edge(f'layer{layer-1}_output', f'comm_{layer-1}_{layer}')
            dot.edge(f'comm_{layer-1}_{layer}', f'layer{layer}_input')
        
        # Complete layer on single GPU
        dot.node(f'layer{layer}_attn_q', f'Q Projection\n[B=1024, H=8192]\nGPU{gpu_id}', 
                fillcolor='lightcoral')
        dot.node(f'layer{layer}_attn_k', f'K Projection\n[B=1024, H=8192]\nGPU{gpu_id}', 
                fillcolor='lightcoral')
        dot.node(f'layer{layer}_attn_v', f'V Projection\n[B=1024, H=8192]\nGPU{gpu_id}', 
                fillcolor='lightcoral')
        
        dot.node(f'layer{layer}_attn_out', f'Attention Output\n[B=1024, H=8192]\nGPU{gpu_id}', 
                fillcolor='lightcoral')
        
        dot.node(f'layer{layer}_mlp_fc1', f'MLP FC1\n[B=1024, H=32768]\nGPU{gpu_id}', 
                fillcolor='lightcyan')
        dot.node(f'layer{layer}_mlp_fc2', f'MLP FC2\n[B=1024, H=8192]\nGPU{gpu_id}', 
                fillcolor='lightcyan')
        
        # Residual connections
        dot.node(f'layer{layer}_res1', f'Residual Add 1\n[B=1024, H=8192]\nGPU{gpu_id}', 
                shape='diamond', fillcolor='lightpink')
        dot.node(f'layer{layer}_res2', f'Residual Add 2\n[B=1024, H=8192]\nGPU{gpu_id}', 
                shape='diamond', fillcolor='lightpink')
        
        # Layer output
        dot.node(f'layer{layer}_output', f'Layer {layer} Output\n[B=1024, H=8192]\nGPU{gpu_id}', 
                shape='ellipse', fillcolor='lightyellow')
        
        # Connections within layer
        dot.edge(f'layer{layer}_input', f'layer{layer}_attn_q')
        dot.edge(f'layer{layer}_input', f'layer{layer}_attn_k')
        dot.edge(f'layer{layer}_input', f'layer{layer}_attn_v')
        
        dot.edge(f'layer{layer}_attn_q', f'layer{layer}_attn_out')
        dot.edge(f'layer{layer}_attn_k', f'layer{layer}_attn_out')
        dot.edge(f'layer{layer}_attn_v', f'layer{layer}_attn_out')
        
        dot.edge(f'layer{layer}_attn_out', f'layer{layer}_res1')
        dot.edge(f'layer{layer}_input', f'layer{layer}_res1')  # Residual connection
        
        dot.edge(f'layer{layer}_res1', f'layer{layer}_mlp_fc1')
        dot.edge(f'layer{layer}_mlp_fc1', f'layer{layer}_mlp_fc2')
        dot.edge(f'layer{layer}_mlp_fc2', f'layer{layer}_res2')
        dot.edge(f'layer{layer}_res1', f'layer{layer}_res2')  # Residual connection
        dot.edge(f'layer{layer}_res2', f'layer{layer}_output')
    
    # Output node
    dot.node('output', 'Final Output\n[B=1024, H=8192]', shape='ellipse', fillcolor='lightgreen')
    dot.edge('layer16_output', 'output')
    
    return dot

if __name__ == "__main__":
    # Create baseline DAG
    baseline = create_baseline_dag()
    baseline.render('/home/wzc/data/file-share/2025-09-08-18-57-30/baseline_dag', format='svg', cleanup=False)
    baseline.save('/home/wzc/data/file-share/2025-09-08-18-57-30/baseline_dag.dot')
    
    # Create proposed DAG
    proposed = create_proposed_dag()
    proposed.render('/home/wzc/data/file-share/2025-09-08-18-57-30/proposed_dag', format='svg', cleanup=False)
    proposed.save('/home/wzc/data/file-share/2025-09-08-18-57-30/proposed_dag.dot')
    
    print("DAGs generated successfully!")
    print("Files created:")
    print("- baseline_dag.svg")
    print("- baseline_dag.dot")
    print("- proposed_dag.svg")
    print("- proposed_dag.dot")