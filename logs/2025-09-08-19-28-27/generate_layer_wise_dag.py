#!/usr/bin/env python3
"""
Generate DAG for the proposed layer-wise deployment strategy
16 layers distributed across 16 GPUs, 1 layer per GPU
"""

import graphviz

def create_layer_wise_dag():
    dot = graphviz.Digraph('layer_wise_deployment', format='svg')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Model Input\nINPUT DIMENSION=[1024, seq_len, 8192]\nGPU=all', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Communication nodes (parallelogram style)
    dot.attr('node', shape='parallelogram', fillcolor='lightyellow')
    
    # Layer nodes (rectangle style)
    dot.attr('node', shape='rectangle', fillcolor='lightcoral')
    
    # Create 16 layers, each on separate GPU
    prev_node = 'input'
    
    for layer_id in range(16):
        gpu_id = layer_id
        
        # Layer norm (pre-attention)
        ln1_name = f'layer{layer_id}_ln1'
        dot.node(ln1_name, f'LayerNorm\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPU={gpu_id}')
        dot.edge(prev_node, ln1_name)
        
        # Multi-head attention
        attn_name = f'layer{layer_id}_attn'
        dot.node(attn_name, f'MultiHeadAttention\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPU={gpu_id}')
        dot.edge(ln1_name, attn_name)
        
        # Residual add after attention
        residual1_name = f'layer{layer_id}_residual1'
        dot.node(residual1_name, f'ResidualAdd\nINPUT1=[1024, seq_len, 8192]\nINPUT2=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPU={gpu_id}')
        dot.edge(ln1_name, residual1_name)
        dot.edge(attn_name, residual1_name)
        
        # Layer norm (pre-FFN)
        ln2_name = f'layer{layer_id}_ln2'
        dot.node(ln2_name, f'LayerNorm\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPU={gpu_id}')
        dot.edge(residual1_name, ln2_name)
        
        # MLP/FFN
        ffn_name = f'layer{layer_id}_ffn'
        dot.node(ffn_name, f'FeedForward\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPU={gpu_id}')
        dot.edge(ln2_name, ffn_name)
        
        # Residual add after FFN
        residual2_name = f'layer{layer_id}_residual2'
        dot.node(residual2_name, f'ResidualAdd\nINPUT1=[1024, seq_len, 8192]\nINPUT2=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPU={gpu_id}')
        dot.edge(residual1_name, residual2_name)
        dot.edge(ffn_name, residual2_name)
        
        # Communication node between layers (if not last layer)
        if layer_id < 15:
            comm_name = f'comm{layer_id}_to_{layer_id+1}'
            dot.node(comm_name, f'P2P Transfer\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nFROM_GPU={gpu_id}\nTO_GPU={layer_id+1}', 
                     shape='parallelogram', fillcolor='lightyellow')
            dot.edge(residual2_name, comm_name)
            prev_node = comm_name
        else:
            prev_node = residual2_name
    
    # Output node
    dot.node('output', 'Model Output\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPU=15', 
             shape='ellipse', fillcolor='lightgreen')
    dot.edge(prev_node, 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_layer_wise_dag()
    dag.render('/home/wzc/data/file-share/2025-09-08-19-28-27/layer_wise_deployment', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-08-19-28-27/layer_wise_deployment.dot')
    print("Layer-wise deployment DAG generated successfully")