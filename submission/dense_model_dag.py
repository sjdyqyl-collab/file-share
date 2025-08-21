#!/usr/bin/env python3

import graphviz

def create_dense_model_dag():
    """Create DAG for 16-layer dense model deployed across 16 GPUs"""
    
    dot = graphviz.Digraph('dense_model_16_gpu', format='svg')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Input node
    dot.node('input', 'Input\n[Batch: 1024, Seq: 512, Hidden: 8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Create nodes for each layer on each GPU
    for layer in range(1, 17):
        gpu_id = layer - 1  # GPU 0 to 15
        
        # Layer input aggregation
        if layer > 1:
            dot.node(f'agg_{layer}', f'Aggregate\nGPU {gpu_id-1}→{gpu_id}', 
                     shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Multi-Head Attention
        dot.node(f'mha_q_{layer}', f'MHA Query\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.node(f'mha_k_{layer}', f'MHA Key\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.node(f'mha_v_{layer}', f'MHA Value\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.node(f'mha_attn_{layer}', f'MHA Attention\n[1024×16×512×512]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.node(f'mha_out_{layer}', f'MHA Output\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Residual connection and LayerNorm
        dot.node(f'residual1_{layer}', f'Residual Add\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightpink')
        dot.node(f'ln1_{layer}', f'LayerNorm\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightpink')
        
        # FFN
        dot.node(f'ffn_up_{layer}', f'FFN Up\n[1024×512×32768]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightcoral')
        dot.node(f'ffn_gate_{layer}', f'FFN Gate\n[1024×512×32768]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightcoral')
        dot.node(f'ffn_down_{layer}', f'FFN Down\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Second residual and LayerNorm
        dot.node(f'residual2_{layer}', f'Residual Add\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightpink')
        dot.node(f'ln2_{layer}', f'LayerNorm\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='rectangle', style='filled', fillcolor='lightpink')
        
        # Inter-GPU communication
        if layer < 16:
            dot.node(f'split_{layer}', f'Split\nGPU {gpu_id}→{gpu_id+1}', 
                     shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 'Output\n[Batch: 1024, Seq: 512, Hidden: 8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect the nodes
    dot.edge('input', 'mha_q_1')
    dot.edge('input', 'mha_k_1')
    dot.edge('input', 'mha_v_1')
    
    for layer in range(1, 17):
        # MHA connections
        dot.edge(f'mha_q_{layer}', f'mha_attn_{layer}')
        dot.edge(f'mha_k_{layer}', f'mha_attn_{layer}')
        dot.edge(f'mha_v_{layer}', f'mha_attn_{layer}')
        dot.edge(f'mha_attn_{layer}', f'mha_out_{layer}')
        
        # Residual connections
        if layer == 1:
            dot.edge('input', f'residual1_{layer}')
        else:
            dot.edge(f'ln2_{layer-1}', f'agg_{layer}')
            dot.edge(f'agg_{layer}', f'mha_q_{layer}')
            dot.edge(f'agg_{layer}', f'mha_k_{layer}')
            dot.edge(f'agg_{layer}', f'mha_v_{layer}')
            dot.edge(f'agg_{layer}', f'residual1_{layer}')
            
        dot.edge(f'mha_out_{layer}', f'residual1_{layer}')
        dot.edge(f'residual1_{layer}', f'ln1_{layer}')
        
        # FFN connections
        dot.edge(f'ln1_{layer}', f'ffn_up_{layer}')
        dot.edge(f'ln1_{layer}', f'ffn_gate_{layer}')
        dot.edge(f'ffn_up_{layer}', f'ffn_down_{layer}')
        dot.edge(f'ffn_gate_{layer}', f'ffn_down_{layer}')
        dot.edge(f'ffn_down_{layer}', f'residual2_{layer}')
        
        # Second residual
        dot.edge(f'ln1_{layer}', f'residual2_{layer}')
        dot.edge(f'residual2_{layer}', f'ln2_{layer}')
        
        # Inter-layer communication
        if layer < 16:
            dot.edge(f'ln2_{layer}', f'split_{layer}')
            dot.edge(f'split_{layer}', f'agg_{layer+1}')
    
    # Final output
    dot.edge('ln2_16', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_dense_model_dag()
    dag.render('/home/wzc/data/file-share/submission/dense_model_16_gpu', cleanup=True)
    print("Dense model DAG saved to /home/wzc/data/file-share/submission/dense_model_16_gpu.svg")