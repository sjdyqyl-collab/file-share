#!/usr/bin/env python3

import graphviz

def create_moe_model_dag():
    """Create DAG for 16-layer MoE model with 8 experts per layer deployed across 16 GPUs"""
    
    dot = graphviz.Digraph('moe_model_16_gpu', format='svg')
    dot.attr(rankdir='TB', size='25,35')
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
        
        # Multi-Head Attention (same as dense)
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
        
        # MoE Gate
        dot.node(f'gate_{layer}', f'MoE Gate\n[1024×512×8]\nGPU {gpu_id}', 
                 shape='diamond', style='filled', fillcolor='gold')
        
        # 8 Experts per layer (each expert is a FFN)
        for expert in range(8):
            dot.node(f'expert_{layer}_{expert}', 
                     f'Expert {expert}\n[1024×512×8192→32768→8192]\nGPU {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Expert aggregation
        dot.node(f'expert_agg_{layer}', f'Expert Aggregation\n[1024×512×8192]\nGPU {gpu_id}', 
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        
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
        
        # MoE connections
        dot.edge(f'ln1_{layer}', f'gate_{layer}')
        dot.edge(f'ln1_{layer}', f'expert_agg_{layer}')
        
        # Expert routing (dashed lines for gate selection)
        for expert in range(8):
            dot.edge(f'gate_{layer}', f'expert_{layer}_{expert}', 
                     style='dashed', label=f'expert_{expert}')
            dot.edge(f'ln1_{layer}', f'expert_{layer}_{expert}')
            dot.edge(f'expert_{layer}_{expert}', f'expert_agg_{layer}')
        
        # Second residual
        dot.edge(f'expert_agg_{layer}', f'residual2_{layer}')
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
    dag = create_moe_model_dag()
    dag.render('/home/wzc/data/file-share/submission/moe_model_16_gpu', cleanup=True)
    print("MoE model DAG saved to /home/wzc/data/file-share/submission/moe_model_16_gpu.svg")