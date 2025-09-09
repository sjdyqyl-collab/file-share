#!/usr/bin/env python3

import os
from graphviz import Digraph

def create_baseline_dag():
    """
    Create DAG for baseline configuration:
    - 16 GPUs total
    - TP=8, PP=2
    - 4 experts per GPU
    - 4-layer MoE model
    """
    dot = Digraph('baseline_moe', comment='Baseline MoE Deployment (TP=8, PP=2, 4 experts/GPU)')
    dot.attr(rankdir='TB', size='20,30')
    
    # Global dimensions
    batch_size = 1024
    seq_len = 10000
    hidden_size = 8192
    ffn_hidden_size = 32768  # typical 4x hidden_size
    num_layers = 4
    num_experts = 16
    experts_per_gpu = 4
    
    # Input node
    dot.node('input', f'Total Input\nShape: [{batch_size}, {seq_len}, {hidden_size}]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline stages
    for stage in range(2):  # PP=2
        with dot.subgraph(name=f'cluster_stage_{stage}') as stage_subgraph:
            stage_subgraph.attr(label=f'Pipeline Stage {stage}', style='dashed')
            
            for layer in range(num_layers // 2):  # 2 layers per stage
                layer_id = stage * (num_layers // 2) + layer
                
                # Layer input
                layer_input = f'layer_{layer_id}_input'
                if layer_id == 0:
                    dot.edge('input', layer_input)
                else:
                    prev_layer = f'layer_{layer_id-1}_output'
                    dot.edge(prev_layer, layer_input)
                
                # Layer norm (duplicated across TP group)
                ln_node = f'layer_{layer_id}_ln'
                dot.node(ln_node, f'LayerNorm\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: 0-7', 
                         shape='rectangle', style='filled', fillcolor='lightyellow')
                dot.edge(layer_input, ln_node)
                
                # Multi-head attention (TP=8)
                attn_nodes = []
                for tp_rank in range(8):
                    attn_node = f'layer_{layer_id}_attn_tp{tp_rank}'
                    # Column-parallel for QKV
                    qkv_node = f'layer_{layer_id}_qkv_tp{tp_rank}'
                    dot.node(qkv_node, f'QKV Linear\nShape: [{batch_size}, {seq_len}, {hidden_size//8*3}]\nGPU: {tp_rank + stage*8}', 
                             shape='rectangle', style='filled', fillcolor='lightgreen')
                    dot.edge(ln_node, qkv_node)
                    
                    # Attention computation
                    attn_comp = f'layer_{layer_id}_attn_comp_tp{tp_rank}'
                    dot.node(attn_comp, f'Attention\nShape: [{batch_size}, {seq_len}, {hidden_size//8}]\nGPU: {tp_rank + stage*8}', 
                             shape='rectangle', style='filled', fillcolor='lightgreen')
                    dot.edge(qkv_node, attn_comp)
                    
                    # Output projection (row-parallel)
                    out_proj = f'layer_{layer_id}_out_proj_tp{tp_rank}'
                    dot.node(out_proj, f'Output Proj\nShape: [{batch_size}, {seq_len}, {hidden_size//8}]\nGPU: {tp_rank + stage*8}', 
                             shape='rectangle', style='filled', fillcolor='lightgreen')
                    dot.edge(attn_comp, out_proj)
                    attn_nodes.append(out_proj)
                
                # Attention all-reduce
                attn_ar = f'layer_{layer_id}_attn_ar'
                dot.node(attn_ar, f'All-Reduce Sum\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: 0-7', 
                         shape='parallelogram', style='filled', fillcolor='lightcoral')
                for attn_node in attn_nodes:
                    dot.edge(attn_node, attn_ar)
                
                # Residual add 1
                residual1 = f'layer_{layer_id}_residual1'
                dot.node(residual1, f'Residual Add\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: 0-7', 
                         shape='ellipse', style='filled', fillcolor='lightgray')
                dot.edge(layer_input, residual1)
                dot.edge(attn_ar, residual1)
                
                # Layer norm 2
                ln2_node = f'layer_{layer_id}_ln2'
                dot.node(ln2_node, f'LayerNorm\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: 0-7', 
                         shape='rectangle', style='filled', fillcolor='lightyellow')
                dot.edge(residual1, ln2_node)
                
                # Expert routing
                gate_node = f'layer_{layer_id}_gate'
                dot.node(gate_node, f'Gate\nShape: [{batch_size * seq_len}, {num_experts}]\nGPU: 0-7', 
                         shape='parallelogram', style='filled', fillcolor='orange')
                dot.edge(ln2_node, gate_node)
                
                # Expert computation (4 experts per GPU, 16 total)
                expert_nodes = []
                for expert_id in range(num_experts):
                    gpu_id = expert_id // experts_per_gpu + stage * 8
                    expert_node = f'layer_{layer_id}_expert_{expert_id}'
                    dot.node(expert_node, f'Expert {expert_id}\nShape: [{batch_size * seq_len // 4}, {ffn_hidden_size}]\nGPU: {gpu_id}', 
                             shape='rectangle', style='filled', fillcolor='lightcyan')
                    
                    # Routing connection (dashed)
                    dot.edge(gate_node, expert_node, style='dashed', label=f'route tokens')
                    expert_nodes.append(expert_node)
                
                # Expert aggregation
                expert_agg = f'layer_{layer_id}_expert_agg'
                dot.node(expert_agg, f'Expert Aggregation\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: 0-7', 
                         shape='parallelogram', style='filled', fillcolor='lightcoral')
                for expert_node in expert_nodes:
                    dot.edge(expert_node, expert_agg)
                
                # Final residual add
                final_residual = f'layer_{layer_id}_output'
                dot.node(final_residual, f'Final Residual\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: 0-7', 
                         shape='ellipse', style='filled', fillcolor='lightgray')
                dot.edge(residual1, final_residual)
                dot.edge(expert_agg, final_residual)
    
    # Output node
    dot.node('output', f'Total Output\nShape: [{batch_size}, {seq_len}, {hidden_size}]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    dot.edge('layer_3_output', 'output')
    
    return dot

def create_proposed_dag():
    """
    Create DAG for proposed configuration:
    - 64 GPUs total
    - EP=64 (1 expert per GPU)
    - 4-layer MoE model
    """
    dot = Digraph('proposed_moe', comment='Proposed MoE Deployment (EP=64, 1 expert/GPU)')
    dot.attr(rankdir='TB', size='30,40')
    
    # Global dimensions
    batch_size = 1024
    seq_len = 10000
    hidden_size = 8192
    ffn_hidden_size = 32768
    num_layers = 4
    num_experts = 64  # EP=64
    
    # Input node
    dot.node('input', f'Total Input\nShape: [{batch_size}, {seq_len}, {hidden_size}]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    for layer in range(num_layers):
        # Layer input
        layer_input = f'layer_{layer}_input'
        if layer == 0:
            dot.edge('input', layer_input)
        else:
            prev_layer = f'layer_{layer-1}_output'
            dot.edge(prev_layer, layer_input)
        
        # Layer norm (replicated across all GPUs for load balancing)
        ln_node = f'layer_{layer}_ln'
        dot.node(ln_node, f'LayerNorm\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: all GPUs', 
                 shape='rectangle', style='filled', fillcolor='lightyellow')
        dot.edge(layer_input, ln_node)
        
        # Multi-head attention (replicated for load balancing)
        attn_nodes = []
        for gpu_id in range(64):
            attn_node = f'layer_{layer}_attn_gpu{gpu_id}'
            dot.node(attn_node, f'Multi-Head Attention\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.edge(ln_node, attn_node)
            attn_nodes.append(attn_node)
        
        # Attention all-reduce across GPUs
        attn_ar = f'layer_{layer}_attn_ar'
        dot.node(attn_ar, f'All-Reduce Sum\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: all GPUs', 
                 shape='parallelogram', style='filled', fillcolor='lightcoral')
        for attn_node in attn_nodes:
            dot.edge(attn_node, attn_ar)
        
        # Residual add 1
        residual1 = f'layer_{layer}_residual1'
        dot.node(residual1, f'Residual Add\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: all GPUs', 
                 shape='ellipse', style='filled', fillcolor='lightgray')
        dot.edge(layer_input, residual1)
        dot.edge(attn_ar, residual1)
        
        # Layer norm 2
        ln2_node = f'layer_{layer}_ln2'
        dot.node(ln2_node, f'LayerNorm\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: all GPUs', 
                 shape='rectangle', style='filled', fillcolor='lightyellow')
        dot.edge(residual1, ln2_node)
        
        # Expert routing (distributed)
        gate_node = f'layer_{layer}_gate'
        dot.node(gate_node, f'Gate\nShape: [{batch_size * seq_len}, {num_experts}]\nGPU: all GPUs', 
                 shape='parallelogram', style='filled', fillcolor='orange')
        dot.edge(ln2_node, gate_node)
        
        # Expert computation (1 expert per GPU)
        expert_nodes = []
        for expert_id in range(num_experts):
            gpu_id = expert_id
            expert_node = f'layer_{layer}_expert_{expert_id}'
            dot.node(expert_node, f'Expert {expert_id}\nShape: [{batch_size * seq_len // 64}, {ffn_hidden_size}]\nGPU: {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightcyan')
            
            # Routing connection with token distribution
            dot.edge(gate_node, expert_node, style='dashed', label=f'route tokens')
            expert_nodes.append(expert_node)
        
        # Expert aggregation across all GPUs
        expert_agg = f'layer_{layer}_expert_agg'
        dot.node(expert_agg, f'Expert Aggregation\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: all GPUs', 
                 shape='parallelogram', style='filled', fillcolor='lightcoral')
        for expert_node in expert_nodes:
            dot.edge(expert_node, expert_agg)
        
        # Final residual add
        final_residual = f'layer_{layer}_output'
        dot.node(final_residual, f'Final Residual\nShape: [{batch_size}, {seq_len}, {hidden_size}]\nGPU: all GPUs', 
                 shape='ellipse', style='filled', fillcolor='lightgray')
        dot.edge(residual1, final_residual)
        dot.edge(expert_agg, final_residual)
    
    # Output node
    dot.node('output', f'Total Output\nShape: [{batch_size}, {seq_len}, {hidden_size}]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    dot.edge('layer_3_output', 'output')
    
    return dot

if __name__ == '__main__':
    # Create output directory
    os.makedirs('/home/wzc/data/file-share/2025-09-09-09-01-50', exist_ok=True)
    
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    baseline_dag.render('/home/wzc/data/file-share/2025-09-09-09-01-50/baseline_moe', format='svg', cleanup=False)
    baseline_dag.save('/home/wzc/data/file-share/2025-09-09-09-01-50/baseline_moe.dot')
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag()
    proposed_dag.render('/home/wzc/data/file-share/2025-09-09-09-01-50/proposed_moe', format='svg', cleanup=False)
    proposed_dag.save('/home/wzc/data/file-share/2025-09-09-09-01-50/proposed_moe.dot')
    
    print("DAGs generated successfully!")
    print("Files saved:")
    print("- /home/wzc/data/file-share/2025-09-09-09-01-50/baseline_moe.svg")
    print("- /home/wzc/data/file-share/2025-09-09-09-01-50/baseline_moe.dot")
    print("- /home/wzc/data/file-share/2025-09-09-09-01-50/proposed_moe.svg")
    print("- /home/wzc/data/file-share/2025-09-09-09-01-50/proposed_moe.dot")