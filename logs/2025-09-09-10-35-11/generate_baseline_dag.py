#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_baseline_dag():
    """
    Create DAG for baseline MoE model with TP=8, PP=2, 16 GPUs total
    Each GPU hosts 4 experts with tensor parallelism applied
    """
    
    dot = Digraph(comment='Baseline MoE DAG (TP=8, PP=2)', format='svg')
    dot.attr(rankdir='TB', size='20,30', ranksep='1.5', nodesep='0.5')
    
    # Input node
    dot.node('input', 'Model Input\n[1024, 10000, 8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Global variables
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192
    ffn_hidden = 32768
    experts_per_layer = 16
    tp_degree = 8
    
    # Create layers 0-3 with detailed breakdown
    for layer_idx in range(4):
        layer_prefix = f"layer_{layer_idx}"
        
        # Determine pipeline stage and GPU allocation
        if layer_idx < 2:
            pipeline_stage = 0
            gpu_base = 0
        else:
            pipeline_stage = 1
            gpu_base = 8
            
        # Add layer boundary
        dot.node(f'{layer_prefix}_boundary', f'Layer {layer_idx} Boundary', 
                shape='box', style='dashed', fillcolor='lightgray')
        
        # Gate computation (shared across TP group)
        gate_gpu = gpu_base + (layer_idx % 2) * 4
        dot.node(f'{layer_prefix}_gate', f'Gate Network\n[1024, 10000, 16]\nGPU {gate_gpu}', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Expert routing (dashed lines for gate decisions)
        for expert_idx in range(experts_per_layer):
            expert_gpu = gpu_base + (expert_idx // 4)  # 4 experts per GPU
            expert_local_idx = expert_idx % 4
            
            # Expert computation subgraph
            expert_prefix = f'{layer_prefix}_expert_{expert_idx}'
            
            # Expert input aggregation
            dot.node(f'{expert_prefix}_input_agg', 
                    f'Expert {expert_idx} Input Aggregation\n[batch_tokens, 8192]\nGPU {expert_gpu}', 
                    shape='parallelogram', style='filled', fillcolor='lightgreen')
            
            # Tensor parallel expert computation
            for tp_rank in range(tp_degree):
                tp_gpu = expert_gpu * tp_degree + tp_rank
                
                # First linear (column parallel)
                dot.node(f'{expert_prefix}_linear1_tp{tp_rank}', 
                        f'Expert {expert_idx} Linear1 TP{tp_rank}\n[batch_tokens, 8192] -> [batch_tokens, 4096]\nGPU {tp_gpu}', 
                        shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Activation
                dot.node(f'{expert_prefix}_gelu_tp{tp_rank}', 
                        f'Expert {expert_idx} GELU TP{tp_rank}\n[batch_tokens, 4096]\nGPU {tp_gpu}', 
                        shape='rectangle', style='filled', fillcolor='lightpink')
                
                # Second linear (row parallel)
                dot.node(f'{expert_prefix}_linear2_tp{tp_rank}', 
                        f'Expert {expert_idx} Linear2 TP{tp_rank}\n[batch_tokens, 4096] -> [batch_tokens, 8192]\nGPU {tp_gpu}', 
                        shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # All-reduce for tensor parallel
                dot.node(f'{expert_prefix}_allreduce_tp{tp_rank}', 
                        f'Expert {expert_idx} All-Reduce TP{tp_rank}\n[batch_tokens, 8192]\nGPU {tp_gpu}', 
                        shape='ellipse', style='filled', fillcolor='lightsteelblue')
            
            # Expert output
            dot.node(f'{expert_prefix}_output', 
                    f'Expert {expert_idx} Output\n[batch_tokens, 8192]\nGPU {expert_gpu}', 
                    shape='parallelogram', style='filled', fillcolor='lightgreen')
            
        # Output aggregation for layer
        dot.node(f'{layer_prefix}_output_agg', 
                f'Layer {layer_idx} Output Aggregation\n[1024, 10000, 8192]\nAll GPUs', 
                shape='parallelogram', style='filled', fillcolor='lightsteelblue')
        
        # Residual connection
        dot.node(f'{layer_prefix}_residual', 
                f'Layer {layer_idx} Residual Add\n[1024, 10000, 8192]\nAll GPUs', 
                shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Output node
    dot.node('output', 'Model Output\n[1024, 10000, 8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect the DAG
    dot.edge('input', 'layer_0_boundary')
    
    # Layer 0 connections
    dot.edge('layer_0_boundary', 'layer_0_gate')
    for expert_idx in range(experts_per_layer):
        expert_gpu = (expert_idx // 4)
        expert_prefix = f'layer_0_expert_{expert_idx}'
        
        # Gate to expert routing (dashed)
        dot.edge('layer_0_gate', f'{expert_prefix}_input_agg', style='dashed')
        
        # Expert computation chain
        dot.edge(f'{expert_prefix}_input_agg', f'{expert_prefix}_linear1_tp0')
        for tp_rank in range(tp_degree):
            dot.edge(f'{expert_prefix}_linear1_tp{tp_rank}', f'{expert_prefix}_gelu_tp{tp_rank}')
            dot.edge(f'{expert_prefix}_gelu_tp{tp_rank}', f'{expert_prefix}_linear2_tp{tp_rank}')
            dot.edge(f'{expert_prefix}_linear2_tp{tp_rank}', f'{expert_prefix}_allreduce_tp{tp_rank}')
            if tp_rank < tp_degree - 1:
                dot.edge(f'{expert_prefix}_allreduce_tp{tp_rank}', f'{expert_prefix}_allreduce_tp{tp_rank+1}')
        
        dot.edge(f'{expert_prefix}_allreduce_tp{tp_degree-1}', f'{expert_prefix}_output')
        dot.edge(f'{expert_prefix}_output', 'layer_0_output_agg')
    
    # Residual and layer connections
    dot.edge('layer_0_boundary', 'layer_0_residual', style='dotted')
    dot.edge('layer_0_output_agg', 'layer_0_residual')
    dot.edge('layer_0_residual', 'layer_1_boundary')
    
    # Layer 1 connections
    dot.edge('layer_1_boundary', 'layer_1_gate')
    for expert_idx in range(experts_per_layer):
        expert_gpu = 4 + (expert_idx // 4)
        expert_prefix = f'layer_1_expert_{expert_idx}'
        
        dot.edge('layer_1_gate', f'{expert_prefix}_input_agg', style='dashed')
        dot.edge(f'{expert_prefix}_input_agg', f'{expert_prefix}_linear1_tp0')
        
        for tp_rank in range(tp_degree):
            dot.edge(f'{expert_prefix}_linear1_tp{tp_rank}', f'{expert_prefix}_gelu_tp{tp_rank}')
            dot.edge(f'{expert_prefix}_gelu_tp{tp_rank}', f'{expert_prefix}_linear2_tp{tp_rank}')
            dot.edge(f'{expert_prefix}_linear2_tp{tp_rank}', f'{expert_prefix}_allreduce_tp{tp_rank}')
            if tp_rank < tp_degree - 1:
                dot.edge(f'{expert_prefix}_allreduce_tp{tp_rank}', f'{expert_prefix}_allreduce_tp{tp_rank+1}')
        
        dot.edge(f'{expert_prefix}_allreduce_tp{tp_degree-1}', f'{expert_prefix}_output')
        dot.edge(f'{expert_prefix}_output', 'layer_1_output_agg')
    
    dot.edge('layer_1_boundary', 'layer_1_residual', style='dotted')
    dot.edge('layer_1_output_agg', 'layer_1_residual')
    dot.edge('layer_1_residual', 'layer_2_boundary')
    
    # Layer 2 connections
    dot.edge('layer_2_boundary', 'layer_2_gate')
    for expert_idx in range(experts_per_layer):
        expert_gpu = 8 + (expert_idx // 4)
        expert_prefix = f'layer_2_expert_{expert_idx}'
        
        dot.edge('layer_2_gate', f'{expert_prefix}_input_agg', style='dashed')
        dot.edge(f'{expert_prefix}_input_agg', f'{expert_prefix}_linear1_tp0')
        
        for tp_rank in range(tp_degree):
            dot.edge(f'{expert_prefix}_linear1_tp{tp_rank}', f'{expert_prefix}_gelu_tp{tp_rank}')
            dot.edge(f'{expert_prefix}_gelu_tp{tp_rank}', f'{expert_prefix}_linear2_tp{tp_rank}')
            dot.edge(f'{expert_prefix}_linear2_tp{tp_rank}', f'{expert_prefix}_allreduce_tp{tp_rank}')
            if tp_rank < tp_degree - 1:
                dot.edge(f'{expert_prefix}_allreduce_tp{tp_rank}', f'{expert_prefix}_allreduce_tp{tp_rank+1}')
        
        dot.edge(f'{expert_prefix}_allreduce_tp{tp_degree-1}', f'{expert_prefix}_output')
        dot.edge(f'{expert_prefix}_output', 'layer_2_output_agg')
    
    dot.edge('layer_2_boundary', 'layer_2_residual', style='dotted')
    dot.edge('layer_2_output_agg', 'layer_2_residual')
    dot.edge('layer_2_residual', 'layer_3_boundary')
    
    # Layer 3 connections
    dot.edge('layer_3_boundary', 'layer_3_gate')
    for expert_idx in range(experts_per_layer):
        expert_gpu = 12 + (expert_idx // 4)
        expert_prefix = f'layer_3_expert_{expert_idx}'
        
        dot.edge('layer_3_gate', f'{expert_prefix}_input_agg', style='dashed')
        dot.edge(f'{expert_prefix}_input_agg', f'{expert_prefix}_linear1_tp0')
        
        for tp_rank in range(tp_degree):
            dot.edge(f'{expert_prefix}_linear1_tp{tp_rank}', f'{expert_prefix}_gelu_tp{tp_rank}')
            dot.edge(f'{expert_prefix}_gelu_tp{tp_rank}', f'{expert_prefix}_linear2_tp{tp_rank}')
            dot.edge(f'{expert_prefix}_linear2_tp{tp_rank}', f'{expert_prefix}_allreduce_tp{tp_rank}')
            if tp_rank < tp_degree - 1:
                dot.edge(f'{expert_prefix}_allreduce_tp{tp_rank}', f'{expert_prefix}_allreduce_tp{tp_rank+1}')
        
        dot.edge(f'{expert_prefix}_allreduce_tp{tp_degree-1}', f'{expert_prefix}_output')
        dot.edge(f'{expert_prefix}_output', 'layer_3_output_agg')
    
    dot.edge('layer_3_boundary', 'layer_3_residual', style='dotted')
    dot.edge('layer_3_output_agg', 'layer_3_residual')
    dot.edge('layer_3_residual', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-09-10-35-11/baseline_moe_dag', format='svg', cleanup=True)
    dag.save('/home/wzc/data/file-share/2025-09-09-10-35-11/baseline_moe_dag.dot')
    print("Baseline MoE DAG generated successfully")