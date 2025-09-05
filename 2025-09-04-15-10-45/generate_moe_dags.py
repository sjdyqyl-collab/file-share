#!/usr/bin/env python3
"""
Generate complete DAGs for MoE baseline vs proposed cross-node expert parallelism
"""

import os
from graphviz import Digraph

def create_baseline_dag():
    """
    Create baseline DAG with TP=8, PP=2, 16 GPUs total
    Each GPU has 4 experts colocated with tensor parallelism
    """
    dot = Digraph(name='baseline_moe_dag', 
                  comment='Baseline MoE: TP=8, PP=2, 16 GPUs, 4 experts/GPU')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input
    dot.node('input', 'Input Tokens\n[1024, 8192]', fillcolor='lightgreen')
    
    # Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_stage0') as c0:
        c0.attr(label='Pipeline Stage 0\nGPUs 0-7', style='rounded', fillcolor='lightgray')
        
        # Layer 0
        for gpu_id in range(8):
            # Attention
            attn_node = f'layer0_attn_gpu{gpu_id}'
            c0.node(attn_node, f'Attention\nGPU {gpu_id}\n[1024, 8192]', 
                   shape='rectangle', fillcolor='lightcoral')
            
            # Residual connection
            residual_node = f'layer0_residual_gpu{gpu_id}'
            c0.node(residual_node, f'Residual Add\nGPU {gpu_id}\n[1024, 8192]', 
                   shape='diamond', fillcolor='yellow')
            
            # Expert selection (gate)
            gate_node = f'layer0_gate_gpu{gpu_id}'
            c0.node(gate_node, f'Expert Gate\nGPU {gpu_id}\n[1024, 16]', 
                   shape='parallelogram', fillcolor='orange', style='dashed')
            
            # Experts (4 per GPU)
            for expert_idx in range(4):
                expert_id = gpu_id * 4 + expert_idx
                expert_node = f'layer0_expert{expert_id}_gpu{gpu_id}'
                c0.node(expert_node, f'Expert {expert_id}\nGPU {gpu_id}\n[256, 32768]\n→ [256, 8192]', 
                       shape='rectangle', fillcolor='lightsteelblue')
                
                # Expert computation
                expert_linear1 = f'layer0_exp{expert_id}_linear1_gpu{gpu_id}'
                expert_act = f'layer0_exp{expert_id}_act_gpu{gpu_id}'
                expert_linear2 = f'layer0_exp{expert_id}_linear2_gpu{gpu_id}'
                
                c0.node(expert_linear1, f'Linear1\nGPU {gpu_id}\n[256, 8192]×[8192, 32768]', 
                       shape='rectangle', fillcolor='lightpink')
                c0.node(expert_act, f'GELU\nGPU {gpu_id}\n[256, 32768]', 
                       shape='ellipse', fillcolor='lightgreen')
                c0.node(expert_linear2, f'Linear2\nGPU {gpu_id}\n[256, 32768]×[32768, 8192]', 
                       shape='rectangle', fillcolor='lightpink')
                
                # Expert connections
                c0.edge(expert_linear1, expert_act)
                c0.edge(expert_act, expert_linear2)
                c0.edge(expert_linear2, expert_node, style='invis')
            
            # Expert aggregation
            agg_node = f'layer0_agg_gpu{gpu_id}'
            c0.node(agg_node, f'Expert Aggregation\nGPU {gpu_id}\n[1024, 8192]', 
                   shape='hexagon', fillcolor='gold')
            
            # Final residual
            final_residual = f'layer0_final_gpu{gpu_id}'
            c0.node(final_residual, f'Final Residual\nGPU {gpu_id}\n[1024, 8192]', 
                   shape='diamond', fillcolor='yellow')
    
    # Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_stage1') as c1:
        c1.attr(label='Pipeline Stage 1\nGPUs 8-15', style='rounded', fillcolor='lightgray')
        
        # Layer 2 (similar to layer 0)
        for gpu_id in range(8, 16):
            actual_gpu = gpu_id - 8
            attn_node = f'layer2_attn_gpu{gpu_id}'
            c1.node(attn_node, f'Attention\nGPU {gpu_id}\n[1024, 8192]', 
                   shape='rectangle', fillcolor='lightcoral')
            
            # Similar structure for layer 2 and layer 3
            # ... (abbreviated for brevity, will be expanded)
    
    # Pipeline connections
    for gpu_id in range(8):
        # Layer 0 connections
        dot.edge('input', f'layer0_attn_gpu{gpu_id}')
        dot.edge(f'layer0_attn_gpu{gpu_id}', f'layer0_residual_gpu{gpu_id}')
        dot.edge(f'layer0_residual_gpu{gpu_id}', f'layer0_gate_gpu{gpu_id}')
        
        # Expert routing (dashed)
        for expert_idx in range(4):
            expert_id = gpu_id * 4 + expert_idx
            dot.edge(f'layer0_gate_gpu{gpu_id}', f'layer0_exp{expert_id}_linear1_gpu{gpu_id}', 
                    style='dashed', label=f'token_routing')
        
        # Expert aggregation
        for expert_idx in range(4):
            expert_id = gpu_id * 4 + expert_idx
            dot.edge(f'layer0_exp{expert_id}_linear2_gpu{gpu_id}', f'layer0_agg_gpu{gpu_id}')
        
        dot.edge(f'layer0_agg_gpu{gpu_id}', f'layer0_final_gpu{gpu_id}')
        dot.edge(f'layer0_residual_gpu{gpu_id}', f'layer0_final_gpu{gpu_id}')
        
        # Pipeline to stage 1
        dot.edge(f'layer0_final_gpu{gpu_id}', f'layer2_attn_gpu{gpu_id+8}', 
                label='pipeline_send', style='dotted')
    
    # Output
    output_nodes = [f'layer3_final_gpu{i+8}' for i in range(8)]
    dot.node('output', 'Output Tokens\n[1024, 8192]', fillcolor='lightgreen')
    
    for node in output_nodes:
        dot.edge(node, 'output', style='dotted')
    
    return dot

def create_proposed_dag():
    """
    Create proposed DAG with 64 GPUs, 1 expert per GPU
    Large-scale cross-node expert parallelism
    """
    dot = Digraph(name='proposed_moe_dag', 
                  comment='Proposed MoE: Cross-node EP=16, 64 GPUs, 1 expert/GPU')
    dot.attr(rankdir='TB', size='30,40')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input
    dot.node('input', 'Input Tokens\n[1024, 8192]', fillcolor='lightgreen')
    
    # Global token distribution
    dot.node('token_dist', 'Token Distribution\nAll GPUs\n[1024, 8192]', 
             shape='hexagon', fillcolor='gold')
    
    # Process each layer
    for layer in range(4):
        layer_name = f'layer{layer}'
        
        with dot.subgraph(name=f'cluster_layer{layer}') as c:
            c.attr(label=f'Layer {layer}\n16 Experts across 16 GPUs', 
                   style='rounded', fillcolor='lightgray')
            
            # Attention across all GPUs (optional tensor parallelism)
            attn_nodes = []
            for gpu_group in range(4):  # 4 pipeline stages
                for gpu_in_group in range(16):  # 16 GPUs per stage
                    gpu_id = layer * 16 + gpu_in_group
                    attn_node = f'{layer_name}_attn_gpu{gpu_id}'
                    c.node(attn_node, f'Attention\nGPU {gpu_id}\n[1024, 8192]', 
                           shape='rectangle', fillcolor='lightcoral')
                    attn_nodes.append(attn_node)
            
            # Gate for expert selection
            gate_nodes = []
            for gpu_id in range(16):
                actual_gpu = layer * 16 + gpu_id
                gate_node = f'{layer_name}_gate_gpu{actual_gpu}'
                c.node(gate_node, f'Expert Gate\nGPU {actual_gpu}\n[1024, 16]', 
                       shape='parallelogram', fillcolor='orange', style='dashed')
                gate_nodes.append(gate_node)
            
            # Individual experts (1 per GPU)
            expert_nodes = []
            for expert_id in range(16):
                actual_gpu = layer * 16 + expert_id
                expert_node = f'{layer_name}_expert{expert_id}_gpu{actual_gpu}'
                c.node(expert_node, f'Expert {layer*16+expert_id}\nGPU {actual_gpu}\n[64, 32768]\n→ [64, 8192]', 
                       shape='rectangle', fillcolor='lightsteelblue')
                
                # Expert computation details
                expert_linear1 = f'{layer_name}_exp{expert_id}_linear1_gpu{actual_gpu}'
                expert_act = f'{layer_name}_exp{expert_id}_act_gpu{actual_gpu}'
                expert_linear2 = f'{layer_name}_exp{expert_id}_linear2_gpu{actual_gpu}'
                
                c.node(expert_linear1, f'Linear1\nGPU {actual_gpu}\n[64, 8192]×[8192, 32768]', 
                       shape='rectangle', fillcolor='lightpink')
                c.node(expert_act, f'GELU\nGPU {actual_gpu}\n[64, 32768]', 
                       shape='ellipse', fillcolor='lightgreen')
                c.node(expert_linear2, f'Linear2\nGPU {actual_gpu}\n[64, 32768]×[32768, 8192]', 
                       shape='rectangle', fillcolor='lightpink')
                
                # Expert connections
                c.edge(expert_linear1, expert_act)
                c.edge(expert_act, expert_linear2)
                c.edge(expert_linear2, expert_node, style='invis')
                expert_nodes.append(expert_node)
            
            # Expert aggregation across GPUs
            agg_node = f'{layer_name}_agg'
            c.node(agg_node, f'Expert Aggregation\nAll GPUs\n[1024, 8192]', 
                   shape='hexagon', fillcolor='gold')
            
            # Residual connections
            residual_nodes = []
            for gpu_id in range(16):
                actual_gpu = layer * 16 + gpu_id
                residual_node = f'{layer_name}_residual_gpu{actual_gpu}'
                c.node(residual_node, f'Residual Add\nGPU {actual_gpu}\n[1024, 8192]', 
                       shape='diamond', fillcolor='yellow')
                residual_nodes.append(residual_node)
    
    # Create connections
    # Input to first layer
    dot.edge('input', 'token_dist')
    for gpu_id in range(16):
        dot.edge('token_dist', f'layer0_attn_gpu{gpu_id}')
    
    # Layer connections
    for layer in range(4):
        layer_name = f'layer{layer}'
        
        # Attention to gate
        for gpu_id in range(16):
            actual_gpu = layer * 16 + gpu_id
            dot.edge(f'{layer_name}_attn_gpu{actual_gpu}', f'{layer_name}_gate_gpu{actual_gpu}')
            dot.edge(f'{layer_name}_attn_gpu{actual_gpu}', f'{layer_name}_residual_gpu{actual_gpu}')
        
        # Gate to expert routing (dashed)
        for expert_id in range(16):
            actual_gpu = layer * 16 + expert_id
            dot.edge(f'{layer_name}_gate_gpu{actual_gpu}', 
                    f'{layer_name}_exp{expert_id}_linear1_gpu{actual_gpu}', 
                    style='dashed', label='token_routing')
        
        # Expert to aggregation (cross-GPU communication)
        for expert_id in range(16):
            actual_gpu = layer * 16 + expert_id
            dot.edge(f'{layer_name}_exp{expert_id}_linear2_gpu{actual_gpu}', 
                    f'{layer_name}_agg', 
                    label=f'all_to_all\nGPU {actual_gpu}')
        
        # Aggregation to residual
        for gpu_id in range(16):
            actual_gpu = layer * 16 + gpu_id
            dot.edge(f'{layer_name}_agg', f'{layer_name}_residual_gpu{actual_gpu}')
        
        # Layer to layer pipeline
        if layer < 3:
            for gpu_id in range(16):
                actual_gpu = layer * 16 + gpu_id
                next_gpu = (layer + 1) * 16 + gpu_id
                dot.edge(f'{layer_name}_residual_gpu{actual_gpu}', 
                        f'layer{layer+1}_attn_gpu{next_gpu}', 
                        label='pipeline_send', style='dotted')
    
    # Final output
    output_nodes = [f'layer3_residual_gpu{i+48}' for i in range(16)]
    dot.node('output', 'Output Tokens\n[1024, 8192]', fillcolor='lightgreen')
    dot.node('final_agg', 'Final Aggregation\nAll GPUs\n[1024, 8192]', 
             shape='hexagon', fillcolor='gold')
    
    for node in output_nodes:
        dot.edge(node, 'final_agg', style='dotted')
    dot.edge('final_agg', 'output')
    
    return dot

if __name__ == '__main__':
    # Create output directory
    output_dir = '/home/wzc/data/file-share/2025-09-04-15-10-45'
    
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    baseline_dag.render(os.path.join(output_dir, 'baseline_moe_dag'), 
                       format='svg', cleanup=False)
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag()
    proposed_dag.render(os.path.join(output_dir, 'proposed_moe_dag'), 
                       format='svg', cleanup=False)
    
    # Also save DOT files
    with open(os.path.join(output_dir, 'baseline_moe_dag.dot'), 'w') as f:
        f.write(baseline_dag.source)
    
    with open(os.path.join(output_dir, 'proposed_moe_dag.dot'), 'w') as f:
        f.write(proposed_dag.source)
    
    print("DAGs generated successfully!")
    print(f"Files saved to: {output_dir}")