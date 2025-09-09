#!/usr/bin/env python3
"""
Generate complete DAGs for the MoE paper:
1. Baseline: 16 GPUs, TP=8, PP=2, 4 experts per GPU
2. Proposed: 64 GPUs, EP=64, 1 expert per GPU
"""

import os
from graphviz import Digraph

def create_baseline_dag():
    """Create baseline DAG with 16 GPUs, TP=8, PP=2, 4 experts per GPU"""
    dot = Digraph(name='baseline_moe_dag', comment='4-layer MoE Baseline Deployment')
    dot.attr(rankdir='TB', size='100,100', concentrate='true')
    
    # Global dimensions
    batch_size = 1024
    seq_len = 10000
    token_dim = 8192
    mlp_hidden = 32768
    num_heads = 16
    head_dim = 512
    
    # Create pipeline stages
    for stage_id in range(2):
        with dot.subgraph(name=f'cluster_stage_{stage_id}') as stage:
            stage.attr(label=f'Pipeline Stage {stage_id}', style='rounded', color='blue', bgcolor='lightblue')
            
            # Each stage has 2 layers
            for layer_id in range(2):
                global_layer_id = stage_id * 2 + layer_id
                
                with stage.subgraph(name=f'cluster_layer_{global_layer_id}') as layer:
                    layer.attr(label=f'Layer {global_layer_id}', style='dashed', color='green')
                    
                    # Input to layer
                    if global_layer_id == 0:
                        # Global input
                        dot.node('global_input', 
                                f'Global Input\n[B={batch_size}, S={seq_len}, D={token_dim}]',
                                shape='ellipse', style='filled', fillcolor='lightyellow')
                        layer_input = 'global_input'
                    else:
                        # From previous layer
                        prev_layer_id = global_layer_id - 1
                        layer_input = f'layer_{prev_layer_id}_output'
                    
                    # Layer normalization
                    ln_node = f'layer_{global_layer_id}_ln'
                    dot.node(ln_node,
                            f'LayerNorm\n[B={batch_size}, S={seq_len}, D={token_dim}]',
                            shape='rectangle', style='filled', fillcolor='lightcyan')
                    dot.edge(layer_input, ln_node)
                    
                    # Multi-Head Attention across 8 GPUs with TP
                    mha_nodes = []
                    for tp_rank in range(8):
                        gpu_id = stage_id * 8 + tp_rank
                        mha_node = f'layer_{global_layer_id}_mha_tp{tp_rank}'
                        dot.node(mha_node,
                                f'MHA TP{tp_rank}\n[B={batch_size}, S={seq_len}, D={token_dim//8}]\nGPU{gpu_id}',
                                shape='rectangle', style='filled', fillcolor='lightpink')
                        dot.edge(ln_node, mha_node)
                        mha_nodes.append(mha_node)
                    
                    # MHA all-reduce
                    mha_ar_node = f'layer_{global_layer_id}_mha_ar'
                    dot.node(mha_ar_node,
                            f'MHA All-Reduce\n[B={batch_size}, S={seq_len}, D={token_dim}]',
                            shape='parallelogram', style='filled', fillcolor='orange')
                    
                    for mha_node in mha_nodes:
                        dot.edge(mha_node, mha_ar_node)
                    
                    # Residual add
                    residual1_node = f'layer_{global_layer_id}_residual1'
                    dot.node(residual1_node,
                            f'Residual Add\n[B={batch_size}, S={seq_len}, D={token_dim}]',
                            shape='diamond', style='filled', fillcolor='lightgreen')
                    dot.edge(layer_input, residual1_node)
                    dot.edge(mha_ar_node, residual1_node)
                    
                    # Second LayerNorm
                    ln2_node = f'layer_{global_layer_id}_ln2'
                    dot.node(ln2_node,
                            f'LayerNorm\n[B={batch_size}, S={seq_len}, D={token_dim}]',
                            shape='rectangle', style='filled', fillcolor='lightcyan')
                    dot.edge(residual1_node, ln2_node)
                    
                    # MoE - 4 experts per GPU, distributed across 8 GPUs
                    expert_outputs = []
                    for gpu_id in range(stage_id * 8, (stage_id + 1) * 8):
                        for expert_id in range(4):
                            expert_node = f'layer_{global_layer_id}_expert_{gpu_id}_{expert_id}'
                            expert_idx = (gpu_id - stage_id * 8) * 4 + expert_id
                            dot.node(expert_node,
                                    f'Expert {expert_idx}\n[B=?, S=?, D={mlp_hidden}]\nGPU{gpu_id}',
                                    shape='rectangle', style='filled', fillcolor='lightblue')
                            
                            # Gate routing (dashed line)
                            gate_node = f'layer_{global_layer_id}_gate_{gpu_id}_{expert_id}'
                            dot.node(gate_node,
                                    f'Gate {expert_idx}\nSelect tokens\nGPU{gpu_id}',
                                    shape='parallelogram', style='dashed', fillcolor='yellow')
                            dot.edge(ln2_node, expert_node, style='dashed')
                            dot.edge(gate_node, expert_node)
                            
                            expert_outputs.append(expert_node)
                    
                    # Expert aggregation
                    expert_agg_node = f'layer_{global_layer_id}_expert_agg'
                    dot.node(expert_agg_node,
                            f'Expert Aggregation\n[B={batch_size}, S={seq_len}, D={token_dim}]',
                            shape='parallelogram', style='filled', fillcolor='orange')
                    
                    for expert_out in expert_outputs:
                        dot.edge(expert_out, expert_agg_node)
                    
                    # Final residual add
                    residual2_node = f'layer_{global_layer_id}_residual2'
                    dot.node(residual2_node,
                            f'Residual Add\n[B={batch_size}, S={seq_len}, D={token_dim}]',
                            shape='diamond', style='filled', fillcolor='lightgreen')
                    dot.edge(residual1_node, residual2_node)
                    dot.edge(expert_agg_node, residual2_node)
                    
                    # Layer output
                    layer_output = f'layer_{global_layer_id}_output'
                    dot.node(layer_output,
                            f'Layer {global_layer_id} Output\n[B={batch_size}, S={seq_len}, D={token_dim}]',
                            shape='ellipse', style='filled', fillcolor='lightgray')
                    dot.edge(residual2_node, layer_output)
    
    # Global output
    dot.node('global_output',
            f'Global Output\n[B={batch_size}, S={seq_len}, D={token_dim}]',
            shape='ellipse', style='filled', fillcolor='lightyellow')
    dot.edge('layer_3_output', 'global_output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with 64 GPUs, EP=64, 1 expert per GPU"""
    dot = Digraph(name='proposed_large_ep_moe_dag', comment='4-layer MoE Large EP Deployment')
    dot.attr(rankdir='TB', size='150,150', concentrate='true')
    
    # Global dimensions
    batch_size = 1024
    seq_len = 10000
    token_dim = 8192
    mlp_hidden = 32768
    num_heads = 16
    head_dim = 512
    
    # Global input
    dot.node('global_input',
            f'Global Input\n[B={batch_size}, S={seq_len}, D={token_dim}]',
            shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Global token distribution
    dot.node('token_distribution',
            f'Token Distribution\nSplit tokens across GPUs',
            shape='parallelogram', style='filled', fillcolor='orange')
    dot.edge('global_input', 'token_distribution')
    
    # Process each layer
    for layer_id in range(4):
        with dot.subgraph(name=f'cluster_layer_{layer_id}') as layer:
            layer.attr(label=f'Layer {layer_id}', style='rounded', color='blue', bgcolor='lightblue')
            
            # Layer input handling
            if layer_id == 0:
                layer_input = 'token_distribution'
            else:
                layer_input = f'layer_{layer_id-1}_output'
            
            # Layer normalization (replicated across all 64 GPUs)
            ln_nodes = []
            for gpu_id in range(64):
                ln_node = f'layer_{layer_id}_ln_gpu{gpu_id}'
                dot.node(ln_node,
                        f'LayerNorm\n[B=?, S=?, D={token_dim}]\nGPU{gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightcyan')
                dot.edge(layer_input, ln_node)
                ln_nodes.append(ln_node)
            
            # Multi-Head Attention (replicated across all 64 GPUs)
            mha_nodes = []
            for gpu_id in range(64):
                mha_node = f'layer_{layer_id}_mha_gpu{gpu_id}'
                dot.node(mha_node,
                        f'MHA\n[B=?, S=?, D={token_dim}]\nGPU{gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightpink')
                dot.edge(ln_nodes[gpu_id], mha_node)
                mha_nodes.append(mha_node)
            
            # Residual add (local to each GPU)
            residual1_nodes = []
            for gpu_id in range(64):
                residual1_node = f'layer_{layer_id}_residual1_gpu{gpu_id}'
                dot.node(residual1_node,
                        f'Residual Add\n[B=?, S=?, D={token_dim}]\nGPU{gpu_id}',
                        shape='diamond', style='filled', fillcolor='lightgreen')
                dot.edge(layer_input, residual1_node)
                dot.edge(mha_nodes[gpu_id], residual1_node)
                residual1_nodes.append(residual1_node)
            
            # Second LayerNorm
            ln2_nodes = []
            for gpu_id in range(64):
                ln2_node = f'layer_{layer_id}_ln2_gpu{gpu_id}'
                dot.node(ln2_node,
                        f'LayerNorm\n[B=?, S=?, D={token_dim}]\nGPU{gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightcyan')
                dot.edge(residual1_nodes[gpu_id], ln2_node)
                ln2_nodes.append(ln2_node)
            
            # Expert computation - 1 expert per GPU
            expert_outputs = []
            for expert_id in range(16):
                gpu_id = layer_id * 16 + expert_id
                
                # Gate computation
                gate_node = f'layer_{layer_id}_gate_expert{expert_id}'
                dot.node(gate_node,
                        f'Gate Expert{expert_id}\nSelect tokens\nGPU{gpu_id}',
                        shape='parallelogram', style='dashed', fillcolor='yellow')
                
                # Expert computation
                expert_node = f'layer_{layer_id}_expert{expert_id}_gpu{gpu_id}'
                dot.node(expert_node,
                        f'Expert {expert_id}\n[B=?, S=?, D={mlp_hidden}]\nGPU{gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightblue')
                
                # Connect gate to expert (dashed)
                dot.edge(gate_node, expert_node, style='dashed')
                
                # Connect LayerNorm to gate and expert
                for gpu_src in range(64):
                    dot.edge(ln2_nodes[gpu_src], gate_node, constraint='false')
                    dot.edge(ln2_nodes[gpu_src], expert_node)
                
                expert_outputs.append(expert_node)
            
            # Expert aggregation across all GPUs
            expert_agg_nodes = []
            for gpu_id in range(64):
                agg_node = f'layer_{layer_id}_expert_agg_gpu{gpu_id}'
                dot.node(agg_node,
                        f'Expert Aggregation\n[B=?, S=?, D={token_dim}]\nGPU{gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='orange')
                
                # Connect all experts to all aggregation nodes
                for expert_out in expert_outputs:
                    dot.edge(expert_out, agg_node, constraint='false')
                
                expert_agg_nodes.append(agg_node)
            
            # Final residual add
            residual2_nodes = []
            for gpu_id in range(64):
                residual2_node = f'layer_{layer_id}_residual2_gpu{gpu_id}'
                dot.node(residual2_node,
                        f'Residual Add\n[B=?, S=?, D={token_dim}]\nGPU{gpu_id}',
                        shape='diamond', style='filled', fillcolor='lightgreen')
                dot.edge(residual1_nodes[gpu_id], residual2_node)
                dot.edge(expert_agg_nodes[gpu_id], residual2_node)
                residual2_nodes.append(residual2_node)
            
            # Layer output aggregation
            layer_output = f'layer_{layer_id}_output'
            dot.node(layer_output,
                    f'Layer {layer_id} Output\n[B={batch_size}, S={seq_len}, D={token_dim}]',
                    shape='ellipse', style='filled', fillcolor='lightgray')
            
            for gpu_id in range(64):
                dot.edge(residual2_nodes[gpu_id], layer_output)
    
    # Global output
    dot.node('global_output',
            f'Global Output\n[B={batch_size}, S={seq_len}, D={token_dim}]',
            shape='ellipse', style='filled', fillcolor='lightyellow')
    dot.edge('layer_3_output', 'global_output')
    
    return dot

def main():
    # Create output directory
    output_dir = '/home/wzc/data/file-share/2025-09-08-14-29-22'
    
    # Generate baseline DAG
    print("Generating baseline DAG...")
    baseline_dag = create_baseline_dag()
    
    # Save baseline DAG
    baseline_dot_path = os.path.join(output_dir, 'baseline_moe_dag.dot')
    baseline_svg_path = os.path.join(output_dir, 'baseline_moe_dag.svg')
    
    with open(baseline_dot_path, 'w') as f:
        f.write(baseline_dag.source)
    
    baseline_dag.render(baseline_dot_path.replace('.dot', ''), format='svg', cleanup=True)
    
    # Generate proposed DAG
    print("Generating proposed DAG...")
    proposed_dag = create_proposed_dag()
    
    # Save proposed DAG
    proposed_dot_path = os.path.join(output_dir, 'proposed_large_ep_moe_dag.dot')
    proposed_svg_path = os.path.join(output_dir, 'proposed_large_ep_moe_dag.svg')
    
    with open(proposed_dot_path, 'w') as f:
        f.write(proposed_dag.source)
    
    proposed_dag.render(proposed_dot_path.replace('.dot', ''), format='svg', cleanup=True)
    
    print(f"Generated DAGs:")
    print(f"- Baseline: {baseline_dot_path}, {baseline_svg_path}")
    print(f"- Proposed: {proposed_dot_path}, {proposed_svg_path}")
    
    # Create JSON summary
    import json
    summary = {
        "generated_dags": [
            {
                "name": "baseline_moe_dag",
                "description": "4-layer MoE Baseline with 16 GPUs, TP=8, PP=2, 4 experts per GPU",
                "dot_path": baseline_dot_path,
                "svg_path": baseline_svg_path,
                "total_gpus": 16,
                "experts_per_gpu": 4,
                "parallel_strategy": "TP=8, PP=2"
            },
            {
                "name": "proposed_large_ep_moe_dag", 
                "description": "4-layer MoE Large EP with 64 GPUs, EP=64, 1 expert per GPU",
                "dot_path": proposed_dot_path,
                "svg_path": proposed_svg_path,
                "total_gpus": 64,
                "experts_per_gpu": 1,
                "parallel_strategy": "EP=64"
            }
        ]
    }
    
    summary_path = os.path.join(output_dir, 'generated_dags_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

if __name__ == '__main__':
    result = main()
    print(json.dumps(result, indent=2))