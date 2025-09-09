#!/usr/bin/env python3

import graphviz

def create_proposed_dag():
    """
    Create proposed DAG for EP=64 with 64 GPUs
    Each GPU has exactly one expert
    """
    
    dot = graphviz.Digraph('proposed_moe_dag', comment='Proposed MoE DAG - EP=64, 64 GPUs, One Expert per GPU')
    dot.attr(rankdir='TB', size='30,40')
    
    # Define global dimensions
    batch_size = 1024
    seq_len = 10000
    hidden_size = 8192
    mlp_hidden_size = 32768
    num_heads = 16
    head_dim = 512
    num_experts = 16
    total_gpus = 64
    experts_per_gpu = 1
    
    # Input node
    dot.node('input', f'Total Input\\nbatch={batch_size}, seq={seq_len}, hidden={hidden_size}', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Create 4 layers with 16 experts each, distributed across 64 GPUs
    for layer_id in range(4):
        layer_label = f'Layer {layer_id} (EP=64, 16 experts across 64 GPUs)'
        
        with dot.subgraph(name=f'cluster_layer_{layer_id}') as layer_cluster:
            layer_cluster.attr(label=layer_label, style='rounded', fillcolor='lightgray')
            
            # Global token routing for this layer
            layer_cluster.node(f'global_gate_{layer_id}',
                             f'Global Gate Layer {layer_id}\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: routing map\\nAll GPUs',
                             shape='ellipse', style='filled', fillcolor='lightcyan')
            
            # Token splitting by expert destination
            layer_cluster.node(f'token_split_{layer_id}',
                             f'Token Split by Expert\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: expert-specific tokens\\nAll GPUs',
                             shape='parallelogram', style='filled', fillcolor='yellow')
            
            # Expert nodes - 16 experts distributed across 64 GPUs
            # Each expert gets 4 GPUs for redundancy, but only 1 active expert per GPU
            for expert_id in range(num_experts):
                # Calculate GPU range for this expert (4 GPUs per expert for load balancing)
                gpu_start = expert_id * 4
                
                # Expert-specific routing
                layer_cluster.node(f'expert_route_{layer_id}_{expert_id}',
                               f'Route to Expert {expert_id}\\nINPUT: tokens\\nOUTPUT: tokens for expert {expert_id}\\nCross-GPU Transfer',
                               shape='ellipse', style='filled', fillcolor='orange')
                
                # For each expert, show the 4 GPUs it can use
                for gpu_offset in range(4):
                    actual_gpu = gpu_start + gpu_offset
                    
                    # MHA for this expert's tokens (simplified - assuming tokens are routed to expert)
                    layer_cluster.node(f'mha_qkv_{layer_id}_{expert_id}_{gpu_offset}',
                                   f'MHA QKV Linear\\nExpert {expert_id}\\nGPU {actual_gpu}\\nINPUT: tokens\\nOUTPUT: qkv vectors',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    layer_cluster.node(f'mha_attn_{layer_id}_{expert_id}_{gpu_offset}',
                                   f'MHA Attention\\nExpert {expert_id}\\nGPU {actual_gpu}\\nINPUT: qkv vectors\\nOUTPUT: attention output',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    layer_cluster.node(f'mha_out_{layer_id}_{expert_id}_{gpu_offset}',
                                   f'MHA Output Linear\\nExpert {expert_id}\\nGPU {actual_gpu}\\nINPUT: attention output\\nOUTPUT: hidden states',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    # Expert MLP - one expert per GPU
                    layer_cluster.node(f'expert_gate_{layer_id}_{expert_id}_{gpu_offset}',
                                   f'Expert Gate\\nExpert {expert_id}\\nGPU {actual_gpu}\\nINPUT: hidden states\\nOUTPUT: expert selection',
                                   shape='parallelogram', style='filled', fillcolor='lightcyan')
                    
                    layer_cluster.node(f'expert_linear1_{layer_id}_{expert_id}_{gpu_offset}',
                                   f'Expert {expert_id} Linear 1\\nGPU {actual_gpu}\\nINPUT: {batch_size//4},variable_seq,{hidden_size}\\nOUTPUT: {batch_size//4},variable_seq,{mlp_hidden_size}',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    layer_cluster.node(f'expert_activation_{layer_id}_{expert_id}_{gpu_offset}',
                                   f'Expert {expert_id} GELU\\nGPU {actual_gpu}\\nINPUT: {batch_size//4},variable_seq,{mlp_hidden_size}\\nOUTPUT: {batch_size//4},variable_seq,{mlp_hidden_size}',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    layer_cluster.node(f'expert_linear2_{layer_id}_{expert_id}_{gpu_offset}',
                                   f'Expert {expert_id} Linear 2\\nGPU {actual_gpu}\\nINPUT: {batch_size//4},variable_seq,{mlp_hidden_size}\\nOUTPUT: {batch_size//4},variable_seq,{hidden_size}',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    layer_cluster.node(f'expert_weight_{layer_id}_{expert_id}_{gpu_offset}',
                                   f'Expert Weighting\\nExpert {expert_id}\\nGPU {actual_gpu}\\nINPUT: expert output, gate weights\\nOUTPUT: weighted output',
                                   shape='parallelogram', style='filled', fillcolor='yellow')
            
            # Token aggregation after expert processing
            layer_cluster.node(f'token_aggregate_{layer_id}',
                           f'Token Aggregate\\nINPUT: expert outputs from all GPUs\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nCross-GPU Gather',
                           shape='ellipse', style='filled', fillcolor='red')
            
            # Residual connections
            layer_cluster.node(f'residual1_{layer_id}',
                           f'Residual Add 1\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nAll GPUs',
                           shape='diamond', style='filled', fillcolor='pink')
            
            layer_cluster.node(f'residual2_{layer_id}',
                           f'Residual Add 2\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nAll GPUs',
                           shape='diamond', style='filled', fillcolor='pink')
    
    # Output node
    dot.node('output', f'Total Output\\nbatch={batch_size}, seq={seq_len}, hidden={hidden_size}', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect the flow
    # Input to Layer 0
    dot.edge('input', 'global_gate_0')
    dot.edge('global_gate_0', 'token_split_0')
    
    # Layer 0 connections
    for expert_id in range(num_experts):
        dot.edge('token_split_0', f'expert_route_0_{expert_id}')
        
        for gpu_offset in range(4):
            dot.edge(f'expert_route_0_{expert_id}', f'mha_qkv_0_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_qkv_0_{expert_id}_{gpu_offset}', f'mha_attn_0_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_attn_0_{expert_id}_{gpu_offset}', f'mha_out_0_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_out_0_{expert_id}_{gpu_offset}', f'expert_gate_0_{expert_id}_{gpu_offset}', style='dashed')
            dot.edge(f'expert_gate_0_{expert_id}_{gpu_offset}', f'expert_linear1_0_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_linear1_0_{expert_id}_{gpu_offset}', f'expert_activation_0_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_activation_0_{expert_id}_{gpu_offset}', f'expert_linear2_0_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_linear2_0_{expert_id}_{gpu_offset}', f'expert_weight_0_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_weight_0_{expert_id}_{gpu_offset}', f'token_aggregate_0')
    
    dot.edge('token_aggregate_0', 'residual1_0')
    dot.edge('residual1_0', 'residual2_0')
    
    # Connect Layer 0 to Layer 1
    dot.edge('residual2_0', 'global_gate_1')
    dot.edge('global_gate_1', 'token_split_1')
    
    # Layer 1 connections
    for expert_id in range(num_experts):
        dot.edge('token_split_1', f'expert_route_1_{expert_id}')
        
        for gpu_offset in range(4):
            dot.edge(f'expert_route_1_{expert_id}', f'mha_qkv_1_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_qkv_1_{expert_id}_{gpu_offset}', f'mha_attn_1_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_attn_1_{expert_id}_{gpu_offset}', f'mha_out_1_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_out_1_{expert_id}_{gpu_offset}', f'expert_gate_1_{expert_id}_{gpu_offset}', style='dashed')
            dot.edge(f'expert_gate_1_{expert_id}_{gpu_offset}', f'expert_linear1_1_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_linear1_1_{expert_id}_{gpu_offset}', f'expert_activation_1_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_activation_1_{expert_id}_{gpu_offset}', f'expert_linear2_1_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_linear2_1_{expert_id}_{gpu_offset}', f'expert_weight_1_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_weight_1_{expert_id}_{gpu_offset}', f'token_aggregate_1')
    
    dot.edge('token_aggregate_1', 'residual1_1')
    dot.edge('residual1_1', 'residual2_1')
    
    # Connect Layer 1 to Layer 2
    dot.edge('residual2_1', 'global_gate_2')
    dot.edge('global_gate_2', 'token_split_2')
    
    # Layer 2 connections (same pattern as Layer 0)
    for expert_id in range(num_experts):
        dot.edge('token_split_2', f'expert_route_2_{expert_id}')
        
        for gpu_offset in range(4):
            dot.edge(f'expert_route_2_{expert_id}', f'mha_qkv_2_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_qkv_2_{expert_id}_{gpu_offset}', f'mha_attn_2_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_attn_2_{expert_id}_{gpu_offset}', f'mha_out_2_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_out_2_{expert_id}_{gpu_offset}', f'expert_gate_2_{expert_id}_{gpu_offset}', style='dashed')
            dot.edge(f'expert_gate_2_{expert_id}_{gpu_offset}', f'expert_linear1_2_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_linear1_2_{expert_id}_{gpu_offset}', f'expert_activation_2_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_activation_2_{expert_id}_{gpu_offset}', f'expert_linear2_2_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_linear2_2_{expert_id}_{gpu_offset}', f'expert_weight_2_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_weight_2_{expert_id}_{gpu_offset}', f'token_aggregate_2')
    
    dot.edge('token_aggregate_2', 'residual1_2')
    dot.edge('residual1_2', 'residual2_2')
    
    # Connect Layer 2 to Layer 3
    dot.edge('residual2_2', 'global_gate_3')
    dot.edge('global_gate_3', 'token_split_3')
    
    # Layer 3 connections
    for expert_id in range(num_experts):
        dot.edge('token_split_3', f'expert_route_3_{expert_id}')
        
        for gpu_offset in range(4):
            dot.edge(f'expert_route_3_{expert_id}', f'mha_qkv_3_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_qkv_3_{expert_id}_{gpu_offset}', f'mha_attn_3_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_attn_3_{expert_id}_{gpu_offset}', f'mha_out_3_{expert_id}_{gpu_offset}')
            dot.edge(f'mha_out_3_{expert_id}_{gpu_offset}', f'expert_gate_3_{expert_id}_{gpu_offset}', style='dashed')
            dot.edge(f'expert_gate_3_{expert_id}_{gpu_offset}', f'expert_linear1_3_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_linear1_3_{expert_id}_{gpu_offset}', f'expert_activation_3_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_activation_3_{expert_id}_{gpu_offset}', f'expert_linear2_3_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_linear2_3_{expert_id}_{gpu_offset}', f'expert_weight_3_{expert_id}_{gpu_offset}')
            dot.edge(f'expert_weight_3_{expert_id}_{gpu_offset}', f'token_aggregate_3')
    
    dot.edge('token_aggregate_3', 'residual1_3')
    dot.edge('residual1_3', 'residual2_3')
    dot.edge('residual2_3', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-09-11-32-31/proposed_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-09-11-32-31/proposed_moe_dag.dot')
    print("Proposed DAG generated successfully")